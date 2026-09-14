"""User-facing vector environment lifecycle and runtime state."""
from .config import build_step_config, validate_physics
from .transition import evaluate_transition

from smallsat_sim.envs.rendering.vector import VectorRendering

from smallsat_sim.envs.effects.custom import validate_custom_replay
from smallsat_sim.envs.effects import scheduling
from argparse import Namespace
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from mujoco import mjx
from smallsat_sim.envs.rendering.rollout import RLVisualization
from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.envs.vec_env.freeflyer import from_mjx
from smallsat_sim.envs.vec_env.observations import (
    mjx_state_features,
    mjx_observations,
)
from smallsat_sim.envs.vec_env.types import FreeFlyerVecEnvState, VecEnvState
from smallsat_sim.model.mujoco_xml import build_mujoco_xml

from smallsat_sim.envs.vec_env.mjx_backend import (
    advance_physics,
    prepare_effects,
    reset_from_key,
)


class VecEnv(VectorRendering, BaseEnv):
    """Mutable host façade over pure vector simulation kernels."""

    def __init__(self, args) -> None:
        self.use_wandb = args.wandb

        self.train_with_failures = self.env_cfg.environment.train_with_failures

        self.use_adaptive_approach = self.env_cfg.context.enabled

        self.history_len = self.env_cfg.context.history_len
        self.max_episode_len = self.env_cfg.max_episode_len
        super().__init__(args)

        self.run_id = self.env_cfg.run_id

        self.num_envs = self.env_cfg.environment.num_envs

        # Mixer maps thruster commands to body-frame wrench
        mixer = jnp.asarray(self.symbolic_model.mixer, dtype=jnp.float32)
        self._thruster_mixer = jax.device_put(mixer)
        self._thruster_mixer_T = jax.device_put(mixer.T)

        self.using_rl = True

        # Observation and action spaces
        self.obs_dim = 12
        self.act_dim = int(self.model.nu)
        self.adaptive_context_mode = self.env_cfg.context.mode
        if self.adaptive_context_mode != "wrench_context":
            raise ValueError("Only wrench_context adaptation is supported")
        self.ext_dim = self.res_dim = 6 if self.use_adaptive_approach else 0

        self.init_qpos = self.mjx_batch.qpos
        self.init_qvel = self.mjx_batch.qvel

        self.max_start_offset = self.env_cfg.Bodies.max_start_offset
        self.max_start_linear_velocity = float(
            getattr(self.env_cfg.Bodies, "max_start_linear_velocity", 0.0)
        )
        self.max_start_angular_velocity = float(
            getattr(self.env_cfg.Bodies, "max_start_angular_velocity", 0.0)
        )

        self.jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
        self.jit_forward = jax.jit(jax.vmap(mjx.forward, in_axes=(None, 0)))

        # Cache for the reward breakdown after each transition (used for logging)
        self._last_reward_components: dict[str, jnp.ndarray] = {}
        self.collect_reward_components = bool(
            getattr(self.env_cfg.environment, "collect_reward_components", False)
        )
        self.disturbance_states = ()
        self.perturbation_states = ()
        self._terminal_hold_counts = jnp.zeros((self.num_envs,), dtype=jnp.int32)
        self.reset()

    def next_rng_keys(self, count: int = 1) -> jnp.ndarray:
        """Draw ``count`` fresh PRNG keys from the environment stream."""
        if count < 1:
            raise ValueError("count must be >= 1")
        splits = jax.random.split(self._rng, count + 1)
        self._rng = splits[0]
        return splits[1:]

    def reset(self) -> None:
        """Reset the agent in all the environment instances, while randomizing the initial position."""
        new_state = reset_from_key(
            self._rng,
            mjx_model=self.mjx_model,
            init_qpos=self.init_qpos,
            init_qvel=self.init_qvel,
            num_envs=self.num_envs,
            max_start_offset=self.max_start_offset,
            max_start_linear_velocity=self.max_start_linear_velocity,
            max_start_angular_velocity=self.max_start_angular_velocity,
        )
        self._rng = new_state.rng
        self.mjx_batch = new_state.mjx_batch
        self._terminal_hold_counts = new_state.terminal_hold_counts
        self.mjx_data = self.mjx_data.replace(
            qpos=self.mjx_batch.qpos[0], qvel=self.mjx_batch.qvel[0]
        )
        self._refresh_effect_states()

    def transition(
        self,
        actions: jnp.ndarray,
        states_res: jnp.ndarray,
        next_waypoint: jnp.ndarray,
        iter: int | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply input action on the environment. Returns the rewards and whether the terminal state has been reached."""

        previous = self.state_struct
        previous_features = states_res[:, :12]
        previous_residual = states_res[:, 12:12 + self.res_dim] if states_res.shape[1] >= 12 + self.res_dim else None
        self.step(input=actions)
        config = self.build_step_config()
        termination, reward = evaluate_transition(
            previous_features, self.get_states(next_waypoint), actions,
            previous.terminal_hold_counts, previous.mjx_batch.actuator_force,
            previous.mjx_batch.ctrl, self.get_actual_wrench(), config, previous_residual)
        self._terminal_hold_counts = termination.terminal_hold_counts
        self._last_reward_components = dict(reward.components)
        return reward.rewards, termination.terminals

    def get_last_reward_components(self) -> dict[str, jnp.ndarray]:
        """Return the shaped reward and penalty breakdown from the most recent transition."""
        return self._last_reward_components

    def _refresh_effect_states(self) -> None:
        """
        Capture snapshots of the current disturbance and perturbation objects.
        These will later feed the functional rollout helpers.
        """
        if hasattr(self, "disturbances") and self.disturbances is not None:
            self.disturbance_states = tuple(
                getattr(dist, "state", None) for dist in self.disturbances.disturbances
            )
        else:
            self.disturbance_states = ()

        if hasattr(self, "perturbations") and self.perturbations is not None:
            self.perturbation_states = tuple(
                getattr(pert, "state", None)
                for pert in self.perturbations.perturbations
            )
        else:
            self.perturbation_states = ()

    def step(self, input) -> None:
        """Simulate environments for one timestep."""
        self._pre_physics_step(input)

        decimation = self.env_cfg.environment.control_decimation
        if self.viewer is None:
            self.mjx_batch = advance_physics(self.mjx_model, self.mjx_batch, decimation)
        else:
            torque_world = jnp.einsum("bij,bj->bi", self.mjx_batch.xmat[:, 1],
                                     self.mjx_batch.qfrc_applied[:, 3:])
            for substep in range(decimation):
                if substep % self.env_cfg.viewer.viewer_decimation == 0:
                    mjx.get_data_into(self.data_vec, self.model, self.mjx_batch)
                    self._update_viewer()
                torque_body = jnp.einsum("bji,bj->bi", self.mjx_batch.xmat[:, 1], torque_world)
                self.mjx_batch = self.jit_step(self.mjx_model, self.mjx_batch.replace(
                    qfrc_applied=self.mjx_batch.qfrc_applied.at[:, 3:].set(torque_body)))

        self._post_physics_step()

    def get_obs(self) -> jnp.ndarray:
        """Return position, quaternion, body velocity and angular velocity."""
        return mjx_observations(self.mjx_batch)

    def get_states(self, next_waypoint: jnp.ndarray) -> jnp.ndarray:
        """Return position/attitude errors and body velocities for the reference."""
        return mjx_state_features(self.mjx_batch, next_waypoint)

    @property
    def state_struct(self) -> VecEnvState:
        """Expose the current environment state as a lightweight dataclass."""
        self._refresh_effect_states()
        return VecEnvState(self._rng, self.mjx_batch, self._terminal_hold_counts,
                           self.disturbance_states, self.perturbation_states)

    def apply_state_struct(self, state: VecEnvState) -> None:
        """Overwrite the imperative environment state with the provided functional snapshot."""
        self._refresh_effect_states()
        validate_custom_replay(self.perturbation_states, state.perturbation_states)
        validate_custom_replay(self.disturbance_states, state.disturbance_states)
        current = self.state_struct
        if jax.tree.structure(current.mjx_batch) != jax.tree.structure(state.mjx_batch):
            raise ValueError("Snapshot physics structure does not match this environment")
        for expected, restored in zip(jax.tree.leaves(current.mjx_batch),
                                      jax.tree.leaves(state.mjx_batch), strict=True):
            if expected.shape != restored.shape or expected.dtype != restored.dtype:
                raise ValueError("Snapshot physics shapes/dtypes do not match this environment")
        if state.terminal_hold_counts.shape != (self.num_envs,):
            raise ValueError("Snapshot terminal counters must have one entry per environment")
        if state.rng.shape != current.rng.shape or state.rng.dtype != current.rng.dtype:
            raise ValueError("Snapshot RNG layout does not match this environment")
        if state.terminal_hold_counts.dtype != current.terminal_hold_counts.dtype:
            raise ValueError("Snapshot terminal counter dtype does not match this environment")
        self._adopt_state(state)

    def _adopt_state(self, state):
        """Publish a validated or internally produced state to host adapters."""
        self._rng = state.rng
        self.mjx_batch = state.mjx_batch
        self._terminal_hold_counts = state.terminal_hold_counts
        self.disturbance_states = state.disturbance_states
        self.perturbation_states = state.perturbation_states

        if hasattr(self, "disturbances") and self.disturbances is not None:
            for obj, snapshot in zip(
                self.disturbances.disturbances,
                state.disturbance_states,
                strict=True,
            ):
                obj.state = snapshot
        if hasattr(self, "perturbations") and self.perturbations is not None:
            for obj, snapshot in zip(
                self.perturbations.perturbations,
                state.perturbation_states,
                strict=True,
            ):
                obj.restore_state(snapshot)
        self._refresh_effect_states()

    def verify_functional_step(self, *args, **kwargs) -> None:
        """Run the opt-in imperative/functional parity diagnostic."""
        from .diagnostics import verify_functional_step
        verify_functional_step(self, *args, **kwargs)

    def build_step_config(self, *, effects_enabled=True):
        """Capture task and physics parameters for a compiled step."""
        self._refresh_effect_states()
        return build_step_config(self, effects_enabled=effects_enabled)

    def rollout_backend(self):
        """Select the internal step/reset/features interface once, before tracing."""
        from . import mjx_backend, freeflyer
        backend = self.env_cfg.environment.rollout_backend
        if backend == "mjx":
            return mjx_backend
        if backend == "freeflyer":
            return freeflyer
        raise ValueError("rollout_backend must be mjx or freeflyer")

    def freeflyer_state_struct(self) -> FreeFlyerVecEnvState:
        return from_mjx(self.state_struct)


    def schedule_random_faults(
        self,
        key,
        fraction_perturbed_envs: float,
        perturbation_distribution=None,
        start_time: float | None = None,
    ) -> None:
        """Sample actuator faults, preferring rows without existing failures."""
        return scheduling.schedule_random_faults(
            self,
            key=key,
            fraction_perturbed_envs=fraction_perturbed_envs,
            perturbation_distribution=perturbation_distribution,
            start_time=start_time,
        )

    def schedule_random_disturbances(
        self,
        key,
        fraction_disturbed_envs: float,
        start_time: float | None = None,
    ) -> None:
        """Activate constant forces, preferring rows without actuator faults."""
        return scheduling.schedule_random_disturbances(
            self,
            key=key,
            fraction_disturbed_envs=fraction_disturbed_envs,
            start_time=start_time,
        )

    def set_constant_wrench_disturbances(
        self,
        env_indices: jnp.ndarray,
        wrenches: jnp.ndarray,
        start_time: float = 0.0,
    ) -> None:
        """Set deterministic world-frame wrenches for the selected rows."""
        return scheduling.set_constant_wrench_disturbances(
            self,
            env_indices=env_indices,
            wrenches=wrenches,
            start_time=start_time,
        )

    def get_desired_wrench(self, ctrl: jnp.ndarray) -> jnp.ndarray:
        """Compute the net body-frame wrench generated by the commanded thruster forces."""
        ctrl = jnp.asarray(ctrl, dtype=self._thruster_mixer_T.dtype)
        ctrl = jnp.atleast_2d(ctrl)
        wrench = ctrl @ self._thruster_mixer_T
        return wrench

    def get_actual_wrench(self) -> jnp.ndarray:
        """Return the body-frame wrench computed from the forces actually applied by MuJoCo."""
        actuator_force = jnp.asarray(
            self.mjx_batch.actuator_force, dtype=self._thruster_mixer_T.dtype
        )
        actuator_force = jnp.atleast_2d(actuator_force)
        wrench = actuator_force @ self._thruster_mixer_T
        return wrench


    def _setup_sim(self, args: Namespace):
        """
        Prepares simulation according to args.
        Creates a viewer depending on headless flag.
        """
        # Some subclasses may call into BaseEnv before VecEnv.__init__ has assigned num_envs.
        # Fall back to the configured value so batching still works.
        num_envs = getattr(self, "num_envs", self.env_cfg.environment.num_envs)
        self.num_envs = num_envs

        xml = build_mujoco_xml(self.env_cfg, self.model_cfg, scene="training")

        self.model = mujoco.MjModel.from_xml_string(xml)
        validate_physics(self.model, self.env_cfg.environment.rollout_backend)
        self.data = mujoco.MjData(self.model)

        self.mjx_model = mjx.put_model(
            self.model
        )  # Uses the installed MJX backend; add the `warp` extra for Warp.
        self.mjx_data = mjx.put_data(self.model, self.data)
        jax.block_until_ready(self.mjx_data.qpos)

        # Keep this visible because accidentally running RL on CPU is costly.
        print("Devices available to JAX: ", jax.devices())
        print("Device used by JAX: ", self.mjx_data.qpos.devices(), "\n")

        # Batch the data and randomize the starting position
        rng = self.next_rng_keys(num_envs)
        self.mjx_batch = jax.vmap(  # The initial position is randomized when the env is reset (at init and after each epoch)
            lambda rng: self.mjx_data.replace(qpos=self.mjx_data.qpos)
        )(
            rng
        )
        jax.block_until_ready(self.mjx_batch.qpos)


        self._rl_visualization = None
        self.viewer = None
        self.renderer = None
        self.data_vec = None
        self._update_viewer = lambda *args, **kwargs: None
        self._update_renderer = lambda *args, **kwargs: None
        if getattr(args, "viewer", False) or args.video:
            self._rl_visualization = RLVisualization(self.model, args, self.env_cfg.renderer)
        elif not args.headless:
            self.data_vec = mjx.get_data(self.model, self.mjx_batch)
            self._create_viewer(args)
            self._update_viewer = self.visualization.update_viewer


    def _pre_physics_step(self, input: jnp.ndarray) -> None:
        """Prepares the environment for the simulation step in MuJoCo."""
        state, control, force = prepare_effects(self.state_struct, jnp.asarray(input))
        self._adopt_state(state.replace(mjx_batch=state.mjx_batch.replace(
            ctrl=control, qfrc_applied=force)))

    def _post_physics_step(self):
        """Read vector observations directly; host viewer data may be stale."""
        self.obs = self.obs_gt = self.get_obs()
