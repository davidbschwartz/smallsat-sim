"""Classical MuJoCo environment base class and shared setup."""

from smallsat_sim.envs.rendering.classical import ClassicalRendering
from smallsat_sim.envs.config import resolve_env_config
from copy import deepcopy
from smallsat_sim.model.vehicle import load_vehicle, validate_vehicle
from typing import TypeVar
import numpy as np
import mujoco
import jax
import jax.numpy as jnp
from datetime import datetime

from smallsat_sim.model.dynamics import SymbolicModel
from smallsat_sim.model.mujoco_xml import build_mujoco_xml
from smallsat_sim.envs.effects.disturbances import DisturbanceList
from smallsat_sim.envs.effects.classical import PerturbationList
from smallsat_sim.utils.logger import Logger

from argparse import Namespace
from typing import Optional


T = TypeVar("T", np.ndarray, jnp.ndarray)


class BaseEnv(ClassicalRendering):
    def __init__(self, args: Namespace) -> None:
        # Initialize arguments
        self.args = args

        # Environment-owned random streams derived from configuration seed
        self.np_rng = np.random.RandomState(self.env_cfg.sim.seed)
        self._rng = jax.random.PRNGKey(self.env_cfg.sim.seed)
        self._rng, self._noise_key = jax.random.split(self._rng)

        # Setup simulation environment
        self._setup_sim(args)

        # Create symbolic model
        self.symbolic_model = SymbolicModel(self.model_cfg)

        # Initialize disturbance and perturbation to None as default setting
        self.disturbances = None
        self.disturbances_keycodes = DisturbanceList([]).keycode_dict.keys()
        self.perturbations = None
        self.perturbations_keycodes = PerturbationList([]).keycode_dict.keys()

        # Initialize observations
        self.set_obs(v_frame=self.env_cfg.sim.obs.v_frame)

        # Flag to know whether VecEnv is being used
        self.using_rl = False

        # Save time of simulation start (for filenames)
        self.sim_start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Initialize MC iterations
        self.run_id = 0

        # Initialize logger if enabled
        if args.log:
            self.logger = Logger(log_name=self.sim_start_time)

    def reset(self) -> None:
        """
        Resets environment.
        """
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.set_obs(v_frame=self.env_cfg.sim.obs.v_frame)
        print("Environment reset.")

    def reset_to_state(self, pos: np.ndarray, att: np.ndarray) -> None:
        """
        Resets environment to a desired state.

        NOTE: att is in euler angles [roll, pitch, yaw] in radians.
        """

        # Reset position and attitude
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = pos
        # Need to convert att to a quaternion for MuJoCo
        euler = np.array(att, dtype=float)
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, euler, "XYZ")

        self.data.qpos[3:7] = quat

        mujoco.mj_forward(self.model, self.data)

        self.set_obs(v_frame=self.env_cfg.sim.obs.v_frame)
        print("Environment reset.")

    def step(self, input: np.ndarray) -> None:
        """
        Simulate environment for one timestep.
        """
        # Prepare env for simulation step
        self._pre_physics_step(input)

        # Advance simulation
        for substep in range(self.env_cfg.control.control_decimation):
            # Update viewer
            if substep % self.env_cfg.viewer.viewer_decimation == 0:
                self._update_viewer()

            # Step in MuJoCo engine
            mujoco.mj_step(self.model, self.data)

        # Execute post physics steps
        self._post_physics_step()
        self._update_renderer()

    def set_obs(self, v_frame: str = "body") -> None:
        """
        Return all states

        args:
            v_frame (str): Specifies the frame of the velocity in the returned observation.
                             - "body": Return the velocity in the body frame.
                             - "inertial": Return the velocity in the inertial frame.

        returns:
            np.array: Array of observations containing position, orientation, velocity
                      and angular velocity.
        """
        # Read the named free joint, not the last body's derived arrays. qpos
        # and qvel contain the post-integration state even before mj_forward.
        from smallsat_sim.utils.helpers import Rquat
        body = int(self.model.body("body0").id)
        joint = int(self.model.body_jntadr[body])
        qadr = int(self.model.jnt_qposadr[joint])
        vadr = int(self.model.jnt_dofadr[joint])
        pose = self.data.qpos[qadr:qadr + 7].copy()
        velocity = self.data.qvel[vadr:vadr + 3].copy()
        if v_frame == "body":
            velocity = np.asarray(Rquat(pose[3:7])).T @ velocity
        elif v_frame != "inertial":
            raise ValueError(f"Unsupported velocity frame: {v_frame}")
        obs = np.r_[pose, velocity, self.data.qvel[vadr + 3:vadr + 6]]
        self.obs_gt = obs
        self.obs = self._apply_obs_noise(obs).copy()
        norm = np.linalg.norm(self.obs[3:7])
        if norm < 1e-12:
            raise ValueError("Observation quaternion must be nonzero")
        self.obs[3:7] /= norm

    def get_obs(self) -> T:
        """
        Returns the current (noisy) observations
        """
        return self.obs.copy()

    def get_obs_gt(self) -> T:
        """
        Return the GT observations. Use this method for
        visualization and for evaluations.
        """
        return self.obs_gt.copy()

    def _apply_obs_noise(self, obs: T) -> T:
        """
        Applies additive Gaussian noise on top of observations.
        For MuJoCo: obs are of type np.ndarray
        For MJX: obs are of type jnp.ndarray
        """
        if self.env_cfg.sim.noise.add_obs_noise:
            # Single agent in MuJoCo
            if isinstance(obs, np.ndarray):
                # Calculate noise
                noise_r = self.np_rng.normal(0, self.env_cfg.sim.noise.sigma_r, 3)
                noise_q = self.np_rng.normal(0, self.env_cfg.sim.noise.sigma_q, 4)
                noise_v = self.np_rng.normal(0, self.env_cfg.sim.noise.sigma_v, 3)
                noise_w = self.np_rng.normal(0, self.env_cfg.sim.noise.sigma_w, 3)

                # Concatenate noise
                noise = np.concatenate((noise_r, noise_q, noise_v, noise_w))

                # Add noise to observations and return
                return obs + noise

            # Multiple agents in MJX
            else:
                # Set the PRNG keys
                num_envs = obs.shape[0]
                self._noise_key, *noise_subkeys = jax.random.split(
                    self._noise_key, num=5
                )
                noise_r = jax.random.multivariate_normal(
                    noise_subkeys[0],
                    jnp.zeros((num_envs, 3)),
                    self.env_cfg.sim.noise.sigma_r * jnp.identity(3),
                )
                noise_q = jax.random.multivariate_normal(
                    noise_subkeys[1],
                    jnp.zeros((num_envs, 4)),
                    self.env_cfg.sim.noise.sigma_q * jnp.identity(4),
                )
                noise_v = jax.random.multivariate_normal(
                    noise_subkeys[2],
                    jnp.zeros((num_envs, 3)),
                    self.env_cfg.sim.noise.sigma_v * jnp.identity(3),
                )
                noise_w = jax.random.multivariate_normal(
                    noise_subkeys[3],
                    jnp.zeros((num_envs, 3)),
                    self.env_cfg.sim.noise.sigma_w * jnp.identity(3),
                )

                # Concatenate noise along the second dimension
                noise = jnp.concatenate((noise_r, noise_q, noise_v, noise_w), axis=1)

                # Add noise to observations and return
                return obs + noise
        else:
            return obs


    def _load_cfg(
        self, env_name: str, model_name: str | None = None, *, vehicle=None, config=None
    ) -> tuple:
        """
        Load environment settings and the selected physical vehicle independently.

        Episode poses stay in env_cfg.Bodies; an asset does not replace reset settings.
        """
        # Save env and model names for later use
        self.env_name = env_name
        self.model_name = model_name

        # Resolve fresh settings without importing per-vehicle modules.
        if config is None:
            env_cfg = resolve_env_config(env_name)
        else:
            env_cfg = deepcopy(config)

        asset_name = model_name or getattr(env_cfg, "model", None)
        if asset_name is None and vehicle is None:
            raise ValueError(
                "Model name not provided and environment config does not define 'model'."
            )

        model_cfg = (
            load_vehicle(f"vehicles/{asset_name}.yaml")
            if vehicle is None
            else validate_vehicle(vehicle)
        )

        return env_cfg, model_cfg

    def _setup_sim(self, args: Namespace) -> None:
        """
        Prepares simulation according to args.
        Creates a viewer depending on headless flag.
        """
        # Generate xml using env and model config files
        xml = build_mujoco_xml(self.env_cfg, self.model_cfg, scene=getattr(self, "scene", "gateway"))

        # Create model and data instances
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self._setup_rendering(args)


    def __del__(self):
        self.close()

    def _pre_physics_step(self, input: T) -> None:
        """
        Prepares the environment for the simulation step in MuJoCo.
        This includes:
            - Adding external disturbances
            - Adding perturbations to control input and model dynamics
            - ...
        """
        # External disturbances
        if self.disturbances:
            self.data.qfrc_applied = np.asarray(
                self.disturbances.apply(self.data.time).reshape(-1)
            )

        # Perturbations
        if self.perturbations:
            self.data.ctrl = np.asarray(
                self.perturbations.apply(jnp.asarray(input), self.data.time).reshape(-1)
            )
        else:
            self.data.ctrl = input

        # Log perturbed inputs
        if hasattr(self, "logger"):
            self.logger.log(self.run_id, self.data.time, u_actual=self.data.ctrl.copy())

    def _post_physics_step(self) -> None:
        """
        Executes actions after stepping simulation
        """
        # Fetch most recent observations
        self.set_obs(v_frame=self.env_cfg.sim.obs.v_frame)

    def _key_callback(self, keycode) -> None:
        """
        Callback function for keypressed detected in the MuJoCo viewer
        """
        try:
            if chr(keycode) in self.perturbations_keycodes:
                self.perturbations.key_callback(keycode)
            if chr(keycode) in self.disturbances_keycodes:
                self.disturbances.key_callback(keycode)
        except:
            print(
                "No disturbance or perturbation list registered. Keycallback unsuccessful."
            )
