"""Optional classical-controller deployment comparisons (separate environment)."""

from pathlib import Path
import os

import jax
import jax.numpy as jnp
import numpy as np

from smallsat_sim.controllers.lqr.controller import LQRController
from smallsat_sim.envs.vehicles.astrobee_benchmark.env import AstrobeeBenchmarkEnv
from smallsat_sim.envs.vehicles.astrobee_rl import config as rl_config
from smallsat_sim.envs.effects.disturbances import DisturbanceList, ConstantForceDisturbance
from smallsat_sim.planners.oracle.oracle import OraclePlanner
from smallsat_sim.utils.helpers import (
    get_args,
    calc_attitude_error,
    calc_lateral_tracking_error,
)

FAULT_ONSET_STEP = 100


class Benchmarker:
    def __init__(self, run_name="default"):
        self.args = get_args()
        self.run_name = run_name
        self._repo_root = Path(__file__).resolve().parents[2]

    def deploy_and_test_classic(self, controller_type: str) -> None:
        """
        Deploy and test classic controllers (Nominal MPC, LQR) on the oracle trajectory.
        """
        rl_cfg = rl_config.resolve_config().training

        def _seed_for_stage(stage_name: str) -> jax.random.PRNGKey:
            base_seed = int(getattr(env.env_cfg.sim, "seed", 0))
            stage_idx = stages.index(stage_name)
            seed = base_seed + stage_idx
            env.np_rng.seed(seed)
            return jax.random.PRNGKey(seed)

        def _save_video(env, stage_name: str) -> None:
            if self.args.video:
                video_dir = os.path.join(
                    str(self._repo_root),
                    "experiments",
                    "rl_results",
                    self.run_name,
                    stage_name,
                    "videos",
                )
                output_name = f"{self.run_name}_{stage_name}_run{env.run_id:04d}"
                env.get_sim_rendering(output_name, output_dir=video_dir)

        def _log_step(env, planner, stage_name: str) -> None:
            if not hasattr(env, "logger"):
                return
            obs = env.get_obs()
            tracking_error = float(
                calc_lateral_tracking_error(obs=obs, planner=planner)
            )
            angle_error = float(calc_attitude_error(np.array([1, 0, 0, 0]), obs[3:7]))
            env.logger.log(
                env.run_id,
                float(env.data.time),
                run_name=self.run_name,
                stage=stage_name,
                mean_lateral_error=tracking_error,
                mean_angle_error=angle_error,
                mean_extrinsic_error=0.0,
            )

        def _apply_stage_perturbation(env, stage_name: str, start_time: float) -> None:
            if stage_name == "stuck_off_deployment":
                env.perturbations.perturbations[0].stuck_off_thruster(
                    index=None, start_time=start_time
                )
            elif stage_name == "stuck_on_deployment":
                env.perturbations.perturbations[1].stuck_on_thruster(
                    index=None, start_time=start_time
                )
            elif stage_name == "faulty_valve_deployment":
                env.perturbations.perturbations[2].register_perturbation(
                    index=None, start_time=start_time
                )
            elif stage_name == "saturated_thrust_deployment":
                env.perturbations.perturbations[3].register_perturbation(
                    index=None, start_time=start_time
                )
            elif stage_name == "thrust_instability_deployment":
                env.perturbations.perturbations[4].register_perturbation(
                    index=None, start_time=start_time
                )
            elif stage_name == "constant_force_disturbances_deployment":
                disturbance_key = _seed_for_stage(stage_name)
                env.disturbances = DisturbanceList(
                    [ConstantForceDisturbance(env.env_cfg, disturbance_key)]
                )
                env.disturbances.disturbances[0].const_force_disturbance(
                    start_time=start_time
                )

        if controller_type not in {"nominal_mpc", "lqr"}:
            raise ValueError("controller_type must be one of: 'nominal_mpc', 'lqr'.")

        # Create environment
        env = AstrobeeBenchmarkEnv(args=self.args)

        # Create planner (oracle trajectory matching RL layout)
        planner = OraclePlanner(
            env,
            radius=rl_cfg.deployment_radius,
            spacing=rl_cfg.deployment_spacing,
            clearance_dist=0.2,
            plane="xy",
            z_offset=10.17,
        )

        # Create controller
        if controller_type == "nominal_mpc":
            from smallsat_sim.controllers.nominal_mpc.controller import NominalMPCController

            ctrl = NominalMPCController(env, planner)
        else:
            # Benchmark override: use MPC-style costs so LQR actually tracks in nominal runs.
            env.env_cfg.control.LQR.cost.Q = (
                env.env_cfg.control.NominalMPC.cost.Q.copy()
            )
            env.env_cfg.control.LQR.cost.R = (
                env.env_cfg.control.NominalMPC.cost.R.copy()
            )
            ctrl = LQRController(env, planner)

        stages = [
            "deployment",
            "stuck_off_deployment",
            "stuck_on_deployment",
            "faulty_valve_deployment",
            "saturated_thrust_deployment",
            "thrust_instability_deployment",
            "constant_force_disturbances_deployment",
        ]

        for stage_name in stages:
            env.reset()
            env.reset_perturbations()
            env.disturbances = None
            planner.idx_reference_point = 0
            _seed_for_stage(stage_name)

            env.reset_to_state(
                pos=np.array(rl_cfg.deployment_init_pos),
                att=np.array([0.0, 0.0, 0.0]),
            )

            step = 0
            max_steps = rl_cfg.deployment_len
            # Stop at the simulation time limit or configured control-step limit.
            while env.data.time <= env.env_cfg.sim.max_sim_time:
                if max_steps is not None and step >= int(max_steps):
                    break
                # Allow nominal tracking before introducing the stage fault.
                if step == FAULT_ONSET_STEP:
                    _apply_stage_perturbation(env, stage_name, float(env.data.time))

                # Advance one control step and capture requested video and metrics.
                ctrl_input = ctrl.get_control_input(env)
                env.step(input=ctrl_input)
                if self.args.video:
                    if (
                        env.data.time >= env.env_cfg.renderer.start_recording
                        and env.data.time <= env.env_cfg.renderer.end_recording
                    ):
                        env._update_renderer()
                _log_step(env, planner, stage_name)
                step += 1

            _save_video(env, stage_name)

        # Save log if logging is enabled
        if self.args.log:
            env.logger.save_log()

        env.close()
