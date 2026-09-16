"""Experiment runner setup and classical-controller native evaluation.

Learned policies use runner.evaluate() through the scenario/artifact adapter.
The classical evaluation loop owns paired scenario resets, fault onset,
stopping criteria, and artifact recording.
"""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
import os

import mujoco
import yaml
import numpy as np

from smallsat_sim.configuration import settings
from smallsat_sim.controllers.pd.controller import PDController
from smallsat_sim.controllers.rl.runners.factory import make_runner
from smallsat_sim.envs.effects.classical import PerturbationStatus, ThrusterFailureSimulator
from smallsat_sim.envs.vehicles.astrobee_rl.env import AstrobeeEnvVectorized
from smallsat_sim.model.dynamics import SymbolicModel
from smallsat_sim.planners.oracle.oracle_rl import OraclePlannerRL
from smallsat_sim.controllers.nominal_mpc.reference import MPCSolverError
from smallsat_sim.model.mujoco_xml import build_mujoco_xml
from smallsat_sim.model.vehicle import load_vehicle

from .common import evaluation_trial_count, plain
from .effects import training_effects
from .replay import EpisodeRecording
from .task import PoseMetrics, register_task, sample_trial


def vehicle(config, name, directory):
    path = Path(directory) / f"{name}_asset.yaml"
    path.write_text(yaml.safe_dump(config["assets"][name]))
    return load_vehicle(path)


def build_runner(config, job, run, *, saved_run=None):
    register_task()
    common = deepcopy(config["common"])
    env_config = settings(common["env"])
    training_config = settings(common["training"])
    algorithm = job["controller"]
    training_config.algorithm = algorithm
    if algorithm in config["protocol"].get("policy_hidden_sizes", {}):
        training_config.policy_hidden_sizes = deepcopy(config["protocol"]["policy_hidden_sizes"][algorithm])
    training_config.training_checkpoint_interval = config["protocol"].get(
        "checkpoint_intervals", {}
    ).get(algorithm, training_config.training_checkpoint_interval)
    training_config.checkpoint_dir = str((run.path / "checkpoint").resolve())
    if saved_run is not None:
        # Preserve checkpoint identity even when the archived run moved machines.
        saved_settings = yaml.safe_load((saved_run.path / "runtime_config.yaml").read_text())
        training_config.checkpoint_dir = saved_settings["training"]["checkpoint_dir"]
    env_config.sim.seed = job["seed"]
    env_config.model = job["spacecraft"]
    env_config.environment.num_envs = job.get(
        "batch_size", config["protocol"].get("training_num_envs", {}).get(
            algorithm, env_config.environment.num_envs
        )
    )
    env_config.max_episode_len = training_config[algorithm.upper()].max_ep_len
    env_config.environment.train_with_failures = job.get("regime") == "randomized"
    env_config.environment.custom_faults = []
    env_config.environment.custom_disturbances = []
    if env_config.environment.train_with_failures:
        env_config.environment.custom_faults, env_config.environment.custom_disturbances = (
            training_effects(common["train_distribution"])
        )
    if "batch_size" in job:
        steps = config["protocol"]["rollout_steps"]
        env_config.environment.rollout_backend = config["protocol"]["backend"]
        training_config.PPO.steps_per_epoch = steps
        training_config.PPO.max_ep_len = steps
        env_config.max_episode_len = steps
    wandb_mode = "disabled" if saved_run is not None else config["protocol"].get("wandb_mode", "disabled")
    if wandb_mode != "disabled":
        directory = (run.path / "wandb").resolve()
        directory.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_MODE"] = wandb_mode
        os.environ["WANDB_DIR"] = str(directory)
    args = SimpleNamespace(
        headless=True, num_bodies=1, log=False, wandb=wandb_mode != "disabled", video=False, viewer=False
    )
    asset = vehicle(config, job["spacecraft"], run.path)
    env = AstrobeeEnvVectorized(
        args=args, config=env_config, vehicle=asset,
        run_name=saved_run.meta["run_id"] if saved_run is not None else run.path.name,
    )
    try:
        (run.path / "model.xml").write_text(
            build_mujoco_xml(env.env_cfg, env.model_cfg, scene="training")
        )
        planner = OraclePlannerRL(env, radius=0.0)
        runner = make_runner(env, planner, config=training_config)
        run.update(training_num_envs=env.num_envs)
        (run.path / "runtime_config.yaml").write_text(
            yaml.safe_dump(plain({"env": env_config, "training": training_config}))
        )
        return runner
    except BaseException:
        env.close()
        raise


class NativeEnvironment:
    """Headless native MuJoCo adapter using the same vehicle and XML builder."""

    def __init__(self, config, name, run, *, scene="training"):
        self.model_cfg = vehicle(config, name, run.path)
        self.env_cfg = settings(deepcopy(config["common"]["classical"]))
        common = config["common"]
        self.env_cfg.model = name
        self.env_cfg.sim.dt = common["env"]["sim"]["dt"]
        self.env_cfg.sim.obs.v_frame = "inertial"
        self.env_cfg.Bodies.bodies_list[0].pos = common["reference"][:3]
        self.env_cfg.Bodies.bodies_list[0].euler = [0.0, 0.0, 0.0]
        decimation = common["env"]["environment"]["control_decimation"]
        self.env_cfg.control.PD.control_decimation = decimation
        self.env_cfg.control.PD.gains = settings(
            dict(zip(("Kp_x", "Kd_x", "Kp_q", "Kd_q"), common["pd_gains"][name]))
        )
        self.env_cfg.control.NominalMPC.control_decimation = decimation
        self.env_cfg.control.NominalMPC.Ts = decimation * self.env_cfg.sim.dt
        for key in ("Q", "R", "T"):
            self.env_cfg.control.NominalMPC.cost[key] = np.asarray(
                self.env_cfg.control.NominalMPC.cost[key]
            )
        self.env_cfg.control.NominalMPC.code_export_directory = str(
            (run.path / "mpc_solver").resolve()
        )
        xml = build_mujoco_xml(self.env_cfg, self.model_cfg, scene=scene)
        (run.path / "model.xml").write_text(xml)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.chaser = int(self.model.body("body0").id)
        self.nominal_mass = float(self.model.body_mass[self.chaser])
        self.nominal_inertia = self.model.body_inertia[self.chaser].copy()
        self.visualization = None
        self.using_rl = False
        self.run_id = 0
        self.args = SimpleNamespace(video=False)
        mujoco.mj_forward(self.model, self.data)

    def get_obs(self):
        return np.r_[self.data.qpos[:7], self.data.qvel[:6]]

    def reset(self, sample):
        mujoco.mj_resetData(self.model, self.data)
        self.model.body_mass[self.chaser] = self.nominal_mass * sample["mass_scale"]
        self.model.body_inertia[self.chaser] = self.nominal_inertia * sample["inertia_scale"]
        mujoco.mj_setConst(self.model, self.data)
        self.data.qpos[:7] = sample["initial_qpos"]
        self.data.qvel[:6] = sample["initial_qvel"]
        mujoco.mj_forward(self.model, self.data)


class SetpointPlanner:
    def __init__(self, reference):
        self.reference = np.asarray(reference)

    def get_reference(self, obs):
        return self.reference[:3, None], self.reference[3:7, None]


def controller(env, planner, name):
    if name == "pd":
        return PDController(env, planner)
    if name == "mpc":
        # Keep the optional acados dependency local so PD/RL work without the MPC extra.
        try:
            from smallsat_sim.controllers.nominal_mpc.controller import NominalMPCController
        except ImportError as error:
            raise RuntimeError(
                "MPC requires the repository mpc extra and native acados setup; see .setup/smallsat"
            ) from error
        env.symbolic_model = SymbolicModel(env.model_cfg)
        return NominalMPCController(env, planner)
    raise ValueError(f"Unknown classical controller {name}")


def fault_mapping(vehicle, condition, sample, common):
    """Reuse the existing seeded GP sampler; store its complete realized curve."""
    thruster_index = sample["affected_thruster"]
    max_thrust = vehicle.actuators[thruster_index].forcerange[1]
    fault = common["faults"]
    if condition in ("faulty_valve", "saturated_thrust", "thrust_instability"):
        sampler = ThrusterFailureSimulator(
            num_points=fault["num_points"],
            subset_size=fault["subset_size"],
            upper_bound=max_thrust,
            valve_min=fault["valve_min_fraction"] * max_thrust,
            valve_max=fault["valve_max_fraction"] * max_thrust,
            rng=np.random.RandomState(sample["effect_seed"]),
        )
        status = getattr(PerturbationStatus, condition.upper())
        # Freeze kernel choices explicitly rather than inheriting mutable profiles.
        lengthscale, outputscale = fault["profiles"][condition]
        sampler.failure_modes[status].update(lengthscale=lengthscale, outputscale=outputscale)
        x, y = sampler.generate_failure_data(status)
        sample["fault_curve_input"] = x.numpy()
        sample["fault_curve_output"] = y.numpy()

    def apply(command, time):
        applied = np.asarray(command).copy() * sample["thrust_scale"]
        if time < fault["onset_seconds"]:
            return applied
        if condition == "stuck_off":
            applied[thruster_index] = 0.0
        if condition == "stuck_on":
            applied[thruster_index] = max_thrust
        if "fault_curve_input" in sample:
            applied[thruster_index] = np.interp(
                applied[thruster_index], sample["fault_curve_input"], sample["fault_curve_output"]
            )
        return applied

    return apply


def evaluate(config, job, run):
    """Evaluate classical controllers with native MuJoCo."""
    evaluation_start = perf_counter()
    common, protocol = config["common"], config["protocol"]
    env = NativeEnvironment(config, job["spacecraft"], run)
    reference = np.asarray(common["reference"])
    planner = SetpointPlanner(reference)
    policy = controller(env, planner, job["controller"])
    conditions = (
        [job["condition"]] if "condition" in job else list(common["evaluation_distributions"])
    )
    decimation = common["env"]["environment"]["control_decimation"]
    run.update(task="pose_regulation", evaluation_num_envs=1, evaluation_backend="mujoco_native")
    for condition in conditions:
        distribution = common["evaluation_distributions"].get(condition, {})
        # Exp2 nominal includes the same randomized initial conditions as every fault.
        if protocol["experiment"] == "exp2_fault_robustness" and condition == "nominal":
            distribution = {}
        for trial in range(evaluation_trial_count(config, condition)):
            trial_start = perf_counter()
            seed = common["evaluation_seed_start"] + trial
            sample = sample_trial(common, seed, distribution, env.model.nu)
            env.reset(sample)
            recording = EpisodeRecording(env)
            if policy is not None and hasattr(policy, "_initialize_solver"):
                policy._initialize_solver(env)
            apply_fault = fault_mapping(env.model_cfg, condition, sample, common)
            run.append(dict(condition=condition, trial=trial, **sample), "samples.jsonl")
            metrics = PoseMetrics(common)
            reason = None
            solver_status = 0
            solve_times = []
            for _ in range(common["episode_steps"]):
                obs = env.get_obs()
                try:
                    command = policy.get_control_input(env)
                except MPCSolverError as error:
                    solver_status = error.status
                    reason = "solver_failure"
                    break
                if policy is not None and hasattr(policy, "ocp_solver"):
                    solver_status = int(policy.ocp_solver.status)
                    solve_times.append(float(policy.ocp_solver.get_stats("time_tot")))
                    if solver_status != 0:
                        reason = "solver_failure"
                        break
                command = np.clip(
                    command, env.model.actuator_ctrlrange[:, 0], env.model.actuator_ctrlrange[:, 1]
                )
                if not np.all(np.isfinite(command)):
                    reason = "nonfinite_control"
                    break
                env.data.ctrl[:] = apply_fault(command, env.data.time)
                env.data.xfrc_applied[env.chaser] = sample["wrench"]
                for _ in range(decimation):
                    mujoco.mj_step(env.model, env.data)
                obs = env.get_obs()
                if not np.all(np.isfinite(obs)):
                    reason = "nonfinite_state"
                    break
                recording.add(env)
                if metrics.add(obs, reference, command):
                    break
            recording.save(run.path / "recordings" / condition / f"{trial:04d}.npz")
            row = metrics.row(reason)
            if reason:
                row["success"] = False
            run.append(
                dict(
                    **{**job, "condition": condition},
                    trial=trial,
                    evaluation_seed=seed,
                    initial_condition_seed=seed,
                    training_seed=None,
                    solver_status=solver_status,
                    mean_solve_seconds=float(np.mean(solve_times)) if solve_times else None,
                    max_solve_seconds=max(solve_times) if solve_times else None,
                    affected_thruster=sample["affected_thruster"]
                    if condition
                    in (
                        "stuck_off",
                        "stuck_on",
                        "faulty_valve",
                        "saturated_thrust",
                        "thrust_instability",
                    )
                    else None,
                    fault_type=condition,
                    evaluation_wall_seconds=perf_counter() - trial_start,
                    fault_severity={
                        "curve": condition
                        in ("faulty_valve", "saturated_thrust", "thrust_instability"),
                        "valve_bounds": [
                            common["faults"]["valve_min_fraction"],
                            common["faults"]["valve_max_fraction"],
                        ],
                    },
                    normalized_control_effort=row["control_effort"]
                    / max(float(env.model.actuator_ctrlrange[:, 1].sum()), 1e-12),
                    **row,
                )
            )
    run.update(evaluation_wall_seconds=perf_counter() - evaluation_start)
