"""Port of demonstration-use-cases' Gateway contact characterization draft.

Contact is measured at every physics step and filtered to chaser/Gateway pairs.
No latch or hardware contact validation is implied.
"""

import math
from copy import deepcopy

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from smallsat_sim.controllers.nominal_mpc.reference import MPCSolverError
from smallsat_sim.planners.mission.docking import DockingPlanner

from .common import execute, parser
from .replay import EpisodeRecording
from .runtime import NativeEnvironment, controller
from .task import rotation_error


class ContactMetrics:
    def __init__(self, dt):
        self.dt = dt
        self.first = None
        self.peak = 0.0
        self.impulse = 0.0
        self.events = 0
        self.duration = 0.0
        self.previous = False

    def add(self, time, contact, force):
        if contact:
            if self.first is None:
                self.first = time
            self.events += int(not self.previous)
            self.duration += self.dt
            self.peak = max(self.peak, force)
            self.impulse += force * self.dt
        self.previous = contact

    def row(self):
        return dict(
            first_contact_time=self.first,
            peak_contact_force=self.peak,
            contact_impulse=self.impulse,
            contact_events=self.events,
            cumulative_contact_duration=self.duration,
        )


def contact_force(env, gateway):
    active = False
    total = 0.0
    force = np.zeros(6)
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        bodies = {
            int(env.model.geom_bodyid[contact.geom1]),
            int(env.model.geom_bodyid[contact.geom2]),
        }
        if bodies != {env.chaser, gateway}:
            continue
        mujoco.mj_contactForce(env.model, env.data, index, force)
        active = True
        total += float(np.linalg.norm(force[:3]))
    return active, total


def docking_sample(seed, p, pre_dock, attitude):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi)
    radius = rng.uniform(*p["initial_radius"])
    position = pre_dock + [
        radius * np.cos(theta),
        radius * np.sin(theta),
        rng.uniform(-p["initial_z_span"], p["initial_z_span"]),
    ]
    rotation = Rotation.from_euler(
        "xyz",
        rng.uniform(-p["initial_attitude_degrees"], p["initial_attitude_degrees"], 3),
        degrees=True,
    )
    rotation = rotation * Rotation.from_quat(np.roll(attitude, -1))
    return dict(
        initial_qpos=np.r_[position, np.roll(rotation.as_quat(), 1)],
        initial_qvel=np.r_[
            rng.uniform(-p["initial_linear_velocity"], p["initial_linear_velocity"], 3),
            rng.uniform(-p["initial_angular_velocity"], p["initial_angular_velocity"], 3),
        ],
        mass_scale=1.0,
        inertia_scale=1.0,
    )


def run_trial(env, ctrl, planner, p, seed, gateway, *, recording_path=None):
    sample = docking_sample(seed, p, planner.pre_dock_position, planner.dock_attitude)
    env.reset(sample)
    recording = EpisodeRecording(env)
    planner.reset()
    initial_depth = max(
        (
            max(0.0, -float(c.dist))
            for c in env.data.contact
            if {int(env.model.geom_bodyid[c.geom1]), int(env.model.geom_bodyid[c.geom2])}
            == {env.chaser, gateway}
        ),
        default=0.0,
    )
    if initial_depth > 1e-6:
        # Retain invalid scenarios as failures; never launch a deeply interpenetrating body.
        obs = env.get_obs()
        position = float(np.linalg.norm(obs[:3] - planner.dock_position))
        attitude = float(np.linalg.norm(rotation_error(obs[3:7], planner.dock_attitude)))
        if recording_path is not None:
            recording.save(recording_path)
        return (
            dict(
                evaluation_seed=seed,
                initial_condition_seed=seed,
                success=False,
                docking_success=False,
                rendezvous_success=False,
                settling_time=None,
                final_position_error=position,
                final_attitude_error=attitude,
                mean_position_error=position,
                mean_attitude_error=attitude,
                control_effort=0.0,
                episode_length=0,
                termination_reason="initial_penetration",
                initial_penetration_depth=initial_depth,
                solver_status=0,
                first_contact_time=0.0,
                peak_contact_force=None,
                contact_impulse=None,
                contact_events=0,
                cumulative_contact_duration=0.0,
            ),
            [],
            sample,
        )
    if hasattr(ctrl, "_initialize_solver"):
        ctrl._initialize_solver(env)
    contact = ContactMetrics(env.model.opt.timestep)
    hold = 0
    settling_time = None
    rendezvous = False
    effort = 0.0
    history = []
    success = False
    sim_dt = env.model.opt.timestep
    control_dt = sim_dt * p["control_decimation"]
    required = math.ceil(p["settling_hold_seconds"] / control_dt)
    solver_status = 0
    reason = None
    initial_obs = env.get_obs()
    position = float(np.linalg.norm(initial_obs[:3] - planner.dock_position))
    attitude = float(np.linalg.norm(rotation_error(initial_obs[3:7], planner.dock_attitude)))
    for step in range(p["steps"]):
        try:
            command = np.asarray(ctrl.get_control_input(env))
        except MPCSolverError as error:
            solver_status = error.status
            reason = "solver_failure"
            break
        if hasattr(ctrl, "ocp_solver"):
            solver_status = int(ctrl.ocp_solver.status)
            if solver_status != 0:
                reason = "solver_failure"
                break
        command = np.clip(
            command, env.model.actuator_ctrlrange[:, 0], env.model.actuator_ctrlrange[:, 1]
        )
        if not np.all(np.isfinite(command)):
            raise FloatingPointError("Nonfinite docking control")
        env.data.ctrl[:] = command
        effort += float(np.abs(command).sum()) * control_dt
        interval_peak = 0.0
        interval_contact = True
        for _ in range(p["control_decimation"]):
            mujoco.mj_step(env.model, env.data)
            active, force = contact_force(env, gateway)
            contact.add(float(env.data.time), active, force)
            interval_peak = max(interval_peak, force)
            interval_contact &= active
        obs = env.get_obs()
        recording.add(env)
        position = float(np.linalg.norm(obs[:3] - planner.dock_position))
        attitude = float(np.linalg.norm(rotation_error(obs[3:7], planner.dock_attitude)))
        speed = float(np.linalg.norm(obs[7:10]))
        angular = float(np.linalg.norm(obs[10:13]))
        # Preserve the draft's rendezvous gate, now explicitly before first contact.
        if (
            contact.first is None
            and env.data.time <= p["precontact_deadline"]
            and position <= p["precontact_position_tolerance"]
            and attitude <= p["precontact_attitude_tolerance"]
        ):
            rendezvous = True
        inside = (
            interval_contact
            and position <= p["settled_position_tolerance"]
            and attitude <= p["settled_attitude_tolerance"]
            and speed <= p["settled_speed_tolerance"]
            and angular <= p["settled_angular_speed_tolerance"]
        )
        hold = hold + 1 if inside else 0
        if hold >= required and settling_time is None:
            settling_time = float(env.data.time) - required * control_dt
        history.append(
            dict(
                time=float(env.data.time),
                position_error=position,
                attitude_error=attitude,
                contact_force=interval_peak,
                command_effort=float(np.abs(command).sum()),
            )
        )
        if contact.first is not None and env.data.time >= contact.first + p["post_contact_timeout"]:
            break
    success = (
        reason is None
        and rendezvous
        and hold >= required
        and contact.peak <= p["contact_force_bound"]
    )
    row = dict(
        evaluation_seed=seed,
        initial_condition_seed=seed,
        rendezvous_success=rendezvous,
        docking_success=success,
        success=success,
        settling_time=settling_time,
        final_position_error=position,
        final_attitude_error=attitude,
        mean_position_error=float(np.mean([h["position_error"] for h in history]))
        if history
        else position,
        mean_attitude_error=float(np.mean([h["attitude_error"] for h in history]))
        if history
        else attitude,
        control_effort=effort,
        episode_length=len(history),
        solver_status=solver_status,
        initial_penetration_depth=initial_depth,
        termination_reason=reason
        or (
            "settled_contact"
            if success
            else "post_contact_timeout"
            if contact.first is not None
            else "timeout"
        ),
        **contact.row(),
    )
    if recording_path is not None:
        recording.save(recording_path)
    return row, history, sample


def docking(config, job, run):
    local = deepcopy(config)
    protocol = local["protocol"]
    common = local["common"]
    common["env"]["sim"]["dt"] = protocol["sim_dt"]
    common["env"]["environment"]["control_decimation"] = protocol["control_decimation"]
    env = NativeEnvironment(local, job["spacecraft"], run, scene="gateway")
    run.update(task="gateway_docking", success_definition="demonstration_use_cases/docking/v1")
    mujoco.mj_forward(env.model, env.data)
    site = next(s for s in env.env_cfg.gateway.dock_sites if s["name"] == protocol["dock_site"])
    site_position = env.data.site_xpos[env.model.site(protocol["dock_site"]).id].copy()
    axis = np.asarray(site["approach_axis"])
    axis = axis / np.linalg.norm(axis)
    dock = site_position - axis * protocol["dock_surface_offset"]
    pre_dock = dock + axis * protocol["approach_offset"]
    planner = DockingPlanner(
        env,
        pre_dock,
        dock,
        np.asarray(site["quat"]),
        switch_distance=protocol["switch_distance"],
        dock_distance=protocol["dock_distance"],
        approach_speed=protocol.get("approach_speed"),
        validate_geometry=False,  # run_trial records penetration as a failed trial
    )
    policy = controller(env, planner, job["controller"])
    gateway = int(env.model.body("gateway_full").id)
    original = env.model.geom_solref.copy()
    env.model.geom_solref[:] = protocol["settings"][job["condition"]]
    # All other contact attributes come from the compiled model, saved explicitly.
    from .common import write_json

    write_json(
        run.path / "contact_model.json",
        dict(
            original_solref=original,
            solref=env.model.geom_solref,
            solimp=env.model.geom_solimp,
            friction=env.model.geom_friction,
            dock=dock,
            pre_dock=pre_dock,
            site=protocol["dock_site"],
        ),
    )
    trials = (
        protocol["default_trials"]
        if job["condition"] == "default"
        else protocol["sensitivity_trials"]
    )
    for trial in range(trials):
        seed = common["evaluation_seed_start"] + trial
        row, trace, sample = run_trial(
            env,
            policy,
            planner,
            protocol,
            seed,
            gateway,
            recording_path=run.path / "recordings" / job["condition"] / f"{trial:04d}.npz",
        )
        run.append(
            dict(
                **job,
                **row,
                trial=trial,
                contact_setting=job["condition"],
                solref_time_constant=protocol["settings"][job["condition"]][0],
                solref_damping=protocol["settings"][job["condition"]][1],
            )
        )
        run.append(dict(seed=seed, **sample), "samples.jsonl")
        # Every trace is retained; representative selection belongs to postprocessing.
        write_json(run.path / "traces" / f"{trial:04d}.json", trace)


def main(argv=None):
    execute("exp4_docking", parser("exp4_docking").parse_args(argv), docking)


if __name__ == "__main__":
    main()
