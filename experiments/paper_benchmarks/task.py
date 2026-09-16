"""Shared versioned pose task and deterministic, paired evaluation samples."""

import numpy as np
from scipy.spatial.transform import Rotation

SUCCESS_ID = "demonstration_use_cases/full_pose/v1"


def register_task():
    from smallsat_sim.api import register_termination
    from smallsat_sim.api.registry import list_terminations
    from smallsat_sim.envs.termination import full_pose_termination

    if SUCCESS_ID not in list_terminations():
        register_termination(SUCCESS_ID, full_pose_termination)


def rotation_error(quaternion, desired):
    current = Rotation.from_quat(np.roll(quaternion, -1))
    target = Rotation.from_quat(np.roll(desired, -1))
    return (target * current.inv()).as_rotvec()


def features(obs, reference):
    rotation = Rotation.from_quat(np.roll(obs[3:7], -1))
    return np.concatenate(
        (
            obs[:3] - reference[:3],
            rotation_error(obs[3:7], reference[3:7]),
            rotation.inv().apply(obs[7:10]),
            obs[10:13],
        )
    ).astype(np.float32)


def sample_trial(common, seed, distribution, actuator_count):
    # Independent streams keep initial states paired even if an effect changes its draw count.
    init_rng, dynamics_rng, fault_rng = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
    ]
    ic = common["initial_conditions"]
    reference = np.asarray(common["reference"])
    u = init_rng.uniform(size=2)
    radius, angle = ic["max_offset"] * np.sqrt(u[0]), 2 * np.pi * u[1]
    position = reference[:3] + [radius * np.cos(angle), radius * np.sin(angle), 0.0]
    # Same support/formula as vec_env.reset.sample_initial_quaternion; independent NumPy draws.
    a, b, c = init_rng.uniform(0, 2 * np.pi, 3)
    q = np.array(
        [
            np.sin(a) * np.cos(b) * np.cos(c) + np.cos(a) * np.sin(b) * np.sin(c),
            np.cos(a) * np.sin(b) * np.cos(c) - np.sin(a) * np.cos(b) * np.sin(c),
            np.sin(a) * np.sin(b) * np.cos(c) + np.cos(a) * np.cos(b) * np.sin(c),
            np.cos(a) * np.cos(b) * np.cos(c) - np.sin(a) * np.sin(b) * np.sin(c),
        ]
    )
    q /= np.linalg.norm(q)
    velocity = np.r_[
        init_rng.uniform(-ic["max_linear_velocity"], ic["max_linear_velocity"], 3),
        init_rng.uniform(-ic["max_angular_velocity"], ic["max_angular_velocity"], 3),
    ]
    if distribution.get("initial") == "fixed":
        position = reference[:3] + [1.0, 0.0, 0.0]
        q = np.roll(Rotation.from_euler("xyz", [0.2, -0.2, 0.2]).as_quat(), 1)
        velocity = np.zeros(6)
    return dict(
        initial_qpos=np.r_[position, q],
        initial_qvel=velocity,
        mass_scale=dynamics_rng.uniform(*distribution.get("mass", [1.0, 1.0])),
        inertia_scale=dynamics_rng.uniform(*distribution.get("inertia", [1.0, 1.0])),
        thrust_scale=dynamics_rng.uniform(*distribution.get("thrust", [1.0, 1.0]), actuator_count),
        wrench=np.r_[
            dynamics_rng.uniform(
                -distribution.get("force", 0.0), distribution.get("force", 0.0), 3
            ),
            dynamics_rng.uniform(
                -distribution.get("torque", 0.0), distribution.get("torque", 0.0), 3
            ),
        ],
        affected_thruster=int(fault_rng.integers(actuator_count)),
        evaluation_seed=seed,
        initial_condition_seed=seed,
        effect_seed=int(fault_rng.integers(2**31)),
    )


class PoseMetrics:
    def __init__(self, common):
        self.task = common["env"]["environment"]
        self.dt = common["env"]["sim"]["dt"] * self.task["control_decimation"]
        self.positions, self.attitudes = [], []
        self.effort = 0.0
        self.hold = 0
        self.success = False

    def add(self, obs, reference, command):
        pos = float(np.linalg.norm(obs[:3] - reference[:3]))
        att = float(np.linalg.norm(rotation_error(obs[3:7], reference[3:7])))
        self.positions.append(pos)
        self.attitudes.append(att)
        self.effort += float(np.abs(command).sum()) * self.dt
        inside = pos <= self.task["terminal_radius"] and att <= self.task["terminal_max_att_error"]
        self.hold = self.hold + 1 if inside else 0
        self.success = self.hold >= self.task["terminal_hold_steps"]
        return self.success

    def row(self, reason=None):
        row = dict(
            success=self.success,
            completion_fraction=float(self.success),
            completion_time=len(self.positions) * self.dt if self.success else None,
            episode_length=len(self.positions),
            control_effort=self.effort,
            termination_reason=reason or ("success" if self.success else "timeout"),
        )
        for name, values in [
            ("position_error", self.positions),
            ("attitude_error", self.attitudes),
        ]:
            row.update(
                {
                    f"mean_{name}": float(np.mean(values)) if values else None,
                    f"max_{name}": float(np.max(values)) if values else None,
                    f"final_{name}": values[-1] if values else None,
                }
            )
        return row
