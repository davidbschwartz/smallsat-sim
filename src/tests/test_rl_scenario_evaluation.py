"""Saved-policy evaluation uses the library collector, task and MJX physics."""

from copy import copy
from types import SimpleNamespace

from flax import nnx
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from smallsat_sim.api.experiments import make_experiment
from smallsat_sim.controllers.rl.runners.runner_metrics import scenario_metrics
from smallsat_sim.controllers.rl.runners.runner_utils import model_fingerprint
from smallsat_sim.controllers.rl.runners.scenario_evaluation import evaluate_scenarios
from smallsat_sim.envs.vec_env.observations import state_features
from smallsat_sim.utils.quaternions_jax import quaternion_to_rotation_matrix


def experiment(path, algorithm):
    return make_experiment(
        vehicle="astrobee_rl",
        algorithm=algorithm,
        planner_radius=0.0,
        log=False,
        run_name="scenario_test",
        overrides={
            "RL": {
                "num_envs": 2,
                "rollout_backend": "freeflyer",
                "train_with_failures": False,
                "use_adaptive_approach": False,
                "use_pretrained": False,
                "policy_hidden_sizes": [8, 8],
                "checkpoint_dir": str(path),
                "episode_len": 8,
                "n_evals": 1,
                "terminal_hold_steps": 2,
                "terminal_radius": 0.25,
                "terminal_max_att_error": 0.25,
                "PPO": {
                    "steps_per_epoch": 4,
                    "epochs": 1,
                    "max_ep_len": 4,
                    "actor_training_epochs": 1,
                    "critic_training_epochs": 1,
                    "num_minibatches": 1,
                },
                "SAC": {
                    "total_transitions": 8,
                    "collection_steps": 2,
                    "replay_capacity": 16,
                    "batch_size": 4,
                    "random_steps": 4,
                    "learning_starts": 4,
                    "updates_per_collection": 1,
                    "max_ep_len": 4,
                    "randomization_pool_size": 4,
                },
            }
        },
    )


def scenarios(runner):
    result = []
    for seed, offset in enumerate([0.0, 2.0, -2.0]):
        pose = np.asarray(runner.reference_point)[0].copy()
        pose[0] += offset
        result.append(
            dict(
                initial_qpos=pose,
                initial_qvel=np.zeros(6),
                mass_scale=1.0 + seed * 0.2,
                inertia_scale=1.0 + seed * 0.1,
                thrust_scale=np.full(runner.env.act_dim, 0.7),
                wrench=np.array([0.01, -0.01, 0.0, 0.001, 0.0, 0.0]),
                evaluation_seed=10000 + seed,
            )
        )
    return result


@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
def test_saved_policy_evaluates_each_scenario_once(tmp_path, algorithm):
    exp = experiment(tmp_path, algorithm)
    runner = exp.runner
    try:
        runner.learn()
        saved = model_fingerprint(runner.agent.actor)
        original_mass = runner.env.model.body_mass.copy()
        original_state = jax.tree.map(lambda x: np.asarray(x).copy(), runner.env.state_struct)
        samples = scenarios(runner)
        # Loading must win over arbitrary changes to the live actor.
        nnx.update(runner.agent.actor, jax.tree.map(jnp.zeros_like, nnx.state(runner.agent.actor)))
        rows, timing = runner.evaluate(scenarios=samples, batch_size=2)
        assert model_fingerprint(runner.agent.actor) == saved
        assert timing["backend"] == "mjx"
        assert [b["scenarios"] for b in timing["batches"]] == [2, 1]
        assert len(rows) == 3
        assert [r["scenario"]["evaluation_seed"] for r in rows] == [10000, 10001, 10002]
        assert rows[0]["metrics"]["success"]
        assert rows[0]["metrics"]["episode_length"] == 2
        assert [r["metrics"]["termination_reason"] for r in rows[1:]] == ["timeout", "timeout"]
        assert len(rows[0]["time"]) == 3
        again, _ = runner.evaluate(scenarios=samples, batch_size=1)
        for a, b in zip(rows, again, strict=True):
            np.testing.assert_allclose(a["qpos"], b["qpos"], atol=2e-6)
            assert a["metrics"]["episode_length"] == b["metrics"]["episode_length"]
        np.testing.assert_array_equal(runner.env.model.body_mass, original_mass)
        for before, after in zip(
            jax.tree.leaves(original_state), jax.tree.leaves(runner.env.state_struct), strict=True
        ):
            np.testing.assert_array_equal(before, after)
        with pytest.raises(ValueError, match="randomize"):
            runner.evaluate(scenarios=samples, randomize=True)
    finally:
        exp.env.close()


class FeedbackActor(nnx.Module):
    def deterministic_action(self, obs):
        return jnp.clip(0.025 + 0.01 * obs, 0.0, 0.1)


class InfiniteActor(nnx.Module):
    def deterministic_action(self, obs):
        return jnp.full_like(obs, jnp.inf)


def test_shared_mjx_collector_matches_native_randomized_physics(tmp_path):
    exp = experiment(tmp_path, "ppo")
    runner = exp.runner
    try:
        runner.agent.actor = FeedbackActor()
        samples = scenarios(runner)[1:]
        rows, _ = evaluate_scenarios(runner, samples, source="zero", steps=512, batch_size=2)
        assert all(row["metrics"]["episode_length"] == 512 for row in rows)
        model = copy(runner.env.model)
        original_mass, original_inertia = model.body_mass[1], model.body_inertia[1].copy()
        reference = jnp.asarray(runner.reference_point[0])
        decimation = runner.env.env_cfg.environment.control_decimation
        for sample, row in zip(samples, rows, strict=True):
            data = mujoco.MjData(model)
            model.body_mass[1] = original_mass * sample["mass_scale"]
            model.body_inertia[1] = original_inertia * sample["inertia_scale"]
            mujoco.mj_setConst(model, data)
            data.qpos[:] = sample["initial_qpos"]
            data.qvel[:] = sample["initial_qvel"]
            data.xfrc_applied[1] = sample["wrench"]
            mujoco.mj_forward(model, data)
            for pose in row["qpos"][1:]:
                qpos, qvel = jnp.asarray(data.qpos[None]), jnp.asarray(data.qvel[None])
                rot = quaternion_to_rotation_matrix(qpos[:, 3:7])
                velocity = jnp.einsum("bji,bj->bi", rot, qvel[:, :3])
                obs = state_features(qpos, velocity, qvel[:, 3:6], reference)
                data.ctrl[:] = (
                    np.asarray(runner.agent.actor.deterministic_action(obs))[0]
                    * sample["thrust_scale"]
                )
                for _ in range(decimation):
                    mujoco.mj_step(model, data)
                np.testing.assert_allclose(pose, data.qpos, atol=5e-4)
        runner.agent.actor = InfiniteActor()
        rows, timing = evaluate_scenarios(
            runner, samples + samples[:1], source="zero", steps=2, batch_size=2
        )
        assert len(rows) == 3
        assert [b["scenarios"] for b in timing["batches"]] == [2, 1]
        assert all(r["metrics"]["termination_reason"] == "nonfinite_control" for r in rows)
        assert all(r["metrics"]["episode_length"] == 0 for r in rows)
    finally:
        exp.env.close()


def test_reduction_uses_task_terminals_and_preserves_first_event():
    poses = np.tile([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], (6, 5, 1))
    commands = np.ones((6, 5, 2))
    commands[3, 0] = np.nan  # Already successful: later failure is irrelevant.
    commands[0, 1] = np.nan
    poses[1, 3, 0] = np.nan
    success = np.zeros((6, 5), dtype=bool)
    success[1, 0] = True
    failure = np.zeros_like(success)
    failure[2, 4] = True  # A task-defined failure with a finite state.
    output = SimpleNamespace(
        terminals=success | failure,
        success_terminals=success,
        failure_terminals=failure,
        rewards=np.ones((6, 5)),
        next_position_error=np.zeros((6, 5, 3)),
        next_attitude_error=np.zeros((6, 5)),
    )
    rows = scenario_metrics(
        output, commands, (poses, np.zeros((6, 5, 6))), initial_poses=poses[0].copy(), dt=0.05
    )
    assert [r["metrics"]["termination_reason"] for r in rows] == [
        "success",
        "nonfinite_control",
        "timeout",
        "nonfinite_state",
        "failure",
    ]
    assert [r["metrics"]["episode_length"] for r in rows] == [2, 0, 6, 1, 3]
    np.testing.assert_allclose(
        [r["metrics"]["control_effort"] for r in rows], [0.2, 0, 0.6, 0.1, 0.3]
    )
