"""Small integration checks for retained training capabilities."""

from smallsat_sim.controllers.rl.storage.rollout_batch import make_training_batch
from smallsat_sim.envs.effects.scheduling import reset_and_randomize
from smallsat_sim.controllers.rl.runners.runner_utils import (
    save_training_data,
    load_training_data,
)
from smallsat_sim.controllers.rl.runners.on_policy_runner import OnPolicyRunner
from copy import deepcopy
from pathlib import Path
from smallsat_sim.controllers.rl.runners.runner_utils import (
    checkpoint_exists,
    model_fingerprint,
)
import jax
import numpy as np
import pytest
from flax import nnx
from experiments.rl_benchmarking.benchmark import build_runner
from experiments.rl_benchmarking.evaluation import Scenario, prepare
from smallsat_sim.controllers.rl.runners.pretraining import pd_demonstrator
from smallsat_sim.controllers.pd.vectorized_controller import VectorizedPDController


@pytest.fixture
def runner(tmp_path):
    hp = dict(
        steps_per_epoch=6,
        max_ep_len=6,
        epochs=2,
        num_minibatches=2,
        actor_training_epochs=1,
        critic_training_epochs=1,
    )
    runner = build_runner(
        "rma_cnn",
        13,
        log=False,
        overrides={
            "num_envs": 4,
            "rollout_backend": "freeflyer",
            "context_window_len": 3,
            "policy_hidden_sizes": [8, 8],
            "PPO": hp,
            "am_collection_epochs": 1,
            "am_epochs": 1,
            "am_batch_size": 4,
            "pretraining_epochs": 1,
            "pretraining_batch_size": 4,
            "failure_fraction": 0.0,
            "disturbance_fraction": 0.0,
        },
    )
    runner.ckpt_dir = str(tmp_path / "checkpoints")
    yield runner
    runner.env.close()


def test_resume_reproduces_next_rollout_and_update(runner):
    collector = runner.collector("privileged", stochastic=True)
    result = runner.collect(collector, randomize=False)
    runner.agent.update(
        make_training_batch(
            result, runner.context_scale, runner.agent.gamma, runner.agent.lam
        )
    )
    runner.policy_epoch = 1
    runner.save("policy")
    first = runner.collect(collector, randomize=False)
    expected = runner.agent.update(
        make_training_batch(
            first, runner.context_scale, runner.agent.gamma, runner.agent.lam
        )
    )
    expected_state = nnx.state(runner.agent.actor)
    runner.restore("policy")
    assert runner.policy_epoch == 1
    second = runner.collect(collector, randomize=False)
    np.testing.assert_array_equal(first.actions, second.actions)
    actual = runner.agent.update(
        make_training_batch(
            second, runner.context_scale, runner.agent.gamma, runner.agent.lam
        )
    )
    np.testing.assert_array_equal(expected.actor_steps, actual.actor_steps)
    jax.tree.map(
        np.testing.assert_array_equal, expected_state, nnx.state(runner.agent.actor)
    )
    with pytest.raises(FileExistsError):
        runner.begin("policy", "fresh")
    assert collector._cache_size() == 1


def test_estimator_controls_collection_and_teacher_is_frozen(runner):
    # The current history estimate affects the next action; teacher labels are not inputs.
    reset_and_randomize(runner.env, jax.random.PRNGKey(4), enabled=False)
    collector = runner.collector("estimated", stochastic=False)
    state = runner.env.freeflyer_state_struct()
    args = (state, nnx.state(runner.agent.actor), nnx.state(runner.agent.critic))
    original_actor = nnx.state(runner.agent.actor)
    first = collector(
        *args, nnx.state(runner.am), jax.random.PRNGKey(5), runner.reference_point
    )
    runner.am.output.bias.value += 5.0
    second = collector(
        *args, nnx.state(runner.am), jax.random.PRNGKey(5), runner.reference_point
    )
    np.testing.assert_array_equal(first.actions[:3], second.actions[:3])
    assert not np.allclose(first.actions[3:], second.actions[3:])
    # The label at t is from the completed transition t, not the context entering t.
    expected = (
        second.step_outputs.actual_wrench
        - second.actions @ runner.env._thruster_mixer_T
    ) / runner.context_scale
    np.testing.assert_allclose(second.labels.normalized_context, expected, atol=1e-6)
    jax.tree.map(
        np.testing.assert_array_equal, original_actor, nnx.state(runner.agent.actor)
    )


def test_pretraining_matches_pd_and_can_be_followed_by_rl(runner):
    runner.env.reset()
    teacher = VectorizedPDController(runner.env, runner.planner)
    demonstration = pd_demonstrator(runner.env, runner.planner, runner.reference_point)
    np.testing.assert_allclose(
        demonstration(runner.env.get_states(runner.reference_point)),
        teacher.get_control_input(runner.env, runner.reference_point),
        atol=1e-6,
    )
    before = np.asarray(runner.agent.actor.mu_net.layers[0].kernel).copy()
    runner.pretrain()
    assert not np.array_equal(before, runner.agent.actor.mu_net.layers[0].kernel)
    assert int(runner.agent.actor_optimizer.step.value) == 0
    runner.learn()
    teacher_state = nnx.state(runner.agent.actor)
    runner.train_adaptation_module_on_policy()
    jax.tree.map(
        np.testing.assert_array_equal, teacher_state, nnx.state(runner.agent.actor)
    )
    runner.restore("adaptation")
    assert runner.adaptation_epoch == 1


def test_scenarios_reproduce_initial_state_and_failure_realization(runner):
    scenario = Scenario("stuck_off", ("stuck_off",), onset_seconds=0.0)
    first = prepare(runner.env, scenario, 100)
    # Consume unrelated RNGs to ensure the scenario seed really resets all streams.
    runner.collect(runner.collector("privileged", stochastic=True), randomize=False)
    second = prepare(runner.env, scenario, 100)
    jax.tree.map(np.testing.assert_array_equal, first, second)


def test_realized_scenario_checkpoint_roundtrip(runner, tmp_path):
    state = prepare(runner.env, Scenario("off", ("stuck_off",), onset_seconds=0.0), 100)
    save_training_data(tmp_path, "scenario.ckpt", {"state": state})
    restored = load_training_data(
        tmp_path, "scenario.ckpt", mjx_batch_template=state.mjx_batch
    )["state"]
    jax.tree.map(np.testing.assert_array_equal, state, restored)


def test_adaptation_rejects_replaced_teacher(runner):
    runner.save("policy")
    runner.save("adaptation")
    runner.agent.actor.log_std.value += 0.1
    runner.save("policy")
    with pytest.raises(ValueError, match="different teacher"):
        runner.restore("adaptation")


def test_fresh_learning_honors_saved_pretraining_weights(runner, monkeypatch):

    runner.save("pretraining")
    pretrained_actor = nnx.state(runner.agent.actor)
    runner.training_cfg.use_pretrained = True
    fresh = OnPolicyRunner(runner.env, runner.planner, config=runner.training_cfg)
    fresh.ckpt_dir = runner.ckpt_dir
    assert fresh.config_id != runner.config_id
    update = fresh.agent.update
    calls = []

    def checked_update(batch):
        if not calls:
            jax.tree.map(
                np.testing.assert_array_equal,
                pretrained_actor,
                nnx.state(fresh.agent.actor),
            )
        calls.append(True)
        return update(batch)

    monkeypatch.setattr(fresh.agent, "update", checked_update)
    fresh.learn()
    assert len(calls) == fresh.agent.epochs


def test_shared_teacher_allows_new_estimator_but_keeps_strict_resume(runner, tmp_path):

    # One trained teacher supplies both CNN and transformer adaptation runs.
    runner.learn()
    teacher = Path(runner.ckpt_dir) / runner.training_state_file_name
    teacher_id = model_fingerprint(runner.agent.actor)
    original_bytes = {
        p.relative_to(teacher): p.read_bytes()
        for p in teacher.rglob("*")
        if p.is_file()
    }
    for preset in ("rma_cnn", "rma_transformer"):
        cfg = deepcopy(runner.resolved_config["rl"])
        cfg["am_architecture"] = "cnn" if preset == "rma_cnn" else "transformer"
        cfg["checkpoint_dir"] = str(tmp_path / preset)
        student = build_runner(preset, 13, log=False, overrides=cfg)
        try:
            student.initialize_from_teacher(teacher)
            assert model_fingerprint(student.agent.actor) == teacher_id
            student.train_adaptation_module_on_policy()
            assert model_fingerprint(student.agent.actor) == teacher_id
            student.restore("adaptation")
            assert student.teacher_source["model_id"] == teacher_id
            assert student.adaptation_epoch == 1
            with pytest.raises(FileExistsError):
                student.initialize_from_teacher(teacher)

            # A matching shape is not enough to bypass resume configuration validation.
            config_id = student.config_id
            student.config_id = "different"
            with pytest.raises(ValueError, match="configuration differs"):
                student.restore("adaptation")
            student.config_id = config_id
            student.agent.actor.log_std.value += 0.1
            student.save("policy")
            with pytest.raises(ValueError, match="different teacher"):
                student.restore("adaptation")
        finally:
            student.env.close()
    assert original_bytes == {
        p.relative_to(teacher): p.read_bytes()
        for p in teacher.rglob("*")
        if p.is_file()
    }


def test_shared_teacher_rejects_changed_task_before_saving(runner, tmp_path):
    runner.save("policy")
    cfg = deepcopy(runner.resolved_config["rl"])
    cfg["checkpoint_dir"] = str(tmp_path / "different_task")
    cfg["terminal_bonus"] += 1
    other = build_runner("rma_cnn", 13, log=False, overrides=cfg)
    try:
        before = model_fingerprint(other.agent.actor)
        with pytest.raises(ValueError, match="policy, physics or task"):
            other.initialize_from_teacher(
                Path(runner.ckpt_dir) / runner.training_state_file_name
            )
        assert model_fingerprint(other.agent.actor) == before
        assert not checkpoint_exists(other.ckpt_dir, other.training_state_file_name)
    finally:
        other.env.close()


@pytest.mark.parametrize("stage", ["policy", "adaptation"])
def test_deployment_loads_renamed_checkpoint_and_steps(runner, stage):
    from smallsat_sim.controllers.rl.controller import RLController

    runner.save("policy")
    runner.save("adaptation")
    filename = (
        runner.training_state_file_name if stage == "policy"
        else runner.adaptation_module_file_name
    )
    checkpoint = Path(runner.ckpt_dir) / filename
    renamed = checkpoint.with_name(f"renamed-{stage}.ckpt")
    checkpoint.rename(renamed)
    controller = RLController(
        runner.env, runner.planner, config=runner.training_cfg, checkpoint=renamed,
    )
    controller.deployment_len = 2
    controller.control()
    assert model_fingerprint(controller.agent.actor) == model_fingerprint(runner.agent.actor)
    assert model_fingerprint(controller.am) == model_fingerprint(runner.am)
    assert np.all(np.isfinite(runner.env.get_obs()))


def test_deployment_rejects_mismatched_teacher(runner):
    from smallsat_sim.controllers.rl.controller import RLController
    from smallsat_sim.controllers.rl.runners.runner_utils import save_adaptation_module

    runner.save("policy")
    save_adaptation_module(
        runner.am, runner.ckpt_dir, runner.adaptation_module_file_name,
        metadata={"teacher_id": "different-policy"},
    )
    controller = RLController(
        runner.env, runner.planner, config=runner.training_cfg,
        checkpoint=Path(runner.ckpt_dir) / runner.training_state_file_name,
    )
    with pytest.raises(ValueError, match="different teacher policy"):
        controller.control()
