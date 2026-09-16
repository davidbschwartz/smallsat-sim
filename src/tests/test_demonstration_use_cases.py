"""Protocol integrity, paired sampling, artifact coverage and native smoke checks."""

from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
from experiments.paper_benchmarks.common import evaluation_trial_count
from experiments.paper_benchmarks.common import (
    EXPERIMENTS,
    load_config,
    Run,
    run_directory,
    jobs,
    digest,
    write_json,
)
from experiments.paper_benchmarks.task import sample_trial, PoseMetrics
from experiments.paper_benchmarks.aggregate import (
    wilson,
    validate_rows,
    summarize,
    discover,
    learning_curves,
)
from experiments.paper_benchmarks.make_tables import tabular, escape


def test_legacy_package_preserves_saved_config_and_effect_references():
    from importlib import import_module
    from experiments.demonstration_use_cases.common import load_config as legacy_load
    from experiments.paper_benchmarks.effects import training_effects

    config = load_config("exp3_rl_robustness", mode="paper")
    assert legacy_load("exp3_rl_robustness", mode="paper") == config
    faults, wrenches = training_effects(config["common"]["train_distribution"])
    for effect in faults + wrenches:
        for key in ["sample", "apply"]:
            module, name = effect[key].split(":")
            assert module == "experiments.demonstration_use_cases.effects"
            assert callable(getattr(import_module(module), name))


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_configs_load_and_smoke_is_separate(experiment):
    paper = load_config(experiment, mode="paper")
    smoke = load_config(experiment, mode="smoke")
    assert jobs(paper) and jobs(smoke)
    assert digest(paper) != digest(smoke)
    assert load_config(experiment, mode="paper") == paper


def test_equal_training_budget_and_moderate_robustness_support():
    config = load_config("exp3_rl_robustness")
    c = config["common"]
    t = c["training"]
    n = c["env"]["environment"]["num_envs"]
    assert n == config["protocol"]["training_num_envs"]["sac"] == 4096
    assert (
        t["PPO"]["steps_per_epoch"] * t["PPO"]["epochs"] * n
        == t["VPG"]["steps_per_epoch"] * t["VPG"]["epochs"] * n
        == t["SAC"]["total_transitions"]
    )
    assert set(c['evaluation_distributions']) == {
        'nominal', 'randomized_initial', 'thrust_variation', 'disturbance', 'combined'}
    assert c['train_distribution']['thrust'] == [0.9, 1.1]
    for distribution in c['evaluation_distributions'].values():
        assert distribution.get('mass', [1., 1.]) == [1., 1.]
        assert distribution.get('inertia', [1., 1.]) == [1., 1.]
        assert distribution.get('force', 0.) <= 0.005
        assert distribution.get('torque', 0.) <= 0.0005


def test_run_directories_and_failure_retention(tmp_path):
    c = load_config("spacecraft_portability", mode="smoke")
    job = jobs(c)[0]
    a = Run(tmp_path, c, job)
    b = Run(tmp_path, c, job)
    assert a.path != b.path
    assert (a.path / "resolved_config.yaml").exists()
    with pytest.raises(RuntimeError):
        with run_directory(tmp_path, c, job) as run:
            raise RuntimeError("test failure")
    assert json.loads((run.path / "metadata.json").read_text())["status"] == "failed"


def test_paired_seeds_and_pose_hold():
    c = load_config("exp2_fault_robustness")["common"]
    a = sample_trial(c, 10000, {}, 12)
    b = sample_trial(c, 10000, c["evaluation_distributions"]["compound"], 12)
    np.testing.assert_array_equal(a["initial_qpos"], b["initial_qpos"])
    np.testing.assert_array_equal(a["initial_qvel"], b["initial_qvel"])
    assert a["affected_thruster"] == b["affected_thruster"]
    m = PoseMetrics(c)
    reference = np.array(c["reference"])
    obs = np.r_[reference, np.zeros(6)]
    for _ in range(4):
        assert not m.add(obs, reference, np.zeros(12))
    assert m.add(obs, reference, np.zeros(12))
    obs[0] += 1.0
    assert not m.add(obs, reference, np.zeros(12))


def test_statistics_and_learning_axis():
    low, high = wilson(0, 10)
    assert low == pytest.approx(0.0) and high > 0
    low, high = wilson(10, 10)
    assert low < 1 and high == pytest.approx(1.0)
    frame = pd.DataFrame(
        [dict(condition="nominal", success=x, control_effort=y) for x, y in [(1, 1.0), (0, 3.0)]]
    )
    s = summarize(frame, ["condition"]).iloc[0]
    assert s.n == 2 and s.success_rate == 0.5 and s.control_effort_mean == 2.0
    curves = pd.DataFrame(
        [
            dict(
                controller="ppo",
                regime="nominal",
                training_seed=seed,
                environment_steps=step,
                mean_episodic_returns=value,
            )
            for seed, step, value in [(0, 10, 1.0), (0, 20, 3.0), (1, 10, 3.0), (1, 20, 5.0)]
        ]
    )
    summary = learning_curves(curves)
    np.testing.assert_allclose(summary["mean"], [2.0, 4.0])
    assert list(summary.n_seeds) == [2, 2]


def test_missing_and_duplicate_rows(tmp_path):
    with pytest.raises(ValueError, match="Missing"):
        discover(tmp_path, "paper")
    assert discover(tmp_path, "paper", True)[1]
    c = load_config("exp1_scaling", mode="smoke")
    job = jobs(c)[0]
    rows = [dict(repeat=i, num_envs=1, status="ok") for i in range(2)]
    validate_rows(c, job, rows)
    with pytest.raises(ValueError):
        validate_rows(c, job, rows + rows)
    validate_rows(c, job, [dict(status="oom")])


def test_latex_structure_and_escaping():
    text = tabular(["Method", "$n$"], [[escape("a_b & c"), "10"]])
    assert text.startswith("\\begin{tabular}{ll}") and text.endswith("\\end{tabular}\n")
    assert r"a\_b \& c" in text and text.count(r"\\") == 2


@pytest.mark.parametrize("name", ["astrobee", "cubesat", "sprint"])
def test_native_vehicle_portability(tmp_path, name):
    from experiments.paper_benchmarks.runtime import NativeEnvironment, SetpointPlanner, controller

    c = load_config("spacecraft_portability", mode="smoke")
    env = NativeEnvironment(c, name, SimpleNamespace(path=tmp_path))
    sample = sample_trial(c["common"], 10000, {}, env.model.nu)
    env.reset(sample)
    ctrl = controller(env, SetpointPlanner(c["common"]["reference"]), "pd")
    assert np.all(np.isfinite(ctrl.get_control_input(env)))


@pytest.mark.parametrize(
    "condition",
    [
        "nominal",
        "stuck_off",
        "stuck_on",
        "saturated_thrust",
        "thrust_instability",
        "faulty_valve",
        "disturbance",
    ],
)
def test_fault_conditions(tmp_path, condition):
    from experiments.paper_benchmarks.runtime import vehicle, fault_mapping

    c = load_config("exp2_fault_robustness")
    v = vehicle(c, "astrobee", tmp_path)
    sample = sample_trial(c["common"], 10000, {}, len(v.actuators))
    apply = fault_mapping(v, condition, sample, c["common"])
    output = apply(np.ones(len(v.actuators)) * 0.1, 1.0)
    assert np.all(np.isfinite(output))
    if condition == "stuck_off":
        assert output[sample["affected_thruster"]] == 0
    if condition == "stuck_on":
        assert (
            output[sample["affected_thruster"]]
            == v.actuators[sample["affected_thruster"]].forcerange[1]
        )
    if "fault_curve_input" in sample:
        assert len(sample["fault_curve_input"]) == 100


def test_contact_accumulator():
    from experiments.paper_benchmarks.exp4_docking import ContactMetrics

    m = ContactMetrics(0.01)
    for time, active, force in [
        (0.0, False, 0.0),
        (0.01, True, 2.0),
        (0.02, True, 4.0),
        (0.03, False, 0.0),
        (0.04, True, 1.0),
    ]:
        m.add(time, active, force)
    assert m.row() == pytest.approx(
        dict(
            first_contact_time=0.01,
            peak_contact_force=4.0,
            contact_impulse=0.07,
            contact_events=2,
            cumulative_contact_duration=0.03,
        )
    )


def test_docking_short_execution(tmp_path):
    from experiments.paper_benchmarks.exp4_docking import docking

    c = load_config("exp4_docking", mode="smoke")
    c["protocol"]["controller"] = "pd"
    c["protocol"].update(default_trials=1, sensitivity_trials=1, steps=2)
    c["protocol"].update(initial_radius=[0.1, 0.2], initial_z_span=0.1)
    job = jobs(c)[0]
    with run_directory(tmp_path, c, job) as run:
        docking(c, job, run)
    row = json.loads((run.path / "metrics.jsonl").read_text())
    assert row["episode_length"] == 2 and row["peak_contact_force"] >= 0
    assert (run.path / "traces/0000.json").exists()
    contact_model = json.loads((run.path / "contact_model.json").read_text())
    assert contact_model["site"] == "dock_crew_airlock_a"
    np.testing.assert_allclose(contact_model["dock"], [-3.11, -7.12740877759179, -0.02])
    np.testing.assert_allclose(
        np.asarray(contact_model["pre_dock"]) - contact_model["dock"], [0.0, -1.0, 0.0]
    )
    with np.load(run.path / "recordings/default/0000.npz") as recording:
        assert len(recording["time"]) == 3
        assert recording["time"][0] == 0.0
        assert recording["qpos"].shape[0] == 3


def test_publication_files_deterministic(tmp_path):
    from experiments.paper_benchmarks.make_figures import make_figures
    from experiments.paper_benchmarks.make_tables import make_tables

    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame(
        [
            dict(
                num_envs=1,
                status="ok",
                env_steps_per_second=100.0,
                sim_seconds_per_second=5.0,
                sim_seconds_per_second_per_env=5.0,
                speedup=1.0,
                parallel_efficiency=1.0,
            )
        ]
    ).to_csv(data / "exp1_scaling.csv", index=False)
    a = make_figures(data, tmp_path / "a") + make_tables(data, tmp_path / "a")
    b = make_figures(data, tmp_path / "b") + make_tables(data, tmp_path / "b")
    assert [path.read_bytes() for path, _ in a] == [path.read_bytes() for path, _ in b]


def test_complete_synthetic_campaign_and_missing_run(tmp_path):
    """Coverage checks, aggregation and publications share one fixed mock campaign."""
    from experiments.paper_benchmarks.aggregate import aggregate

    root = tmp_path / "raw"
    for experiment in EXPERIMENTS:
        config = load_config(experiment, mode="smoke")
        p = config["protocol"]
        for job in jobs(config):
            with run_directory(root, config, job) as run:
                if experiment == "exp1_scaling":
                    n = job["batch_size"]
                    for repeat in range(p["repeats"]):
                        run.append(
                            dict(
                                repeat=repeat,
                                num_envs=n,
                                status="ok",
                                env_steps_per_second=100.0 * n,
                                sim_seconds_per_second=5.0 * n,
                                sim_seconds_per_second_per_env=5.0,
                            )
                        )
                    continue
                conditions = (
                    [job["condition"]]
                    if "condition" in job
                    else list(config["common"]["evaluation_distributions"])
                )
                count = p.get("trials", p.get("default_trials"))
                for condition in conditions:
                    for trial in range(count if experiment == "exp4_docking" else evaluation_trial_count(config, condition)):
                        row = {
                            **job,
                            "condition": condition,
                            "trial": trial,
                            "evaluation_seed": 10000 + trial,
                            "success": trial == 0,
                            "final_position_error": float(trial),
                            "mean_position_error": float(trial),
                            "final_attitude_error": 0.1,
                            "mean_attitude_error": 0.1,
                            "control_effort": 2.0,
                            "termination_reason": "timeout",
                        }
                        if experiment == "exp3_rl_robustness" and job["regime"] != "baseline":
                            row["training_seed"] = job["seed"]
                        if experiment == "exp4_docking":
                            row.update(peak_contact_force=1.0 + trial, settling_time=1.0)
                            write_json(
                                run.path / "traces" / f"{trial:04d}.json",
                                [
                                    dict(
                                        time=0.01,
                                        position_error=0.1,
                                        attitude_error=0.1,
                                        contact_force=1.0,
                                        command_effort=0.1,
                                    )
                                ],
                            )
                        run.append(row)
                if experiment == "exp3_rl_robustness" and job["regime"] != "baseline":
                    (run.path / "checkpoint").mkdir()
                    (run.path / "checkpoint" / "mock").write_text(
                        "synthetic fixture, not policy weights"
                    )
                    run.update(checkpoint_path="checkpoint/mock")
                    run.append(
                        dict(
                            controller=job["controller"],
                            regime=job["regime"],
                            environment_steps=16,
                            mean_episodic_returns=1.0,
                            success_rate=0.5,
                        ),
                        "training.jsonl",
                    )
    manifest = aggregate(root, tmp_path / "artifacts", mode="smoke")
    assert not manifest["partial"]
    summary = pd.read_csv(tmp_path / "artifacts/data/exp3_summary.csv")
    assert len(summary) == 6 * 5
    assert set(summary.loc[summary.condition == "nominal", "n"]) == {1}
    assert set(summary.loc[summary.condition != "nominal", "n"]) == {2}
    assert (tmp_path / "artifacts/paper/figures/exp4_contact_trajectory.pdf").exists()
    assert (tmp_path / "artifacts/paper/tables/exp3_robustness.tex").exists()
    for item in manifest["artifacts"]:
        assert (tmp_path / "artifacts" / item["artifact"]).exists()
        assert item["input_runs"]
    meta_path = next(root.glob("smoke/spacecraft_portability/*/metadata.json"))
    meta = json.loads(meta_path.read_text())
    meta["status"] = "failed"
    write_json(meta_path, meta)
    with pytest.raises(ValueError, match="Missing spacecraft_portability"):
        aggregate(root, tmp_path / "missing-artifacts", mode="smoke")


def test_actual_contact_force_is_filtered_and_integrated():
    import mujoco
    from experiments.paper_benchmarks.exp4_docking import contact_force, ContactMetrics

    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <option timestep="0.002" gravity="0 0 0"/>
      <worldbody>
        <body name="gateway_full"><geom type="plane" size="1 1 .1"/></body>
        <body name="body0" pos="0 0 .09"><freejoint/>
          <geom type="sphere" size=".1" mass="1"/>
        </body>
      </worldbody>
    </mujoco>""")
    data = mujoco.MjData(model)
    env = SimpleNamespace(model=model, data=data, chaser=model.body("body0").id)
    metrics = ContactMetrics(model.opt.timestep)
    for _ in range(3):
        mujoco.mj_step(model, data)
        active, force = contact_force(env, model.body("gateway_full").id)
        metrics.add(float(data.time), active, force)
    assert metrics.peak > 0 and metrics.impulse > 0 and metrics.events == 1
    assert contact_force(env, 0) == (False, 0.0)


def test_mpc_quaternion_reference_is_sign_invariant():
    from smallsat_sim.controllers.nominal_mpc.reference import align_quaternion_reference

    q = np.array([-0.8, 0.6, 0.0, 0.0])
    target = np.array([[1.0], [0.0], [0.0], [0.0]])
    original = target.copy()
    aligned = align_quaternion_reference(q, target)
    np.testing.assert_allclose(aligned, -target)
    np.testing.assert_allclose(
        np.linalg.norm(q - aligned[:, 0]),
        np.linalg.norm(-q - align_quaternion_reference(-q, target)[:, 0]),
    )
    np.testing.assert_array_equal(target, original)


def test_rate_limited_docking_reference_uses_simulation_time():
    from smallsat_sim.planners.mission.docking import DockingPlanner

    env = SimpleNamespace(using_rl=False, data=SimpleNamespace(time=0.0))
    planner = DockingPlanner(env, np.array([1.0, 0, 0]), np.zeros(3), approach_speed=0.02)
    obs = np.r_[1., 0., 0., 1., np.zeros(9)]
    np.testing.assert_allclose(planner.get_reference(obs)[0][:, 0], obs[:3])
    env.data.time = 2.0
    np.testing.assert_allclose(planner.get_reference(obs)[0][:, 0], [0.96, 0, 0])
    np.testing.assert_allclose(planner.get_reference(obs)[0][:, 0], [0.96, 0, 0])
    planner.reset()
    np.testing.assert_allclose(planner.get_reference(obs)[0][:, 0], obs[:3])


def test_initial_docking_penetration_is_retained_without_control(tmp_path):
    from unittest.mock import Mock
    from experiments.paper_benchmarks.exp4_docking import run_trial
    from experiments.paper_benchmarks.runtime import NativeEnvironment
    from smallsat_sim.planners.mission.docking import DockingPlanner

    c = load_config("exp4_docking")
    env = NativeEnvironment(c, "astrobee", SimpleNamespace(path=tmp_path), scene="gateway")
    dock = env.data.site_xpos[env.model.site("dock_orion_interface_a").id].copy()
    planner = DockingPlanner(env, dock + [-2.0, 0, 0], dock, validate_geometry=False)
    ctrl = SimpleNamespace(
        get_control_input=Mock(side_effect=AssertionError("Must not apply control"))
    )
    row, trace, sample = run_trial(
        env, ctrl, planner, c["protocol"], 20000, env.model.body("gateway_full").id
    )
    assert row["termination_reason"] == "initial_penetration" and not row["success"]
    assert row["initial_penetration_depth"] > 1.0 and row["episode_length"] == 0
    assert trace == []
    ctrl.get_control_input.assert_not_called()


@pytest.mark.parametrize("status,command", [(4, np.zeros(12)), (0, np.full(12, np.nan))])
def test_failed_mpc_command_is_rejected(status, command):
    from smallsat_sim.controllers.nominal_mpc.reference import require_valid_control, MPCSolverError

    with pytest.raises(MPCSolverError):
        require_valid_control(status, command)


def test_valid_mpc_command_is_preserved():
    from smallsat_sim.controllers.nominal_mpc.reference import require_valid_control

    command = np.arange(12) * 0.01
    np.testing.assert_array_equal(require_valid_control(0, command), command)


def test_rl_demonstration_training_matrix():
    config = load_config("exp3_rl_robustness", mode="paper")
    training = [job for job in jobs(config) if job["regime"] != "baseline"]
    assert len(training) == 20
    assert {(job["controller"], job["regime"], job["seed"]) for job in training} == {
        (algorithm, regime, seed)
        for algorithm in ("ppo", "sac")
        for regime in ("nominal", "randomized")
        for seed in range(5)
    }


def test_experiment_overrides_are_explicit_and_reject_typos():
    from experiments.paper_benchmarks.common import _apply_overrides
    base = {'training': {'PPO': {'epochs': 80}}, 'episode_steps': 512}
    _apply_overrides(base, {'training': {'PPO': {'epochs': 320}}})
    assert base == {'training': {'PPO': {'epochs': 320}}, 'episode_steps': 512}
    with pytest.raises(ValueError, match='Unknown common override'):
        _apply_overrides(base, {'training': {'PPO': {'epochz': 320}}})


def test_resume_validates_complete_jobs_and_rejects_duplicates(tmp_path):
    from experiments.paper_benchmarks.common import completed_job
    from copy import deepcopy
    config = load_config('spacecraft_portability', mode='smoke')
    job = jobs(config)[0]
    def complete():
        with run_directory(tmp_path, config, job) as run:
            for trial in range(config['protocol']['trials']):
                run.append(dict(condition=job['condition'], evaluation_seed=10000 + trial,
                                success=False, final_position_error=1., final_attitude_error=1.,
                                control_effort=0., termination_reason='timeout'))
    assert not completed_job(tmp_path, config, job)
    complete()
    assert completed_job(tmp_path, config, job)
    changed = deepcopy(config)
    changed['common']['episode_steps'] += 1
    with pytest.raises(ValueError, match='different configuration'):
        completed_job(tmp_path, changed, job)
    complete()
    with pytest.raises(ValueError, match='Duplicate completed job'):
        completed_job(tmp_path, config, job)


def test_resume_rejects_active_job_and_smoke_limits_algorithm_batches(tmp_path):
    from experiments.paper_benchmarks.common import completed_job
    config = load_config('exp3_rl_robustness', mode='smoke')
    assert config['protocol']['training_num_envs']['sac'] == 2
    job = jobs(config)[0]
    Run(tmp_path, config, job)
    with pytest.raises(ValueError, match='already marked running'):
        completed_job(tmp_path, config, job)


def test_isolated_rl_worker_metadata_survives_parent_commit(tmp_path, monkeypatch):
    from experiments.paper_benchmarks import exp3_rl_robustness as experiment
    config = load_config('exp3_rl_robustness', mode='smoke')
    job = jobs(config)[0]
    def worker(command, **kwargs):
        path = Path(command[-1]) / 'metadata.json'
        meta = json.loads(path.read_text())
        meta.update(checkpoint_path='checkpoint/actor', evaluation_backend='mjx',
                    evaluation_wall_seconds=1.25)
        write_json(path, meta)
        return SimpleNamespace(returncode=0)
    with run_directory(tmp_path, config, job) as run:
        monkeypatch.setattr(experiment.subprocess, 'run', worker)
        experiment.isolated_training(config, job, run)
    meta = json.loads((run.path / 'metadata.json').read_text())
    assert meta['status'] == 'complete'
    assert meta['checkpoint_path'] == 'checkpoint/actor'
    assert meta['evaluation_backend'] == 'mjx'
    assert meta['evaluation_wall_seconds'] == 1.25


def test_training_health_distinguishes_numerics_from_performance():
    from experiments.paper_benchmarks.aggregate import training_health
    base = dict(run_id='run', controller='sac', regime='nominal', training_seed=0,
                completed_episode_count=0, success_termination_step_count=0,
                mean_lateral_error=2., mean_angle_error=100., mean_episodic_returns=-1.)
    rows = [dict(base, environment_steps=i+1, epoch_seconds=1.) for i in range(20)]
    rows[-2].update(completed_episode_count=1, success_termination_step_count=1)
    rows[-1].update(completed_episode_count=9, success_termination_step_count=0)
    health = training_health(rows).iloc[0]
    assert health.numerical_health_ok
    assert health.late_success_rate == pytest.approx(.1)  # Not the .5 mean of update rates.
    assert pd.isna(health.early_success_rate)  # No completed episodes is not zero success.
    rows[-1]['epoch_seconds'] = None
    assert not training_health(rows).iloc[0].numerical_health_ok


@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
@pytest.mark.parametrize("defect", [None, "checkpoint", "budget", "axis", "training"])
def test_resume_and_aggregation_share_training_validation(tmp_path, algorithm, defect):
    from experiments.paper_benchmarks.common import completed_job

    config = load_config("exp3_rl_robustness", mode="smoke")
    job = next(job for job in jobs(config)
               if job["controller"] == algorithm and job["regime"] == "nominal")
    hp = config["common"]["training"][algorithm.upper()]
    count = config["protocol"].get("training_num_envs", {}).get(
        algorithm, config["common"]["env"]["environment"]["num_envs"])
    budget = hp.get("total_transitions", hp.get("epochs", 0) * hp.get("steps_per_epoch", 0) * count)
    with run_directory(tmp_path, config, job) as run:
        for condition in config["common"]["evaluation_distributions"]:
            for trial in range(evaluation_trial_count(config, condition)):
                run.append(dict(condition=condition,
                                evaluation_seed=config["common"]["evaluation_seed_start"] + trial,
                                success=False, final_position_error=1., final_attitude_error=1.,
                                control_effort=0., termination_reason="timeout"))
        run.update(checkpoint_path="actor.ckpt")
        if defect != "checkpoint":
            (run.path / "actor.ckpt").mkdir()
        if defect != "training":
            axis = [budget, budget] if defect == "axis" else [budget - 1 if defect == "budget" else budget]
            for step in axis:
                run.append(dict(environment_steps=step), "training.jsonl")
    accepted, errors, _ = discover(tmp_path, "smoke", allow_partial=True)
    if defect is None:
        assert completed_job(tmp_path, config, job)
        assert len(accepted) == 1
    else:
        with pytest.raises(ValueError, match="missing checkpoint, incomplete budget"):
            completed_job(tmp_path, config, job)
        assert not accepted
        assert any("missing checkpoint, incomplete budget" in error for error in errors)


def test_removed_curriculum_is_rejected_including_smoke():
    from experiments.paper_benchmarks.common import validate_config, _configure_smoke

    config = load_config("exp3_rl_robustness", mode="development")
    config["protocol"]["sac_curriculum"] = [{"from_transition": 0, "scale": 1.0}]
    for smoke in (False, True):
        if smoke:
            _configure_smoke(config, "exp3_rl_robustness")
        with pytest.raises(ValueError, match="no longer supported"):
            validate_config(config)


@pytest.mark.parametrize("key", ["checkpoint_intervals", "training_num_envs", "trials"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5])
def test_protocol_integer_settings_reject_invalid_values(key, value):
    from experiments.paper_benchmarks.common import validate_config

    config = load_config("exp3_rl_robustness", mode="development")
    config["protocol"][key] = value if key == "trials" else {"sac": value}
    with pytest.raises(ValueError, match="positive integer"):
        validate_config(config)


def test_removed_training_evaluation_is_rejected():
    from experiments.paper_benchmarks.common import validate_config

    config = load_config("exp3_rl_robustness", mode="development")
    config["protocol"]["sac_training_evaluation"] = dict(
        interval_transitions=10, trials=2, seed_start=30000)
    with pytest.raises(ValueError, match="no longer supported"):
        validate_config(config)


def test_rl_suite_distinctness_pairing_and_single_fixed_probe():
    config = load_config('exp3_rl_robustness', mode='paper')
    common = config['common']
    scenarios = {}
    for condition, distribution in common['evaluation_distributions'].items():
        count = evaluation_trial_count(config, condition)
        scenarios[condition] = [sample_trial(common, common['evaluation_seed_start']+i, distribution, 12)
                                for i in range(count)]
        physical = [{k:s[k] for k in ('initial_qpos','initial_qvel','mass_scale','inertia_scale','thrust_scale','wrench')}
                    for s in scenarios[condition]]
        assert len({digest(s) for s in physical})==count
    assert len(scenarios['nominal'])==1
    assert sum(map(len, scenarios.values()))==401
    for condition in ['thrust_variation','disturbance','combined']:
        for nominal, perturbed in zip(scenarios['randomized_initial'], scenarios[condition], strict=True):
            np.testing.assert_array_equal(nominal['initial_qpos'],perturbed['initial_qpos'])
            np.testing.assert_array_equal(nominal['initial_qvel'],perturbed['initial_qvel'])


@pytest.mark.parametrize('overrides', [{'nominal':0}, {'missing':1}, {'nominal':True}, []])
def test_invalid_condition_counts_rejected(overrides):
    from experiments.paper_benchmarks.common import validate_config
    config = load_config('exp3_rl_robustness', mode='smoke')
    config['protocol']['evaluation_trials']=overrides
    with pytest.raises(ValueError,match='evaluation_trials'):
        validate_config(config)


def test_historical_uniform_trial_counts_remain_valid():
    config = load_config('exp3_rl_robustness', mode='paper')
    config['protocol'].pop('evaluation_trials')
    assert evaluation_trial_count(config,'nominal')==100
