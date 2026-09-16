"""Paper scenarios and artifacts for the standard runner.evaluate() API."""

from time import perf_counter

import numpy as np

from .common import digest, evaluation_trial_count, write_json
from .task import sample_trial


def evaluate_policy(config, job, run, runner):
    started = perf_counter()
    common, protocol = config["common"], config["protocol"]
    model = runner.env.model
    samples, identities = [], []
    counts = {}
    for condition, distribution in common["evaluation_distributions"].items():
        count = evaluation_trial_count(config, condition)
        counts[condition] = count
        seen = set()
        for trial in range(count):
            sample = sample_trial(
                common, common["evaluation_seed_start"] + trial, distribution, model.nu
            )
            physical = {
                k: sample[k]
                for k in (
                    "initial_qpos",
                    "initial_qvel",
                    "mass_scale",
                    "inertia_scale",
                    "thrust_scale",
                    "wrench",
                )
            }
            scenario_id = digest(physical)
            if scenario_id in seen and protocol.get("evaluation_trials"):
                raise ValueError(
                    f"{condition}: duplicate physical evaluation scenario; check sampling/counts"
                )
            seen.add(scenario_id)
            identities.append(
                dict(
                    condition=condition,
                    trial=trial,
                    scenario_id=scenario_id,
                    initial_state_id=digest(
                        {k: sample[k] for k in ("initial_qpos", "initial_qvel")}
                    ),
                )
            )
            samples.append(sample)
            run.append(dict(**identities[-1], **sample), "samples.jsonl")
    results, timings = runner.evaluate(
        scenarios=samples, batch_size=protocol.get("evaluation_batch_size", 128),
        steps=common["episode_steps"],
    )
    run.update(
        task="pose_regulation",
        evaluation_backend=timings["backend"],
        evaluation_device=timings["device"],
        evaluation_num_envs=timings["batch_size"],
        evaluation_episode_counts=counts,
        evaluation_scenario_count=len(samples),
    )
    write_json(run.path / "evaluation_timing.json", timings)
    for sample, identity, result in zip(samples, identities, results, strict=True):
        path = run.path / "recordings" / identity["condition"] / f"{identity['trial']:04d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, time=result["time"], qpos=result["qpos"])
        row = result["metrics"]
        run.append(
            dict(
                **job,
                **identity,
                evaluation_seed=sample["evaluation_seed"],
                initial_condition_seed=sample["initial_condition_seed"],
                training_seed=job["seed"],
                solver_status=0,
                mean_solve_seconds=None,
                max_solve_seconds=None,
                affected_thruster=None,
                fault_type=identity["condition"],
                normalized_control_effort=row["control_effort"]
                / max(float(model.actuator_ctrlrange[:, 1].sum()), 1e-12),
                **row,
            )
        )
    run.update(evaluation_wall_seconds=perf_counter() - started)
    print(
        f"{timings['backend']} evaluation: {len(samples)} scenarios, "
        f"{timings['batch_size']} lanes, {timings['wall_seconds']:.2f}s including compilation; "
        "artifact writing timed separately",
        flush=True,
    )
