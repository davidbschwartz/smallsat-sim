"""Fixed evaluation cases shared by learned and PD controllers."""

from dataclasses import dataclass

from flax import nnx
import jax
import jax.numpy as jnp

from smallsat_sim.configuration import read_yaml
from smallsat_sim.controllers.rl.runners.runner_metrics import report, episode_metrics
from smallsat_sim.controllers.rl.runners.runner_utils import save_training_data


@dataclass(frozen=True)
class Scenario:
    name: str
    failure_types: tuple[str, ...] = ()
    disturbance: bool = False
    onset_seconds: float = 5.0


_cases = read_yaml("benchmarks.yaml")
CORE = tuple(Scenario(**case) for case in _cases["core"])
STRESS = tuple(Scenario(**case) for case in _cases["stress"])


def prepare(env, scenario, seed):
    """Reset *all* environment RNG state before sampling identical realizations.

    Callers can persist the returned simulation state to preserve sampled
    actuator indices, fault functions, disturbances and initial conditions.
    """
    env._rng = jax.random.PRNGKey(seed)
    env.reset()
    env.reset_perturbations()
    env.reset_disturbances()
    keys = jax.random.split(
        jax.random.PRNGKey(seed + 1), len(scenario.failure_types) + 1
    )
    for index, kind in enumerate(scenario.failure_types):
        env.schedule_random_faults(
            keys[index], 1.0, {kind: 1.0}, start_time=scenario.onset_seconds
        )
    if scenario.disturbance:
        env.schedule_random_disturbances(keys[-1], 1.0, start_time=scenario.onset_seconds)
    return env.state_struct


def evaluate_suite(
    runner, *, source="estimated", suite="core", seed=10000, output=None
):
    if runner.am is None:
        source = "zero"
    runner.restore_for_evaluation(source)
    collector = runner.collector(
        source,
        stochastic=False,
        steps=runner.training_cfg.episode_len,
        mode="evaluation",
    )
    rows = []
    for index, scenario in enumerate(CORE + (STRESS if suite == "stress" else ())):
        visualization = runner.env.rollout_visualization(
            "evaluation",
            runner.env.build_step_config(),
        )
        state = prepare(runner.env, scenario, seed + index)
        if output is not None:
            save_training_data(
                output,
                f"scenario_{scenario.name}_seed{seed+index}.ckpt",
                {"state": state, "seed": seed + index},
            )
        if runner.env.env_cfg.environment.rollout_backend == "freeflyer":
            state = runner.env.freeflyer_state_struct()
        result = runner.collect_state(collector, state, jax.random.PRNGKey(seed + index))
        if visualization is not None:
            jax.block_until_ready(result.actions)
            jax.effects_barrier()
            visualization.finish()
        row = report(
            runner,
            "evaluation_" + source,
            index,
            episode_metrics(
                result,
                dt=runner.env.model.opt.timestep * runner.env.env_cfg.environment.control_decimation,
                onset_seconds=scenario.onset_seconds,
            ),
        )
        rows.append(
            {
                "scenario": scenario.name,
                "seed": seed + index,
                "context_source": source,
                "backend": runner.env.env_cfg.environment.rollout_backend,
                "experiment": runner.env.run_name,
                "training_seed": runner.env.env_cfg.sim.seed,
                **row,
            }
        )
    return rows
