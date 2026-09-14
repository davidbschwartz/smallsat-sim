"""Synchronized rollout/update timings; reports hardware without claiming speedups."""

from pathlib import Path
import argparse
import json
import time

from flax import nnx
import jax
import numpy as np

from .benchmark import build_runner
from smallsat_sim.controllers.rl.algorithms.ppo import update_ppo
from smallsat_sim.controllers.rl.storage.rollout_batch import make_training_batch
from smallsat_sim.envs.effects.scheduling import reset_and_randomize



def timed(call):
    start = time.perf_counter()
    result = call()
    jax.block_until_ready(result)
    return result, time.perf_counter() - start


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mjx", "freeflyer"), default="freeflyer")
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("docs/rl_cpu_profile.json"))
    args = parser.parse_args(argv)
    if min(args.envs, args.steps, args.repeats) < 1:
        parser.error("envs, steps and repeats must be positive")
    runner = build_runner(
        "ppo_nominal",
        0,
        log=False,
        overrides={
            "num_envs": args.envs,
            "rollout_backend": args.backend,
            "PPO": {
                "steps_per_epoch": args.steps,
                "max_ep_len": args.steps,
                "epochs": 1,
                "num_minibatches": 1,
                "actor_training_epochs": 1,
                "critic_training_epochs": 1,
            },
        },
    )
    try:
        collector = runner.collector("zero", stochastic=True)
        reset_and_randomize(runner.env, jax.random.PRNGKey(10), enabled=False)
        state = (
            runner.env.freeflyer_state_struct()
            if args.backend == "freeflyer"
            else runner.env.state_struct
        )
        state = jax.device_put(state, runner.agent.key.sharding)
        inputs = (
            state,
            nnx.state(runner.agent.actor),
            nnx.state(runner.agent.critic),
            None,
        )

        def rollout():
            key = jax.device_put(runner._take_keys(), runner.agent.key.sharding)
            return collector(*inputs, key, runner.reference_point)

        result, rollout_first = timed(rollout)
        rollout_times = [timed(rollout)[1] for _ in range(args.repeats)]
        batch = make_training_batch(
            result,
            runner.context_scale,
            runner.agent.gamma,
            runner.agent.lam,
        )
        _, update_first = timed(lambda: runner.agent.update(batch))
        update_times = [
            timed(lambda: runner.agent.update(batch))[1] for _ in range(args.repeats)
        ]

        def epoch():
            result = runner.collect(collector, randomize=False)
            batch = make_training_batch(
                result,
                runner.context_scale,
                runner.agent.gamma,
                runner.agent.lam,
            )
            return runner.agent.update(batch)

        timed(epoch)
        epoch_times = [timed(epoch)[1] for _ in range(args.repeats)]
        metrics = {
            "device": str(jax.devices()),
            "jax": jax.__version__,
            "backend": args.backend,
            "num_envs": args.envs,
            "steps": args.steps,
            "repeats": args.repeats,
            "actor_parameters": sum(
                x.size
                for x in jax.tree.leaves(nnx.state(runner.agent.actor, nnx.Param))
            ),
            "rollout_first_call_seconds": rollout_first,
            "rollout_steady_seconds": float(np.median(rollout_times)),
            "transitions_per_second": args.envs
            * args.steps
            / float(np.median(rollout_times)),
            "update_first_call_seconds": update_first,
            "update_steady_seconds": float(np.median(update_times)),
            "epoch_without_reporting_seconds": float(np.median(epoch_times)),
            "rollout_compilations": collector._cache_size(),
            "update_compilations": update_ppo._cache_size(),
            "device_memory_stats": jax.devices()[0].memory_stats(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2))
    finally:
        runner.env.close()


if __name__ == "__main__":
    main()
