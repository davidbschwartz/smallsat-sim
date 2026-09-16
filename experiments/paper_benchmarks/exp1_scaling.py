"""Synchronized steady-state policy rollouts; one isolated process per batch size."""

import subprocess
import json
import sys
import time

from .common import execute, parser, write_json


def benchmark(config, job, run):
    import jax
    from flax import nnx

    from .runtime import build_runner

    protocol = config["protocol"]
    if protocol["require_gpu"] and jax.default_backend() != "gpu":
        raise RuntimeError(
            "Paper GPU scaling requires a GPU backend. --smoke allows CPU verification."
        )
    runner = build_runner(config, job, run)
    try:
        collector = runner.collector("zero", stochastic=True, steps=protocol["rollout_steps"])
        env = runner.env
        state = (
            env.freeflyer_state_struct() if protocol["backend"] == "freeflyer" else env.state_struct
        )
        inputs = (state, nnx.state(runner.agent.actor), nnx.state(runner.agent.critic), None)

        def rollout():
            result = collector(*inputs, runner._take_keys(), runner.reference_point)
            jax.block_until_ready(result)

        for _ in range(protocol["warmups"]):
            rollout()
        for repeat in range(protocol["repeats"]):
            start = time.perf_counter()
            rollout()
            elapsed = time.perf_counter() - start
            num_envs = job["batch_size"]
            steps = protocol["rollout_steps"]
            dt = float(env.model.opt.timestep)
            decimation = int(env.env_cfg.environment.control_decimation)
            throughput = num_envs * steps / elapsed
            run.append(
                dict(
                    repeat=repeat,
                    status="ok",
                    num_envs=num_envs,
                    rollout_steps=steps,
                    sim_dt=dt,
                    control_decimation=decimation,
                    wall_seconds=elapsed,
                    env_steps_per_second=throughput,
                    env_steps_per_second_per_env=throughput / num_envs,
                    sim_seconds_per_second=throughput * dt * decimation,
                    sim_seconds_per_second_per_env=throughput * dt * decimation / num_envs,
                    device_memory_stats=jax.devices()[0].memory_stats(),
                )
            )
    finally:
        runner.env.close()


def isolated(config, job, run):
    # Allocator pools and OOM failures cannot contaminate the next requested batch.
    write_json(run.path / "job.json", job)
    with (run.path / "worker.log").open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.paper_benchmarks.exp1_scaling",
                "--worker",
                str(run.path.resolve()),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    run.meta = json.loads((run.path / "metadata.json").read_text())
    if result.returncode:
        message = (run.path / "worker.log").read_text()
        if any(
            s in message.lower()
            for s in ("out of memory", "resource_exhausted", "cuda_error_out_of_memory")
        ):
            run.append(dict(status="oom", num_envs=job["batch_size"], error=message[-2000:]))
        else:
            raise RuntimeError(
                f"Worker failed ({result.returncode}); see {run.path}/worker.log\n{message[-1000:]}"
            )


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--worker":
        from .common import Run

        run = Run.open_existing(argv[1])
        benchmark(run.config, run.job, run)
        return
    args = parser("exp1_scaling").parse_args(argv)
    execute("exp1_scaling", args, isolated)


if __name__ == "__main__":
    main()
