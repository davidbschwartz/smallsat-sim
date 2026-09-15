"""One sequential campaign, retaining failed jobs and reporting incomplete artifacts."""

import subprocess
import sys

from .common import EXPERIMENTS, mode_of, parser


def main(argv=None):
    p = parser("reproduce_all")
    args = p.parse_args(argv)
    if args.config:
        p.error("Run per-experiment entry points for custom configurations")
    if args.job_id:
        p.error("Use per-experiment entry points for --job-id")
    mode = "--paper" if args.paper else "--smoke" if args.smoke else None
    failures = []
    for experiment in EXPERIMENTS:
        cmd = [
            sys.executable,
            "-m",
            "experiments.demonstration_use_cases." + experiment,
            "--output",
            str(args.output),
        ]
        if mode:
            cmd.append(mode)
        if args.plan:
            cmd.append("--plan")
        if args.resume:
            cmd.append("--resume")
        if subprocess.run(cmd).returncode:
            failures.append(experiment)
    if args.plan:
        return
    if failures:
        raise RuntimeError(
            "Incomplete campaign: " + ", ".join(failures) + ". Failed run records retained."
        )
    from .aggregate import aggregate

    aggregate(
        args.output,
        args.output / "artifacts",
        mode=mode_of(args),
    )


if __name__ == "__main__":
    main()
