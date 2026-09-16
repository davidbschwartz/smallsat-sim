"""Plot explicit benchmark JSON or training JSONL files; never infer run variants."""

from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--metric", default="success_rate")
    parser.add_argument("--stage", default="policy_training")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/rl_results/comparison.png"),
    )
    args = parser.parse_args(argv)
    records = []
    for path in args.files:
        records.extend(
            [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if path.suffix == ".jsonl"
            else json.loads(path.read_text())
        )
    data = pd.DataFrame(records)
    if args.metric not in data:
        parser.error(f"Metric {args.metric} is absent from these results")
    figure, axes = plt.subplots(figsize=(10, 5), layout="constrained")
    if "scenario" in data:
        data["label"] = (
            data["experiment"]
            + " / "
            + data["context_source"]
            + " / "
            + data["backend"]
        )
        # Average evaluation replicates within each training seed first.
        seeds = data.groupby(["label", "scenario", "training_seed"])[args.metric].mean()
        means = seeds.groupby(["label", "scenario"]).mean().unstack("label")
        means.plot.bar(ax=axes)
        axes.set_xlabel("Scenario (bars average independent training seeds)")
    else:
        data = data[data.stage == args.stage]
        for (experiment, seed), group in data.groupby(["experiment", "seed"]):
            axes.plot(
                group.epoch, group[args.metric], label=f"{experiment}, seed {seed}"
            )
        axes.set_xlabel("Epoch")
        axes.legend()
    axes.set_ylabel(args.metric)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
