"""Consistent vector publication plots, generated only from canonical CSVs."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "smallsat-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STYLE = {
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
}


def make_figures(data, output, *, label=None):
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    annotation = label

    def save(fig, name, exp):
        if annotation:
            fig.text(0.5, 0.005, annotation, ha="center", fontsize=7, color="0.35")
        fig.tight_layout(rect=(0, 0.035 if annotation else 0, 1, 1))
        for suffix in ("pdf", "png"):
            path = output / f"{name}.{suffix}"
            kwargs = (
                {"metadata": {"CreationDate": None, "ModDate": None}}
                if suffix == "pdf"
                else {"dpi": 200}
            )
            fig.savefig(path, **kwargs)
            artifacts.append((path, exp))
        plt.close(fig)

    with plt.rc_context(STYLE):
        path = data / "exp1_scaling.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame = frame[frame.status == "ok"]
            if not frame.empty:
                stats = frame.groupby("num_envs").env_steps_per_second.agg(["mean", "std"])
                fig, ax = plt.subplots(figsize=(3.5, 2.5))
                ax.errorbar(
                    stats.index, stats["mean"], yerr=stats["std"].fillna(0), fmt="ko-", capsize=2
                )
                ax.set(
                    xscale="log",
                    yscale="log",
                    xlabel="Parallel environments",
                    ylabel="Environment steps / s",
                )
                save(fig, "exp1_scaling", "exp1_scaling")
        for prefix, exp, title in [
            ("exp2", "exp2_fault_robustness", "exp2_fault_robustness"),
            ("exp3", "exp3_rl_robustness", "exp3_robustness"),
            ("exp4", "exp4_docking", "exp4_docking_statistics"),
        ]:
            path = data / f"{prefix}_summary.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            if prefix == "exp3":
                frame["method"] = frame.controller + " " + frame.regime
                fig, ax = plt.subplots(figsize=(7.1, 3.3))
                conditions = list(frame.condition.unique())
                methods = list(frame.method.unique())
                width = 0.8 / len(methods)
                hatches = ["", "//", "xx", "..", "\\\\", "++", "oo", "--"]
                for i, method in enumerate(methods):
                    rows = frame[frame.method == method].set_index("condition").reindex(conditions)
                    # Between-training-seed uncertainty for learned policies; Wilson for baselines.
                    low = rows.success_ci_low
                    high = rows.success_ci_high
                    if "seed_success_ci_low" in rows:
                        low = rows.seed_success_ci_low.fillna(low)
                        high = rows.seed_success_ci_high.fillna(high)
                    if "success_seed_ci_low" in rows:
                        low = rows.success_seed_ci_low.fillna(low)
                        high = rows.success_seed_ci_high.fillna(high)
                    ax.bar(
                        np.arange(len(conditions)) - 0.4 + (i + 0.5) * width,
                        rows.success_rate,
                        width,
                        label=method,
                        hatch=hatches[i % len(hatches)],
                        edgecolor="black",
                        linewidth=0.4,
                        yerr=np.maximum(
                            0, np.array([rows.success_rate - low, high - rows.success_rate])
                        ),
                        capsize=1,
                    )
                ax.set_xticks(
                    np.arange(len(conditions)), [c.replace("_", "\n") for c in conditions]
                )
                ax.set(ylabel="Success proportion", ylim=(0, 1.05))
                ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.4))
            else:
                metrics = (
                    ["success_rate", "final_position_error_mean", "control_effort_mean"]
                    if prefix == "exp2"
                    else ["success_rate", "peak_contact_force_mean", "final_position_error_mean"]
                )
                labels = (
                    ["Success proportion", "Final position error [m]", "Command impulse [N s]"]
                    if prefix == "exp2"
                    else [
                        "Success proportion",
                        "Peak summed contact force [N]",
                        "Final position error [m]",
                    ]
                )
                fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.9))
                for ax, metric, label in zip(axes, metrics, labels):
                    trial_path = data / f"{prefix}_trials.csv"
                    if metric != "success_rate" and trial_path.exists():
                        trials = pd.read_csv(trial_path)
                        raw_metric = metric.removesuffix("_mean")
                        samples = [
                            trials.loc[trials.condition == condition, raw_metric].dropna()
                            for condition in frame.condition
                        ]
                        ax.boxplot(
                            samples,
                            positions=np.arange(len(frame)),
                            widths=0.6,
                            patch_artist=True,
                            boxprops={"facecolor": "0.8"},
                            medianprops={"color": "black"},
                            flierprops={"marker": ".", "markersize": 2},
                        )
                    else:
                        ax.bar(
                            np.arange(len(frame)),
                            frame[metric],
                            color="0.7",
                            edgecolor="black",
                            hatch="//",
                        )
                    ax.set_xticks(
                        np.arange(len(frame)),
                        frame.condition.str.replace("_", "\n"),
                        rotation=45,
                        ha="right",
                    )
                    ax.set_ylabel(label)
                    if metric == "success_rate":
                        ax.errorbar(
                            np.arange(len(frame)),
                            frame[metric],
                            yerr=np.maximum(
                                0,
                                np.array(
                                    [
                                        frame[metric] - frame.success_ci_low,
                                        frame.success_ci_high - frame[metric],
                                    ]
                                ),
                            ),
                            fmt="none",
                            ecolor="black",
                            capsize=2,
                        )
                        ax.set_ylim(0, 1.05)
            save(fig, title, exp)
            if prefix == "exp4":
                fig, ax = plt.subplots(figsize=(3.5, 2.5))
                ax.errorbar(
                    np.arange(len(frame)),
                    frame.peak_contact_force_mean,
                    yerr=frame.peak_contact_force_std.fillna(0),
                    fmt="ks",
                    capsize=3,
                )
                ax.set_xticks(np.arange(len(frame)), frame.condition)
                ax.set_ylabel("Peak summed contact force [N]")
                save(fig, "exp4_contact_sensitivity", exp)
        path = data / "exp3_learning_summary.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame = frame[frame.metric == "mean_episodic_returns"]
            algorithms = list(frame.controller.unique())
            fig, axes = plt.subplots(
                1, len(algorithms), figsize=(7.1, 2.5), sharey=True, squeeze=False
            )
            axes = axes[0]
            for ax, algorithm in zip(axes, algorithms):
                for regime, style in [("nominal", "-"), ("randomized", "--")]:
                    rows = frame[(frame.controller == algorithm) & (frame.regime == regime)]
                    x = rows.environment_steps.to_numpy()
                    y = rows["mean"].to_numpy()
                    std = rows["std"].fillna(0).to_numpy()
                    ax.plot(x, y, style, marker="o" if len(x) < 3 else None, label=regime)
                    ax.fill_between(x, y - std, y + std, alpha=0.15)
                ax.set(title=algorithm.upper(), xlabel="Environment steps")
                ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axes[0].set_ylabel("Episode return (mean ± SD)")
            axes[-1].legend()
            save(fig, "exp3_learning_curves", "exp3_rl_robustness")
        path = data / "exp4_representative.csv"
        if path.exists():
            frame = pd.read_csv(path)
            fig, axes = plt.subplots(3, 1, figsize=(3.5, 4.4), sharex=True)
            for ax, metric, label in zip(
                axes,
                ("position_error", "attitude_error", "contact_force"),
                ("Position error [m]", "Attitude error [rad]", "Contact force [N]"),
            ):
                ax.plot(frame.time, frame[metric], "k-")
                ax.set_ylabel(label)
            axes[-1].set_xlabel("Simulation time [s]")
            save(fig, "exp4_contact_trajectory", "exp4_docking")
    return artifacts
