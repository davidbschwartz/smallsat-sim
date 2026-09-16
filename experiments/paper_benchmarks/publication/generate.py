"""Build ICRA figures from frozen raw artifacts; never run simulation or interpolate data."""

from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import importlib
import json
import os
import shlex
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "smallsat-paper-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXPS = [
    "exp1_scaling",
    "exp2_fault_robustness",
    "exp3_rl_robustness",
    "exp4_docking",
    "spacecraft_portability",
]
METRICS = {
    "final_position_error": "Terminal position error [m]",
    "final_attitude_error": "Terminal attitude error [rad]",
    "control_effort": "Command impulse [N s]",
    "first_contact_time": "First contact time [s]",
    "peak_contact_force": "Peak contact force [N]",
}
STYLE = {
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}
METHODS = [("ppo", "nominal"), ("ppo", "randomized"), ("sac", "nominal"), ("sac", "randomized")]
COLORS = ["#0072B2", "#0072B2", "#D55E00", "#D55E00"]
MARKERS = ["o", "s", "^", "D"]


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def exact_learning(frame):
    """Aggregate observed steps only; retain varying seed counts, never interpolate."""
    keys = ["controller", "regime", "training_seed", "environment_steps"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate training seed/step")
    return (
        frame.groupby(["controller", "regime", "environment_steps"])
        .mean_episodic_returns.agg(mean="mean", std="std", n_seeds="count")
        .reset_index()
    )


def windowed_learning(frame, width):
    """Average every logged return within right-closed interaction windows, per seed."""
    frame = frame.copy()
    frame["window_end"] = ((frame.environment_steps - 1) // width + 1) * width
    per_seed = (
        frame.groupby(["controller", "regime", "training_seed", "window_end"])
        .agg(
            mean_episodic_returns=("mean_episodic_returns", "mean"),
            recorded_updates=("mean_episodic_returns", "size"),
            finite_updates=("mean_episodic_returns", "count"),
        )
        .reset_index()
    )
    summary = (
        per_seed.groupby(["controller", "regime", "window_end"])
        .mean_episodic_returns.agg(mean="mean", std="std", n_seeds="count")
        .reset_index()
    )
    return per_seed, summary


def representative(frame, metric):
    successful = frame[frame.success & frame[metric].notna()]
    candidates = successful if len(successful) else frame[frame[metric].notna()]
    if candidates.empty:
        raise ValueError(f"No executed representative for {metric}")
    selected = candidates.sort_values([metric, "evaluation_seed", "run_id"]).iloc[
        (len(candidates) - 1) // 2
    ]
    return selected, "lower median successful" if len(
        successful
    ) else "lower median failed (no successes)"


class Build:
    def __init__(self, source, output, docking_results=None):
        self.source, self.output = source.resolve(), output.resolve()
        self.docking_results = docking_results.resolve() if docking_results else None
        self.command = "python -m experiments.paper_benchmarks.publication.generate"
        if self.docking_results:
            self.command += " --docking-results " + shlex.quote(os.path.relpath(self.docking_results, ROOT))
        if self.output == self.source or self.output.is_relative_to(self.source):
            raise ValueError("Output must be outside source artifacts")
        self.artifacts, self.issues, self.inputs = [], [], defaultdict(set)
        self.runs, self.frames, self.configs = {}, {}, {}
        sys.path.insert(0, str(self.source / "paper-results"))
        self.common = importlib.import_module("experiment_source.common")
        self.aggregate = importlib.import_module("experiment_source.aggregate")

    def relative(self, path):
        return os.path.relpath(path, ROOT)

    def register(self, path, exp, purpose, counts="", convention="", extra=()):
        self.artifacts.append(
            dict(
                filename=str(path.relative_to(self.output)),
                purpose=purpose,
                sources=sorted(self.relative(p) for p in self.inputs[exp] | set(extra)),
                script="experiments/paper_benchmarks/publication/generate.py",
                command=self.command,
                samples=counts,
                statistics=convention,
                caveats=self.issues.copy(),
            )
        )

    def csv(self, frame, name, exp, purpose, convention="Raw values; blank means unavailable."):
        path = self.output / "tables" / (name + ".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, float_format="%.17g")
        self.register(path, exp, purpose, f"{len(frame)} rows", convention)

    def figure(self, fig, name, exp, purpose, counts="", convention="", extra=()):
        fig.tight_layout(pad=0.6, rect=(0, 0, 1, 0.90) if fig.legends else (0, 0, 1, 1))
        for suffix in ["pdf", "png"]:
            path = self.output / (name + "." + suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                path,
                dpi=400,
                metadata={"CreationDate": None, "ModDate": None} if suffix == "pdf" else None,
            )
            self.register(path, exp, purpose, counts, convention, extra)
        plt.close(fig)

    def load(self):
        training, audit = [], []
        for exp in EXPS:
            rows, seen = [], set()
            protocol_path = (
                self.source / "paper-results/experiment_source/configs" / exp / "paper.yaml"
            )
            runs_root = self.source / "paper-results/raw_runs" / exp
            if exp == "exp4_docking" and self.docking_results:
                runs_root = self.docking_results / "paper" / exp
                protocol_path = next(iter(sorted(runs_root.glob("*/resolved_config.yaml"))))
            protocol = yaml.safe_load(protocol_path.read_text())
            if exp == "exp4_docking" and self.docking_results:
                protocol = protocol["protocol"]
            self.inputs[exp].add(protocol_path)
            for path in sorted(runs_root.iterdir()):
                meta = json.loads((path / "metadata.json").read_text())
                cfg = yaml.safe_load((path / "resolved_config.yaml").read_text())
                if exp in self.configs and cfg != self.configs[exp]:
                    raise ValueError(f"Mixed frozen configs: {exp}")
                self.configs[exp] = cfg
                if meta["job_id"] in seen:
                    raise ValueError(f"Duplicate run: {path}")
                seen.add(meta["job_id"])
                self.runs[meta["run_id"]] = (path, meta)
                raw = records(path / "metrics.jsonl")
                self.inputs[exp].update(
                    [path / "metadata.json", path / "resolved_config.yaml", path / "metrics.jsonl"]
                )
                if meta["status"] != "complete":
                    self.issues.append(f"{meta['run_id']}: status={meta['status']}")
                try:
                    self.aggregate.validate_rows(cfg, meta["job"], raw)
                except ValueError as error:
                    self.issues.append(str(error))
                identities = [
                    (r.get("condition"), r.get("evaluation_seed"), r.get("repeat")) for r in raw
                ]
                if len(set(identities)) != len(identities):
                    raise ValueError(f"Duplicate measurement: {path}")
                for r in raw:
                    if r.get("status") not in (None, "ok"):
                        self.issues.append(f"{meta['run_id']}: measurement status={r['status']}")
                    for metric in METRICS:
                        value = r.get(metric)
                        if value is not None and not np.isfinite(value):
                            raise ValueError(f"{path}: nonfinite {metric}; source requires repair")
                    rows.append({**r, "run_id": meta["run_id"]})
                audit.append(
                    dict(
                        experiment=exp,
                        run_id=meta["run_id"],
                        status=meta["status"],
                        rows=len(raw),
                        failed_trials=sum(r.get("success") is False for r in raw),
                    )
                )
                if exp == EXPS[2] and meta["job"]["regime"] != "baseline":
                    p = path / "training.jsonl"
                    if not p.exists():
                        self.issues.append(f"Missing training log: {p}")
                        continue
                    self.inputs[exp].add(p)
                    tr = records(p)
                    steps = [r["environment_steps"] for r in tr]
                    if any(b <= a for a, b in zip([0] + steps, steps)):
                        raise ValueError(f"Nonmonotonic training steps: {p}")
                    tc = cfg["common"]["training"][meta["controller"].upper()]
                    budget = tc.get(
                        "total_transitions",
                        tc.get("epochs", 0)
                        * tc.get("steps_per_epoch", 0)
                        * meta["training_num_envs"],
                    )
                    if not steps or steps[-1] != budget:
                        self.issues.append(
                            f"Incomplete training budget: {meta['run_id']}, expected {budget}"
                        )
                    training.extend(
                        {**r, "training_seed": meta["seed"], "run_id": meta["run_id"]} for r in tr
                    )
            cfg = self.configs.get(exp, {"protocol": protocol})
            for job in self.common.jobs(cfg):
                if self.common.job_id(job) not in seen:
                    self.issues.append(f"Missing job: {exp}/{self.common.job_id(job)}")
            self.frames[exp] = pd.DataFrame(rows)
        self.training = pd.DataFrame(training)
        for identity, group in self.training.groupby(["controller", "regime"]):
            expected = set(self.configs[EXPS[2]]["protocol"]["seeds"])
            actual = set(group.training_seed.unique())
            if expected != actual:
                self.issues.append(
                    f"{identity}: expected seeds {sorted(expected)}, observed {sorted(actual)}"
                )
            finite = np.isfinite(group.mean_episodic_returns)
            if not finite.all():
                self.issues.append(
                    f"{identity}: {(~finite).sum()} unavailable training return measurements"
                )
            counts = group.groupby("environment_steps").training_seed.nunique()
            if (counts != len(expected)).any():
                self.issues.append(
                    f"{identity}: {(counts != len(expected)).sum()} observed steps lack all expected seeds; no interpolation"
                )
        self.inputs["audit"] = set().union(*self.inputs.values())
        self.csv(
            pd.DataFrame(audit),
            "run_audit",
            "audit",
            "All run completion states and failed trial counts",
        )

    def distributions(self, frame, order, metrics, name, exp):
        # Share readable condition labels instead of repeating tiny rotated text.
        fig, axes = plt.subplots(
            1, len(metrics), figsize=(7.1, 3.4), squeeze=False, sharey=True
        )
        for ax, metric in zip(axes[0], metrics):
            samples = [frame.loc[frame.condition == c, metric].dropna().to_numpy() for c in order]
            ax.boxplot(
                samples,
                orientation="horizontal",
                tick_labels=[c.replace("_", " ").capitalize() for c in order],
                widths=0.55,
                showfliers=False,
                medianprops={"color": "black"},
                boxprops={"color": "0.35"},
            )
            for i, values in enumerate(samples):
                # Deterministic jitter; every observed value shown, including failures.
                jitter = np.random.default_rng(0).uniform(-0.17, 0.17, len(values))
                ax.scatter(values, i + 1 + jitter, s=3, alpha=0.35, color="#0072B2", linewidths=0)
            label = METRICS[metric].replace(" error [", " error\n[").replace(" impulse [", " impulse\n[")
            ax.set_xlabel(label, fontsize=10)
            ax.tick_params(axis="x", labelsize=9)
            ax.tick_params(axis="y", labelsize=10, length=0)
            ax.spines["left"].set_visible(False)
            ax.grid(axis="x", alpha=0.15)
        axes[0, 0].invert_yaxis()
        self.figure(
            fig,
            name,
            exp,
            "Monte Carlo distributions",
            f"{len(frame)} trials",
            "All trials; boxes Q1/median/Q3, whiskers 1.5 IQR; every finite sample plotted; unavailable counts in tables.",
        )

    def quantitative(self):
        exp = EXPS[0]
        f = self.frames[exp].copy()
        metas = [self.runs[r][1] for r in f.run_id]
        f["hardware"] = ["; ".join(m["devices"]) for m in metas]
        f["backend"] = [
            self.configs[exp]["protocol"]["backend"] + " / " + m["jax_backend"] for m in metas
        ]
        f["controller_timestep_s"] = [m["controller_timestep"] for m in metas]
        f["aggregate_sim_s_per_wall_s"] = f.env_steps_per_second * f.controller_timestep_s
        f["per_environment_sim_s_per_wall_s"] = f.aggregate_sim_s_per_wall_s / f.num_envs
        good = f[f.status == "ok"]
        baseline = good.loc[good.num_envs == 1, "env_steps_per_second"].mean()
        f["speedup_vs_one_environment"] = f.env_steps_per_second / baseline
        if not np.allclose(good.sim_seconds_per_second, good.aggregate_sim_s_per_wall_s):
            raise ValueError("Scaling time conversion disagrees with source")
        self.csv(
            f,
            "scaling_measurements",
            exp,
            "Every measured repeat; simulator and controller timesteps distinguished",
        )
        stats = (
            f.groupby(
                ["num_envs", "hardware", "backend", "sim_dt", "controller_timestep_s"], dropna=False
            )
            .agg(
                repeats=("env_steps_per_second", "count"),
                aggregate_env_steps_per_s=("env_steps_per_second", "mean"),
                throughput_sd=("env_steps_per_second", "std"),
                aggregate_sim_s_per_wall_s=("aggregate_sim_s_per_wall_s", "mean"),
                per_environment_sim_s_per_wall_s=("per_environment_sim_s_per_wall_s", "mean"),
                speedup=("speedup_vs_one_environment", "mean"),
            )
            .reset_index()
        )
        self.csv(
            stats,
            "scaling",
            exp,
            "Scaling table",
            "Arithmetic mean over repeats; sample SD (ddof=1); speedup vs N=1 same backend.",
        )
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        ax.errorbar(
            stats.num_envs,
            stats.aggregate_env_steps_per_s,
            yerr=stats.throughput_sd,
            fmt="ko-",
            capsize=2,
            markersize=3,
        )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Parallel environments",
            ylabel="Aggregate environment steps / s",
        )
        ax.set_xticks(stats.num_envs, stats.num_envs.astype(str), rotation=45)
        self.figure(
            fig,
            "figures/scaling/throughput",
            exp,
            "GPU throughput",
            f"{len(f)} repeats",
            "Mean ± sample SD; A100; freeflyer backend; 0.05 s control step.",
        )
        for exp in EXPS[1:]:
            f = self.frames[exp]
            groups = [c for c in ["spacecraft", "controller", "regime", "condition"] if c in f]
            summary = self.aggregate.summarize(f, groups)
            # Add contact time, success counts, episode counts per seed, and missing-value counts.
            for idx, row in summary.iterrows():
                subset = f
                for key in groups:
                    subset = subset[subset[key] == row[key]]
                summary.loc[idx, "success_count"] = int(subset.success.sum())
                summary.loc[idx, "failed_count"] = int((~subset.success).sum())
                for metric in METRICS:
                    if metric not in subset:
                        continue
                    v = subset[metric]
                    finite = v.notna() & np.isfinite(pd.to_numeric(v))
                    summary.loc[idx, metric + "_missing"] = int((~finite).sum())
                    if (~finite).any():
                        self.issues.append(
                            f"{exp}/{row['condition']}/{metric}: {(~finite).sum()} unavailable of {len(v)}"
                        )
                    if metric == "first_contact_time":
                        for stat, val in [
                            ("n", finite.sum()),
                            ("mean", v[finite].mean()),
                            ("std", v[finite].std()),
                            ("median", v[finite].median()),
                            ("q25", v[finite].quantile(0.25)),
                            ("q75", v[finite].quantile(0.75)),
                        ]:
                            summary.loc[idx, metric + "_" + stat] = val
                if "training_seed" in subset and subset.training_seed.notna().any():
                    counts = subset.groupby("training_seed").size()
                    summary.loc[idx, "episodes_per_seed"] = json.dumps(
                        {int(k): int(v) for k, v in counts.items()}, sort_keys=True
                    )
                if exp == EXPS[3]:
                    for key in ["solref_damping", "solref_time_constant"]:
                        summary.loc[idx, key] = subset[key].iloc[0]
            self.csv(f, exp + "_trials", exp, "All trial-level results including failures")
            self.csv(
                summary,
                exp + "_summary",
                exp,
                "Exact numerical summaries",
                "Success count/n with Wilson 95% interval; learned-policy seed mean ± Student-t 95% interval (clipped to [0,1]); continuous metrics mean, sample SD, median, Q1/Q3 over finite trials with n and missing counts.",
            )
        exp = EXPS[1]
        self.distributions(
            self.frames[exp],
            self.configs[exp]["protocol"]["conditions"],
            list(METRICS)[:3],
            "figures/model_based_robustness/distributions",
            exp,
        )
        self.rl()
        self.docking()
        self.portability()

    def rl(self, protocol_path=None, evaluation_points=False):
        exp = EXPS[2]
        f = self.frames[exp]
        # Preserve ordering in the authored frozen protocol (resolved YAML was key-sorted).
        p = protocol_path or self.source / "paper-results/experiment_source/configs/exp3_rl_robustness/paper.yaml"
        order = list(yaml.safe_load(p.read_text())["evaluation_distributions"])
        self.csv(
            pd.DataFrame(
                [
                    dict(condition=k, parameters=json.dumps(v, sort_keys=True))
                    for k, v in yaml.safe_load(p.read_text())["evaluation_distributions"].items()
                ]
            ),
            "rl_suite",
            exp,
            "Actual frozen evaluation suite",
        )
        learning = exact_learning(self.training)
        self.csv(
            learning,
            "learning_observed_steps",
            exp,
            "Learning curve values without interpolation",
            "Mean ± sample SD at exactly shared recorded steps; n_seeds reported for every point.",
        )
        # Match the coarsest saved logging cadence, retaining every update in a window.
        width = int(
            max(
                g.environment_steps.sort_values().diff().median()
                for _, g in self.training.groupby(["controller", "regime", "training_seed"])
            )
        )
        per_seed, learning = windowed_learning(self.training, width)
        self.csv(
            per_seed,
            "learning_window_seed_means",
            exp,
            "All recorded training updates aggregated per seed",
            f"Right-closed {width}-interaction windows; arithmetic mean of logged returns per seed, then equal-weight seed mean and sample SD; no interpolation. Counts include every update.",
        )
        self.csv(
            learning,
            "learning_window_summary",
            exp,
            "Main learning figure values",
            f"Mean ± sample SD across seed means in {width}-interaction windows.",
        )
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1), gridspec_kw={"width_ratios": [1, 1.08]})
        diag, daxes = plt.subplots(2, 2, figsize=(7.1, 4.3), sharex=True)
        eval_diag, eval_axes = plt.subplots(2, 2, figsize=(7.1, 4.5), sharex=True, sharey=True)
        for i, (method, regime) in enumerate(METHODS):
            label = f"{method.upper()} {regime}"
            color = COLORS[i]
            style = "-" if regime == "nominal" else "--"
            rows = learning[(learning.controller == method) & (learning.regime == regime)]
            if rows.empty:
                self.issues.append(f"Missing learning condition: {label}")
                continue
            x = rows.window_end.to_numpy() / 1e6
            y = rows["mean"].to_numpy()
            sd = rows["std"].to_numpy()
            axes[0].plot(
                x,
                y,
                style,
                color=color,
                label=label,
                linewidth=0.7,
                marker=MARKERS[i],
                markevery=max(1, len(x) // 10),
                markersize=2,
            )
            axes[0].fill_between(x, y - sd, y + sd, color=color, alpha=0.12)
            for seed, g in self.training[
                (self.training.controller == method) & (self.training.regime == regime)
            ].groupby("training_seed"):
                dax = daxes.flat[i]
                dax.plot(
                    g.environment_steps,
                    g.mean_episodic_returns,
                    label=f"seed {seed}",
                    linestyle=["-", "--", ":", "-.", (0, (3, 1, 1, 1, 1, 1))][int(seed) % 5],
                    linewidth=0.6,
                )
            dax.set(ylabel=f"{label}\nTraining return", xlabel="Environment interactions")
            dax.legend(ncol=3)
            rates = (
                f[(f.controller == method) & (f.regime == regime)]
                .groupby(["condition", "training_seed"])
                .success.mean()
            )
            means = rates.groupby("condition").mean().reindex(order)
            seed_sd = rates.groupby("condition").std().reindex(order)
            seed_count = rates.groupby("condition").count().reindex(order)
            half_width = self.aggregate.t.ppf(0.975, seed_count - 1) * seed_sd / np.sqrt(seed_count)
            lower = (means - half_width).clip(lower=0)
            upper = (means + half_width).clip(upper=1)
            xx = np.arange(len(order))
            # Small categorical offsets separate overlapping markers; y values are exact.
            display_x = xx + (i - 1.5) * (0.14 if evaluation_points else 0.055)
            axes[1].errorbar(
                display_x,
                means,
                yerr=np.array([means - lower, upper - means]),
                linestyle="none" if evaluation_points else style,
                color=color,
                marker=MARKERS[i],
                markersize=3.5,
                markerfacecolor="white",
                markeredgewidth=0.7,
                linewidth=1,
                elinewidth=0.65,
                capsize=2,
                capthick=0.65,
                label=label,
            )
            for seed, values in rates.unstack("condition").reindex(columns=order).iterrows():
                eval_axes.flat[i].plot(
                    xx,
                    values,
                    linewidth=0.7,
                    marker=".",
                    markersize=3,
                    label=f"Seed {int(seed)}",
                    linestyle=["-", "--", ":", "-.", (0, (3, 1, 1, 1, 1, 1))][int(seed) % 5],
                )
            eval_axes.flat[i].set(ylabel=label + "\nSuccess", ylim=(-0.04, 1.05))
            eval_axes.flat[i].set_xticks(xx, [c.replace("_", "\n") for c in order], fontsize=6)
            eval_axes.flat[i].legend(fontsize=6, ncol=3, loc="lower right", frameon=False)
        self.figure(
            eval_diag,
            "diagnostics/rl/evaluation_seeds",
            exp,
            "All evaluation seed rates, including extreme runs",
            "5 seeds per learned condition",
            "Each curve is one training seed, 100 evaluation episodes per suite condition. No smoothing or omitted seeds.",
        )
        axes[0].set(
            xlabel="Environment interactions [millions]",
            ylabel="Training episode return",
            xlim=(0, 680),
            ylim=(-25, 95),
        )
        axes[0].set_xticks([0, 200, 400, 600])
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=4,
            frameon=False,
            fontsize=7,
            handlelength=2.4,
            columnspacing=1.5,
        )
        axes[1].set_xticks(
            range(len(order)), [c.replace("_", "\n").capitalize() for c in order], fontsize=6.5
        )
        axes[1].set(ylabel="Evaluation success", ylim=(-0.035, 1.14), xlim=(-0.22, 4.22))
        axes[1].set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
        for ax in axes:
            ax.grid(axis="y", color="0.92", linewidth=0.5)
            ax.set_axisbelow(True)
        self.figure(
            fig,
            "figures/rl/learning_and_shift",
            exp,
            "Training and distribution-shift response",
            "5 training seeds per method/regime; 100 evaluation episodes per seed/condition",
            "Learning: mean ± sample SD across per-seed window means, using the coarsest recorded interaction cadence (2,097,152 interactions); every logged update contributes, no interpolation. Raw unsmoothed curves remain in diagnostics. Evaluation: lines show equal-weight seed means; thin capped bars show Student-t 95% confidence intervals across training seeds, clipped to [0,1]. Hollow markers use small horizontal categorical offsets to separate overlapping methods, without changing success values. The full 0–100% range and shared legend are retained. All individual seed rates remain in diagnostics/rl/evaluation_seeds. Suite has no dedicated OOD condition; thrust variation is in training range.",
        )
        self.figure(
            diag,
            "diagnostics/rl/individual_seeds",
            exp,
            "All individual training curves",
            "20 training seeds",
            "No smoothing or interpolation.",
        )

    def docking(self):
        exp = EXPS[3]
        f = self.frames[exp]
        default = f[f.condition == "default"]
        chosen, rule = representative(default, "peak_contact_force")
        self.selection = chosen.to_dict()
        root = (
            self.source
            / "demonstration_all_recordings_traces/results/demonstration_campaign_final/paper"
            / exp
            / chosen.run_id
        )
        if self.docking_results:
            root = self.runs[chosen.run_id][0]
        trace_path = root / "traces" / f"{int(chosen.trial):04d}.json"
        trace = pd.DataFrame(json.loads(trace_path.read_text()))
        self.inputs[exp].add(trace_path)
        self.csv(trace, "docking_representative_trace", exp, "Selected docking time history")
        self.csv(
            pd.DataFrame(
                [
                    {
                        **chosen.to_dict(),
                        "selection_rule": rule
                        + " peak contact force, tie by evaluation seed/run ID",
                    }
                ]
            ),
            "docking_selection",
            exp,
            "Deterministic representative used for both plot and renders",
        )
        fig, axes = plt.subplots(3, 1, figsize=(3.5, 4), sharex=True)
        for ax, key, label in zip(
            axes,
            ["position_error", "attitude_error", "contact_force"],
            ["Position error [m]", "Attitude error [rad]", "Contact force [N]"],
        ):
            ax.plot(trace.time, trace[key], "k-", linewidth=0.8)
            if pd.notna(chosen.first_contact_time):
                ax.axvline(
                    chosen.first_contact_time,
                    color="#D55E00",
                    linestyle="--",
                    linewidth=0.8,
                    label="First contact",
                )
            ax.set_ylabel(label)
            ax.grid(alpha=0.15)
        axes[0].legend()
        axes[-1].set_xlabel("Simulation time [s]")
        self.figure(
            fig,
            "figures/docking/time_history",
            exp,
            "Pose convergence and contact evolution",
            "1 of 100 default trials",
            rule
            + " peak force. Contact trace is maximum summed normal contact force in each 0.25 s control interval; marker uses saved physics-step first-contact time.",
        )
        self.distributions(
            default,
            ["default"],
            ["peak_contact_force", "final_position_error", "final_attitude_error"],
            "diagnostics/docking/default_distributions",
            exp,
        )
        self.distributions(
            f,
            ["default", "soft", "damped"],
            ["peak_contact_force", "final_position_error", "final_attitude_error"],
            "diagnostics/docking/contact_sensitivity",
            exp,
        )

    def portability(self):
        exp = EXPS[4]
        rows = []
        import xml.etree.ElementTree as ET

        for run, (path, meta) in self.runs.items():
            if meta["experiment"] != exp:
                continue
            asset_path = path / (meta["spacecraft"] + "_asset.yaml")
            asset = yaml.safe_load(asset_path.read_text())
            model_path = path / "model.xml"
            model = ET.parse(model_path)
            self.inputs[exp].update([asset_path, model_path])
            rows.append(
                dict(
                    vehicle=meta["spacecraft"],
                    mass_kg=asset["physical"]["mass"],
                    configuration="free rigid body; 6 DoF"
                    if model.find(".//freejoint") is not None
                    or model.find('.//joint[@type="free"]') is not None
                    else "see model XML",
                    actuator_count=len(asset["actuators"]),
                    actuator_type="thrusters",
                    task=meta["task"],
                    controller=meta["controller"].upper(),
                    cpu_demonstrated=meta.get("evaluation_backend") == "mujoco_native",
                    gpu_demonstrated="not demonstrated in portability campaign",
                    fidelity=asset.get("metadata", {}).get(
                        "fidelity", "see saved asset configuration"
                    ),
                )
            )
        self.csv(
            pd.DataFrame(rows),
            "spacecraft_api",
            exp,
            "Demonstrated spacecraft and execution evidence",
        )

    def finish(self):
        definitions = {
            exp: {
                "success_definition": cfg["common"].get("success_id"),
                "task": cfg["common"]["env"]["environment"],
                "protocol": cfg["protocol"],
            }
            for exp, cfg in self.configs.items()
        }
        definition_path = self.output / "tables/statistical_definitions.json"
        definition_path.write_text(json.dumps(definitions, indent=2) + "\n")
        self.register(
            definition_path,
            "audit",
            "Exact saved task thresholds and protocols; metric units are documented in README",
            "5 experiment configurations",
        )
        conventions = """## Statistical and scientific conventions

- Source measurements are authoritative; source files are never modified. Tables use 17 significant digits, with empty fields for unavailable values. No LaTeX is generated.
- Units: position errors m; attitude errors rad; control_effort commanded absolute thruster force integrated over time, N s; contact force N; first-contact/settling/completion times s. Exact saved thresholds and contact settings are exported in `tables/statistical_definitions.json`.
- Terminal pose errors match the saved full-pose success definition. All failed trials remain in summaries and distributions. Metric-specific sample/missing counts accompany finite-value summaries.
- Continuous summaries: arithmetic mean, sample SD (ddof=1), median and linear Q1/Q3. Box whiskers extend to observed values within 1.5 IQR; dots show all finite trials.
- Success: successes / all trials with Wilson 95% intervals. Learned policies additionally report equal-weight training-seed means and Student-t 95% intervals, clipped to [0,1]. The RL evaluation figure shows equal-weight seed means and thin capped Student-t 95% confidence intervals across five training seeds, clipped to [0,1]. Hollow markers have small horizontal offsets; measured success values are unchanged. No evaluation smoothing or interpolation is used. All five individual seed curves appear in `diagnostics/rl/evaluation_seeds`. Both PPO variants have 100% means. Confidence intervals remain in the numerical tables. Fixed nominal evaluations repeat a deterministic scenario; Wilson intervals there are descriptive, not evidence of independent scenario coverage.
- Learning curves show reported training episode return because held-out evaluation is final-only. The main learning figure averages every logged update within right-closed 2,097,152-interaction windows per seed, then shows the seed mean ± sample SD. This matches PPO’s 320-update logging cadence; SAC logs 20,480 updates (64 per window). No interpolation. Window counts/values and all exact-step statistics are exported; unsmoothed individual curves remain in diagnostics. Algorithm return scales can differ; compare nominal/randomized within each algorithm.
- RL suite order comes from the authored frozen protocol: nominal, randomized_initial, thrust_variation, disturbance, combined. There is no dedicated OOD condition. Thrust variation is within the randomized training range. PD/MPC results are included in tables.
- Scaling counts control/environment steps: simulated time = throughput × simulator timestep × control decimation. Per-environment rate divides by environment count. Hardware and backend are recorded in CSV; repeats use sample SD.
- Docking distributions and contact sensitivity are retained as diagnostics; the main docking figure shows a representative pose/contact history. The main set contains four figures.
- Docking selection: lower median successful default trial by peak force, tie by evaluation seed/run ID; explicit failed-trial fallback only if no successes. The same trial supplies time history and render sequence. Contact traces store per-control-interval peak summed normal force, not instantaneous force samples. First contact uses the raw physics-step timestamp.
- Renders replay measured poses without integrating dynamics. Camera/layout changes are presentation only. A batch montage shows saved randomized states and is not evidence of a newly executed GPU batch.

## Regenerate

From repository root, with the project's paper dependencies installed:

```sh
python -m experiments.paper_benchmarks.publication.generate
python -m experiments.paper_benchmarks.publication.render
python -m pytest experiments/paper_benchmarks/publication/test_pipeline.py
```

Optional `--source` and `--output` paths are supported. Analysis requires NumPy, pandas, SciPy, PyYAML and Matplotlib; replay additionally requires MuJoCo and Pillow. Raw source bundle must be present. Corrupt JSON and duplicate measurements stop generation; coverage gaps and unavailable metrics are reported below and in `manifest.json`.
"""
        self.output.mkdir(parents=True, exist_ok=True)
        if self.docking_results:
            option = " --docking-results " + shlex.quote(os.path.relpath(self.docking_results, ROOT))
            conventions = conventions.replace("python -m experiments.paper_benchmarks.publication.generate", self.command)
            conventions = conventions.replace("python -m experiments.paper_benchmarks.publication.render", "python -m experiments.paper_benchmarks.publication.render" + option)
            conventions += "\nDocking uses the corrected crew-airlock campaign at `" + os.path.relpath(self.docking_results, ROOT) + "`; other experiments retain the frozen source bundle.\n"
        issues = sorted(set(self.issues))
        for a in self.artifacts:
            a["caveats"] = issues
        paths = set().union(*self.inputs.values())
        paths.update(
            [
                Path(__file__),
                self.source / "paper-results/experiment_source/aggregate.py",
                self.source / "paper-results/experiment_source/common.py",
            ]
        )
        previous = self.output / "manifest.json"
        if previous.exists():
            current = {a["filename"] for a in self.artifacts}
            for artifact in json.loads(previous.read_text()).get("artifacts", []):
                path = self.output / artifact["filename"]
                if artifact["filename"] not in current and path.resolve().is_relative_to(
                    self.output
                ):
                    path.unlink(missing_ok=True)
        manifest = dict(
            issues=issues,
            artifacts=self.artifacts,
            source_sha256={
                self.relative(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)
            },
        )
        (self.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        text = (
            "# SmallSatSim ICRA 2027 artifacts\n\n"
            "## Main-paper set\n\n"
            "- [GPU scaling](figures/scaling/throughput.pdf)\n"
            "- [Model-based fault robustness](figures/model_based_robustness/distributions.pdf)\n"
            "- [Policy learning and distribution shift](figures/rl/learning_and_shift.pdf)\n"
            "- [Docking pose/contact history](figures/docking/time_history.pdf)\n\n"
            "Every plot has a 400 dpi PNG alongside its vector PDF. Exact tables are in `tables/`; render candidates are indexed below.\n\n"
            + conventions
            + "\n## Completeness and unavailable data\n\n"
            + (
                "\n".join("- " + s for s in issues)
                if issues
                else "All expected runs and trial identities are present."
            )
            + "\n\n## Artifact provenance\n\n"
        )
        for a in self.artifacts:
            text += (
                f"### `{a['filename']}`\n\n{a['purpose']}. Samples: {a['samples']}. {a['statistics']}\n\nScript: `{a['script']}`; command: `{a['command']}`. Sources (paths relative to repository root):\n\n"
                + "\n".join("- `" + p + "`" for p in a["sources"])
                + "\n\nCaveats: see completeness list above and statistical conventions.\n\n"
            )
        readme = self.output / "README.md"
        if readme.exists() and "\n## Candidate renders\n" in readme.read_text():
            text += (
                "\n## Candidate renders\n"
                + readme.read_text().split("\n## Candidate renders\n", 1)[1]
            )
        readme.write_text(text)
        print(
            f"Wrote {len(self.artifacts)} artifacts; {len(issues)} explicit caveats. See {self.output}/README.md"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "smallsat-demonstration")
    parser.add_argument("--output", type=Path, default=ROOT / "paper")
    parser.add_argument("--docking-results", type=Path, help="Replacement docking campaign root containing paper/exp4_docking")
    args = parser.parse_args()
    with plt.rc_context(STYLE):
        build = Build(args.source, args.output, args.docking_results)
        build.load()
        build.quantitative()
        build.finish()


if __name__ == "__main__":
    main()
