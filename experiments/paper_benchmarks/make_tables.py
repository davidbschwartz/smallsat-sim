"""Standalone LaTeX tabular fragments; round only for presentation."""

from pathlib import Path

import pandas as pd


def escape(text):
    mapping = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(mapping.get(char, char) for char in str(text))


def number(value):
    return "--" if pd.isna(value) else f"{value:.3g}"


def tabular(headers, rows):
    return (
        "\\begin{tabular}{"
        + "l" * len(headers)
        + "}\n\\hline\n"
        + " & ".join(headers)
        + r" \\"
        + "\n\\hline\n"
        + "\n".join(" & ".join(row) + r" \\" for row in rows)
        + "\n\\hline\n\\end{tabular}\n"
    )


def make_tables(data, output):
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for prefix, exp, title in [
        ("exp2", "exp2_fault_robustness", "exp2_fault_robustness"),
        ("exp3", "exp3_rl_robustness", "exp3_robustness"),
        ("exp4", "exp4_docking", "exp4_docking"),
    ]:
        source = data / f"{prefix}_summary.csv"
        if not source.exists():
            continue
        frame = pd.read_csv(source)
        rows = []
        metrics = ["final_position_error", "final_attitude_error", "control_effort"]
        headers = [
            "Method / condition",
            "$n$",
            r"Success [95\% CI]",
            "Position [m]",
            "Attitude [rad]",
            "Command [N s]",
        ]
        if prefix == "exp4":
            metrics = [
                "peak_contact_force",
                "final_position_error",
                "final_attitude_error",
                "settling_time",
            ]
            headers = [
                "Setting",
                "$n$",
                r"Success [95\% CI]",
                "Peak force [N]",
                "Position [m]",
                "Attitude [rad]",
                "Settling [s]",
            ]
        for _, row in frame.iterrows():
            label = " / ".join(
                str(row[k])
                for k in ("controller", "regime", "condition")
                if k in row and pd.notna(row[k])
            )
            low = row.get("success_seed_ci_low", float("nan"))
            high = row.get("success_seed_ci_high", float("nan"))
            if pd.isna(low):
                low, high = row.success_ci_low, row.success_ci_high
            values = [
                escape(label),
                str(int(row.n)),
                f"{row.success_rate:.3f} [{low:.3f}, {high:.3f}]",
            ]
            for metric in metrics:
                values.append(
                    "$"
                    + number(row[metric + "_mean"])
                    + r" \pm "
                    + number(row[metric + "_std"])
                    + "$"
                )
            rows.append(values)
        path = output / f"{title}.tex"
        path.write_text(tabular(headers, rows))
        artifacts.append((path, exp))
        if prefix == "exp3":
            conditions = list(frame.condition.unique())
            compact = []
            for (algorithm, regime), group in frame.groupby(["controller", "regime"]):
                by_condition = group.set_index("condition")
                seeds = group.get("n_training_seeds", pd.Series(dtype=float)).dropna()
                compact.append(
                    [
                        escape(f"{algorithm} {regime}"),
                        str(int(seeds.iloc[0])) if len(seeds) else "--",
                    ]
                    + [
                        number(by_condition.loc[condition, "success_rate"])
                        if condition in by_condition.index
                        else "--"
                        for condition in conditions
                    ]
                )
            path = output / "exp3_robustness_compact.tex"
            path.write_text(
                "% Success proportions; uncertainty and episode counts are in the full table.\n"
                + tabular(
                    ["Method", "Seeds", *[escape(c.replace("_", " ")) for c in conditions]],
                    compact,
                )
            )
            artifacts.append((path, exp))
    source = data / "exp1_scaling.csv"
    if source.exists():
        frame = pd.read_csv(source)
        rows = []
        for n, group in frame.groupby("num_envs"):
            good = group[group.status == "ok"]
            if good.empty:
                rows.append([str(n), "OOM", "--", "--", "--", "--"])
                continue
            rows.append(
                [str(n)]
                + [
                    number(good[k].mean())
                    for k in (
                        "env_steps_per_second",
                        "sim_seconds_per_second",
                        "sim_seconds_per_second_per_env",
                        "speedup",
                        "parallel_efficiency",
                    )
                ]
            )
        path = output / "exp1_scaling.tex"
        path.write_text(
            tabular(["Envs", "Steps/s", "Sim s/s", "Sim s/s/env", "Speedup", "Efficiency"], rows)
        )
        artifacts.append((path, "exp1_scaling"))
    return artifacts
