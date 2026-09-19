"""Regenerate the paper RL figure from an exported experiment 3 CSV bundle."""

import argparse
import copy
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from scipy.stats import t

from .generate import Build, EXPS, ROOT, STYLE, plt, pd


class RLBuild(Build):
    def __init__(self, source, output):
        self.source, self.output = source.resolve(), output.resolve()
        self.command = f".venv/bin/python -m experiments.paper_benchmarks.publication.generate_rl --source {source} --output {output}"
        self.artifacts, self.issues, self.inputs = [], [], defaultdict(set)
        self.aggregate = SimpleNamespace(t=t)
        training = self.source / "data/exp3_training_curves.csv"
        trials = self.source / "data/exp3_evaluation_trials.csv"
        self.training = pd.read_csv(training)
        self.frames = {EXPS[2]: pd.read_csv(trials)}
        self.inputs[EXPS[2]].update([training, trials])
        learned = self.frames[EXPS[2]].query("controller in ['ppo', 'sac']")
        counts = learned.groupby(['controller', 'regime', 'condition', 'training_seed']).size()
        assert learned.success.isin([True, False]).all(), "Invalid success values"
        assert not learned.duplicated(['run_id', 'condition', 'trial']).any(), "Duplicate evaluation trials"
        assert self.training.groupby(['controller', 'regime']).training_seed.nunique().eq(5).all()
        assert counts.groupby(level='condition').nunique().eq(1).all(), "Unequal evaluation counts"
        self.episode_counts = counts.groupby(level='condition').first().astype(int).to_dict()
        self.count_description = "5 training seeds per method/regime; episodes per seed: " + ", ".join(
            f"{key}={value}" for key, value in self.episode_counts.items()
        )

    def figure(self, fig, name, exp, purpose, counts="", convention="", extra=()):
        if name == "figures/rl/learning_and_shift":
            for axis in fig.axes:
                axis.xaxis.label.set_fontsize(11)
                axis.yaxis.label.set_fontsize(11)
                axis.tick_params(axis='both', labelsize=10)
            fig.axes[1].tick_params(axis='x', labelsize=9)
            for label in fig.axes[1].get_xticklabels():
                label.set_rotation(25)
                label.set_horizontalalignment('right')
                label.set_rotation_mode('anchor')
            handles, labels = fig.axes[0].get_legend_handles_labels()
            for legend in list(fig.legends):
                legend.remove()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                       ncol=4, frameon=False, fontsize=10, handlelength=1.5,
                       handletextpad=0.5, columnspacing=0.8)
            # Include all displayed seed-mean bands instead of the old run's fixed limits.
            ax = fig.axes[0]
            ax.set_ylim(auto=True)
            ax.autoscale_view(scalex=False, scaley=True)
            low, high = ax.get_ylim()
            ax.set_ylim(low, high + 0.18 * (high - low))
            evaluation_ax = fig.axes[1]
            evaluation_ax.set(ylim=(0.70, 1.025), xlim=(-0.38, 4.38))
            evaluation_ax.set_yticks([0.70, 0.80, 0.90, 1.0], ['70%', '80%', '90%', '100%'])
            convention = (
                "Learning: equal-weight mean and sample SD across per-seed window averages; "
                "window width is the coarsest median logging cadence, with no interpolation. "
                "Evaluation: equal-weight seed means and Student-t 95% confidence intervals "
                "clipped to [0,1]. Evaluation axis starts at 70%; all confidence intervals "
                "are visible. Grouped markers have no connecting lines; horizontal categorical "
                "offsets do not alter success values."
            )
            counts = self.count_description
            for index, stem, purpose in [
                (0, "training", "Training episode returns"),
                (1, "evaluation", "Evaluation success under distribution shift"),
            ]:
                # Copy the plotted artists so both exports retain identical data.
                standalone = copy.deepcopy(fig)
                axis = standalone.axes[index]
                for other in list(standalone.axes):
                    if other is not axis:
                        standalone.delaxes(other)
                axis.set_subplotspec(standalone.add_gridspec(1, 1)[0])
                standalone.set_size_inches(5.2, 3.8)
                standalone.legends.clear()
                panel_handles, panel_labels = axis.get_legend_handles_labels()
                standalone.legend(
                    panel_handles, panel_labels, loc="upper center", ncol=2,
                    frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0),
                )
                super().figure(
                    standalone, f"figures/rl/{stem}", exp, purpose,
                    counts, convention, extra,
                )
        elif name == "diagnostics/rl/evaluation_seeds":
            counts = self.count_description
            convention = "Each curve is one training seed; no smoothing or omitted seeds."
        super().figure(fig, name, exp, purpose, counts, convention, extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'paper')
    args = parser.parse_args()
    protocol = ROOT / 'experiments/paper_benchmarks/configs/exp3_rl_robustness/paper.yaml'
    with plt.rc_context(STYLE):
        build = RLBuild(args.source, args.output)
        build.inputs[EXPS[2]].add(protocol)
        build.rl(protocol, evaluation_points=True)
    for artifact in build.artifacts:
        artifact['script'] = 'experiments/paper_benchmarks/publication/generate_rl.py'
    manifest = {
        'artifacts': build.artifacts,
        'episode_counts_per_seed': build.episode_counts,
        'input_sha256': {build.relative(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(build.inputs[EXPS[2]])},
    }
    (args.output / 'rl_v3_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(args.output / 'figures/rl/learning_and_shift.pdf')
    print(args.output / 'figures/rl/training.pdf')
    print(args.output / 'figures/rl/evaluation.pdf')
    print(build.count_description)


if __name__ == '__main__':
    main()
