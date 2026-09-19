# Publication figures

These version-controlled tools generate the paper figures from saved experiment
data. Generated PDFs, PNGs, tables, and provenance manifests remain in the ignored
`paper/` directory. Run the commands below from the repository root using the
project environment with the `paper` extra installed. Analysis uses Matplotlib,
NumPy, pandas, SciPy, and PyYAML; saved-pose rendering also uses MuJoCo and Pillow.

## Current RL figure

```sh
python -m experiments.paper_benchmarks.publication.generate_rl \
  --source exp3_rl_robustness_v3 --output paper
```

The source directory must contain `data/exp3_training_curves.csv` and
`data/exp3_evaluation_trials.csv`. The evaluation order comes from
`../configs/exp3_rl_robustness/paper.yaml`. Outputs include:

- `paper/figures/rl/learning_and_shift.{pdf,png}`: the two-panel paper figure.
- `paper/figures/rl/training.{pdf,png}`: standalone training figure with its own legend.
- `paper/figures/rl/evaluation.{pdf,png}`: standalone evaluation figure with its own legend.
- `paper/diagnostics/rl/`: individual training and evaluation seed plots.
- `paper/tables/learning_*.csv`: exact observed-step and windowed summaries.
- `paper/rl_v3_manifest.json`: source hashes, generation command, episode counts,
  and statistical conventions.

The figure uses every logged training update, aggregated within windows per seed
before averaging across seeds. Evaluation uses seed means and Student-t 95%
confidence intervals. Its zoomed axis begins at 70%. The current nominal suite
has one deterministic episode per seed; other suites have 100.

## Full archived campaign and saved-pose renders

```sh
python -m experiments.paper_benchmarks.publication.generate \
  --source smallsat-demonstration --output paper
python -m experiments.paper_benchmarks.publication.render \
  --source smallsat-demonstration --output paper
```

Both commands support `--docking-results results/docking_crew_airlock` to select
the replacement docking campaign. The source bundle must retain its original
layout: `paper-results/experiment_source/`, `paper-results/raw_runs/`, and
`demonstration_all_recordings_traces/results/demonstration_campaign_final/paper/`.
Rendering requires saved models, recordings, and traces as well as the generated
tables. It replays saved poses without running controllers.

The generator exports both `figures/model_based_robustness/distributions.{pdf,png}`
and `distributions_vertical.{pdf,png}` in that directory, using the same trial data.

The full generator reproduces the older frozen campaign, including its original
RL results. Run `generate_rl` **after** it to replace the RL figure with the v3
results. `rl_v3_manifest.json` describes that replacement; the full campaign's
`manifest.json` and generated README describe the archived campaign.

Run the publication pipeline checks with:

```sh
python -m pytest experiments/paper_benchmarks/publication/test_pipeline.py -q
```

The numerical unit tests run without the archive. The exported-campaign test
requires the full campaign outputs and source bundle.

## Data availability

The CSV bundle and frozen raw archive are separate inputs, not supplied by these
scripts or downloaded automatically. A clone alone cannot reproduce the measured
figures. A release should provide a versioned data archive and download location,
including the matching configuration files; generated manifests record input
checksums. The local names above are defaults, not public download locations.
