# Demonstration use cases

Run experiments for parallel simulation throughput, classical fault robustness,
reinforcement-learning robustness, Gateway contact, and spacecraft portability.
The suite records reproducible run configurations and generates figures and tables
from retained raw data. Experiment settings and run matrices are in `configs/`.

`--paper` selects the checked-in evaluation protocol; it does not certify that
a campaign has completed. Docking measures a local approach and compliant contact;
it does not establish hardware docking performance.

## Installation and execution

Run from the repository root with the documented SmallSatSim installation.
Linux/NVIDIA is the primary platform for the full suite; scaling requires a JAX GPU backend.
MPC needs the `mpc` extra and native acados libraries (`bash .setup/smallsat setup`
on Linux; on macOS, `uv sync --locked --extra mpc --extra paper` followed by
`bash .setup/native/install_mpc.sh`). Source `.setup/env.sh` when running Python
directly. Use the checked-in `uv.lock`.
Matplotlib is required for artifact generation and is included in the `paper`
extra. Rendering and W&B are disabled by default.

```bash
python -m experiments.demonstration_use_cases.exp1_scaling --paper
python -m experiments.demonstration_use_cases.exp2_fault_robustness --paper
python -m experiments.demonstration_use_cases.exp3_rl_robustness --paper
python -m experiments.demonstration_use_cases.exp4_docking --paper
python -m experiments.demonstration_use_cases.spacecraft_portability --paper

# Does not run simulation or training:
python -m experiments.demonstration_use_cases.aggregate

# Sequential full campaign, followed by strict artifact generation:
python -m experiments.demonstration_use_cases.reproduce_all --paper
```

Use `--plan` to inspect all jobs without allocating environments. Use `--smoke`
on any experiment to reduce budgets through the same execution path. Smoke still
requires acados for MPC; it never substitutes another controller. CPU-only users can run smoke checks, including MPC when its native dependencies
are installed. GPU scaling requires a GPU; CPU smoke throughput does not validate it.

```bash
python -m experiments.demonstration_use_cases.spacecraft_portability --smoke
python -m experiments.demonstration_use_cases.exp1_scaling --smoke
python -m experiments.demonstration_use_cases.exp3_rl_robustness --smoke
python -m experiments.demonstration_use_cases.aggregate --mode smoke --allow-partial \
  --output artifacts/demonstration_use_cases_smoke
```

Allow seconds to minutes for smoke checks and potentially hours to days for
full campaigns, depending on hardware. These are planning estimates, not measured
runtime guarantees. The learning campaign includes 20 training runs and 8,020
learned-policy evaluation episodes: PPO and SAC, each trained nominally and with
randomization on seeds 0–4. PD and MPC remain evaluation baselines. All requested jobs are attempted and failures
are reported, so a missing prerequisite does not discard other
methods' results. GPU OOMs have explicit records; unexplained worker termination
is a failure, not an inferred OOM.

Both algorithms train with 4,096 parallel environments. PPO retains 320 updates
of 512 control steps per environment (671,088,640 transitions); SAC has the same
transition budget. Smoke runs use two environments.
SAC uses separate 256-by-256 networks via `protocol.policy_hidden_sizes`; PPO
retains its existing 64-by-64 networks.

Learned-policy evaluation restores the saved checkpoint through `runner.evaluate(scenarios=...)`.
All conditions are packed into batches of up to 128 MJX environments with
batched deterministic actor inference and no episode resets. The experiment's
`mjx_evaluation.py` only samples scenarios and writes results; `runtime.py`'s
control loop is for classical controllers. Training retains its configured
backend (currently `freeflyer`). PD/MPC baselines and contact experiments use
native MuJoCo; they are not silently replaced with a different GPU controller.
The library uses its existing rollout collector, MJX stepping and registered task
termination. Native MuJoCo parity checks live in tests.

`protocol.evaluation_trials.nominal: 1` overrides the default `trials: 100`.
Each learned policy and each PD/MPC baseline therefore evaluates **401 scenarios**:
one fixed nominal reference and 100 each for randomized initial state, thrust
variation, disturbance and combined effects. Initial conditions are paired across
policies and the four randomized suites. This changes neither training nor the
success definition. There is no newly invented OOD condition.
The final batch can have padding lanes for compilation reuse; they are discarded,
never exported and never counted as trials. `samples.jsonl` records stable
`scenario_id` and `initial_state_id` hashes for learned-policy evaluations; duplicate
physical scenarios within a new suite are rejected. `evaluation_timing.json`
separates preparation, synchronized rollout (including first-use compilation),
and metric reduction. Run metadata additionally records total evaluation wall time
including file output. These timings are not pure steady-state throughput benchmarks.

### GPU verification and the new campaign

```bash
# Confirm that this process actually sees the accelerator.
python -c "import jax; print(jax.devices()); assert jax.default_backend() == 'gpu'"

# Cross-backend dynamics, first-terminal-event, padding and failure checks.
python -m pytest src/tests/test_rl_scenario_evaluation.py -q

# Small end-to-end runs for both algorithms, including real checkpoint writes.
python -m experiments.demonstration_use_cases.exp3_rl_robustness --smoke \
  --output results/demonstration_mjx_v2_smoke \
  --job-id controller-ppo_regime-nominal_seed-0_spacecraft-astrobee \
  --job-id controller-sac_regime-nominal_seed-0_spacecraft-astrobee

# Inspect the 20 policy jobs + two classical baselines, then launch.
python -m experiments.demonstration_use_cases.exp3_rl_robustness --paper --plan
python -m experiments.demonstration_use_cases.exp3_rl_robustness --paper \
  --output results/demonstration_mjx_v2 --resume
```

Use a new results root: the changed evaluation protocol has a new config identity,
and resume correctly refuses to mix it with the old 100-identical-reference-trial
campaign. Historical configs without per-condition overrides retain their old
counts when validated. The full command also runs PD/MPC, requiring the MPC setup
above; `--job-id` can select just the learned policies.


### Re-evaluate a saved policy without training

```bash
python -m experiments.demonstration_use_cases.exp3_rl_robustness \
  --evaluate-run PATH_TO_SAVED_RUN \
  --output results/demonstration_reevaluation
```

The saved run must include `metadata.json`, `job.json`, `resolved_config.yaml`,
`runtime_config.yaml`, and its policy checkpoint. A run whose evaluation failed
can be used as long as its checkpoint was saved. The command creates a fresh run,
records the source/checkpoint paths, and copies the available training trace for
aggregation. It neither trains nor modifies the source run. Archived runs can be
moved between machines; their original configuration identity is preserved.
Keep the source checkpoint with the new results: it is referenced, not duplicated.
Use a separate output root to avoid duplicate policies in campaign aggregation.

By default, evaluation uses the saved protocol. To adjust it, copy the saved
`resolved_config.yaml`, edit only the following fields, and pass `--config FILE`:

- `common.evaluation_distributions`, `evaluation_seed_start`, `episode_steps`.
- `protocol.trials`, `evaluation_trials`, `evaluation_batch_size`, `evaluation_backend` (MJX).

Overrides are recorded as development runs. Training settings, physical assets,
and task thresholds must match the saved run. Do not combine `--evaluate-run`
with `--paper`, `--smoke`, `--resume`, `--plan`, or `--job-id`.

For a GPU check, re-evaluate one full saved policy with the 401-scenario protocol
before launching the campaign. Inspect `evaluation_timing.json` for preparation
and synchronized batch times (first-use compilation is included).

Reference timings from the saved A100-SXM4-40GB campaign were 4.5–4.6 minutes per
PPO policy, 24.7–25.3 minutes per SAC policy, and 18–27 seconds per old MJX
evaluation. Twenty training jobs were roughly five hours sequentially. New GPU
measurements are required; compilation, output storage and hardware matter.

Archive **every seed's** `policy_checkpoint_path` from metadata, together with
`resolved_config.yaml`, `runtime_config.yaml`, the asset, model XML and source
hashes. SAC already writes a small `_actor` checkpoint as well as its full training
state; the new metadata points explicitly to that inference checkpoint. PPO's
policy path is its standard training checkpoint; evaluation restores only actor
weights (plus estimator weights for adaptive policies), leaving training state intact. The full SAC checkpoint (including
replay/optimizer state) is needed to resume training, but not to retain the trained
actor for evaluation. No checkpoint is selected by evaluation performance.

The main RL demonstration compares fixed nominal and randomized starting poses,
±10% thrust variation, bounded world-frame disturbances (±0.005 N per force axis,
±0.0005 N m per torque axis), and their combination. Randomized training uses
these same bounds. Mass and inertia stay fixed.
The fixed nominal condition runs once per checkpoint. Across five training seeds,
its success rate describes five policies on one starting situation, not 500
independent scenarios. Use `randomized_initial`
to assess nominal-dynamics reliability across starting poses, rather than
pooling the repeated fixed probe into an overall success rate.
This is a demonstration of a usable robust-control workflow, not a claim to solve
all actuator failures or large model mismatch.

Interpret success rates alongside continuous position/attitude errors and command
impulse. `exp3_continuous_performance` plots medians and interquartile ranges;
the full tables also report trajectory-mean errors. Final errors are measured at
episode termination (success or deadline), not at a common fixed time. Failed
episodes remain in these summaries. Changing the task tolerance to make a policy
pass is not part of tuning.

Aggregation writes `exp3_training_health.csv` with finite-value checks and early/late
training summaries. Window success rates are weighted by completed episode counts;
a window with no completed episodes has no success estimate. Numerical health
alone does not establish controller quality. Inspect the PPO/SAC diagnostic plots
alongside held-out performance, including position and attitude errors, losses,
PPO KL and explained variance, and SAC entropy and temperature.
Training errors average rollout states, including new randomized starts after
automatic resets. They need not approach zero when episodes succeed quickly;
use the separate held-out final-error summaries to assess terminal accuracy.

Optional W&B logging and replay diagnostics are documented
in [SAC tuning](TUNING.md).

The PPO/SAC comparison matches task rewards, initial-state distributions,
environment transitions, parallel environment count, and evaluation criteria.
It compares tuned controllers: SAC uses 256-by-256 networks and discount 0.999,
whereas PPO uses 64-by-64 networks and discount 0.995. Compute and tuning effort
are not matched. Report wall time alongside success and continuous errors, and
use the same held-out evaluation seeds for the multi-seed comparison.

Use `--plan` to obtain exact `job_ids`, then repeat `--job-id ID` to select jobs.
`--resume` skips validated completed jobs, rejects duplicate or mismatched runs,
and retains failed attempts. It refuses to start another attempt for a job still
marked running. `reproduce_all --resume --paper` applies this to the full campaign.

## Configurations and raw data

`configs/common.yaml` freezes simulation, task, reward, optimization and classical
controller settings. `configs/assets/` freezes the physical spacecraft definitions.
Per-experiment `paper.yaml` files define the run matrix. They are loaded directly,
without resolving runtime parameters through mutable package defaults.
`common_overrides` changes only that experiment's common settings;
`evaluation_distributions` replaces its evaluation matrix explicitly.

Raw runs go to `results/demonstration_use_cases/{paper,smoke,development}/<experiment>/`.
Each attempt has its job identity, config hash and unique suffix. It includes:

- `resolved_config.yaml`, `metadata.json`, instantiated asset and `model.xml`;
- `metrics.jsonl` per trial/repeat, with complete sampled scenarios in `samples.jsonl`;
- training `runtime_config.yaml`, raw update diagnostics, `training.jsonl`, checkpoints;
- docking compiled contact parameters and all pose/contact/control traces.

Checkpoints and raw run attempts are never reused or overwritten automatically.
Use separate `--output` roots for separate campaigns. Aggregation rejects two
completed attempts for the same job rather than picking a favorable repeat.
A failed job can be rerun; incomplete attempts remain visible in the manifest.

For custom development runs, export a complete config and edit it:

```python
from pathlib import Path
import yaml
from experiments.demonstration_use_cases.common import load_config

cfg = load_config("exp2_fault_robustness", mode="smoke")
cfg["protocol"]["controllers"] = ["pd"]  # explicit development comparison
Path("/tmp/experiment.yaml").write_text(yaml.safe_dump(cfg))
```

```bash
python -m experiments.demonstration_use_cases.exp2_fault_robustness \
  --config /tmp/experiment.yaml --output results/my_pilot
```

`--paper --config` is rejected. Custom runs are labeled development. For the same
portability workflow, only the `spacecraft` selection changes among `astrobee`,
`cubesat`, and `sprint`; the asset loader, task, metrics and PD interface are shared.
Vehicle gains are specified in `configs/common.yaml`.

## Generate artifacts

`aggregate` validates every expected job and every trial/repeat identity, writes
canonical CSVs under `artifacts/demonstration_use_cases/data`, then produces PDF/PNG figures,
LaTeX tabular fragments, JSON summaries and a source-run manifest. Missing data
fails by default. `--allow-partial` explicitly permits incomplete development
figures; inspect `manifest.json` before using any output in the final report.

```bash
python -m pytest src/tests/test_demonstration_use_cases.py -q
```

## Videos and screenshots

New evaluation runs for experiments 2–4 save compressed pose recordings under
`<run>/recordings/<condition>/<trial>.npz`. Experiment 3 records held-out evaluation
episodes for learned policies and baselines. Rendering runs separately:

```bash
python -m experiments.demonstration_use_cases.replay \
  results/demonstration_use_cases/paper/<experiment>/<run>/recordings/<condition>/0000.npz \
  --output artifacts/episode --width 1920 --height 1080
```

The command writes `episode.mp4` and start/middle/end PNG screenshots. Choose the
view with `--azimuth`, `--elevation`, and `--distance`; the default fixed camera
centers on the recorded trajectory. `--fps` defaults to 30. A working MuJoCo OpenGL
rendering backend is required. Keep the run's `model.xml` and the spacecraft mesh
assets available when replaying.

Playback uses saved poses at control boundaries, holding each until the next
sample. It preserves simulation timing but does not reconstruct motion between
samples or visualize instantaneous contact forces. Timestamps identify the saved
state. Older runs without pose recordings must be rerun to produce these visuals.
Select and report the episode/seed explicitly when using a video in the paper.
