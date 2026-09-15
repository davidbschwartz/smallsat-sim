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
runtime guarantees. The learning campaign includes 20 training runs and 10,000
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

Learned-policy evaluation uses batched MJX: all trials for a condition advance
together, with deterministic actions and no episode resets. Training retains its
configured backend (currently `freeflyer`). Classical PD/MPC baselines and contact
experiments use native MuJoCo. A development config can set
`protocol.evaluation_backend: mujoco_native` for cross-backend checks. Both learned
evaluation paths use identical sampled scenarios, pose metrics, first-episode
termination, and recording formats. Timings are retained in run metadata;
MJX batch timings include preparation and compilation.

The main RL demonstration compares fixed nominal and randomized starting poses,
±10% thrust variation, bounded world-frame disturbances (±0.005 N per force axis,
±0.0005 N m per torque axis), and their combination. Randomized training uses
these same bounds. Mass and inertia stay fixed.
The fixed nominal condition repeats the same deterministic starting state; its
trial count is not a count of independent scenarios. Use `randomized_initial`
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
