```text
   _____                 _________       __  _____ _
  / ___/____ ___  ____ _/ / / ___/____ _/ /_/ ___/(_)___ ___
  \__ \/ __ `__ \/ __ `/ / /\__ \/ __ `/ __/\__ \/ / __ `__ \
 ___/ / / / / / / /_/ / / /___/ / /_/ / /_ ___/ / / / / / / /
/____/_/ /_/ /_/\__,_/_/_//____/\__,_/\__//____/_/_/ /_/ /_/
```


SmallSatSim emerged from the SmallSat Steward project, a collaboration between researchers at Caltech's Jet Propulsion Laboratory (now University of Southern California) and the University of Michigan's Space Systems Laboratory. This project is open-sourced under an [Apache 2.0 license](LICENSE).

**SmallSatSim: A GPU-Accelerated Microgravity Robotics Toolkit for Planning, Control, and Policy Learning**

[**Project page**](https://smallsatsim.github.io)

SmallSatSim combines MuJoCo simulation with batched JAX rollouts for developing
and evaluating spacecraft controllers. It includes:

- **Spacecraft and tasks:** Astrobee, CubeSat, and nominal AERCam Sprint assets, setpoint control, path
  tracking, and mission planning.
- **Classical and learned control:** PD, LQR, MPC, MPCC, GP-MPC, and PPO
  training by default, with optional CNN or transformer adaptation. VPG and SAC
  are also available.
- **Robustness experiments:** actuator faults, external disturbances, custom
  rewards and termination conditions, and seeded evaluation scenarios.
- **Visualization and results:** native and browser viewers, MP4 recording,
  checkpoints, and per-scenario evaluation metrics.

[Install](#installation) · [Quick start](#quick-start) ·
[Define an experiment](#define-your-own-experiment) ·
[Extend the API](#extend-the-api) ·
[Train and evaluate](#training-and-evaluation) · [View and record](#viewers-and-video) ·
[Paper benchmarks](#paper-benchmarks) · [Development](#development)

## Installation

Run commands from a local checkout of this repository. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) first; the repository
selects Python 3.10 and pins dependencies in `uv.lock`.

| Setup | Command |
| --- | --- |
| Ubuntu 22.04+, NVIDIA GPU, including MPC | `bash .setup/ubuntu/setup.sh --system-deps` |
| Ubuntu CPU, including MPC | `bash .setup/ubuntu/setup.sh --cpu --system-deps` |
| Ubuntu NVIDIA GPU, simulation/RL without MPC | `bash .setup/ubuntu/setup.sh --without-mpc --system-deps` |
| Python-only CPU development, including macOS | `uv sync --locked` |

Linux/NVIDIA is the primary target. GPU setup requires a compatible driver and a
working `nvidia-smi`. `--system-deps` installs system packages using `sudo apt-get`;
omit it when those dependencies are already installed. On macOS, install Xcode
command-line tools, CMake, and Rust 1.75+, then build the CPU/MPC stack with:

```bash
uv sync --locked --extra mpc --extra paper
bash .setup/native/install_mpc.sh
bash .setup/smallsat check --mpc
```

The native build and a generated Astrobee MPC solver have been checked on Apple
Silicon. CUDA remains Linux-only; Windows and VOXL deployment are not validated.

Verify the capabilities you installed:

```bash
bash .setup/smallsat check             # Python dependencies
bash .setup/smallsat check --mpc --gpu # Full Linux GPU/MPC installation
```

Use `bash .setup/smallsat run COMMAND` for the examples below. It sets native
library paths and runs `uv run --no-sync`, preserving installed extras. For direct
commands or IDE debugging, source `.setup/env.sh` and select `.venv/bin/python`.

When updating dependencies, retain your installation's extras. For the full stack:

```bash
uv sync --locked --extra cuda12 --extra warp --extra mpc
```

<details>
<summary>MPC installation details and troubleshooting</summary>

The Linux installer builds acados and `t_renderer` in `deps/acados` and installs
the pinned l4acados and GP dependencies. Building the renderer requires Rust 1.75+.
Set `SMALLSAT_BUILD_JOBS` to change compilation concurrency (default: 2).

For missing native libraries, use the launcher or source `.setup/env.sh`, then run
`bash .setup/smallsat check --mpc`. If the pinned acados revision changes, move the
old checkout aside and rerun setup; the installer preserves existing local changes.
See [.setup/native/install_mpc.sh](.setup/native/install_mpc.sh) for build details.

</details>

## Paper benchmarks

**Learn SmallSatSim** with the quick start and editable [examples](examples).
**Run reproducible experiments** with the [paper benchmark package](experiments/paper_benchmarks/README.md).

The [publication tools](experiments/paper_benchmarks/publication/README.md) reproduce
paper figures from retained data; generated files remain in the ignored `paper/` directory.
The package includes Astrobee (`astrobee`), CubeSat (`cubesat`), and Sprint (`sprint`)
portability checks. Campaign manifests record completion and missing runs.
The experiment README documents execution, outputs, and validation limits.

Use the Linux/NVIDIA installation with native MPC for the full suite. Retain your
installed extras when adding the plotting dependency:

```bash
uv sync --locked --extra cuda12 --extra warp --extra mpc --extra paper
bash .setup/smallsat run python -m experiments.paper_benchmarks.exp1_scaling --paper
bash .setup/smallsat run python -m experiments.paper_benchmarks.exp2_fault_robustness --paper
bash .setup/smallsat run python -m experiments.paper_benchmarks.exp3_rl_robustness --paper
bash .setup/smallsat run python -m experiments.paper_benchmarks.exp4_docking --paper
bash .setup/smallsat run python -m experiments.paper_benchmarks.spacecraft_portability --paper

# Regenerate figures/tables from raw data, without retraining:
bash .setup/smallsat run python -m experiments.paper_benchmarks.aggregate
# Or run the full campaign and generate its artifacts:
bash .setup/smallsat run python -m experiments.paper_benchmarks.reproduce_all --paper
```

Raw runs are retained under `results/demonstration_use_cases/paper`, identified by configuration,
seed, and unique attempt. Postprocessing writes `artifacts/demonstration_use_cases`; the full
campaign writes artifacts alongside its results. Scaling takes minutes; Monte Carlo
and the 20-run learning program can take hours/days, depending on hardware. Use
`--plan` to inspect jobs or `--smoke` for small validation runs. CPU-only smoke runs
cannot validate GPU scaling or MPC without its native dependency.

## Quick start

Check that simulation works with a cube under sinusoidal thrust:

```bash
bash .setup/smallsat run python experiments/test.py --headless
```

Press Ctrl+C to stop. To open a native window on a Linux desktop, omit
`--headless` and prefix the command with `MUJOCO_GL=glfw`.

For an existing spacecraft experiment, start with one of these scripts:

| Example | What it runs |
| --- | --- |
| [astrobee_CL.py](experiments/astrobee_CL.py) | Astrobee mission tracking with MPCC; requires the MPC stack |
| [cubesat_CL.py](experiments/cubesat_CL.py) | CubeSat with nominal MPC; requires the MPC stack |
| [train_astrobee.py](experiments/rl_training/train_astrobee.py) | Editable RL training and evaluation workflow |
| [astrobee_RL.py](experiments/astrobee_RL.py) | Deploy a learned controller on a circular path |

```bash
bash .setup/smallsat run python experiments/astrobee_CL.py --headless --log
```

These scripts explicitly construct the environment, planner, and controller or
runner. Change those choices in Python, or use a YAML experiment as shown next.

## Define your own experiment

An experiment combines **a physical asset**, **an environment/task**, **a planner**,
and **a controller or learning algorithm**. YAML selects components and settings;
Python constructs them and decides when to simulate, train, or evaluate.

### 1. Describe the task and training settings

For the nominal **AERCam Sprint** model, a ready-to-run setpoint preset is in
[examples/sprint.yaml](examples/sprint.yaml):

```bash
bash .setup/smallsat run python examples/run_sprint.py
```

Use `vehicle: sprint_rl` for RL or `vehicle: sprint` for classical controllers.
Both select [vehicles/sprint.yaml](src/smallsat_sim/config/vehicles/sprint.yaml).
The model uses a 15.88 kg body, 0.356 m diameter, twelve 0.378 N thrusters and
published nominal inertia. Its spherical shell and camera features are approximate;
thruster lines of action reproduce an idealized six-axis arrangement. Commands
represent averaged force, not flight valve pulses. See the
[validation notes](docs/spacecraft_candidate_validation.md) for sources and limits.
The example runs one PPO epoch as an integration check, not a trained controller.

Suppose you want to train PPO, the default RL method, to hold Astrobee at a fixed
pose without faults or adaptation.
Save the following as `my_experiment.yaml` in the repository root:

```yaml
vehicle: astrobee_rl             # Registered environment/task
asset: vehicles/astrobee.yaml    # Physical spacecraft definition
controller: rl
algorithm: ppo
planner: oracle_rl
planner_radius: 0.0              # Fixed setpoint rather than a circular path
reward: vec_env/full_pose        # Position, attitude, velocity, and effort reward
failures: "off"
run_name: my_setpoint
headless: true
log: false

overrides:
  sim.seed: 0
  Bodies.max_start_offset: 0.5
  RL:
    num_envs: 8
    rollout_backend: freeflyer
    use_adaptive_approach: false
    policy_hidden_sizes: [32, 32]
    checkpoint_dir: experiments/rl_results/my_setpoint
    episode_len: 128
    n_evals: 1
    PPO:
      epochs: 2
      steps_per_epoch: 128
      num_minibatches: 4
      actor_training_epochs: 2
      critic_training_epochs: 2
      max_ep_len: 128
```

This is a small execution check, not a convergence recipe. Each PPO epoch collects
128 steps from each of 8 environments, giving 1,024 transitions before updating
the policy. Increase `PPO.epochs` for a learning experiment. `freeflyer` uses compact
free-flying rigid-body dynamics; select `mjx` for the MuJoCo MJX rollout backend.

### 2. Construct, train, and evaluate

Save this as `run_experiment.py` alongside the YAML:

```python
from pathlib import Path
from smallsat_sim.api.experiments import make_experiment

experiment = make_experiment("my_experiment.yaml")
try:
    runner = experiment.runner
    runner.learn()
    runner.evaluate()
    print("Checkpoint:", Path(runner.ckpt_dir) / runner.training_state_file_name)
finally:
    experiment.env.close()
```

Run it from the repository root:

```bash
bash .setup/smallsat run python run_experiment.py
```

The factory builds the components; `learn()` runs training and writes a checkpoint,
and `evaluate()` runs the configured evaluation episodes. Fresh training refuses
to overwrite an existing checkpoint. For the same run, use `runner.learn(mode="resume")`;
for an independent run, choose a new checkpoint directory. Resume restores training
state and progress toward the configured total budget.

You can also construct directly in Python or override a YAML value at the call site:

```python
experiment = make_experiment(
    "my_experiment.yaml",
    run_name="my_second_run",
    overrides={"RL.checkpoint_dir": "experiments/rl_results/my_second_run"},
)
```

Keyword overrides take precedence over YAML overrides. The returned object exposes
`env`, `planner`, and either `runner` for RL or `controller` for classical control.
Always close `experiment.env` when finished.

### 3. Adapt the experiment

Change the part that corresponds to your research question:

| Change | Where to start |
| --- | --- |
| Spacecraft mass, inertia, geometry, or thrusters | Copy a [vehicle asset](src/smallsat_sim/config/vehicles) and set `asset` to your YAML path |
| Initial conditions, observations, reward weights, or fault distribution | Add overrides based on the [environment defaults](src/smallsat_sim/config/environments/astrobee_rl.yaml) |
| Algorithm, network size, or training budget | Set `algorithm` and override the [training defaults](src/smallsat_sim/config/training/on_policy.yaml) |
| A new reward function | Follow [custom_rewards.py](examples/custom_rewards.py), register it, and select its name with `reward` |
| A new task or success condition | Follow [position_task.py](examples/position_task.py) for task and termination registration |
| Custom actuator faults or external forces | Follow [custom_effects.py](examples/custom_effects.py) and [custom_effects.yaml](examples/custom_effects.yaml) |
| A different classical controller or planner | Select a registered component using the [classical controller guide](examples/classical_controllers.md) |

For example, the bundled [integrated experiment](examples/integrated_experiment.yaml)
combines a custom spacecraft, position-only task, reward, motor loss, and external
wrench. Import its registrations before constructing it:

```python
import examples.position_task
from smallsat_sim.api.experiments import make_experiment

experiment = make_experiment("examples/integrated_experiment.yaml")
try:
    runner = experiment.runner
    result = runner.collect(
        runner.collector("zero", stochastic=False), randomize=False
    )
    print(result.actions.shape)  # (8, 2, 12): steps, environments, actuators
finally:
    experiment.env.close()
```

User asset paths resolve relative to the experiment YAML. Custom Python modules
must remain importable when loading a run; checkpoints do not embed their source.
Use unique registration names and version changed reward/effect semantics.
A robot with different joints or state/action conventions needs an appropriate
environment implementation in addition to its asset definition.

## Extend the API

The API separates physical definitions, task behavior, component construction,
and execution. You can extend each independently:

| API surface | Entry points | Selected or used through |
| --- | --- | --- |
| Experiment construction | `ExperimentSpec`, `make_experiment`, `make_rl_experiment` | YAML, a specification object, or Python keyword arguments |
| Physical vehicles | `load_vehicle`, `register_vehicle_file`, `VehicleSpec` | `asset` and Python asset lookup |
| Environments and tasks | `EnvEntry`, `register_env` | `vehicle` (or explicit `env`) |
| Rewards | `RewardContext`, `RewardResult`, `compose_reward`, `register_reward` | `reward` |
| Episode endings | `TerminationContext`, `TerminationResult`, `register_termination` | `overrides.RL.termination` |
| Planners | `register_planner` | `planner` and `planner_options` |
| Controllers | `register_controller` | `controller` |
| Actuator faults and disturbances | `EffectSampleContext`, sample/apply functions | `overrides.RL.custom_faults` / `custom_disturbances` |
| Rollouts and training | Returned runner's `collector`, `collect`, `learn`, `evaluate` | Explicit Python calls |
| Run identity | `RunSpec`, `build_run_name`, `build_checkpoint_file_names` | Run metadata and checkpoint naming |

Experiment and component builders live in `smallsat_sim.api.experiments`; the
asset, reward, termination, and run helpers are exported by `smallsat_sim.api`.
Discover registered components instead of guessing their names:

```python
from smallsat_sim.api import list_rewards, list_terminations
from smallsat_sim.api.experiments import list_envs, list_planners, list_controllers

print("Environments:", list_envs())
print("Planners:", list_planners())
print("Controllers:", list_controllers())
print("Rewards:", list_rewards())
print("Terminations:", list_terminations())
```

Registrations are process-local: import your extension modules before calling the
factory. Duplicate names raise an error unless you explicitly pass `replace=True`.

### Build experiments in Python

Use an `ExperimentSpec` when generating configurations programmatically. For
example, vary a seed without modifying YAML files:

```python
from dataclasses import replace
from smallsat_sim.api.experiments import ExperimentSpec, make_experiment

base_spec = ExperimentSpec(vehicle="cubesat", controller="pd", log=False)
trial = replace(base_spec, run_name="seed_7", overrides={"sim.seed": 7})
experiment = make_experiment(trial)
try:
    command = experiment.controller.get_control_input(experiment.env)
    experiment.env.step(input=command)
    print(experiment.run_spec)
finally:
    experiment.env.close()
```

`ExperimentSpec` describes construction; `experiment.run_spec` is the run metadata.
The public `build_run_name(run_spec)` and `build_checkpoint_file_names(run_spec)`
helpers derive names without writing files. If a script only needs the RL objects,
`make_rl_experiment(...)` returns `(env, planner, runner)`; the caller still closes
`env` when finished.

### Add a custom vehicle

For a new free-flying spacecraft, start with a complete asset definition and edit
its physical properties, geometry, and thruster layout:

```bash
cp examples/demo_spacecraft.yaml my_spacecraft.yaml
```

In `my_spacecraft.yaml`, change `name` to `my_spacecraft` and edit these fields:

| Field | What it defines |
| --- | --- |
| `physical` | Mass, diagonal inertia, dimensions, and center-of-mass offset |
| `geoms` | Visual/collision geometry, including primitives or mesh references |
| `actuators` | Thruster positions, force directions (`gear`), and command/force ranges |
| `assets` | Asset kind, meshes, and materials |

Use SI units (kg, m, kg·m², N) for the physical quantities and thrust commands.
Load the file to validate its definition and inspect the resulting `VehicleSpec`:

```python
from smallsat_sim.api import load_vehicle

spacecraft = load_vehicle("my_spacecraft.yaml")
print(spacecraft.name, spacecraft.physical.mass, len(spacecraft.actuators))
```

Then set the asset in the `my_experiment.yaml` from above:

```yaml
vehicle: astrobee_rl
asset: my_spacecraft.yaml
```

`vehicle` selects the registered environment/task; `asset` supplies the physical
spacecraft. This reuses the existing free-flyer RL environment with your vehicle.
Keep the remaining training settings from the earlier example, then run
`run_experiment.py`. Asset paths in YAML resolve relative to that experiment file.

For Python code that needs named asset lookup, use
`register_vehicle_file("my_spacecraft.yaml")`, `get_vehicle("my_spacecraft")`, and
`describe_vehicle("my_spacecraft")` from `smallsat_sim.api`. Asset registration is
optional and does not register a new environment. For new joints, observations,
or task behavior, see the environment registration in
[position_task.py](examples/position_task.py).

### Compose a custom reward

Save this as `my_rewards.py` next to `run_experiment.py`:

```python
import jax.numpy as jnp
from smallsat_sim.api import (
    RewardTerm, compose_reward, register_reward, validate_reward,
)


def position_error(context):
    return jnp.linalg.norm(context.next_states[:, :3], axis=-1)


def command_effort(context):
    return jnp.sum(context.actions ** 2, axis=-1)


position_effort = compose_reward({
    "position": RewardTerm(position_error, weight=-1.0),
    "effort": RewardTerm(command_effort, weight=-0.02),
})
validate_reward(position_effort, num_envs=2, action_dim=12)
register_reward("my_lab/position_effort/v1", position_effort)
```

Each term takes a `RewardContext` and returns one scalar per environment, shape
`(num_envs,)`. In the built-in free-flyer RL task, `next_states[:, :3]` is position
error relative to the target. `actions` contains requested actuator commands,
before faults change the realized thrust. Use JAX array operations so terms can
run in compiled, batched rollouts.

`compose_reward` applies the signed weights and returns a `RewardResult` with the
total reward and weighted component diagnostics. `validate_reward` checks the
return type and reward shape on dummy inputs; it does not test learning quality.
This example rewards position accuracy and low effort. Episode success and failure
remain governed by the separately configured termination function.

Import the module **before** constructing the experiment and select its registered
name (or set `reward: my_lab/position_effort/v1` in YAML):

```python
import my_rewards
from smallsat_sim.api.experiments import make_experiment

experiment = make_experiment(
    "my_experiment.yaml", reward="my_lab/position_effort/v1",
)
try:
    experiment.runner.learn()
finally:
    experiment.env.close()
```

### Replace one built-in reward term

To retain the full-pose reward while changing its running fuel cost, add this to
`my_rewards.py`:

```python
from smallsat_sim.envs.rewards import full_pose_reward


@register_reward("my_lab/quadratic_fuel/v1")
def quadratic_fuel_reward(context):
    return full_pose_reward(
        context,
        fuel=lambda c: c.config.lam_fuel * jnp.sum(c.actions ** 2, axis=-1),
    )
```

Select `reward: my_lab/quadratic_fuel/v1` to use it. The other full-pose terms,
including terminal costs, retain their defaults. Costs are subtracted by
`full_pose_reward`, so the replacement returns a positive cost. Its
[function signature](src/smallsat_sim/envs/rewards.py) exposes the other replaceable
potentials, costs, and success bonus; [custom_rewards.py](examples/custom_rewards.py)
contains both reward patterns in an importable example.

### Define task success and failure

Reward values and episode endings have separate contracts. A termination function
receives transition features and per-environment hold counts; it returns terminal,
success, and failure masks plus updated counts. For example, save this as
`my_task.py` to require low position error and speed for several consecutive steps:

```python
import jax.numpy as jnp
from smallsat_sim.api import register_termination, TerminationResult
from smallsat_sim.envs.termination import full_pose_termination


@register_termination("my_lab/position_only/v1")
def position_termination(context):
    cfg = context.config
    states = context.next_states
    within = jnp.linalg.norm(states[:, :3], axis=-1) < cfg.terminal_radius
    within &= jnp.linalg.norm(states[:, 6:9], axis=-1) < cfg.terminal_max_speed
    counts = jnp.where(within, context.terminal_hold_counts + 1, 0)
    success = counts >= cfg.terminal_hold_steps
    failure = full_pose_termination(context).failure_terminals
    return TerminationResult(success | failure, success, failure, counts)
```

Import `my_task` before construction and add
`RL.termination: my_lab/position_only/v1` to your experiment's `overrides` mapping.
This changes success criteria while retaining the built-in failure checks.
All four returned arrays have shape `(num_envs,)`; keep the calculation JAX-compatible.

### Register an environment or task preset

An `EnvEntry` couples an environment class to its default asset, configuration
builder, planner, and controller. Reuse an entry when only task configuration
changes. Add this to `my_task.py`:

```python
from dataclasses import replace
from smallsat_sim.api.experiments import get_env, register_env

base = get_env("astrobee_rl")


def position_config(spec):
    config = base.config_builder(spec)
    config.env.environment.task = "position_only"
    config.env.environment.termination = "my_lab/position_only/v1"
    return config


register_env(
    "my_lab/position_only",
    replace(base, config_builder=position_config),
)
```

Now select `vehicle: my_lab/position_only` in YAML. This preset retains the base
observations and dynamics. To change those, supply your own `env_cls` in an
`EnvEntry` and implement the interface required by your controller or runner.
The factory passes `args` and `vehicle` to the environment constructor; with a
configuration builder, it also passes `run_name` and `config=config.env`.
The builder returns an object with `env` and `training` configuration fields.
See [position_task.py](examples/position_task.py) for a complete task extension.

### Register planners and controllers

Factories let you select your Python implementations from the same experiment
YAML. A planner builder takes `(env, spec)`; a controller builder takes
`(env, planner, spec, run_spec, training_config)`. For example, save this as
`my_components.py` to register a fixed-point planner and a PD controller factory:

```python
from smallsat_sim.api.experiments import register_planner, register_controller
from smallsat_sim.planners.oracle.oracle import OraclePlanner
from smallsat_sim.controllers.pd.controller import PDController


@register_planner("my_lab/hold")
def build_hold(env, spec):
    return OraclePlanner(env, radius=0.0, **spec.planner_options)


@register_controller("my_lab/pd")
def build_pd(env, planner, spec, run_spec, training_config):
    return PDController(env, planner)
```

These factories reuse existing implementations; replace their returned objects
with your own planner/controller to introduce new behavior. Planner contracts are
defined by [BasePlanner](src/smallsat_sim/planners/base_planner.py); classical
controllers expose `get_control_input(env)`. Planner reference formats must match
the controller consuming them. Run the registered components explicitly:

```python
import my_components
from smallsat_sim.api.experiments import make_experiment

experiment = make_experiment(
    vehicle="cubesat", planner="my_lab/hold", controller="my_lab/pd",
    planner_options={"z_offset": 10.0}, log=False,
)
try:
    while experiment.env.data.time < 2.0:
        command = experiment.controller.get_control_input(experiment.env)
        experiment.env.step(input=command)
finally:
    experiment.env.close()
```

See the [classical controller guide](examples/classical_controllers.md) for built-in
planner/controller compatibility and tuning. The factory exposes custom controller
registrations as `experiment.controller`; the built-in `rl` and `rl/on_policy`
names expose their training object as `experiment.runner`.

### Add actuator faults and external disturbances

Effects use importable functions rather than a named registry. A sampler
`sample(key, context, **params)` creates batched state from a JAX random key and
`EffectSampleContext` (vehicle, number of environments, timestep). An application
function `apply(state, values, time, dt)` returns `(new_values, new_state)` using
pure JAX operations, preserving array shapes and dtypes.

For faults, `values` has shape `(num_envs, num_actuators)`; disturbances produce
world-frame force and torque with shape `(num_envs, 6)`. Every sampled state array
has a leading environment dimension. The bundled example implements both motor
efficiency loss and a constant external wrench. Add this under your experiment's
`overrides.RL` settings to select the motor-loss implementation:

```yaml
custom_faults:
  - sample: examples.custom_effects:sample_motor_loss
    apply: examples.custom_effects:apply_motor_loss
    version: '1'
    probability: 0.5
    start_time: 0.2
    params:
      actuator: thruster1
      minimum: 0.2
      maximum: 0.8
```

`probability` controls activation per environment and `start_time` delays the
effect. To add your own behavior, implement the two functions in an importable
module and change the `module:function` paths. See
[custom_effects.py](examples/custom_effects.py) for the function bodies and
[custom_effects.yaml](examples/custom_effects.yaml) for a complete experiment with
both `custom_faults` and `custom_disturbances`.

### Collect rollouts without training

The runner also supports data collection independently of optimization. For an RL
experiment constructed above, use this inside its `try` block:

```python
runner = experiment.runner
batch = runner.collect(
    runner.collector("zero", stochastic=False), randomize=False,
)
print(batch.actions.shape)  # (steps, environments, actuators)
print(batch.step_outputs.rewards.shape)  # (steps, environments)
```

This runs the configured rollout length with zero commands, useful for inspecting
task rewards and dynamics before training. The
[integrated experiment](examples/integrated_experiment.yaml) combines asset, task,
reward, and effect extensions in one such collection run.

## Training and evaluation

For paired, independently sampled scenarios, both PPO and SAC expose the same
saved-policy API. Construct the runner with its saved training configuration and
checkpoint directory, then call:

```python
import numpy as np

scenarios = []
for seed in range(10000, 10100):
    rng = np.random.default_rng(seed)
    pose = np.asarray(runner.reference_point[0]).copy()
    pose[:3] += rng.uniform(-1.0, 1.0, 3)
    scenarios.append(dict(
        evaluation_seed=seed,
        initial_qpos=pose, initial_qvel=np.zeros(6),
        mass_scale=1.0, inertia_scale=1.0,
        thrust_scale=np.ones(runner.env.act_dim), wrench=np.zeros(6),
    ))

episodes, timing = runner.evaluate(scenarios=scenarios, batch_size=128)
```

This restores inference weights without rewinding training optimizers, counters,
or RNGs, and evaluates deterministic actions in parallel MJX
lanes, including when training used `freeflyer`. Each supplied scenario produces
exactly one result, in input order, containing `scenario`, `metrics`, `time`, and
`qpos`. Metrics and recordings end at the first task termination, invalid
transition, or horizon; padding lanes are discarded. `steps` optionally overrides
`training.episode_len`. Scenarios specify the complete physical realization;
training faults and randomization are not added. Linear initial velocity and the
constant applied wrench use world coordinates; angular initial velocity uses the
body frame. This interface currently supports a single free body with a shared
pose reference. Reuse the same scenario list across policies for paired comparisons.
Evaluation seeds identify supplied samples; they do not trigger additional sampling.
Calling `runner.evaluate()` without scenarios retains the existing batch-summary
behavior for compatibility.

Use the [benchmark CLI](experiments/rl_benchmarking/benchmark.py) for named presets
and seeded comparisons:

```bash
bash .setup/smallsat run python -m experiments.rl_benchmarking.benchmark train \
  --experiment ppo_nominal --seed 0

bash .setup/smallsat run python -m experiments.rl_benchmarking.benchmark evaluate \
  --checkpoint PATH --suite core --seed 10000

bash .setup/smallsat run python -m experiments.rl_benchmarking.benchmark deploy \
  --checkpoint PATH --viewer
```

Replace `PATH` with a checkpoint from that benchmark run. Available presets are
`ppo_nominal`, `ppo_randomized`, `vpg_nominal`, `vpg_randomized`, `sac_nominal`,
`sac_randomized`, `rma_cnn`, and `rma_transformer`; see
[benchmarks.yaml](src/smallsat_sim/config/benchmarks.yaml). Default presets can launch
large training runs. Use `--config overrides.yaml` to adjust their budget; this CLI
expects an **override mapping**, not a full experiment YAML:

```yaml
RL:
  num_envs: 8
  rollout_backend: freeflyer
  checkpoint_dir: experiments/rl_results/ppo_trial
  PPO:
    epochs: 2
```

`--mode resume` resumes training. Adaptive PPO runs support `--stage policy` or
`--stage adaptation`; a fresh adaptation stage can use `--teacher PATH` to compare
estimators using the same trained policy. SAC supports neither adaptation nor
supervised pretraining. Use `--help` for the full command interface.

The `core` evaluation suite covers nominal operation, individual actuator faults,
and an external wrench; `stress` adds compound faults. Adaptive evaluation accepts
`--context-source estimated`, `privileged`, or `zero`. Results and per-scenario
outputs go under `experiments/rl_results/`. Compare matching tasks, seeds, budgets,
and evaluation settings. The [PD baseline](experiments/rl_benchmarking/pd.py) and
[result plotter](experiments/rl_benchmarking/rl_plotter.py) have their own `--help`.

For an editable stage-by-stage workflow, use
[train_astrobee.py](experiments/rl_training/train_astrobee.py). It can run supervised
pretraining, learning, adaptation, and evaluation; the benchmark `train` command
only runs learning and applicable adaptation stages. W&B is opt-in with `--wandb`.
The benchmark enables local logging by default (`--no-log` disables it); the
editable script enables it with `--log`.

## Viewers and video

### Browser viewer, locally or over SSH

Add `--viewer` to the RL training script or benchmark commands:

```bash
bash .setup/smallsat run python -m experiments.rl_training.train_astrobee \
  --headless --viewer
```

This runs the configured training workflow. Open the printed URL, normally
`http://localhost:8080`. For a remote server, run this tunnel on your local machine:

```bash
ssh -N -L 8080:127.0.0.1:8080 USER@REMOTE_HOST
```

Then open the same local URL. `--viewer-port` changes the port; update the tunnel
to match. `--view-env` selects the environment to watch (default: 0). Training shows
sampled live states; evaluation is paced while a browser is connected. The viewer
remains available while the process runs and does not require server OpenGL.

### Record an MP4

`--viewer` alone does not save video. Add `--video` to record, with or without a browser:

```bash
bash .setup/smallsat run python -m experiments.rl_benchmarking.benchmark evaluate \
  --checkpoint PATH --video --video-dir videos --view-env 0
```

Evaluation records an episode per rollout. Training records one clip, capped at
10 seconds by default (`--video-duration` changes the cap). MP4 recording requires
working EGL/OpenGL; the Linux launcher defaults to `MUJOCO_GL=egl`. Disable viewers
and recording when measuring training throughput.

### Native and classical viewers

Omit `--headless` for a classical simulation window. On a Linux desktop, use
`MUJOCO_GL=glfw`; on macOS, MuJoCo's native viewer may require `mjpython` in place of
`python`. Classical environments can instead use a browser with
`viewer: {use_viser: true}` in their configuration, also without `--headless`.
Shared defaults live in [simulation.yaml](src/smallsat_sim/config/simulation.yaml).

## Development

Add a regression test for new behavior or a bug fix, run the relevant tests, then
the full suite. For example:

```bash
bash .setup/smallsat run pytest -q src/tests/test_yaml_experiments.py
bash .setup/smallsat run pytest -q src/tests
bash .setup/smallsat run ruff check src experiments examples
```

Install pytest and Ruff by adding `--group dev` to your installation's `uv sync --locked`
command, retaining its extras. MPC tests need the native stack. For viewer changes,
also inspect a live run or short recording on the target platform.

For training or performance changes, use the focused
[SAC regression tests](src/tests/test_sac.py),
[spacecraft portability experiments](experiments/paper_benchmarks/README.md), or
[throughput profiler](experiments/rl_benchmarking/measure_performance.py).
Their short runs do not establish spacecraft convergence or algorithm superiority.

The code is organized into [experiment construction](src/smallsat_sim/api),
[physical models](src/smallsat_sim/model), [environments and effects](src/smallsat_sim/envs),
[controllers and learning](src/smallsat_sim/controllers), and
[planners](src/smallsat_sim/planners). Runnable examples live in [experiments/](experiments)
and [examples/](examples); tests live in [src/tests/](src/tests).

## License

Copyright (c) 2024, Jet Propulsion Laboratory, California Institute of Technology. All rights reserved. JPL NTR 53088.

This software is licensed under the Apache License, version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## Release artifacts

Build a wheel from a fresh temporary source tree and check its contents and
three spacecraft simulation smoke tests:

```bash
bash .setup/smallsat run python .setup/build_release.py
```

The validated wheel is written to `dist/`. This command uses the installed
setuptools build backend and dependencies; run it after the documented locked
installation. Temporary staging avoids carrying deleted modules from a previous
`build/` directory. CI also runs the CPU tests and this wheel check. Native MPC
and GPU validation remain separate target-platform checks.

The wheel records the pinned Git dependencies used by the MPC/GP stack. Install
from the documented checkout with uv, or distribute the wheel as a repository
release artifact. These direct Git requirements require Git/network access when
installing dependencies and are not suitable for a PyPI upload. A PyPI release
requires separately published, compatible dependency versions first.
