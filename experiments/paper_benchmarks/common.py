"""Configuration snapshots, immutable run identity and durable raw records."""

import argparse
import hashlib
import json
import platform
import subprocess
import uuid
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
EXPERIMENTS = (
    "exp1_scaling",
    "exp2_fault_robustness",
    "exp3_rl_robustness",
    "exp4_docking",
    "spacecraft_portability",
)


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True).encode()).hexdigest()


def _apply_overrides(base, overrides):
    """Apply explicit experiment settings without mutating other experiments."""
    for key, value in overrides.items():
        if key not in base:
            raise ValueError(f"Unknown common override: {key}")
        if isinstance(value, dict) and isinstance(base[key], dict):
            _apply_overrides(base[key], value)
        else:
            base[key] = deepcopy(value)


def load_config(experiment, *, mode="development", path=None):
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")
    if mode == "paper" and path:
        raise ValueError("--paper uses checked-in configs; use development mode for --config")
    if path:
        config = yaml.safe_load(Path(path).read_text())
    else:
        common = yaml.safe_load((HERE / "configs/common.yaml").read_text())
        protocol = yaml.safe_load((HERE / "configs" / experiment / "paper.yaml").read_text())
        _apply_overrides(common, protocol.get("common_overrides", {}))
        if "evaluation_distributions" in protocol:
            common["evaluation_distributions"] = deepcopy(protocol["evaluation_distributions"])
        assets = {
            name: yaml.safe_load((HERE / "configs/assets" / f"{name}.yaml").read_text())
            for name in ("astrobee", "cubesat", "sprint")
        }
        config = dict(common=common, protocol=protocol, assets=assets)
    config = deepcopy(config)
    config["mode"] = mode
    if config["protocol"]["experiment"] != experiment:
        raise ValueError("Configuration experiment does not match entry point")
    if mode == "smoke":
        _configure_smoke(config, experiment)
    validate_config(config)
    return config


def _configure_smoke(config, experiment):
    """Reduce execution budgets without changing the checked-in paper protocol."""
    protocol, common = config["protocol"], config["common"]
    common["episode_steps"] = 8
    common["env"]["environment"]["num_envs"] = 2
    for algorithm in ("PPO", "VPG"):
        common["training"][algorithm].update(
            steps_per_epoch=8,
            max_ep_len=8,
            epochs=1,
            num_minibatches=1,
            critic_training_epochs=1,
        )
    common["training"]["PPO"]["actor_training_epochs"] = 1
    common["training"]["SAC"].update(
        total_transitions=16,
        collection_steps=4,
        replay_capacity=32,
        randomization_pool_size=4,
        batch_size=4,
        random_steps=0,
        learning_starts=4,
        updates_per_collection=1,
        max_ep_len=8,
    )
    common["training"]["episode_len"] = 8
    if "trials" in protocol:
        protocol["trials"] = 2
        if "evaluation_trials" in protocol:
            protocol["evaluation_trials"] = {
                k: min(v, 2) for k, v in protocol["evaluation_trials"].items()
            }
        if "evaluation_batch_size" in protocol:
            protocol["evaluation_batch_size"] = 2
    if experiment == "exp1_scaling":
        protocol.update(
            batch_sizes=[1, 8], rollout_steps=8, repeats=2, warmups=1, require_gpu=False
        )
    if experiment == "exp3_rl_robustness":
        protocol["seeds"] = [0]
        protocol["training_num_envs"] = {key: 2 for key in protocol.get("training_num_envs", {})}
    if experiment == "exp4_docking":
        protocol.update(default_trials=2, sensitivity_trials=2, steps=8)


def evaluation_trial_count(config, condition):
    """Per-condition episode counts; absent overrides preserve historical protocols."""
    protocol = config["protocol"]
    return protocol.get("evaluation_trials", {}).get(condition, protocol["trials"])


def validate_config(config):
    common, protocol = config["common"], config["protocol"]
    if "sac_training_evaluation" in protocol:
        raise ValueError("sac_training_evaluation is no longer supported")
    for sizes in protocol.get("policy_hidden_sizes", {}).values():
        if not isinstance(sizes, list) or not sizes or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in sizes
        ):
            raise ValueError("policy_hidden_sizes must contain nonempty lists of positive integers")
    if "sac_curriculum" in protocol:
        raise ValueError("sac_curriculum is no longer supported; use a configuration without curriculum")
    if protocol.get("evaluation_backend", "mujoco_native") not in ("mjx", "mujoco_native"):
        raise ValueError("evaluation_backend must be mjx or mujoco_native")
    if (protocol["experiment"] == "exp3_rl_robustness"
            and protocol.get("evaluation_backend", "mjx") != "mjx"):
        raise ValueError("RL scenario evaluation uses MJX; classical baselines use native MuJoCo")
    overrides = protocol.get('evaluation_trials', {})
    if not isinstance(overrides, dict) or set(overrides) - set(common['evaluation_distributions']):
        raise ValueError('evaluation_trials must map known conditions to positive counts')
    for count in overrides.values():
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError('evaluation_trials must contain positive integers')
    batch_size = protocol.get('evaluation_batch_size', 128)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('evaluation_batch_size must be a positive integer')
    if protocol.get("wandb_mode", "disabled") not in ("disabled", "offline", "online"):
        raise ValueError("wandb_mode must be disabled, offline or online")
    for interval in protocol.get("checkpoint_intervals", {}).values():
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError("checkpoint_intervals must contain positive integers")
    for count in protocol.get("training_num_envs", {}).values():
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("training_num_envs must contain positive integers")
    if (
        common["schema_version"] != 1
        or common["success_id"] != "demonstration_use_cases/full_pose/v1"
    ):
        raise ValueError("Unsupported protocol/success version")
    if common["env"]["sim"]["dt"] <= 0 or common["episode_steps"] < 1:
        raise ValueError("Positive timestep and episode length required")
    for key in (
        "trials",
        "repeats",
        "rollout_steps",
        "steps",
        "default_trials",
        "sensitivity_trials",
    ):
        if key in protocol and (
            isinstance(protocol[key], bool) or not isinstance(protocol[key], int) or protocol[key] < 1
        ):
            raise ValueError(f"{key} must be a positive integer")
    for key in ("batch_sizes", "seeds", "conditions", "controllers", "algorithms", "regimes"):
        if key in protocol and (not protocol[key] or len(set(protocol[key])) != len(protocol[key])):
            raise ValueError(f"{key} must be nonempty and unique")
    for distribution in [
        common["train_distribution"],
        *common["evaluation_distributions"].values(),
    ]:
        for key in ("mass", "inertia", "thrust"):
            low, high = distribution.get(key, [1.0, 1.0])
            if not 0 < low <= high:
                raise ValueError(f"Invalid {key} support")
    return config


def parser(experiment):
    result = argparse.ArgumentParser(description=experiment.replace("_", " "))
    modes = result.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--paper", action="store_true")
    result.add_argument("--config", type=Path, help="Full resolved YAML, development mode only")
    result.add_argument("--output", type=Path, default=Path("results/demonstration_use_cases"))
    result.add_argument("--job-id", action="append", help="Run only these exact --plan job identities")
    result.add_argument("--resume", action="store_true", help="Skip validated completed jobs; retain failed attempts")
    result.add_argument(
        "--plan", action="store_true", help="Print required jobs without simulation"
    )
    return result


def mode_of(args):
    if args.paper:
        return "paper"
    if args.smoke:
        return "smoke"
    return "development"


def jobs(config):
    protocol = config["protocol"]
    experiment = protocol["experiment"]
    if experiment == "exp1_scaling":
        return [
            dict(batch_size=batch_size, seed=0, spacecraft=protocol["spacecraft"], controller="ppo")
            for batch_size in protocol["batch_sizes"]
        ]
    if experiment == "exp2_fault_robustness":
        return [
            dict(
                controller=controller,
                condition=condition,
                seed=0,
                spacecraft=protocol["spacecraft"],
            )
            for controller in protocol["controllers"]
            for condition in protocol["conditions"]
        ]
    if experiment == "exp3_rl_robustness":
        return [
            dict(controller=algorithm, regime=regime, seed=seed, spacecraft=protocol["spacecraft"])
            for algorithm in protocol["algorithms"]
            for regime in protocol["regimes"]
            for seed in protocol["seeds"]
        ] + [
            dict(controller=algorithm, regime="baseline", seed=0, spacecraft=protocol["spacecraft"])
            for algorithm in protocol["baselines"]
        ]
    if experiment == "exp4_docking":
        return [
            dict(
                controller=protocol["controller"],
                condition=condition,
                seed=0,
                spacecraft=protocol["spacecraft"],
            )
            for condition in protocol["settings"]
        ]
    return [
        dict(controller=controller, condition=condition, seed=0, spacecraft=spacecraft)
        for spacecraft in protocol["spacecraft"]
        for controller in protocol["controllers"]
        for condition in protocol["conditions"]
    ]


def job_id(job):
    return "_".join(f"{key}-{value}" for key, value in sorted(job.items()))


def git_info():
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=HERE, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain"))}


@lru_cache(maxsize=1)
def source_fingerprint():
    """Capture uncommitted implementation changes as well as the commit identity."""
    repository = HERE.parents[1]
    files = sorted((repository / "src/smallsat_sim").rglob("*.py"))
    files += sorted(HERE.glob("*.py"))
    files += sorted((repository / "src/smallsat_sim/model/xml").glob("*.xml"))
    hashes = {
        str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    return digest(hashes), hashes


def metadata(config, job):
    import jax

    versions = {}
    for name in (
        "mujoco",
        "jax",
        "jaxlib",
        "numpy",
        "flax",
        "optax",
        "casadi",
        "gpytorch",
        "torch",
    ):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    common = config["common"]
    sim_dt = config["protocol"].get("sim_dt", common["env"]["sim"]["dt"])
    decimation = config["protocol"].get(
        "control_decimation", common["env"]["environment"]["control_decimation"]
    )
    return dict(
        experiment=config["protocol"]["experiment"],
        job=job,
        job_id=job_id(job),
        config_id=digest(config),
        mode=config["mode"],
        timestamp=datetime.now(timezone.utc).isoformat(),
        **git_info(),
        python=platform.python_version(),
        hardware=platform.platform(),
        devices=[str(d) + ":" + d.device_kind for d in jax.devices()],
        jax_backend=jax.default_backend(),
        versions=versions,
        source_fingerprint=source_fingerprint()[0],
        success_definition=common["success_id"],
        task="pose_regulation",
        simulator_timestep=sim_dt,
        control_decimation=decimation,
        controller_timestep=sim_dt * decimation,
        num_envs=job.get(
            "batch_size",
            common["env"]["environment"]["num_envs"]
            if job.get("regime") not in (None, "baseline")
            else 1,
        ),
        checkpoint_path=None,
        **job,
    )


class Run:
    """A single job's resolved configuration, metadata, and append-only records."""

    def __init__(self, root, config, job):
        self.config, self.job = config, job
        self.path = (
            Path(root)
            / config["mode"]
            / config["protocol"]["experiment"]
            / (job_id(job) + "_" + digest(config)[:12] + "_" + uuid.uuid4().hex[:12])
        )
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "resolved_config.yaml").write_text(
            yaml.safe_dump(plain(config), sort_keys=True)
        )
        write_json(self.path / "source_hashes.json", source_fingerprint()[1])
        self.meta = metadata(config, job)
        self.meta.update(run_id=self.path.name, status="running")
        write_json(self.path / "metadata.json", self.meta)

    @classmethod
    def open_existing(cls, directory):
        """Open a prepared run in a worker without creating a new run identity."""
        run = cls.__new__(cls)
        run.path = Path(directory)
        run.config = yaml.safe_load((run.path / "resolved_config.yaml").read_text())
        run.job = json.loads((run.path / "job.json").read_text())
        run.meta = json.loads((run.path / "metadata.json").read_text())
        return run

    def append(self, row, filename="metrics.jsonl"):
        with (self.path / filename).open("a") as out:
            out.write(json.dumps(plain(row), sort_keys=True, allow_nan=False) + "\n")
            out.flush()

    def update(self, **values):
        self.meta.update(values)
        write_json(self.path / "metadata.json", self.meta)


@contextmanager
def run_directory(root, config, job):
    run = Run(root, config, job)
    try:
        yield run
    except BaseException as error:
        run.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    else:
        run.update(status="complete")


def execute(experiment, args, handler):
    config = load_config(experiment, mode=mode_of(args), path=args.config)
    selected = jobs(config)
    requested = getattr(args, "job_id", None)
    if requested:
        unknown = set(requested) - {job_id(job) for job in selected}
        if unknown:
            raise ValueError(f"Unknown job identities: {sorted(unknown)}")
        selected = [job for job in selected if job_id(job) in requested]
    if args.plan:
        print(
            json.dumps(
                {"mode": config["mode"], "config_id": digest(config), "jobs": selected,
                 "job_ids": [job_id(job) for job in selected]}, indent=2
            )
        )
        return
    failures = []
    for job in selected:
        print(f"{experiment}: {job_id(job)}", flush=True)
        try:
            if getattr(args, "resume", False) and completed_job(args.output, config, job):
                print("Validated completed job; skipping", flush=True)
                continue
            with run_directory(args.output, config, job) as run:
                handler(config, job, run)
        except Exception as error:
            failures.append(f"{job_id(job)}: {error}")
            print(f"FAILED (raw directory retained): {failures[-1]}", flush=True)
    if failures:
        raise RuntimeError(
            "Some jobs failed; all other requested jobs were attempted:\n" + "\n".join(failures)
        )


def completed_job(root, config, job):
    """Never select a favorable repeat or resume a different protocol."""
    from .aggregate import records, validate_run_records

    directory = Path(root) / config["mode"] / config["protocol"]["experiment"]
    completed = []
    for path in directory.glob("*/metadata.json"):
        meta = json.loads(path.read_text())
        if meta["job_id"] != job_id(job):
            continue
        if meta["status"] == "running":
            raise ValueError(f"Job already marked running: {path.parent}")
        if meta["status"] != "complete":
            continue
        resolved = yaml.safe_load((path.parent / "resolved_config.yaml").read_text())
        if digest(resolved) != digest(config) or meta["config_id"] != digest(config):
            raise ValueError(f"Completed job has a different configuration: {path.parent}")
        rows = records(path.parent / "metrics.jsonl")
        validate_run_records(path.parent, config, meta, rows)
        completed.append(path.parent)
    if len(completed) > 1:
        raise ValueError(f"Duplicate completed job {job_id(job)}")
    return bool(completed)
