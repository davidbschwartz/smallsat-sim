"""Structured run configuration for experiment construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class RunSpec:
    vehicle: str
    task: str = "setpoint"
    reward: str = "full_pose"
    failure_policy: str | None = None
    algorithm: str = "ppo"
    name: str | None = None
    tags: tuple[str, ...] = ()
    checkpoint_prefix: str | None = None
    overrides: Mapping[str, Any] = field(default_factory=dict)
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_mode: str | None = None


def build_run_name(spec: RunSpec) -> str:
    if spec.name:
        return spec.name
    parts = [
        spec.vehicle,
        spec.task,
        spec.reward,
        spec.failure_policy,
        spec.algorithm,
        *spec.tags,
    ]
    return "_".join(str(part) for part in parts if part)


def build_checkpoint_file_names(spec: RunSpec) -> dict[str, str]:
    prefix = spec.checkpoint_prefix or build_run_name(spec)
    return {
        "pretraining_data_file_name": f"{prefix}_pretraining_data.ckpt",
        "pretraining_state_file_name": f"{prefix}_pretraining_state.ckpt",
        "training_data_file_name": f"{prefix}_training_data.ckpt",
        "training_state_file_name": f"{prefix}_training_state.ckpt",
        "adaptation_module_file_name": f"{prefix}_adapt_module_state.ckpt",
    }


def run_spec_from_env(env, *, algorithm="ppo") -> RunSpec:
    reward = env.env_cfg.environment.reward
    if bool(getattr(env, "train_with_failures", False)):
        failure_policy = "domain_randomization"
    else:
        failure_policy = None
    vehicle = getattr(getattr(env, "model_cfg", None), "name", None)
    if vehicle is None:
        vehicle = getattr(env.env_cfg, "model", "unknown_vehicle")
    return RunSpec(
        vehicle=str(vehicle),
        task=str(getattr(env.env_cfg.environment, "task", "setpoint")),
        reward=str(reward),
        failure_policy=None if failure_policy is None else str(failure_policy),
        algorithm=algorithm,
    )
