"""Construct environments, planners and controllers from one experiment description.

Use make_experiment for YAML-driven runs; register_* adds Python behavior.
The returned objects expose their usual explicit train/control methods.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from smallsat_sim.api.registry import NamedRegistry, RegistryError, get_reward, get_termination
from smallsat_sim.api.runs import RunSpec
from smallsat_sim.model.vehicle import load_vehicle
from smallsat_sim.configuration import (
    read_yaml,
    settings,
    flatten_overrides,
)
from smallsat_sim.controllers.pd.controller import PDController
from smallsat_sim.controllers.pd.vectorized_controller import VectorizedPDController
from smallsat_sim.controllers.rl.runners.factory import make_runner
from smallsat_sim.envs.vehicles.astrobee.env import AstrobeeEnv
from smallsat_sim.envs.config import resolve_astrobee_config
from smallsat_sim.envs.vehicles.astrobee_rl.env import AstrobeeEnvVectorized
from smallsat_sim.envs.vehicles.astrobee_rl.config import resolve_config
from smallsat_sim.envs.vehicles.cubesat.env import CubesatEnv
from smallsat_sim.envs.config import resolve_cubesat_config
from smallsat_sim.planners.oracle.oracle import OraclePlanner
from smallsat_sim.planners.oracle.oracle_rl import OraclePlannerRL


@dataclass(frozen=True)
class ExperimentSpec:
    vehicle: str
    asset: str | None = None
    env: str | None = None
    controller: str | None = None
    planner: str | None = None
    reward: str | None = None
    failures: str | None = None
    algorithm: str | None = None
    run_name: str = "default"
    overrides: Mapping[str, Any] = field(default_factory=dict)
    wandb: bool = False
    log: bool = True
    video: bool = False
    headless: bool = True
    planner_radius: float = 3.0
    planner_spacing: float = 1.0
    planner_clearance_dist: float = 0.2
    planner_options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Experiment:
    env: Any
    planner: Any | None
    controller: Any | None = None
    runner: Any | None = None
    run_spec: RunSpec | None = None


@dataclass(frozen=True)
class EnvEntry:
    """Environment constructor and its default physical asset."""

    env_cls: type
    default_asset: str
    config_builder: Callable[[ExperimentSpec], Any] | None = None
    default_controller: str = "pd"
    default_planner: str = "oracle"


PlannerBuilder = Callable[[Any, ExperimentSpec], Any]
ControllerBuilder = Callable[[Any, Any | None, ExperimentSpec, RunSpec, Any], Any]


_ENVS = NamedRegistry[EnvEntry]("env")
_PLANNERS = NamedRegistry[PlannerBuilder]("planner")
_CONTROLLERS = NamedRegistry[ControllerBuilder]("controller")


def register_env(name: str, entry: EnvEntry, *, replace: bool = False) -> EnvEntry:
    return _ENVS.register(name, entry, replace=replace)


def get_env(name: str) -> EnvEntry:
    return _ENVS.get(name)


def list_envs() -> tuple[str, ...]:
    return _ENVS.names()


def register_planner(
    name: str | None = None,
    planner_fn: PlannerBuilder | None = None,
    *,
    replace: bool = False,
):
    return _PLANNERS.decorator(name, planner_fn, replace=replace)


def get_planner(name: str) -> PlannerBuilder:
    return _PLANNERS.get(name)


def list_planners() -> tuple[str, ...]:
    return _PLANNERS.names()


def register_controller(
    name: str | None = None,
    controller_fn: ControllerBuilder | None = None,
    *,
    replace: bool = False,
):
    return _CONTROLLERS.decorator(name, controller_fn, replace=replace)


def get_controller(name: str) -> ControllerBuilder:
    return _CONTROLLERS.get(name)


def list_controllers() -> tuple[str, ...]:
    return _CONTROLLERS.names()


def _env_args(spec: ExperimentSpec) -> SimpleNamespace:
    return SimpleNamespace(
        headless=spec.headless,
        num_bodies=1,
        video=spec.video,
        log=spec.log,
        wandb=spec.wandb,
    )


def _train_with_failures(failures: str | None) -> bool:
    return failures is not None and str(failures).lower() not in (
        "none",
        "off",
        "false",
    )


def _astrobee_config(spec: ExperimentSpec):
    return settings({
        "env": resolve_astrobee_config(overrides=spec.overrides), "training": None,
    })


def _cubesat_config(spec: ExperimentSpec):
    return settings({
        "env": resolve_cubesat_config(overrides=spec.overrides), "training": None,
    })


def _sprint_defaults(spec: ExperimentSpec):
    # Preserve both supported spellings of explicit PD overrides.
    overrides = {
        (f"control.{key}" if key.startswith("PD.") else key): value
        for key, value in flatten_overrides(spec.overrides).items()
    }
    for key, value in {"Kp_x": 0.4, "Kd_x": 5.0, "Kp_q": 0.08, "Kd_q": 0.2}.items():
        overrides.setdefault(f"control.PD.gains.{key}", value)
    return replace(spec, overrides=overrides)


def _physical_pd_defaults(spec, asset):
    overrides = {(f"control.{key}" if key.startswith("PD.") else key): value
                 for key, value in flatten_overrides(spec.overrides).items()}
    m, inertia = asset.physical.mass, max(asset.physical.diag_inertia)
    for key, value in {"Kp_x": m * .35**2, "Kd_x": 2 * m * .35,
                       "Kp_q": 2 * inertia * .8**2, "Kd_q": 2 * inertia * .8}.items():
        overrides.setdefault(f"control.PD.gains.{key}", value)
    return replace(spec, overrides=overrides)


def _sprint_config(spec: ExperimentSpec):
    return _astrobee_config(_physical_pd_defaults(spec, load_vehicle(spec.asset or "vehicles/sprint.yaml")))


def _sprint_rl_config(spec: ExperimentSpec):
    return _astrobee_rl_config(_sprint_defaults(spec))


def _astrobee_rl_config(spec: ExperimentSpec):
    config = resolve_config(
        overrides=spec.overrides,
        algorithm=spec.algorithm,
        train_with_failures=(
            None if spec.failures is None else _train_with_failures(spec.failures)
        ),
    )
    if spec.reward is not None:
        config.env.environment.reward = spec.reward
    get_termination(config.env.environment.termination)
    return config


def _build_env(spec: ExperimentSpec, env_entry: EnvEntry, config) -> Any:
    vehicle = load_vehicle(spec.asset or env_entry.default_asset)
    if config is not None:
        return env_entry.env_cls(
            args=_env_args(spec), run_name=spec.run_name,
            config=config.env, vehicle=vehicle,
        )
    return env_entry.env_cls(args=_env_args(spec), vehicle=vehicle)


def _build_run_spec(env: Any, spec: ExperimentSpec, training_config) -> RunSpec:
    return RunSpec(
        vehicle=spec.vehicle,
        task=(
            str(getattr(env.env_cfg.environment, "task", "setpoint"))
            if spec.reward and hasattr(env.env_cfg, "environment")
            else str(spec.controller)
        ),
        reward=spec.reward,
        failure_policy=spec.failures,
        algorithm=(
            training_config.algorithm
            if spec.controller in ("rl", "rl/on_policy")
            else spec.algorithm or spec.controller
        ),
        name=spec.run_name,
        overrides=spec.overrides,
        wandb_mode="online" if spec.wandb else None,
    )


@register_planner("oracle")
def _build_oracle_planner(env: Any, spec: ExperimentSpec) -> OraclePlanner:
    return OraclePlanner(
        env,
        radius=spec.planner_radius,
        spacing=spec.planner_spacing,
        clearance_dist=spec.planner_clearance_dist,
        **spec.planner_options,
    )


@register_planner("oracle_rl")
def _build_oracle_rl_planner(env: Any, spec: ExperimentSpec) -> OraclePlannerRL:
    return OraclePlannerRL(
        env,
        radius=spec.planner_radius,
        spacing=spec.planner_spacing,
        clearance_dist=spec.planner_clearance_dist,
        **spec.planner_options,
    )


@register_planner("mission")
def _build_mission(env, spec):
    from smallsat_sim.planners.mission.mission import MissionPlanner
    return MissionPlanner(env, spacing=spec.planner_spacing, **spec.planner_options)


@register_planner("mission_spline")
def _build_spline(env, spec):
    from smallsat_sim.planners.mission.mission import MissionPlannerCubicSpline
    return MissionPlannerCubicSpline(env, spacing=spec.planner_spacing, **spec.planner_options)


@register_planner("docking")
def _build_docking(env, spec):
    from smallsat_sim.planners.mission.docking import DockingPlanner
    return DockingPlanner(env, **spec.planner_options)


@register_planner("ad_star")
def _build_ad_star(env, spec):
    if spec.planner_options:
        raise RegistryError("AD* settings belong in overrides.planner")
    from smallsat_sim.planners.ad_star.ad_star import ADStarPlanner
    return ADStarPlanner(env)


def _classical_builder(module, class_name):
    def build(env, planner, spec, run_spec, training_config):
        if getattr(env, "using_rl", False):
            raise RegistryError(f"{spec.controller} requires a single classical environment")
        if planner is None:
            raise RegistryError(f"{spec.controller} requires a planner")
        if spec.controller in ("mpcc", "gp_mpc"):
            from smallsat_sim.planners.validation import require_contouring_planner
            require_contouring_planner(planner)
        from importlib import import_module
        cls = getattr(import_module(module), class_name)
        return cls(env, planner)
    return build


for _name, _module, _class in (
    ("lqr", "lqr", "LQRController"),
    ("mpc", "nominal_mpc", "NominalMPCController"),
    ("mpcc", "mpcc", "NominalMPCCController"),
    ("gp_mpc", "gp_mpc", "GPMPC"),
):
    register_controller(_name, _classical_builder(f"smallsat_sim.controllers.{_module}.controller", _class))


@register_controller("pd")
def _build_pd_controller(
    env: Any, planner: Any | None, spec: ExperimentSpec, run_spec: RunSpec, training_config
) -> PDController:
    del spec, run_spec
    if planner is None:
        raise RegistryError("Controller 'pd' requires a planner.")
    return PDController(env, planner)


@register_controller("vectorized_pd")
def _build_vectorized_pd_controller(
    env: Any, planner: Any | None, spec: ExperimentSpec, run_spec: RunSpec, training_config
) -> VectorizedPDController:
    del spec, run_spec
    if planner is None:
        raise RegistryError("Controller 'vectorized_pd' requires a planner.")
    return VectorizedPDController(env, planner)


@register_controller("rl")
@register_controller("rl/on_policy")
def _build_on_policy_runner(
    env: Any, planner: Any | None, spec: ExperimentSpec, run_spec: RunSpec, training_config
) -> Any:
    if planner is None:
        raise RegistryError("Controller 'rl/on_policy' requires a planner.")
    runner = make_runner(env, planner, config=training_config)
    runner.run_spec = run_spec
    return runner


register_env(
    "astrobee",
    EnvEntry(
        env_cls=AstrobeeEnv,
        default_asset="vehicles/astrobee.yaml",
        config_builder=_astrobee_config,
    ),
)
register_env(
    "cubesat",
    EnvEntry(
        env_cls=CubesatEnv,
        default_asset="vehicles/cubesat.yaml",
        config_builder=_cubesat_config,
    ),
)
register_env(
    "astrobee_rl",
    EnvEntry(
        env_cls=AstrobeeEnvVectorized,
        default_asset="vehicles/astrobee.yaml",
        config_builder=_astrobee_rl_config,
        default_controller="rl/on_policy",
        default_planner="oracle_rl",
    ),
)


register_env(
    "sprint",
    EnvEntry(
        env_cls=AstrobeeEnv,
        default_asset="vehicles/sprint.yaml",
        config_builder=_sprint_config,
    ),
)
register_env(
    "sprint_rl",
    EnvEntry(
        env_cls=AstrobeeEnvVectorized,
        default_asset="vehicles/sprint.yaml",
        config_builder=_sprint_rl_config,
        default_controller="rl/on_policy",
        default_planner="oracle_rl",
    ),
)


def make_experiment(
    config: str | Path | ExperimentSpec | None = None,
    **options,
) -> Experiment:
    """Build an experiment from YAML, an ExperimentSpec, or explicit keyword settings.

    Nothing runs here: callers explicitly train, evaluate, or control the result.
    """
    if isinstance(config, ExperimentSpec):
        spec = replace(config, **options)
    else:
        values = {} if config is None else read_yaml(config)
        # User experiment files can keep their asset YAML alongside them.
        if config is not None and values.get("asset") and Path(config).is_file():
            asset_path = Path(config).parent / values["asset"]
            if asset_path.is_file():
                values["asset"] = str(asset_path.resolve())
        overrides = {
            **flatten_overrides(values.get("overrides", {})),
            **flatten_overrides(options.get("overrides", {})),
        }
        spec = ExperimentSpec(**{**values, **options, "overrides": overrides})

    # Resolve names before allocating the simulator or optional controller.
    env_name = spec.env or spec.vehicle
    env_entry = get_env(env_name)
    controller_name = spec.controller or env_entry.default_controller
    planner_name = spec.planner or env_entry.default_planner
    reward = spec.reward
    spec = replace(
        spec,
        env=env_name,
        controller=controller_name,
        planner=planner_name,
        reward=reward,
    )
    planner_builder = get_planner(planner_name)
    controller_builder = get_controller(controller_name)
    if reward is not None:
        get_reward(reward)

    # Build in dependency order, and close the environment if setup fails.
    if spec.asset is not None and controller_name in ("pd", "vectorized_pd"):
        spec = _physical_pd_defaults(spec, load_vehicle(spec.asset))
    config = env_entry.config_builder(spec) if env_entry.config_builder else None
    if config is None and spec.overrides:
        raise RegistryError(
            "Register a config_builder to resolve environment overrides before allocation"
        )
    if config is not None and hasattr(config.env, "environment"):
        resolved_reward = getattr(config.env.environment, "reward", None)
        if resolved_reward is not None:
            get_reward(resolved_reward)
            spec = replace(spec, reward=resolved_reward)
    training_config = config.training if config is not None else None
    runtime_env = _build_env(spec, env_entry, config)
    try:
        run_spec = _build_run_spec(runtime_env, spec, training_config)
        runtime_env.run_spec = run_spec
        runtime_planner = planner_builder(runtime_env, spec)
        runtime_controller = controller_builder(
            runtime_env, runtime_planner, spec, run_spec, training_config
        )
        is_rl = controller_name in ("rl", "rl/on_policy")
        return Experiment(
            env=runtime_env,
            planner=runtime_planner,
            run_spec=run_spec,
            controller=None if is_rl else runtime_controller,
            runner=runtime_controller if is_rl else None,
        )
    except BaseException:
        runtime_env.close()
        raise


def make_rl_experiment(*, vehicle="astrobee_rl", **options):
    """Convenience tuple for scripts that explicitly own the RL workflow."""
    entry = get_env(options.get("env") or vehicle)
    if (options.get("controller") or entry.default_controller) not in ("rl", "rl/on_policy"):
        raise RegistryError("make_rl_experiment requires an rl/on_policy environment/controller")
    experiment = make_experiment(vehicle=vehicle, **options)
    return experiment.env, experiment.planner, experiment.runner
