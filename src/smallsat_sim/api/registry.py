"""Typed registries for user-provided API extensions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

from smallsat_sim.api.rewards import RewardCallable
from smallsat_sim.envs.termination import TerminationCallable, full_pose_termination
from smallsat_sim.model.vehicle import VehicleSpec


class RegistryError(ValueError):
    pass


T = TypeVar("T")


class NamedRegistry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, value: T, *, replace: bool = False) -> T:
        key = str(name).strip()
        if not key:
            raise RegistryError(f"{self.kind} names must be non-empty.")
        if key in self._items and not replace:
            raise RegistryError(f"{self.kind} entry already exists: {key}")
        self._items[key] = value
        return value

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError as exc:
            raise RegistryError(f"Unknown {self.kind}: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def decorator(
        self,
        name: str | None = None,
        value: T | None = None,
        *,
        replace: bool = False,
    ):
        def register_value(candidate: T) -> T:
            candidate_name = name or getattr(candidate, "__name__", "")
            return self.register(candidate_name, candidate, replace=replace)

        if value is not None:
            return register_value(value)
        return register_value


_vehicles = NamedRegistry[VehicleSpec]("vehicle")
_rewards = NamedRegistry[RewardCallable]("reward")
_terminations = NamedRegistry[TerminationCallable]("termination")
_terminations.register("full_pose", full_pose_termination)


def register_vehicle(vehicle: VehicleSpec, *, replace: bool = False) -> VehicleSpec:
    return _vehicles.register(vehicle.name, vehicle, replace=replace)


def get_vehicle(name: str) -> VehicleSpec:
    return _vehicles.get(name)


def list_vehicles() -> tuple[str, ...]:
    return _vehicles.names()


def describe_vehicle(name: str) -> dict[str, object]:
    vehicle = get_vehicle(name)
    return {
        "name": vehicle.name,
        "bodies": len(vehicle.bodies),
        "actuators": len(vehicle.actuators),
        "geoms": len(vehicle.geoms),
        "asset_kind": vehicle.assets.kind,
        "default_env": vehicle.default_env,
        "default_model": vehicle.default_model,
    }


def register_reward(
    name: str | None = None,
    reward_fn: RewardCallable | None = None,
    *,
    replace: bool = False,
):
    return _rewards.decorator(name, reward_fn, replace=replace)


def get_reward(name: str) -> RewardCallable:
    return _rewards.get(name)


def list_rewards() -> tuple[str, ...]:
    return _rewards.names()


def register_termination(
    name: str | None = None,
    termination_fn: TerminationCallable | None = None,
    *, replace: bool = False,
):
    return _terminations.decorator(name, termination_fn, replace=replace)


def get_termination(name: str) -> TerminationCallable:
    return _terminations.get(name)


def list_terminations() -> tuple[str, ...]:
    return _terminations.names()
