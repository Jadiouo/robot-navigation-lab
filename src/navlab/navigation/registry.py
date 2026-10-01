"""Planner registry: adding a planner to the benchmark is one ``register_planner`` call.

A factory has the signature ``factory(vehicle, v_pref, seed, **overrides) -> LocalPlanner`` (the planner contract of
:mod:`navlab.local.interface`).  ``overrides`` are hyper-parameter overrides (the frozen benchmark config supplies them).
A planner may set the attribute ``uses_global_layer = False`` to run without replanning / recovery (the reference
``pp`` baseline); the default is ``True``.  P4's learned planner would be, e.g.::

    register_planner("rl", lambda vehicle, v_pref, seed, **kw: RLPlanner(vehicle, **kw))
"""
from __future__ import annotations

from typing import Callable

from navlab.core import VehicleConfig
from navlab.local import (DWAConfig, DWAPlanner, LocalPlanner, MPPIConfig, MPPIPlanner, PPStopConfig, PPStopPlanner,
                          PurePursuitTracker)

PlannerFactory = Callable[..., LocalPlanner]
PLANNER_REGISTRY: dict[str, PlannerFactory] = {}
ALIASES = {"pure_pursuit": "pp"}


def register_planner(name: str, factory: PlannerFactory) -> None:
    if name in PLANNER_REGISTRY:
        raise ValueError(f"planner {name!r} is already registered")
    PLANNER_REGISTRY[name] = factory


def planner_names() -> tuple[str, ...]:
    return tuple(PLANNER_REGISTRY)


def make_planner(name: str, vehicle: VehicleConfig, v_pref: float = 4.0, seed: int = 0, **overrides) -> LocalPlanner:
    name = ALIASES.get(name, name)
    if name not in PLANNER_REGISTRY:
        raise ValueError(f"planner must be one of {planner_names()}")
    return PLANNER_REGISTRY[name](vehicle, v_pref, seed, **overrides)


register_planner("pp", lambda vehicle, v_pref, seed, **kw: PurePursuitTracker(vehicle, v_pref))
register_planner("pp_stop", lambda vehicle, v_pref, seed, **kw: PPStopPlanner(vehicle, PPStopConfig(v_pref=v_pref, **kw)))
register_planner("dwa", lambda vehicle, v_pref, seed, **kw: DWAPlanner(vehicle, DWAConfig(v_pref=v_pref, **kw)))
register_planner("mppi", lambda vehicle, v_pref, seed, **kw: MPPIPlanner(vehicle, MPPIConfig(v_pref=v_pref, seed=seed, **kw)))
