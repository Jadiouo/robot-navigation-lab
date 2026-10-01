"""Stress conditions: one-factor-at-a-time axes around a nominal condition, plus the MCL-breaking suite.

A :class:`Condition` is everything that differs between two episodes of the same scenario seed besides the
planner and the pose source: the scenario-side stress (agents, hidden obstacles) and the sensing / odometry stress.
Level 0 of every axis *is* the nominal condition, so the nominal episodes are reused as level 0 of the stress curves.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from navlab.world.generator import ScenarioStress


@dataclass(frozen=True)
class Condition:
    name: str
    axis: str = "nominal"
    level: int = 0
    value: float = 0.0              # the axis value (for plotting)
    agent_density: float = 1.0
    agent_speed: float = 1.0
    hidden_density: float = 1.0
    lidar_sigma: float = 0.05       # m
    lidar_p_short: float = 0.04
    lidar_p_max: float = 0.02       # dropout
    gyro_bias: float = 0.01         # rad/s
    speed_scale: float = 1.0

    @property
    def stress(self) -> ScenarioStress:
        return ScenarioStress(self.agent_density, self.agent_speed, self.hidden_density)


NOMINAL = Condition("nominal")

# axis name -> (Condition field, levels); the first level is the nominal value.
AXES: dict[str, tuple[str, tuple[float, ...]]] = {
    "lidar_sigma": ("lidar_sigma", (0.05, 0.10, 0.20, 0.40, 0.80)),
    "lidar_dropout": ("lidar_p_max", (0.02, 0.10, 0.20, 0.35, 0.50)),
    "lidar_short": ("lidar_p_short", (0.04, 0.10, 0.20, 0.30, 0.40)),
    "gyro_bias": ("gyro_bias", (0.01, 0.03, 0.06, 0.10, 0.15)),
    "speed_scale": ("speed_scale", (1.0, 1.05, 1.10, 1.20, 1.30)),
    "agent_density": ("agent_density", (1.0, 2.0, 3.0, 4.5, 6.0)),
    "agent_speed": ("agent_speed", (1.0, 1.5, 2.0, 2.5, 3.0)),
    "hidden_density": ("hidden_density", (1.0, 2.0, 4.0, 7.0, 10.0)),
}
# Axes on which the ground-truth-pose arm is also run (the scan-side axes, where localisation is a mediator).
GT_AXES = ("lidar_short", "agent_density", "agent_speed", "hidden_density")


def axis_conditions(axis: str) -> list[Condition]:
    field_name, levels = AXES[axis]
    out = []
    for i, v in enumerate(levels):
        out.append(Condition(f"{axis}={v:g}", axis, i, v, **{field_name: v}))
    return out


def stress_conditions() -> list[Condition]:
    """All non-nominal levels of all axes (nominal is level 0 of each and comes from the nominal suite)."""
    return [c for ax in AXES for c in axis_conditions(ax)[1:]]


# MCL-breaking suite: (family, condition).  Designed to remove the information the filter lives on, in a world that is
# otherwise quiet (no agents in the hall) so that a failure there points at localization:
# a featureless corridor with a biased odometer (the along-track drift is unobservable), the same with unmapped
# boxes (part of every scan is unexplained by the map), and heavy occlusion by a crowd with many spurious / dropped returns.
MCL_BREAK: tuple[tuple[str, Condition], ...] = (
    ("hall", Condition("hall/clean", "mclbreak", 0, 0.0, agent_density=0.0, hidden_density=0.0)),
    ("hall", Condition("hall/odom_bias", "mclbreak", 1, 1.0, agent_density=0.0, hidden_density=0.0, gyro_bias=0.10, speed_scale=1.20)),
    ("hall", Condition("hall/odom_bias+hidden", "mclbreak", 2, 2.0, agent_density=0.0, hidden_density=10.0, gyro_bias=0.10, speed_scale=1.20)),
    ("rooms", Condition("rooms/crowd+outliers", "mclbreak", 3, 3.0, agent_density=8.0, lidar_p_short=0.30, lidar_p_max=0.30)),
)
