"""Scenarios with unmapped obstacles.  The known map (what the robot owns) is always the static one.

=================  ===================================================================================
``corridor_box``   straight 11.5 m corridor, a 2 m hidden box on the centre line (static, unmapped)
``crossing_ped``   open map, a pedestrian steps out into the lane as the car approaches
``patrol``         open map, two pedestrians patrolling across the route at different phases
``crowd``          open map, four bouncing constant-velocity agents
``hidden_wall``    open map, a hidden 14 m wall across the route (needs a global replan)
=================  ===================================================================================
All geometry jitter is drawn from ``np.random.default_rng(seed)``; the same seed reproduces the world.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from navlab.core import GridMap, Scenario, VehicleConfig, VehicleState
from navlab.maps.scenarios import _HEIGHT_M, _RESOLUTION, _WIDTH_M, _empty_grid, _fill_rect
from navlab.world.dynamic import ConstantVelocityAgent, CrossingPedestrian, DiscAgent, PatrolAgent
from navlab.world.scene import DynamicWorld

SCENARIO_NAMES = ("corridor_box", "crossing_ped", "patrol", "crowd", "hidden_wall")
_LANE_Y = 13.25


@dataclass
class DynamicScenario:
    name: str
    scenario: Scenario                      # the KNOWN static scenario (start, goal, map)
    hidden: np.ndarray                      # unmapped static occupancy (bool, same shape as the map)
    agent_factory: Callable[[], list[DiscAgent]]
    description: str = ""
    v_pref: float = 4.0                     # cruise speed used by the planners and the baseline
    max_time: float = 45.0

    def build_world(self, vehicle: VehicleConfig | None = None) -> DynamicWorld:
        return DynamicWorld(self.scenario.grid, self.hidden, self.agent_factory(), vehicle or VehicleConfig())


def make_dynamic_scenario(name: str, seed: int = 0) -> DynamicScenario:
    if name not in SCENARIO_NAMES:
        raise ValueError(f"scenario must be one of {SCENARIO_NAMES}")
    rng = np.random.default_rng(seed)
    known, hidden = _empty_grid(), _empty_grid()
    agents: Callable[[], list[DiscAgent]] = lambda: []
    if name == "corridor_box":
        _fill_rect(known, 6.0, 0.0, 36.0, 7.5)
        _fill_rect(known, 6.0, 19.0, 36.0, _HEIGHT_M)
        cy = float(np.round((_LANE_Y + rng.uniform(-0.6, 0.6)) * 2) / 2)
        _fill_rect(hidden, 20.0, cy - 1.0, 22.0, cy + 1.0)
        text = "Straight corridor; a 2 m box on the centre line is missing from the map."
    elif name == "crossing_ped":
        y0 = _LANE_Y + 3.6 + rng.uniform(-0.3, 0.3)
        speed = 1.3 + rng.uniform(-0.1, 0.1)
        trigger = 11.0 + rng.uniform(-0.5, 0.5)
        x = 21.0 + rng.uniform(-1.0, 1.0)
        agents = lambda: [CrossingPedestrian(pos=[x, y0], radius=0.35, end=[x, _LANE_Y - 6.0], speed=speed, trigger_distance=trigger)]
        text = "A pedestrian steps into the lane ~11 m ahead of the car, timed to meet it at cruise speed."
    elif name == "patrol":
        ph = rng.uniform(0.0, 1.0, 2)
        agents = lambda: [
            PatrolAgent(pos=[14.0, 4.0], radius=0.4, waypoints=[[14.0, 4.0], [14.0, 22.0]], speed=1.2, phase=float(ph[0])),
            PatrolAgent(pos=[27.0, 22.0], radius=0.4, waypoints=[[27.0, 22.0], [27.0, 4.0]], speed=1.0, phase=float(ph[1])),
        ]
        text = "Two pedestrians patrol back and forth across the route."
    elif name == "crowd":
        n = 4
        pos = np.column_stack((rng.uniform(11.0, 33.0, n), rng.uniform(3.0, 23.0, n)))
        ang = rng.uniform(0.0, 2.0 * np.pi, n)
        spd = rng.uniform(0.6, 1.3, n)
        rad = rng.uniform(0.3, 0.5, n)
        vel = np.column_stack((np.cos(ang), np.sin(ang))) * spd[:, None]
        agents = lambda: [ConstantVelocityAgent(pos=pos[i], radius=float(rad[i]), vel=vel[i]) for i in range(n)]
        text = "Four discs bounce around the free space at 0.6-1.3 m/s."
    else:  # hidden_wall
        y_lo = float(np.round(rng.uniform(3.5, 5.5) * 2) / 2)
        _fill_rect(hidden, 20.0, y_lo, 22.0, y_lo + 14.0)
        text = "A 14 m wall across the route is absent from the map; the robot must discover it and replan."
    start = VehicleState(4.25, _LANE_Y, 0.0)
    scenario = Scenario(name=name, grid=GridMap(known, _RESOLUTION), start=start, goal=(37.25, _LANE_Y),
                        metadata={"description": text, "seed": seed, "width_m": _WIDTH_M})
    return DynamicScenario(name, scenario, hidden, agents, text)
