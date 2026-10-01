"""Ground-truth simulation world: hidden static obstacles and moving disc agents."""

from .dynamic import ConstantVelocityAgent, CrossingPedestrian, DiscAgent, PatrolAgent
from .scenarios import SCENARIO_NAMES, DynamicScenario, make_dynamic_scenario
from .scene import DynamicWorld, rect_disc_distance, rect_disc_overlap

__all__ = [
    "ConstantVelocityAgent", "CrossingPedestrian", "DiscAgent", "PatrolAgent",
    "SCENARIO_NAMES", "DynamicScenario", "make_dynamic_scenario",
    "DynamicWorld", "rect_disc_distance", "rect_disc_overlap",
]
