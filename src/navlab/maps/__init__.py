"""Procedural maps and the shared vehicle-footprint collision geometry."""

from .collision import CollisionChecker, cell_to_world, world_to_cell
from .scenarios import make_scenario, make_tracking_scenario

__all__ = [
    "CollisionChecker",
    "cell_to_world",
    "world_to_cell",
    "make_scenario",
    "make_tracking_scenario",
]
