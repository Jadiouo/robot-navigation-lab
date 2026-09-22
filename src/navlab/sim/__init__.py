"""Kinematic simulation and the shared path-tracking environment."""

from .bicycle import BicycleModel, wrap_angle

__all__ = ["BicycleModel", "TrackingEnv", "wrap_angle"]


def __getattr__(name: str):
    if name == "TrackingEnv":
        from .env import TrackingEnv
        return TrackingEnv
    raise AttributeError(name)
