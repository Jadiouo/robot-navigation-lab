"""Classical lateral policies and the longitudinal controller shared with PPO."""

from .longitudinal import LongitudinalController

__all__ = ["LongitudinalController", "make_controller"]


def __getattr__(name: str):
    # Keep `navlab.sim.bicycle` importable while lateral policies import it.
    if name == "make_controller":
        from .lateral import make_controller
        return make_controller
    raise AttributeError(name)
