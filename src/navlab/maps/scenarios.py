"""Small deterministic maps designed for end-to-end, not arbitrary-world, claims."""
from __future__ import annotations

import numpy as np

from navlab.core import GridMap, Scenario, VehicleState


_RESOLUTION = 0.5
_WIDTH_M, _HEIGHT_M = 42.0, 26.0


def _empty_grid() -> np.ndarray:
    return np.zeros((round(_HEIGHT_M / _RESOLUTION), round(_WIDTH_M / _RESOLUTION)), dtype=bool)


def _fill_rect(occupancy: np.ndarray, x0: float, y0: float, x1: float, y1: float, value: bool = True) -> None:
    c0, c1 = int(round(x0 / _RESOLUTION)), int(round(x1 / _RESOLUTION))
    r0, r1 = int(round(y0 / _RESOLUTION)), int(round(y1 / _RESOLUTION))
    occupancy[r0:r1, c0:c1] = value


def make_scenario(name: str, seed: int = 0) -> Scenario:
    """Return a deterministic static scenario; seed is retained for the public API."""
    occupancy = _empty_grid()
    # Cell centres avoid an artificial first half-cell diagonal caused by
    # snapping a rear-axle pose to the planning lattice.
    start = VehicleState(4.25, 13.25, 0.0)
    goal = (37.25, 13.25)
    if name == "open":
        description = "Open space baseline."
    elif name == "detour":
        # A vertical wall forces a generous upper detour.  Its end clearance is
        # intentionally large enough for the vehicle's conservative planning disc.
        _fill_rect(occupancy, 18.0, 0.0, 24.0, 18.0)
        description = "Static known map with one obstacle that requires an upper detour."
    elif name == "narrow":
        # The only through passage is 1.25 m, below width + both safety margins.
        _fill_rect(occupancy, 19.0, 0.0, 23.0, _HEIGHT_M)
        _fill_rect(occupancy, 19.0, 12.5, 23.0, 13.5, value=False)
        description = "The sole passage is intentionally narrower than the vehicle footprint."
    else:
        raise ValueError("scenario must be one of: open, detour, narrow")
    grid = GridMap(occupancy=occupancy, resolution=_RESOLUTION)
    return Scenario(name=name, grid=grid, start=start, goal=goal, metadata={"description": description, "seed": seed})


def make_tracking_scenario(seed: int = 0, split: str = "train", curriculum_stage: str = "standard") -> Scenario:
    """Generate a broad, gentle corridor suitable for deterministic PPO episodes.

    It is deliberately a tracking scenario, not a promise of arbitrary-map
    navigation.  Split offsets prevent held-out routes from duplicating train
    routes while preserving reproducibility.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if curriculum_stage not in {"standard", "easy"}:
        raise ValueError("curriculum_stage must be standard or easy")
    if curriculum_stage == "easy" and split != "train":
        raise ValueError("the easy curriculum stage is available only for the train split")
    split_offset = {"train": 0, "validation": 10_000, "test": 20_000}[split]
    rng = np.random.default_rng(seed + split_offset)
    occupancy = _empty_grid()
    if curriculum_stage == "easy":
        # A single distant wall gives a controller at least 14 m of forward
        # progress and preview before a deliberately non-zero turn.  It is not
        # an open-road shortcut: zero steering still drives into the wall.
        bottom_depth = float(rng.uniform(14.5, 15.0))
        _fill_rect(occupancy, 20.0, 0.0, 22.0, bottom_depth)
        upper_y = bottom_depth + 5.0
        reference_points = np.array([
            [4.25, 13.25], [10.0, 13.25], [16.0, upper_y], [27.0, upper_y], [37.25, 13.25],
        ])
        grid = GridMap(occupancy=occupancy, resolution=_RESOLUTION)
        return Scenario(
            name="tracking-train-easy", grid=grid, start=VehicleState(4.25, 13.25, 0.0), goal=(37.25, 13.25),
            metadata={
                "geometry_seed": seed, "split": split, "curriculum_stage": "easy",
                "reference_points": reference_points.tolist(),
                "description": "Train-only wide-radius single-detour curriculum route; steering is required.",
            },
        )
    # Two alternating walls force a broad S-route.  Their large longitudinal
    # separation leaves enough room for a 3.65 m minimum-radius bicycle turn.
    bottom_depth = float(rng.uniform(12.0, 13.5))
    top_depth = float(rng.uniform(12.0, 13.5))
    _fill_rect(occupancy, 12.0, 0.0, 14.0, bottom_depth)
    _fill_rect(occupancy, 27.0, _HEIGHT_M - top_depth, 29.0, _HEIGHT_M)
    grid = GridMap(occupancy=occupancy, resolution=_RESOLUTION)
    # The extra 5.5 m centreline clearance is intentionally larger than the
    # conservative circumscribed planning radius, leaving room for the fillet
    # to turn without clipping either wall across every public seed/split.
    upper_y = min(_HEIGHT_M - 5.5, bottom_depth + 5.5)
    lower_y = max(5.5, _HEIGHT_M - top_depth - 5.5)
    reference_points = np.array([
        [4.25, 13.25], [8.25, 13.25], [16.0, upper_y], [21.0, upper_y],
        [25.5, lower_y], [31.0, lower_y], [37.25, 13.25],
    ])
    return Scenario(
        name=f"tracking-{split}", grid=grid, start=VehicleState(4.25, 13.25, 0.0), goal=(37.25, 13.25),
        metadata={
            "geometry_seed": seed, "split": split, "curriculum_stage": "standard", "reference_points": reference_points.tolist(),
            "description": "Procedural broad-corridor tracking route with a held-out S-route geometry.",
        },
    )
