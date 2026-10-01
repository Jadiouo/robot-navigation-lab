"""Strict tuning / test split by scenario seed.

``seed in [0, 1000)``      tuning set: the only seeds a planner parameter may ever be looked at on;
``seed >= 100000``         test set: run once, after the configuration is frozen (see :mod:`navlab.benchmark.config`).

Every scenario of a split is ``(family, seed)`` with ``family`` cycling through the family list, so a
list of ``n`` scenarios is balanced across families.  The functions refuse seeds of the wrong split.
"""
from __future__ import annotations

from navlab.world.generator import FAMILIES

TUNING_RANGE = (0, 1000)
TEST_BASE = 100_000


def split_of(seed: int) -> str:
    if TUNING_RANGE[0] <= seed < TUNING_RANGE[1]:
        return "tuning"
    if seed >= TEST_BASE:
        return "test"
    raise ValueError(f"seed {seed} belongs to no split (tuning: 0-999, test: >= {TEST_BASE})")


def scenario_list(split: str, n: int, families: tuple[str, ...] = FAMILIES, start: int = 0) -> list[tuple[str, int]]:
    """``n`` ``(family, seed)`` pairs of ``split``; the i-th uses seed base + i and family ``families[i % len]``."""
    if split == "tuning":
        if start + n > TUNING_RANGE[1]:
            raise ValueError("the tuning split has only 1000 seeds")
        base = TUNING_RANGE[0]
    elif split == "test":
        base = TEST_BASE
    else:
        raise ValueError("split must be 'tuning' or 'test'")
    return [(families[i % len(families)], base + i) for i in range(start, start + n)]


def assert_split(pairs, split: str) -> None:
    for _, seed in pairs:
        if split_of(seed) != split:
            raise ValueError(f"seed {seed} is not in the {split} split")
