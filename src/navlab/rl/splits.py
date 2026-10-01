"""Seed ranges for the learned planner (kept out of :mod:`navlab.benchmark.splits`, whose source is hashed by the P3 freeze).

``[0, 1000)``        tuning split: also used for PPO checkpoint selection (validation) -- never for gradient updates;
``[1000, 100000)``   PPO training seeds (scenario generator seeds);
``>= 100000``        test split (never touched before the freeze).
"""
from __future__ import annotations

from navlab.benchmark.splits import TEST_BASE, TUNING_RANGE

TRAIN_RANGE = (1000, TEST_BASE)

assert TUNING_RANGE[1] <= TRAIN_RANGE[0] and TRAIN_RANGE[1] <= TEST_BASE      # disjoint: tuning < train < test


def is_train_seed(seed: int) -> bool:
    return TRAIN_RANGE[0] <= seed < TRAIN_RANGE[1]


def assert_train_seed(seed: int) -> None:
    if not is_train_seed(seed):
        raise ValueError(f"seed {seed} is not a PPO training seed (allowed: {TRAIN_RANGE[0]}..{TRAIN_RANGE[1] - 1})")
