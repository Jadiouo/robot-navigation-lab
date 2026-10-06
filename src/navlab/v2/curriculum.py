"""v2 training curriculum: the v1 recipe, one notch harder, plus the hall family.

* ``sample_condition2``: 30 % exactly nominal; otherwise every stress axis is active with p = 0.4 and drawn uniformly between its nominal
  value and ``u * top`` where ``top`` is the **level 3.5** value, i.e. the midpoint of the benchmark's level-3 and level-4 values
  (v1 stopped at level 3).  The benchmark's level 4 itself is never trained on.  ``navlab.benchmark.conditions`` is only read.
* ``sample_family``: ``hall`` with probability ``HALL_FRAC`` (0.18, inside the required 15-20 %), the other three uniformly.
* ``assert_training_pairs``: the training guard -- seeds in ``[1000, 100000)`` only, never ``warehouse`` (and, via
  :func:`navlab.v2.splits.assert_not_warehouse_for_training`, never a v2-test seed).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from navlab.benchmark.conditions import AXES, NOMINAL, Condition
from navlab.rl.splits import TRAIN_RANGE, is_train_seed
from navlab.v2.splits import assert_not_warehouse_for_training
from navlab.world.generator import FAMILIES

LEVEL_TOP = 3.5                 # curriculum ceiling, in benchmark axis levels
HALL_FRAC = 0.18
P_NOMINAL, P_AXIS = 0.3, 0.4
POSE_GT_PROB = 0.4
TRAIN_FAMILIES = FAMILIES + ("hall",)
VAL_RANGE = (0, 1000)           # tuning split: validation / selection only


def level_value(levels: tuple[float, ...], level: float = LEVEL_TOP) -> float:
    """Linear interpolation of an axis value at a (fractional) level."""
    lo = int(np.floor(level))
    hi = min(lo + 1, len(levels) - 1)
    return float(levels[lo] + (level - lo) * (levels[hi] - levels[lo]))


def sample_condition2(rng: np.random.Generator, u: float) -> Condition:
    if u <= 0.0 or rng.random() < P_NOMINAL:
        return NOMINAL
    kw = {}
    for _, (fname, levels) in AXES.items():
        if rng.random() < P_AXIS:
            nominal, top = levels[0], level_value(levels)
            kw[fname] = float(nominal + rng.uniform(0.0, u) * (top - nominal))
    return replace(NOMINAL, name="train2", **kw)


def sample_family(rng: np.random.Generator, hall_frac: float = HALL_FRAC) -> str:
    if rng.random() < hall_frac:
        return "hall"
    return FAMILIES[int(rng.integers(len(FAMILIES)))]


def assert_training_pairs(pairs) -> None:
    pairs = list(pairs)
    assert_not_warehouse_for_training(pairs)
    for fam, seed in pairs:
        if fam not in TRAIN_FAMILIES:
            raise ValueError(f"family {fam!r} is not a training family {TRAIN_FAMILIES}")
        if not is_train_seed(int(seed)):
            raise ValueError(f"seed {seed} is not a v2 training seed (allowed {TRAIN_RANGE[0]}..{TRAIN_RANGE[1] - 1})")


def assert_validation_pairs(pairs) -> None:
    """Validation / pilot-validation scenarios: tuning seeds only (< 1000), no warehouse, nothing >= 100000."""
    pairs = list(pairs)
    assert_not_warehouse_for_training(pairs)
    for fam, seed in pairs:
        if fam not in TRAIN_FAMILIES:
            raise ValueError(f"family {fam!r} is not allowed in validation {TRAIN_FAMILIES}")
        if not (VAL_RANGE[0] <= int(seed) < VAL_RANGE[1]):
            raise ValueError(f"seed {seed} is not a tuning seed (validation allowed {VAL_RANGE[0]}..{VAL_RANGE[1] - 1})")
