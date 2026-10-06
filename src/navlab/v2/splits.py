"""v2 seed ranges (kept outside :mod:`navlab.benchmark.splits`, whose source is hashed by the P3 freeze).

``[0, 1000)``            tuning (pilot validation 900-999 is a subset, independent of 200-259 validation and 500-659 selection);
``[1000, 100000)``       PPO training;
``[100000, 100240)``     P3 test (read-only history);
``[200000, 200240)``     v2 test, old families (family = FAMILIES[i % 3]); stress axes use the first 60;
``[200000, 200060)``     v2 mclbreak (hall and rooms, each on its own family);
``[300000, 300120)``     v2 warehouse nominal; the first 60 also run warehouse/crowd.
Warehouse seeds exist only in test and QA: training / selection code must call :func:`assert_not_warehouse_for_training`.
"""
from __future__ import annotations

from dataclasses import replace

from navlab.benchmark.conditions import NOMINAL, Condition
from navlab.benchmark.splits import TEST_BASE, TUNING_RANGE
from navlab.rl.splits import TRAIN_RANGE
from navlab.world.generator import FAMILIES

V2_TEST_BASE = 200_000
WAREHOUSE_BASE = 300_000
N_NOMINAL, N_STRESS, N_MCLBREAK = 240, 60, 60
N_WAREHOUSE, N_WAREHOUSE_CROWD = 120, 60
P3_TEST_RANGE = (TEST_BASE, TEST_BASE + 240)
PILOT_VALIDATION_RANGE = (900, 1000)
OLD_VALIDATION_RANGE, OLD_SELECTION_RANGE = (200, 260), (500, 660)

WAREHOUSE_CROWD = replace(NOMINAL, name="warehouse/crowd", axis="warehouse_crowd", level=1, value=3.0, agent_density=3.0)
WAREHOUSE_NOMINAL = NOMINAL


def nominal_list() -> list[tuple[str, int]]:
    return [(FAMILIES[i % 3], V2_TEST_BASE + i) for i in range(N_NOMINAL)]


def stress_list() -> list[tuple[str, int]]:
    return nominal_list()[:N_STRESS]


def mclbreak_list(family: str) -> list[tuple[str, int]]:
    return [(family, V2_TEST_BASE + i) for i in range(N_MCLBREAK)]


def warehouse_list() -> list[tuple[str, int]]:
    return [("warehouse", WAREHOUSE_BASE + i) for i in range(N_WAREHOUSE)]


def warehouse_crowd_list() -> list[tuple[str, int]]:
    return warehouse_list()[:N_WAREHOUSE_CROWD]


def pilot_validation_list() -> list[tuple[str, int]]:
    lo, hi = PILOT_VALIDATION_RANGE
    return [(FAMILIES[i % 3], lo + i) for i in range(hi - lo)]


def v2_test_seeds() -> set[int]:
    out = {s for _, s in nominal_list()} | {s for _, s in warehouse_list()}
    for fam in ("hall", "rooms"):
        out |= {s for _, s in mclbreak_list(fam)}
    return out


def check_disjoint() -> None:
    seeds = v2_test_seeds()
    ranges = {"tuning": TUNING_RANGE, "train": TRAIN_RANGE, "p3_test": P3_TEST_RANGE}
    for name, (lo, hi) in ranges.items():
        bad = [s for s in seeds if lo <= s < hi]
        assert not bad, f"v2 test seeds overlap {name}: {bad[:5]}"
    assert min(seeds) >= V2_TEST_BASE
    wh = {s for _, s in warehouse_list()}
    assert not wh & ({s for _, s in nominal_list()} | {V2_TEST_BASE + i for i in range(N_MCLBREAK)}), "warehouse seeds overlap other v2 test seeds"
    lo, hi = PILOT_VALIDATION_RANGE
    assert TUNING_RANGE[0] <= lo and hi <= TUNING_RANGE[1]
    for a, b in (OLD_VALIDATION_RANGE, OLD_SELECTION_RANGE):
        assert hi <= a or b <= lo, "pilot validation overlaps an existing validation / selection range"


check_disjoint()


def assert_not_warehouse_for_training(pairs) -> None:
    """Guard for train / select code: refuse warehouse scenarios and every v2-test seed (``>= 200000``)."""
    for fam, seed in pairs:
        if fam == "warehouse" or int(seed) >= V2_TEST_BASE:
            raise ValueError(f"({fam}, {seed}): warehouse / v2-test scenarios are test- and QA-only, never for training or selection")
