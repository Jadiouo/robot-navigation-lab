"""Addendum-1 cell table: DWA x {mcl, ba0.05, ba0.10, ba0.20} x S = 1.00, hall_v3, seeds 603000-603399 (n = 400).

Arms are built by the same factory as in navlab.v3b.cells (``_cell``) and run by the same localiser factory
(navlab.v3b.runner._loc_class), so an arm of a given name is parameter-identical to v3b's.
"""
from __future__ import annotations

from navlab.v3b.cells import ALL_CELLS, RESERVED_RERUN, SEED_BLOCKS, TUNING_SEEDS, Cell, _cell

BLOCK_ADD1 = "BLOCK_ADD1_EXPLORATORY"
ADD1_SEEDS = range(603000, 603400)      # n = 400
ADD1_PLANNER = "dwa"
ADD1_SCALE = 1.00
ADD1_BA_SIGS = (0.05, 0.10, 0.20)


def _check_seed_range() -> None:
    assert ADD1_SEEDS.start == 603000 and ADD1_SEEDS.stop == 603400 and ADD1_SEEDS.step == 1 and len(ADD1_SEEDS) == 400
    others = [TUNING_SEEDS, *SEED_BLOCKS.values(), range(RESERVED_RERUN[0], RESERVED_RERUN[1] + 1),
              range(0, 1000), range(600000, 601400), range(602000, 603000), range(604000, 604120)]
    for o in others:
        assert ADD1_SEEDS.stop <= o.start or o.stop <= ADD1_SEEDS.start, f"addendum1 seeds overlap {o}"


_check_seed_range()


def add1_cells() -> list[Cell]:
    out = [_cell(BLOCK_ADD1, ADD1_PLANNER, "mcl", ADD1_SCALE)]
    out += [_cell(BLOCK_ADD1, ADD1_PLANNER, "ba", ADD1_SCALE, sig_s=s) for s in ADD1_BA_SIGS]
    return out


ADD1_CELLS: list[Cell] = add1_cells()
ARMS = tuple(c.arm for c in ADD1_CELLS)   # ('mcl', 'ba0.05', 'ba0.10', 'ba0.20')


def check_add1_seed(seed: int) -> None:
    if seed not in ADD1_SEEDS:
        raise ValueError(f"seed {seed} outside the addendum-1 segment {ADD1_SEEDS}")


def jobs() -> list[tuple[Cell, int]]:
    return [(c, s) for c in ADD1_CELLS for s in ADD1_SEEDS]


def v3b_counterpart(cell: Cell) -> Cell:
    """The v3b cell with the same planner/filt/sig_s/mu_s/alpha (any block, any scale); used by the tests."""
    return next(c for c in ALL_CELLS if (c.planner, c.filt, c.sig_s, c.mu_s, c.alpha) == (cell.planner, cell.filt, cell.sig_s, cell.mu_s, cell.alpha))
