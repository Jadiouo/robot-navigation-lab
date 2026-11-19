"""The experiment cell table -- the single source of truth for direction-B v3b (hall_v3 only).

A *cell* is (planner, localiser/pose arm, odometer scale S) inside a seed *block*.  Every episode is identified by
``(cell_id, seed)``.  Seeds only come from the three blocks below; tuning seeds 0-999 and any other range are rejected.

Arms (``filt``):  mcl | ba (``sig_s`` = scale-prior std) | gt | amcl (``alpha``) | mclc_{m,p}NN (compensation, see faults.py).
``mclc`` (e = 0) exists for the identity test only and is NOT part of any block.
"""
from __future__ import annotations

from dataclasses import dataclass

from .faults import COMP_E, comp_name

FAMILY = "hall_v3"
TUNING_SEEDS = range(0, 1000)

BLOCK_GONOGO = "BLOCK_GONOGO"
BLOCK_CONFIRM_A = "BLOCK_CONFIRM_A"
BLOCK_CONFIRM_B = "BLOCK_CONFIRM_B"
SEED_BLOCKS: dict[str, range] = {
    BLOCK_GONOGO: range(604000, 604120),     # n = 120
    BLOCK_CONFIRM_A: range(600000, 600400),  # n = 400
    BLOCK_CONFIRM_B: range(601000, 601400),  # n = 400
}

BA_SIG_S = 0.10
BA_MU_MISPLACED = 0.95
SCALES_A = (1.05, 1.075, 1.10, 1.15, 1.20, 1.30)
SCALES_BA_SIG = (1.10, 1.20, 1.30)
SCALES_B = (1.00, 1.10, 1.20)
R_LIST = (1.0, 1.5, 2.0)

# Reserved seed range for RE-RUNNING episodes lost to a program crash or an infrastructure fault ONLY.
# It must never be used to add samples; it is not a block, and the runner never draws seeds from it (check_seed rejects it).
RESERVED_RERUN = (602000, 602999)

# primary hypothesis family: MCL vs BA(0.10) at four BLOCK_CONFIRM_A scales, R = 2.0, McNemar exact two-sided, Holm over the four.
# 1.05 and 1.075 are descriptive (raw p and paired-difference CI are still reported)
PRIMARY_BLOCK = BLOCK_CONFIRM_A
PRIMARY_R = 2.0
PRIMARY_SCALES = (1.10, 1.15, 1.20, 1.30)
DESCRIPTIVE_SCALES = (1.05, 1.075)
PRIMARY_PLANNER = "dwa"
PRIMARY_ARMS = ("mcl", f"ba{BA_SIG_S:.2f}")   # (a, b): FAR difference a - b; everything else is descriptive


@dataclass(frozen=True)
class Cell:
    block: str
    planner: str        # navlab planner registry name: dwa | pp_stop
    filt: str           # mcl | ba | gt | amcl | mclc_*
    scale: float
    sig_s: float | None = None    # ba only
    alpha: float | None = None    # amcl only
    mu_s: float = 1.0             # ba only: prior centre of the odometer scale

    @property
    def arm(self) -> str:
        if self.filt == "ba":
            return f"ba{self.sig_s:.2f}" + ("" if self.mu_s == 1.0 else f"_mu{self.mu_s:g}")
        if self.filt == "amcl":
            return f"amcl{self.alpha:g}"
        return self.filt

    @property
    def cell_id(self) -> str:
        return f"{self.block}|{self.planner}|{self.arm}|S{self.scale:.3f}"

    @property
    def seeds(self) -> range:
        return SEED_BLOCKS[self.block]


def _cell(block, planner, filt, scale, **kw) -> Cell:
    return Cell(block, planner, filt, float(scale), **kw)


def _gonogo() -> list[Cell]:
    b = BLOCK_GONOGO
    return [
        _cell(b, "dwa", "mcl", 1.00),
        _cell(b, "dwa", "mcl", 1.10),
        _cell(b, "dwa", "ba", 1.10, sig_s=BA_SIG_S),
        _cell(b, "dwa", "gt", 1.10),
        _cell(b, "pp_stop", "mcl", 1.10),
        _cell(b, "dwa", "amcl", 1.10, alpha=0.2),
    ]


def _confirm_a() -> list[Cell]:
    b, out = BLOCK_CONFIRM_A, []
    for filt in ("mcl", "ba", "gt"):
        for s in SCALES_A:
            out.append(_cell(b, "dwa", filt, s, sig_s=BA_SIG_S if filt == "ba" else None))
    for sig in (0.05, 0.20):
        for s in SCALES_BA_SIG:
            out.append(_cell(b, "dwa", "ba", s, sig_s=sig))
    for s in SCALES_BA_SIG:
        out.append(_cell(b, "dwa", "ba", s, sig_s=BA_SIG_S, mu_s=BA_MU_MISPLACED))
    for e in COMP_E:
        for s in SCALES_BA_SIG:
            out.append(_cell(b, "dwa", comp_name(e), s))
    return out


def _confirm_b() -> list[Cell]:
    b, out = BLOCK_CONFIRM_B, []
    for filt in ("mcl", "ba", "gt"):
        for s in SCALES_B:
            out.append(_cell(b, "pp_stop", filt, s, sig_s=BA_SIG_S if filt == "ba" else None))
    for a in (0.05, 0.2):
        for s in SCALES_B:
            out.append(_cell(b, "dwa", "amcl", s, alpha=a))
    return out


BLOCK_CELLS: dict[str, list[Cell]] = {BLOCK_GONOGO: _gonogo(), BLOCK_CONFIRM_A: _confirm_a(), BLOCK_CONFIRM_B: _confirm_b()}
ALL_CELLS: list[Cell] = [c for cs in BLOCK_CELLS.values() for c in cs]
CELL_BY_ID: dict[str, Cell] = {c.cell_id: c for c in ALL_CELLS}


def identity_cell(scale: float, e: float = 0.0, block: str = BLOCK_CONFIRM_A) -> Cell:
    """The e = 0 'mclc' arm (identity test only; deliberately not part of any block)."""
    return Cell(block, "dwa", comp_name(e), float(scale))


def all_cells_valid_check() -> None:
    blocks = list(SEED_BLOCKS.values())
    for name, r in SEED_BLOCKS.items():
        assert r.step == 1 and len(r) > 0, name
        assert r.start >= TUNING_SEEDS.stop, f"{name} overlaps the tuning seeds 0-999"
    assert RESERVED_RERUN[0] >= TUNING_SEEDS.stop
    blocks.append(range(RESERVED_RERUN[0], RESERVED_RERUN[1] + 1))
    for i, a in enumerate(blocks):
        for b in blocks[i + 1:]:
            assert a.stop <= b.start or b.stop <= a.start, "seed blocks overlap"
    assert [len(BLOCK_CELLS[k]) for k in (BLOCK_GONOGO, BLOCK_CONFIRM_A, BLOCK_CONFIRM_B)] == [6, 45, 15]
    assert len(CELL_BY_ID) == len(ALL_CELLS), "duplicate cell_id"
    assert all(c.filt != "mclc" for c in ALL_CELLS), "mclc (e=0) is identity-test only"
    assert all(c.seeds is SEED_BLOCKS[c.block] for c in ALL_CELLS)


def check_seed(cell: Cell, seed: int, *, allow_tuning: bool = False) -> None:
    """Raise unless ``seed`` lies in the cell's own block (or, only with ``allow_tuning``, in the tuning range 0-999)."""
    if seed in SEED_BLOCKS[cell.block]:
        return
    if allow_tuning and seed in TUNING_SEEDS:
        return
    where = next((n for n, r in SEED_BLOCKS.items() if seed in r), "RESERVED_RERUN" if RESERVED_RERUN[0] <= seed <= RESERVED_RERUN[1] else "no block")
    raise ValueError(f"seed {seed} not allowed for {cell.cell_id}: it lies in {where}, the cell's block is {cell.block} {SEED_BLOCKS[cell.block]}")


all_cells_valid_check()
