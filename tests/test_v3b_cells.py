"""v3b cell table and seed blocks."""
from __future__ import annotations

import pytest

from navlab.v3b import cells as C
from navlab.v3b import faults as F


def test_counts_and_blocks():
    assert [len(C.BLOCK_CELLS[b]) for b in (C.BLOCK_GONOGO, C.BLOCK_CONFIRM_A, C.BLOCK_CONFIRM_B)] == [6, 45, 15]
    assert [len(r) for r in C.SEED_BLOCKS.values()] == [120, 400, 400]
    assert C.SEED_BLOCKS[C.BLOCK_GONOGO] == range(604000, 604120) and C.SEED_BLOCKS[C.BLOCK_CONFIRM_A] == range(600000, 600400)
    assert C.SEED_BLOCKS[C.BLOCK_CONFIRM_B] == range(601000, 601400) and C.RESERVED_RERUN == (602000, 602999)
    assert len({c.cell_id for c in C.ALL_CELLS}) == 66
    assert not any(c.filt == "mclc" for c in C.ALL_CELLS)


def test_seed_blocks_disjoint_and_outside_tuning():
    rs = list(C.SEED_BLOCKS.values()) + [range(C.RESERVED_RERUN[0], C.RESERVED_RERUN[1] + 1)]
    seen = set()
    for r in rs:
        assert r.start >= 1000
        assert not (seen & set(r))
        seen |= set(r)


def test_check_seed_guards():
    c = C.BLOCK_CELLS[C.BLOCK_GONOGO][0]
    C.check_seed(c, 604000); C.check_seed(c, 604119)
    for bad in (-1, 0, 999, 1000, 600000, 604120, 602000, 602999, 10 ** 7):
        with pytest.raises(ValueError):
            C.check_seed(c, bad)
    with pytest.raises(ValueError):
        C.check_seed(c, 600000, allow_tuning=True)          # other blocks stay closed
    C.check_seed(c, 3, allow_tuning=True)                   # tuning only on explicit request
    with pytest.raises(ValueError):
        C.check_seed(c, 1000, allow_tuning=True)


def test_compensation_arms_and_names():
    assert [F.comp_name(e) for e in F.COMP_E] == ["mclc_m50", "mclc_m25", "mclc_m10", "mclc_p10", "mclc_p25", "mclc_p50"]
    assert all(F.comp_e(F.comp_name(e)) == e for e in F.COMP_E) and F.comp_name(0.0) == "mclc"
    assert abs(F.s_hat(1.10, 0.0) - 1.10) < 1e-12 and abs(F.s_hat(1.20, -0.5) - 1.10) < 1e-12 and abs(F.s_hat(1.30, 0.5) - 1.45) < 1e-12
    a = [c for c in C.BLOCK_CELLS[C.BLOCK_CONFIRM_A] if c.filt.startswith("mclc")]
    assert len(a) == 18 and {c.scale for c in a} == {1.10, 1.20, 1.30}


def test_ba_cells_and_primary_family():
    ba = [c for c in C.BLOCK_CELLS[C.BLOCK_CONFIRM_A] if c.filt == "ba"]
    assert len(ba) == 6 + 6 + 3
    assert sorted({(c.sig_s, c.mu_s) for c in ba}) == [(0.05, 1.0), (0.10, 0.95), (0.10, 1.0), (0.20, 1.0)]
    assert C.PRIMARY_SCALES == (1.10, 1.15, 1.20, 1.30) and C.PRIMARY_R == 2.0 and C.DESCRIPTIVE_SCALES == (1.05, 1.075)


def test_fault_odo_residual_factor():
    f = F.FaultOdo(1.10, comp_e=0.0)
    assert f.ratio == 1.0
    assert F.FaultOdo(1.10, comp_e=-0.5).ratio == pytest.approx(1.10 / 1.05)
