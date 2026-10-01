"""Statistics helpers against known values; split disjointness; frozen-config guard."""
import json
import math

import numpy as np
import pytest

from navlab.benchmark import splits
from navlab.benchmark.config import default_config, freeze, load_frozen
from navlab.benchmark.stats import holm, mcnemar_exact, paired_bootstrap, paired_counts, wilson


def test_wilson_known_values():
    lo, hi = wilson(0, 10)       # textbook: [0, 0.2775]
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-4)
    lo, hi = wilson(10, 10)
    assert hi == pytest.approx(1.0) and lo == pytest.approx(0.7225, abs=1e-4)
    lo, hi = wilson(50, 100)     # [0.4038, 0.5962]
    assert (lo, hi) == pytest.approx((0.4038, 0.5962), abs=1e-4)
    assert wilson(0, 0) == (0.0, 1.0)
    lo, hi = wilson(1, 20)       # the normal interval would go below 0 here; Wilson stays inside [0, 1]
    assert 0.0 < lo < 0.05 < hi < 0.25


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 10) == pytest.approx(2 * 0.5 ** 10)          # 0.001953125
    assert mcnemar_exact(10, 0) == mcnemar_exact(0, 10)
    assert mcnemar_exact(3, 9) == pytest.approx(299 / 2048)                 # 2 * P(X <= 3 | n = 12, p = 1/2) = 2 * (1 + 12 + 66 + 220) / 4096
    assert mcnemar_exact(8, 2) == pytest.approx(0.109375)                 # n = 10: 2 * 56 / 1024


def test_paired_counts_and_bootstrap_and_holm():
    a = np.array([1, 1, 0, 0, 1, 0], bool)
    b = np.array([1, 0, 0, 1, 0, 0], bool)
    assert paired_counts(a, b) == (1, 2, 1, 2)
    mean, lo, hi = paired_bootstrap(np.array([1.0] * 20), n_boot=200)
    assert mean == lo == hi == 1.0
    mean, lo, hi = paired_bootstrap(np.array([1.0, -1.0] * 50), n_boot=2000, seed=1)
    assert lo < 0.0 < hi and mean == 0.0
    assert math.isnan(paired_bootstrap(np.array([]))[0])
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_splits_are_disjoint_and_enforced():
    tune, test = splits.scenario_list("tuning", 300), splits.scenario_list("test", 300)
    assert {s for _, s in tune}.isdisjoint({s for _, s in test})
    assert max(s for _, s in tune) < 1000 <= splits.TEST_BASE <= min(s for _, s in test)
    assert [f for f, _ in test[:6]] == ["corridors", "rooms", "field"] * 2
    assert splits.split_of(5) == "tuning" and splits.split_of(100_007) == "test"
    with pytest.raises(ValueError):
        splits.split_of(5000)                           # neither split
    with pytest.raises(ValueError):
        splits.assert_split(tune[:2], "test")
    with pytest.raises(ValueError):
        splits.scenario_list("tuning", 1001)


def test_frozen_config_detects_edits_and_missing_files(tmp_path):
    path = tmp_path / "frozen.json"
    with pytest.raises(FileNotFoundError):
        load_frozen(path)
    cfg = freeze(default_config({"pp_stop": {"standoff": 1.2}}), path)
    assert load_frozen(path)["hash"] == cfg["hash"] and len(cfg["hash"]) == 64
    edited = json.loads(path.read_text())
    edited["planners"]["pp_stop"]["standoff"] = 0.5
    path.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="hash"):
        load_frozen(path)
