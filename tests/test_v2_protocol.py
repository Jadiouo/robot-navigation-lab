"""v2 protocol freeze: hash, seed disjointness, and that the P3 / PPO freezes are untouched."""
import json
from pathlib import Path

import pytest

from navlab.benchmark.config import code_digest, config_hash, load_frozen
from navlab.benchmark.ppo_freeze import load_ppo_frozen
from navlab.v2 import protocol as P
from navlab.v2 import splits as S
from navlab.v2.worlds import generate_scenario_v2, map_digest

RES = Path(__file__).resolve().parents[1] / "docs" / "results" / "benchmark"


def test_protocol_hash_valid_and_worlds_locked():
    p = P.load_protocol()
    assert p["hash"] == config_hash(p)
    assert p["reserved"]["reward_version"] is None and p["reserved"]["speed_feature_version"] is None
    assert [p["test_families"][k]["n_tests"] for k in ("G1", "G2", "G3", "G4", "G5")] == [18, 6, 24, 12, 8]
    for k in ("G1", "G2", "G3", "G4", "G5"):
        assert len(p["test_families"][k]["tests"]) == p["test_families"][k]["n_tests"]


def test_seed_ranges_disjoint():
    S.check_disjoint()
    seeds = S.v2_test_seeds()
    assert not any(0 <= s < 100_240 for s in seeds)
    assert len(S.nominal_list()) == 240 and len(S.stress_list()) == 60 and len(S.warehouse_list()) == 120
    assert S.warehouse_list()[0][1] == 300000 and S.warehouse_list()[-1][1] == 300119
    assert all(900 <= s < 1000 for _, s in S.pilot_validation_list())


def test_warehouse_training_guard():
    S.assert_not_warehouse_for_training([("field", 1500), ("rooms", 5)])
    with pytest.raises(ValueError):
        S.assert_not_warehouse_for_training([("warehouse", 10)])
    with pytest.raises(ValueError):
        S.assert_not_warehouse_for_training([("field", 300000)])
    with pytest.raises(ValueError):
        S.assert_not_warehouse_for_training([("field", 200005)])


def test_p3_freezes_untouched():
    assert code_digest() == "89ef94496adeeb0c839a15c8a91671478e076879b43f66103ec378808cd1f617"
    load_frozen(RES / "frozen_config.json")
    art, _ = load_ppo_frozen(RES / "ppo_frozen.json", RES / "frozen_config.json")
    assert art["hash"] == "f7be3e1dee2fc523b85dca401d395e0620b081237171bfa4b1278dc9578ccccf"
    assert P.load_protocol()["baseline_chain"]["code_digest_now"] == code_digest()


def test_tampered_protocol_rejected(tmp_path):
    p = json.loads(P.PROTOCOL_PATH.read_text())
    p["seed_ranges"]["v2_warehouse"] = [300000, 300200]
    f = tmp_path / "p.json"
    f.write_text(json.dumps(p))
    with pytest.raises(ValueError, match="hash"):
        P.load_protocol(f)


def test_fingerprints_reproduce_for_a_sample():
    p = P.load_protocol()["fingerprints"]
    for fam, seed in (("corridors", 200000), ("warehouse", 300000), ("warehouse", 300059), ("hall", 200010)):
        dyn = generate_scenario_v2(fam, seed)
        assert p["occupancy"][f"{fam}/{seed}"] == map_digest(dyn.scenario.grid.occupancy)


def test_protocol_revision_2_records_r1_hash():
    p = P.load_protocol()
    assert p["revision"] == 2 and "a2705ddee13dfb11261a793aa4702c7ac829a0151a94e0d2a0691b47a42d51f4" in p["revision_note"]
    assert p["hash"] != "a2705ddee13dfb11261a793aa4702c7ac829a0151a94e0d2a0691b47a42d51f4"
