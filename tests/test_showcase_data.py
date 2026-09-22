from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from navlab.showcase import dataset


_RESULTS = Path(__file__).resolve().parents[1] / "docs" / "results"


@pytest.fixture(scope="module")
def catalog() -> dict:
    return dataset.build_catalog(_RESULTS)


def test_catalog_has_all_frozen_cases_verified_maps_and_raw_sources(catalog: dict) -> None:
    assert catalog["schema_version"] == 1
    assert len(catalog["cases"]) == 175
    assert len({case["id"] for case in catalog["cases"]}) == 175
    assert [entry["name"] for entry in catalog["provenance"]["source_archives"]] == [
        "classical-raw-records.zip", "heldout-raw-records.zip",
    ]
    assert catalog["maps"]["detour"]["occupancy_sha256"] == "a39d913f3505a16c881ed941f5d7e289b78c2ade93b13205090b358b54656e31"

    case = next(item for item in catalog["cases"] if item["id"] == "end-to-end-detour-astar")
    assert case["source"]["member"] == "end_to_end/detour-astar/run.json"
    assert json.loads(case["raw"]["run_json"]) == case["manifest"]
    assert case["trace"]["columns"] == ["t", "x", "y", "yaw", "v", "delta", "v_ref", "cte", "s"]
    assert case["trace"]["rows"][0][0] == pytest.approx(0.05)
    assert case["trace"]["rows"][1][0] == pytest.approx(0.10)


def test_heldout_comparison_retains_all_twenty_routes_and_matches_freeze(catalog: dict) -> None:
    comparison = catalog["comparisons"]["heldout"]
    assert comparison["route_seeds"] == list(range(20))
    policies = {entry["id"]: entry for entry in comparison["policies"]}
    assert list(policies) == ["pure_pursuit", "stanley", "pid", "lqr", "lqr_rate", "ppo_seed0", "ppo_seed1", "ppo_seed2"]
    assert policies["ppo_seed0"]["summary"]["success_rate"] == pytest.approx(0.5)
    assert policies["ppo_seed1"]["summary"]["successes"] == 0
    assert policies["pure_pursuit"]["summary"]["successes"] == 20

    heldout = [case for case in catalog["cases"] if case["group"] == "heldout"]
    assert len(heldout) == 160
    assert all(case["notes"] for case in heldout)  # vehicle fallback is explicit per old manifest.
    assert all(case["vehicle"] == {
        "wheelbase": 2.5, "width": 1.6, "front_overhang": 0.8, "rear_overhang": 0.8,
        "dt": 0.05, "max_steer": 0.6, "max_steer_rate": 0.8, "max_accel": 2.0,
        "max_decel": 3.0, "max_speed": 8.0, "safety_margin": 0.15,
    } for case in heldout)
    for policy in policies:
        assert sorted(case["route_seed"] for case in heldout if case["policy_id"] == policy) == list(range(20))


def test_no_trace_failure_is_preserved_without_fake_playback(catalog: dict) -> None:
    narrow = [case for case in catalog["cases"] if case["map_id"] == "narrow"]
    assert len(narrow) == 2
    assert all(not case["metrics"]["success"] for case in narrow)
    assert all(case["trace"]["rows"] == [] for case in narrow)
    assert all(case["trajectory"]["rows"] == [] for case in narrow)


def test_ablation_titles_expose_both_independent_guards(catalog: dict) -> None:
    titles = {case["title"] for case in catalog["cases"] if case["group"] == "ablation"}
    assert titles == {
        "消融：後向速度規劃關・前瞻煞車關",
        "消融：後向速度規劃關・前瞻煞車開",
        "消融：後向速度規劃開・前瞻煞車關",
        "消融：後向速度規劃開・前瞻煞車開",
    }


def test_archive_integrity_and_reconstructed_map_mismatch_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _RESULTS / "classical-raw-records.zip"
    sidecar = _RESULTS / "classical-raw-records.zip.sha256"
    shutil.copy(archive, tmp_path / archive.name)
    shutil.copy(sidecar, tmp_path / sidecar.name)
    (tmp_path / archive.name).write_bytes(b"not the recorded archive")
    with pytest.raises(dataset.CatalogError, match="SHA-256 mismatch"):
        dataset._open_verified_archive(tmp_path, archive.name)

    monkeypatch.setattr(dataset, "_occupancy_hash", lambda _occupancy: "0" * 64)
    with pytest.raises(dataset.CatalogError, match="occupancy hash mismatch"):
        dataset.build_catalog(_RESULTS)
