"""Build the static showcase catalog from the frozen result archives.

This module only reads recorded archives.  It deliberately rebuilds maps solely
for occupancy-hash validation; it never invokes a planner, controller, or PPO.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from navlab.core import GridMap, VehicleConfig
from navlab.maps import make_scenario, make_tracking_scenario

_SCHEMA_VERSION = 1
_ARCHIVES = ("classical-raw-records.zip", "heldout-raw-records.zip")
_TRAJECTORY_COLUMNS = ("x", "y", "yaw", "s", "v_ref", "kappa")
_TRACE_COLUMNS = ("t", "x", "y", "yaw", "v", "delta", "v_ref", "cte", "s")
_POLICY_ORDER = (
    "pure_pursuit", "stanley", "pid", "lqr", "lqr_rate", "ppo_seed0", "ppo_seed1", "ppo_seed2",
)
_POLICY_LABELS = {
    "pure_pursuit": "純追蹤（Pure Pursuit）",
    "stanley": "Stanley",
    "pid": "PID",
    "lqr": "LQR",
    "lqr_rate": "LQR（含轉向速率狀態）",
    "ppo_seed0": "PPO（seed 0）",
    "ppo_seed1": "PPO（seed 1）",
    "ppo_seed2": "PPO（seed 2）",
}
_FROZEN_FIRST_RELEASE_VEHICLE = {
    "wheelbase": 2.5,
    "width": 1.6,
    "front_overhang": 0.8,
    "rear_overhang": 0.8,
    "dt": 0.05,
    "max_steer": 0.6,
    "max_steer_rate": 0.8,
    "max_accel": 2.0,
    "max_decel": 3.0,
    "max_speed": 8.0,
    "safety_margin": 0.15,
}


class CatalogError(ValueError):
    """Raised when a frozen source archive cannot support a trustworthy UI."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _occupancy_hash(occupancy: np.ndarray) -> str:
    return _sha256(np.asarray(occupancy, dtype=bool).tobytes())


def _require_finite(value: Any, context: str) -> None:
    """Reject JSON's permissive NaN/Infinity and non-finite nested numbers."""
    if isinstance(value, float) and not math.isfinite(value):
        raise CatalogError(f"{context}: non-finite numeric value")
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite(child, f"{context}[{index}]")


def _json_bytes(data: bytes, context: str) -> dict[str, Any]:
    try:
        decoded = data.decode("utf-8")
        value = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogError(f"{context}: invalid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CatalogError(f"{context}: JSON root must be an object")
    _require_finite(value, context)
    return value


def _archive_sidecar_hash(results_dir: Path, archive_name: str) -> str:
    sidecar = results_dir / f"{archive_name}.sha256"
    try:
        text = sidecar.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise CatalogError(f"missing SHA-256 sidecar for {archive_name}") from exc
    digest = text.split(maxsplit=1)[0] if text else ""
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise CatalogError(f"invalid SHA-256 sidecar for {archive_name}")
    return digest


def _open_verified_archive(results_dir: Path, archive_name: str) -> tuple[zipfile.ZipFile, dict[str, Any]]:
    path = results_dir / archive_name
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise CatalogError(f"missing source archive: {path}") from exc
    expected = _archive_sidecar_hash(results_dir, archive_name)
    actual = _sha256(data)
    if actual != expected:
        raise CatalogError(f"archive SHA-256 mismatch for {archive_name}: expected {expected}, got {actual}")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise CatalogError(f"invalid ZIP archive: {archive_name}") from exc

    names: set[str] = set()
    total_uncompressed = 0
    for info in archive.infolist():
        name = info.filename
        parts = Path(name).parts
        if not name or name.startswith(("/", "\\")) or "\\" in name or ".." in parts:
            archive.close()
            raise CatalogError(f"unsafe ZIP member in {archive_name}: {name!r}")
        if not info.is_dir():
            if name in names:
                archive.close()
                raise CatalogError(f"duplicate ZIP member in {archive_name}: {name}")
            names.add(name)
            total_uncompressed += info.file_size
    # The archived recordings are small.  This prevents a malformed input from
    # expanding into an unexpectedly large in-memory catalog.
    if total_uncompressed > 128 * 1024 * 1024:
        archive.close()
        raise CatalogError(f"uncompressed archive is too large: {archive_name}")
    return archive, {"name": archive_name, "sha256": actual, "bytes": len(data)}


def _read_member(archive: zipfile.ZipFile, member: str, archive_name: str) -> bytes:
    try:
        return archive.read(member)
    except KeyError as exc:
        raise CatalogError(f"{archive_name}: missing required member {member}") from exc


def _read_csv(data: bytes, context: str) -> tuple[str, list[str], list[dict[str, str]]]:
    try:
        raw = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CatalogError(f"{context}: CSV is not UTF-8") from exc
    try:
        reader = csv.DictReader(io.StringIO(raw))
        if not reader.fieldnames or any(name is None or not name.strip() for name in reader.fieldnames):
            raise CatalogError(f"{context}: CSV has no usable header")
        headers = [str(name) for name in reader.fieldnames]
        rows = list(reader)
    except csv.Error as exc:
        raise CatalogError(f"{context}: malformed CSV") from exc
    if any(row is None or None in row for row in rows):
        raise CatalogError(f"{context}: malformed CSV row")
    for row_number, row in enumerate(rows, start=2):
        for name, value in row.items():
            if value in ("", None):
                continue
            try:
                numeric = float(value)
            except ValueError:
                continue
            if not math.isfinite(numeric):
                raise CatalogError(f"{context}: non-finite {name!r} at row {row_number}")
    return raw, headers, rows


def _float_cell(row: dict[str, str], name: str, context: str, *, allow_empty: bool = False) -> float | None:
    value = row.get(name)
    if value in (None, ""):
        if allow_empty:
            return None
        raise CatalogError(f"{context}: missing numeric column {name!r}")
    try:
        number = float(value)
    except ValueError as exc:
        raise CatalogError(f"{context}: invalid numeric {name!r}") from exc
    if not math.isfinite(number):
        raise CatalogError(f"{context}: non-finite numeric {name!r}")
    return number


def _trajectory_payload(rows: list[dict[str, str]], headers: list[str], context: str) -> dict[str, Any]:
    # The runner writes the neutral ``step`` header when a no-path run has no
    # trajectory rows.  Normalize that recorded absence for the UI instead of
    # pretending a reference line exists.
    if not rows and headers == ["step"]:
        return {"columns": list(_TRAJECTORY_COLUMNS), "rows": []}
    for name in _TRAJECTORY_COLUMNS:
        if name not in headers or any(name not in row for row in rows):
            raise CatalogError(f"{context}: missing trajectory column {name!r}")
    payload = [[_float_cell(row, name, context) for name in _TRAJECTORY_COLUMNS] for row in rows]
    if payload and len(payload) < 2:
        raise CatalogError(f"{context}: trajectory must contain at least two samples")
    return {"columns": list(_TRAJECTORY_COLUMNS), "rows": payload}


def _trace_payload(rows: list[dict[str, str]], headers: list[str], context: str) -> dict[str, Any]:
    if not rows and headers == ["step"]:
        return {"columns": list(_TRACE_COLUMNS), "rows": []}
    required = ("step", "dt", "x", "y", "yaw", "v", "delta", "v_ref", "s")
    for name in required:
        if name not in headers or any(name not in row for row in rows):
            raise CatalogError(f"{context}: missing trace column {name!r}")
    output: list[list[float | None]] = []
    for index, row in enumerate(rows):
        step = _float_cell(row, "step", context)
        dt = _float_cell(row, "dt", context)
        if step is None or dt is None or step != int(step) or int(step) != index or dt <= 0.0:
            raise CatalogError(f"{context}: trace must have sequential post-step samples with positive dt")
        values: list[float | None] = [float((int(step) + 1) * dt)]
        values.extend(_float_cell(row, name, context) for name in _TRACE_COLUMNS[1:-2])
        values.append(_float_cell(row, "cte", context, allow_empty=True))
        values.append(_float_cell(row, "s", context))
        output.append(values)
    return {"columns": list(_TRACE_COLUMNS), "rows": output}


def _reconstruct_map(manifest: dict[str, Any], context: str) -> Any:
    scenario_data = manifest.get("scenario")
    if not isinstance(scenario_data, dict) or not isinstance(scenario_data.get("grid"), dict):
        raise CatalogError(f"{context}: manifest has no scenario grid")
    name = scenario_data.get("name")
    metadata = scenario_data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise CatalogError(f"{context}: scenario metadata must be an object")
    try:
        if name in {"open", "detour", "narrow"}:
            scenario = make_scenario(str(name), seed=int(metadata.get("seed", 0)))
        elif isinstance(name, str) and name.startswith("tracking-"):
            scenario = make_tracking_scenario(
                seed=int(metadata["geometry_seed"]),
                split=str(metadata["split"]),
                curriculum_stage=str(metadata.get("curriculum_stage", "standard")),
            )
        else:
            raise CatalogError(f"{context}: unsupported archived scenario {name!r}")
    except (KeyError, TypeError, ValueError) as exc:
        raise CatalogError(f"{context}: cannot reconstruct archived map") from exc

    grid = scenario_data["grid"]
    occupancy = np.asarray(scenario.grid.occupancy, dtype=bool)
    expected_shape = list(occupancy.shape)
    if grid.get("shape_yx") != expected_shape or grid.get("resolution_m") != scenario.grid.resolution or grid.get("origin_xy_m") != list(scenario.grid.origin):
        raise CatalogError(f"{context}: reconstructed map geometry does not match manifest")
    expected_hash = grid.get("occupancy_sha256")
    actual_hash = _occupancy_hash(occupancy)
    if not isinstance(expected_hash, str) or actual_hash != expected_hash:
        raise CatalogError(f"{context}: occupancy hash mismatch (archived map and generator differ)")
    return scenario


def _map_id(manifest: dict[str, Any], scenario: Any) -> str:
    name = str(manifest["scenario"]["name"])
    if name in {"open", "detour", "narrow"}:
        return name
    return f"{name}-{_occupancy_hash(scenario.grid.occupancy)[:12]}"


def _map_payload(scenario: Any) -> dict[str, Any]:
    occupancy = np.asarray(scenario.grid.occupancy, dtype=bool)
    return {
        "origin": [float(v) for v in scenario.grid.origin],
        "resolution": float(scenario.grid.resolution),
        "shape": [int(v) for v in occupancy.shape],
        "occupied": [[int(row), int(column)] for row, column in np.argwhere(occupancy)],
        "occupancy_sha256": _occupancy_hash(occupancy),
    }


def _vehicle_payload(manifest: dict[str, Any], notes: list[str], provenance_notes: list[str]) -> dict[str, float]:
    supplied = manifest.get("vehicle")
    fields = tuple(VehicleConfig.__dataclass_fields__)
    if isinstance(supplied, dict) and all(name in supplied for name in fields):
        try:
            vehicle = VehicleConfig(**{name: float(supplied[name]) for name in fields})
        except (TypeError, ValueError) as exc:
            raise CatalogError("manifest vehicle is invalid") from exc
    elif supplied is None:
        vehicle = VehicleConfig(**_FROZEN_FIRST_RELEASE_VEHICLE)
        note = "此 held-out manifest 未保存 vehicle；使用凍結首版 VehicleConfig 預設值。"
        notes.append(note)
        if note not in provenance_notes:
            provenance_notes.append(note)
    else:
        raise CatalogError("manifest vehicle is incomplete")
    return {name: float(getattr(vehicle, name)) for name in fields}


def _policy_for_member(member: str, manifest: dict[str, Any]) -> str:
    parts = member.split("/")
    if parts[0] == "ppo-seed-" or parts[0].startswith("ppo-seed-"):
        seed = parts[0].removeprefix("ppo-seed-")
        policy = f"ppo_seed{seed}"
    elif parts[0] == "classical":
        policy = parts[1] if len(parts) > 1 else ""
    else:
        command = manifest.get("command", {})
        policy = command.get("controller") if isinstance(command, dict) else ""
    if policy not in _POLICY_LABELS:
        raise CatalogError(f"{member}: unsupported policy {policy!r}")
    return policy


def _case_group(member: str) -> str:
    group = member.split("/", 1)[0]
    if group not in {"end_to_end", "controllers", "ablation", "heldout"}:
        if group in {"classical", "ppo-seed-0", "ppo-seed-1", "ppo-seed-2"}:
            return "heldout"
        raise CatalogError(f"unsupported archive case group in {member}")
    return group


def _case_title(group: str, manifest: dict[str, Any], policy_label: str, route_seed: int) -> str:
    command = manifest.get("command") if isinstance(manifest.get("command"), dict) else {}
    if group == "heldout":
        return f"測試路線 {route_seed}：{policy_label}"
    scenario = str(command.get("scenario", manifest.get("scenario", {}).get("name", "")))
    if group == "end_to_end":
        return f"端到端：{scenario}／{str(command.get('planner', '')).upper()}"
    if group == "controllers":
        return f"控制器比較：{policy_label}"
    backward = "開" if command.get("backward_pass") is True else "關"
    braking = "開" if command.get("lookahead_braking") is True else "關"
    return f"消融：後向速度規劃{backward}・前瞻煞車{braking}"


def _case_id(member: str) -> str:
    stem = member.removesuffix("/run.json")
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")


def _metric_summary(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    if len(manifests) != 20:
        raise CatalogError(f"held-out policy has {len(manifests)} episodes; expected 20")
    metrics = [item["metrics"] for item in manifests]
    if any(not isinstance(metric, dict) for metric in metrics):
        raise CatalogError("held-out manifest is missing metrics")
    successes = [metric for metric in metrics if metric.get("success") is True]

    def mean(field: str, source: list[dict[str, Any]]) -> float | None:
        values = [float(metric[field]) for metric in source if metric.get(field) is not None]
        return float(np.mean(values)) if values else None

    return {
        "episodes": len(metrics),
        "successes": len(successes),
        "success_rate": len(successes) / len(metrics),
        "collision_rate": sum(metric.get("collision") is True for metric in metrics) / len(metrics),
        "timeout_rate": sum(str(metric.get("termination_reason", "")).lower() == "timeout" for metric in metrics) / len(metrics),
        "quality_all_rollouts": {field: mean(field, metrics) for field in (
            "mean_abs_cte_m", "max_abs_cte_m", "speed_rmse_mps", "max_overspeed_mps",
            "max_abs_kinematic_lateral_accel_mps2", "p99_abs_kinematic_lateral_accel_mps2", "final_progress_m",
        )},
        "quality_success_only": {field: mean(field, successes) for field in (
            "mean_abs_cte_m", "max_abs_cte_m", "speed_rmse_mps", "max_overspeed_mps",
            "max_abs_kinematic_lateral_accel_mps2", "p99_abs_kinematic_lateral_accel_mps2", "final_progress_m",
        )},
        "mean_time_to_goal_s_success_only": mean("time_to_goal_s", successes),
    }


def _assert_same_summary(actual: Any, expected: Any, context: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise CatalogError(f"{context}: summary schema differs from archived summary")
        for key in expected:
            _assert_same_summary(actual[key], expected[key], f"{context}.{key}")
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-10):
            raise CatalogError(f"{context}: aggregate does not match archived summary")
    elif actual != expected:
        raise CatalogError(f"{context}: aggregate does not match archived summary")


def _validate_heldout_summary(results_dir: Path, by_policy: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    path = results_dir / "ppo-heldout-reevaluation.json"
    try:
        summary_source = _json_bytes(path.read_bytes(), path.name)
    except OSError as exc:
        raise CatalogError("missing ppo-heldout-reevaluation.json") from exc
    expected_seeds = list(range(20))
    if summary_source.get("route_seeds") != expected_seeds:
        raise CatalogError("held-out summary must retain route seeds 0 through 19")

    policies: list[dict[str, Any]] = []
    for policy in _POLICY_ORDER:
        manifests = by_policy.get(policy, [])
        seeds = sorted(int(item["scenario"]["metadata"]["geometry_seed"]) for item in manifests)
        if seeds != expected_seeds:
            raise CatalogError(f"{policy}: held-out archive does not contain exactly route seeds 0 through 19")
        actual = _metric_summary(manifests)
        if policy.startswith("ppo_seed"):
            expected = summary_source.get("ppo", {}).get(policy.removeprefix("ppo_seed"), {}).get("summary")
        else:
            expected = summary_source.get("classical", {}).get(policy, {}).get("summary")
        if not isinstance(expected, dict):
            raise CatalogError(f"{policy}: missing archived held-out summary")
        _assert_same_summary(actual, expected, policy)
        policies.append({"id": policy, "label": _POLICY_LABELS[policy], "summary": actual})
    if set(by_policy) != set(_POLICY_ORDER):
        raise CatalogError("held-out archive policy set differs from the frozen eight-policy comparison")
    return policies


def build_catalog(results_dir: Path | str) -> dict[str, Any]:
    """Return a fully verified, JSON-serializable catalog from frozen archives."""
    results = Path(results_dir)
    if not results.is_dir():
        raise CatalogError(f"results directory does not exist: {results}")

    provenance_archives: list[dict[str, Any]] = []
    provenance_notes = [
        "Catalog data are copied from frozen source archives; maps are rebuilt only to verify occupancy hashes.",
        "Held-out summaries are re-aggregated from all 160 archived manifests and checked against ppo-heldout-reevaluation.json.",
    ]
    maps: dict[str, dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []
    heldout_manifests: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for archive_name in _ARCHIVES:
        archive, archive_record = _open_verified_archive(results, archive_name)
        provenance_archives.append(archive_record)
        try:
            run_members = sorted(name for name in archive.namelist() if name.endswith("/run.json"))
            expected_count = 15 if archive_name.startswith("classical") else 160
            if len(run_members) != expected_count:
                raise CatalogError(f"{archive_name}: expected {expected_count} run manifests, found {len(run_members)}")
            for member in run_members:
                prefix = member.removesuffix("run.json")
                run_bytes = _read_member(archive, member, archive_name)
                trace_bytes = _read_member(archive, f"{prefix}trace.csv", archive_name)
                trajectory_bytes = _read_member(archive, f"{prefix}trajectory.csv", archive_name)
                manifest = _json_bytes(run_bytes, f"{archive_name}:{member}")
                if manifest.get("schema_version") != 1:
                    raise CatalogError(f"{archive_name}:{member}: unsupported run schema")
                trace_raw, trace_headers, trace_rows = _read_csv(trace_bytes, f"{archive_name}:{prefix}trace.csv")
                trajectory_raw, trajectory_headers, trajectory_rows = _read_csv(trajectory_bytes, f"{archive_name}:{prefix}trajectory.csv")
                trajectory = _trajectory_payload(trajectory_rows, trajectory_headers, f"{archive_name}:{prefix}trajectory.csv")
                trace = _trace_payload(trace_rows, trace_headers, f"{archive_name}:{prefix}trace.csv")
                if not trace["rows"] and manifest.get("metrics", {}).get("success") is True:
                    raise CatalogError(f"{archive_name}:{member}: successful run cannot have an empty trace")

                scenario = _reconstruct_map(manifest, f"{archive_name}:{member}")
                map_id = _map_id(manifest, scenario)
                payload = _map_payload(scenario)
                existing = maps.setdefault(map_id, payload)
                if existing != payload:
                    raise CatalogError(f"{archive_name}:{member}: map id collision with different occupancy")

                group = _case_group(member)
                policy_id = _policy_for_member(member, manifest)
                metadata = manifest["scenario"].get("metadata", {})
                if not isinstance(metadata, dict):
                    raise CatalogError(f"{archive_name}:{member}: scenario metadata must be an object")
                route_seed = int(metadata.get("geometry_seed", metadata.get("seed", 0)))
                notes: list[str] = []
                vehicle = _vehicle_payload(manifest, notes, provenance_notes)
                case = {
                    "id": _case_id(member),
                    "title": _case_title(group, manifest, _POLICY_LABELS[policy_id], route_seed),
                    "group": group,
                    "policy_id": policy_id,
                    "policy_label": _POLICY_LABELS[policy_id],
                    "route_seed": route_seed,
                    "map_id": map_id,
                    "start": {name: float(manifest["scenario"]["start"].get(name, 0.0)) for name in ("x", "y", "yaw", "v", "delta")},
                    "goal": [float(value) for value in manifest["scenario"]["goal"]],
                    "vehicle": vehicle,
                    "metrics": manifest.get("metrics", {}),
                    "manifest": manifest,
                    "trajectory": trajectory,
                    "trace": trace,
                    "raw": {"run_json": run_bytes.decode("utf-8"), "trace_csv": trace_raw, "trajectory_csv": trajectory_raw},
                    "source": {
                        "archive": archive_name,
                        "member": member,
                        "sha256": {"run_json": _sha256(run_bytes), "trace_csv": _sha256(trace_bytes), "trajectory_csv": _sha256(trajectory_bytes)},
                    },
                    "notes": notes,
                }
                if group == "heldout":
                    heldout_manifests[policy_id].append(manifest)
                cases.append(case)
        finally:
            archive.close()

    case_ids = [case["id"] for case in cases]
    if len(cases) != 175 or len(case_ids) != len(set(case_ids)) or any(not value for value in case_ids):
        raise CatalogError("catalog must contain 175 cases with unique, non-empty ids")
    policies = _validate_heldout_summary(results, dict(heldout_manifests))
    return {
        "schema_version": _SCHEMA_VERSION,
        "title": "Robot Navigation Lab",
        "provenance": {"source_archives": provenance_archives, "notes": provenance_notes},
        "maps": maps,
        "comparisons": {"heldout": {"route_seeds": list(range(20)), "policies": policies}},
        "cases": cases,
    }
