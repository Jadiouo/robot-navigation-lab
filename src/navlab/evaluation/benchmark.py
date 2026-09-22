"""Small, explicit benchmark matrices. Every cell runs the real pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from navlab.evaluation.metrics import aggregate_runs
from navlab.evaluation.runner import prepare_navigation, run_navigation


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _flat(record: dict[str, Any], group: str, label: str) -> dict[str, Any]:
    metrics = record["metrics"]
    return {"group": group, "label": label, **record["command"], **metrics}


def run_benchmark(output: Path | str, quick: bool = False, seed: int = 0) -> dict[str, Any]:
    """Produce the three portfolio experiment groups with raw sub-runs retained."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    budget = 900 if quick else 2500
    max_steps = 600 if quick else 2000
    controllers = ["pure_pursuit", "stanley"] if quick else ["pure_pursuit", "stanley", "pid", "lqr", "lqr_rate"]
    planners = ["astar"] if quick else ["astar", "rrtstar"]
    rows: list[dict[str, Any]] = []

    # Group 1: complete navigation funnel; narrow remains in the denominator.
    for scenario in ("open", "detour", "narrow"):
        for planner in planners:
            label = f"{scenario}-{planner}"
            record = run_navigation(output / "end_to_end" / label, scenario, planner, "pure_pursuit", seed, budget, max_steps=max_steps)
            rows.append(_flat(record, "end_to_end", label))

    # Group 2: controller comparison uses one literal immutable trajectory object.
    controller_prepared = prepare_navigation("detour", "astar", seed, budget, backward_pass=True)
    for controller in controllers:
        label = f"detour-{controller}"
        record = run_navigation(
            output / "controllers" / label, "detour", "astar", controller, seed, budget,
            max_steps=max_steps, prepared=controller_prepared,
        )
        rows.append(_flat(record, "controllers", label))

    # Group 3: declared 2x2 speed/braking ablation, same controller and scene.
    for backward_pass in (False, True):
        for lookahead_braking in (False, True):
            label = f"backward-{int(backward_pass)}_braking-{int(lookahead_braking)}"
            record = run_navigation(
                output / "ablation" / label, "detour", "astar", "pure_pursuit", seed, budget,
                backward_pass=backward_pass, lookahead_braking=lookahead_braking, max_steps=max_steps,
            )
            rows.append(_flat(record, "ablation", label))

    _write_rows(output / "results.csv", rows)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "quick": quick,
        "matrix": {
            "end_to_end": {"scenarios": ["open", "detour", "narrow"], "planners": planners},
            "controllers": controllers,
            "ablation": {"backward_pass": [False, True], "lookahead_braking": [False, True]},
        },
        "groups": {group: aggregate_runs([{"metrics": row} for row in rows if row["group"] == group]) for group in {row["group"] for row in rows}},
        "rows": rows,
    }
    (output / "benchmark.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
