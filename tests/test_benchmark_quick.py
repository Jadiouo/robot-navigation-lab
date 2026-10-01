"""Quick benchmark smoke test: the whole pipeline (scenarios -> episodes -> CSV -> tests -> figures -> README) on a handful of tuning-split episodes."""
import csv
import json

import pytest

from navlab.benchmark.report import category, load_rows
from navlab.cli import build_parser, main


@pytest.mark.slow
def test_quick_benchmark_runs_end_to_end_and_never_touches_the_test_split(tmp_path):
    assert main(["benchmark-uncertainty", "--suite", "all", "--quick", "--workers", "4", "--output", str(tmp_path)]) == 0
    rows = load_rows(tmp_path / "episodes_quick.csv")
    assert rows and {r["planner"] for r in rows} == {"pp", "pp_stop", "dwa", "mppi"} and {r["pose"] for r in rows} == {"gt", "mcl"}
    assert {r["suite"] for r in rows} == {"nominal", "stress", "mclbreak"}
    assert all(r["seed"] < 1000 for r in rows)                                    # --quick uses the tuning split only
    for name in ("README_quick.md", "summary_quick.csv", "paired_tests_quick.csv", "fig_outcome_taxonomy_quick.png", "fig_loc_error_vs_failure_quick.png"):
        assert (tmp_path / name).exists() and (tmp_path / name).stat().st_size > 100
    text = (tmp_path / "README_quick.md").read_text()
    assert "QUICK SMOKE RUN" in text and "Wilson" in text and "McNemar" in text
    assert not (tmp_path / "frozen_config.json").exists()                          # quick never freezes
    # paired design: every (family, seed, cond) was run for all four planners and both poses where both exist
    by = {}
    for r in rows:
        by.setdefault((r["suite"], r["cond"], r["family"], r["seed"], r["pose"]), set()).add(r["planner"])
    assert all(v == {"pp", "pp_stop", "dwa", "mppi"} for v in by.values())
    cats = {category(r) for r in rows}
    assert cats <= {"success", "loc_induced", "collision_static", "collision_dynamic", "timeout", "stuck"}


def test_cli_exposes_benchmark_uncertainty():
    args = build_parser().parse_args(["benchmark-uncertainty", "--suite", "nominal", "--workers", "3", "--quick"])
    assert args.suite == "nominal" and args.workers == 3 and args.quick


def test_test_suites_refuse_to_run_without_a_frozen_config(tmp_path):
    from navlab.benchmark.suites import run_suites
    with pytest.raises(FileNotFoundError, match="frozen"):
        run_suites("nominal", tmp_path, workers=1, quick=False)
