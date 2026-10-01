"""The top-level README must not drift from the generated benchmark files."""
import csv
import json
import re
from pathlib import Path

from navlab.benchmark import readme_block

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "docs/results/benchmark"
README = (ROOT / "README.md").read_text()


def test_generated_headline_block_matches_the_csvs():
    assert readme_block.current(README) == readme_block.render(BENCH), "run: python -m navlab.benchmark.readme_block --write"


def test_hand_written_facts_match_the_artifacts():
    cfg = json.loads((BENCH / "frozen_config.json").read_text())
    assert cfg["hash"][:8] in README and "canonical JSON" in README
    n_base = sum(1 for _ in csv.DictReader((BENCH / "episodes.csv").open()))
    n_ppo = sum(1 for _ in csv.DictReader((BENCH / "episodes_ppo.csv").open()))
    assert f"{n_base:,} episodes in the baseline test set and {n_ppo:,} for the three PPO seeds" in README
    cands: dict[str, int] = {}
    for r in csv.DictReader((BENCH / "tuning_log.csv").open()):
        cands[r["planner"]] = cands.get(r["planner"], 0) + 1
    m = re.search(r"pp_stop got (\d+) candidate parameter sets, DWA and MPPI (\d+) each", README)
    assert m, "tuning-asymmetry sentence missing"
    assert (int(m.group(1)), int(m.group(2)), int(m.group(2))) == (cands["pp_stop"], cands["dwa"], cands["mppi"])
