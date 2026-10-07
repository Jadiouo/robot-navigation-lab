"""Add the PPO v2 aggregates to docs/viewer/agg.json under the key "v2".

Reads only docs/results/benchmark_v2/*.csv (via examples/v2_findings_numbers.py); runs no episodes and leaves
data.json untouched.  Idempotent:   python examples/build_viewer_v2_agg.py
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v2n", ROOT / "examples/v2_findings_numbers.py")
v2n = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2n)


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)


def main():
    e, _, p = v2n.load()
    F = v2n.facts()
    nom = {}
    for pl in v2n.ALL:
        nom[pl] = {}
        for po in ("gt", "mcl"):
            d = e[(e.suite == "nominal") & (e.planner == pl) & (e.pose == po)]
            k, n = int((d.outcome == "success").sum()), len(d)
            lo, hi = wilson(k, n)
            nom[pl][po] = {"k": k, "n": n, "rate": round(k / n, 4), "lo": lo, "hi": hi}
    seeds = {"v1": F["seed_v1"], "v2": F["seed_v2"], "nv": F["seed_nv"]}
    ref = {pl: float(np.mean([nom[pl]["gt"]["rate"], nom[pl]["mcl"]["rate"]])) for pl in ("dwa", "pp_stop")}
    fail = {k: {kk: round(vv, 4) for kk, vv in F[f"pooled_{k}"].items()} for k in ("v1", "v2", "dwa", "nv_ok")}
    rows = lambda grp: [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}  # noqa: E731
                        for r in p[p.group == grp][["test", "success_A", "success_B", "diff", "diff_lo", "diff_hi", "p_holm", "significant"]].to_dict("records")]
    out = {
        "nominal": nom, "seed_means": {k: [round(x, 4) for x in v] for k, v in seeds.items()},
        "ref": {k: round(v, 4) for k, v in ref.items()}, "failure": fail,
        "perm_p": round(F["perm_p_v1_v2"], 3), "g1": list(F["g1"]), "g2": list(F["g2"]),
        "g1_vs_baselines": list(F["g1_vs_baselines"]),
        "branch": {k: round(v, 4) for k, v in F["branch"].items()},
        "g6": rows("G6"), "g7": rows("G7"),
        "warehouse": {pl: {po: round(F["warehouse"][pl][po], 4) for po in ("gt", "mcl")} for pl in v2n.ALL},
        "episodes": F["n_episodes"],
    }
    f = ROOT / "docs/viewer/agg.json"
    agg = json.loads(f.read_text())
    agg["v2"] = out
    f.write_text(json.dumps(agg, separators=(",", ":")))
    print("agg.json updated with v2;", f.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
