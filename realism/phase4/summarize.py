#!/usr/bin/env python3
"""summarize.py episodes.jsonl : per-S success, false-arrival rates (R=0.5/1/2), timings, retries, Wilson CI, n=30 time estimate."""
import json, math, sys, collections

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

rows = [json.loads(l) for l in open(sys.argv[1])]
by = collections.defaultdict(list)
for r in rows: by[r["S"]].append(r)
for S, rs in sorted(by.items()):
    n = len(rs); ok = [r for r in rs if r["nav2_result"] == "SUCCEEDED"]
    print(f"S={S}: n={n} SUCCEEDED={len(ok)} infra_fail={sum(r['infra_fail'] for r in rs)} retries={sum(r['retries'] for r in rs)}")
    for R in (0.5, 1.0, 2.0):
        far = [r for r in ok if r["d_goal_truth"] is not None and r["d_goal_truth"] > R]
        lo, hi = wilson(len(far), len(ok)); lo2, hi2 = wilson(len(far), n)
        print(f"  FAR@{R}: {len(far)}/{len(ok)} of successes ({lo:.2f}-{hi:.2f});  {len(far)}/{n} of all episodes ({lo2:.2f}-{hi2:.2f})")
    w = [r["wall_episode_s"] for r in rs]; nw = [r["nav_wall_s"] for r in rs if r["nav_wall_s"]]
    print(f"  wall/episode incl. startup+retries: mean {sum(w)/n:.0f}s  min {min(w):.0f} max {max(w):.0f};  nav wall mean {sum(nw)/max(len(nw),1):.0f}s;  rtf_nav mean {sum(r['rtf_nav'] or 0 for r in rs)/n:.2f}")
tot = sum(r["wall_episode_s"] for r in rows) / len(rows)
print(f"mean wall/episode all = {tot:.0f}s -> per cell n=30: {30*tot/3600:.2f} h ; per 1 cell-pair(2 S x30): {60*tot/3600:.2f} h")
for n in (30,):
    for p in (0.5, 0.25, 0.1, 0.0):
        k = round(p * n); lo, hi = wilson(k, n); print(f"Wilson95 n={n} p={p}: [{lo:.2f},{hi:.2f}] width {hi-lo:.2f}")
