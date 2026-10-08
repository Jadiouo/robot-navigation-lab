"""Aggregate diagnostic CSVs (navlab.v3.diag) into per-(cond, planner, pose) rates.  ``python -m navlab.v3.summarize a.csv [b.csv]``"""
from __future__ import annotations

import sys

import pandas as pd


def table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["v3_success"] = (df.v3_outcome == "success").astype(int)
    df["false_arrival"] = (df.v3_outcome == "false_arrival").astype(int)
    df["lucky"] = (df.v3_outcome == "lucky_success").astype(int)
    df["est_only_success"] = (df.v3_est_only == "success").astype(int)
    df["wall_stop"] = ((df.success == 1) & (df.wall_dist_end < 8.0)).astype(int)
    g = df.groupby(["cond", "planner", "pose"])
    t = g.agg(n=("seed", "count"), success=("success", "mean"), false_success=("false_success", "sum"), v3_success=("v3_success", "mean"), est_only=("est_only_success", "mean"),
              false_arrival=("false_arrival", "mean"), lucky=("lucky", "sum"), loc_last10_med=("loc_last10", "median"),
              end_goal=("true_goal_end", "mean"), wall_dist_end=("wall_dist_end", "median"), wall_dist_goal=("wall_dist_goal", "mean")).reset_index()
    ns = df[df.success == 1].groupby(["cond", "planner", "pose"]).size().rename("n_succ").reset_index()
    t = t.merge(ns, how="left").fillna({"n_succ": 0})
    t["false_succ_rate_of_success"] = (t.false_success / t.n_succ.where(t.n_succ > 0)).round(3)
    t["false_succ_rate_of_all"] = (t.false_success / t.n).round(3)
    return t.round(3)


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 30)
    for p in sys.argv[1:]:
        print(p)
        print(table(pd.read_csv(p)).to_string(index=False))
