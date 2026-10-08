#!/usr/bin/env python3
"""Robust sequential episode runner (run inside ONE gpujob, ROS env sourced by run_pilot.sh).

run_episodes.py OUTDIR PLAN  where PLAN = "S:NNN,S:NNN,..."  (S = odometry scale, NNN = episode number)
  episode id   = gzB-pilot-NNN
  scenario seed= 900000 + (NNN % 100)   (hall_v3 sampler of navlab.v3.worlds; NNN and NNN+100 share a scenario => paired S)
Each episode: fresh stack (gz + Nav2 + injector); up to MAX_RETRY=2 restarts on infrastructure failure
(startup failure, action-server-ack timeout in the BT, robot never moved).  One JSON line per episode -> OUTDIR/episodes.jsonl
"""
import json, math, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hall_world, make_config, stack

OUT, PLAN = sys.argv[1], sys.argv[2]
DOMAIN = int(os.environ.get("P4_DOMAIN", "84"))
MAX_RETRY = 2
VMAX = 0.40
os.makedirs(OUT, exist_ok=True)
cfg = f"{OUT}/cfg"; make_config.main(cfg)


def infra_reason(c, launch_log):
    if c is None: return "client_no_result"
    if c.get("error"): return "client_error:" + c["error"][:120]
    if not c.get("ok_start"): return "not_ready"
    try:
        txt = open(launch_log, errors="ignore").read()
    except Exception:
        txt = ""
    if "acknowledge goal request" in txt: return "bt_action_ack_timeout"
    if c.get("nav2_result") != "SUCCEEDED" and c.get("truth_path_len_m", 0) < 0.5: return "never_moved"
    if "process has died" in txt and c.get("nav2_result") != "SUCCEEDED": return "process_died"
    return None


def attempt(eid, S, seed, k):
    tag = f"{eid}-a{k}"; ld = f"{OUT}/logs/{tag}"; os.makedirs(ld, exist_ok=True)
    scen = hall_world.build(seed, ld)
    L = scen["route_len_m"]
    procs, env = stack.launch_stack(tag, DOMAIN, ld, cfg, scen, S, ld)
    t0 = time.time(); c = None
    try:
        sim_tmo = 3 * L / VMAX + 60
        cp = subprocess.run(["bash", "-c", f"source {stack.ROS_SETUP}; python3 {HERE}/episode_client.py {ld}/client.json {ld}/scenario.json {sim_tmo} 300"],
                            env=env, timeout=700, stdout=open(f"{ld}/client.log", "w"), stderr=subprocess.STDOUT)
        c = json.load(open(f"{ld}/client.json"))
    except Exception as e:
        c = None if not os.path.exists(f"{ld}/client.json") else json.load(open(f"{ld}/client.json"))
        if c is not None: c.setdefault("error", repr(e))
    finally:
        stack.stop(procs, f"gzB_{tag}")
    return scen, c, infra_reason(c, f"{ld}/launch.log"), time.time() - t0


for item in PLAN.split(","):
    Ss, nn = item.split(":"); S = float(Ss); nnn = int(nn)
    eid = f"gzB-pilot-{nnn:03d}"; seed = 900000 + nnn % 100
    attempts = []; t_ep = time.time()
    for k in range(MAX_RETRY + 1):
        scen, c, reason, wall = attempt(eid, S, seed, k)
        attempts.append(dict(attempt=k, infra_reason=reason, wall_s=wall, nav2_result=(c or {}).get("nav2_result"), stage=(c or {}).get("stage")))
        if reason is None: break
    ar = (c or {}).get("at_result", {}); st = (c or {}).get("settled", {})
    row = dict(id=eid, S=S, seed=seed, start=scen["start"], goal=scen["goal"], route_len_m=scen["route_len_m"],
               hall_width_m=scen["hall"]["width_m"], nav2_result=(c or {}).get("nav2_result", "NO_RESULT"),
               infra_fail=reason is not None, retries=len(attempts) - 1,
               d_goal_amcl_pose=ar.get("d_goal_amcl"), d_goal_tf_est=ar.get("d_goal_tf"), d_goal_truth=ar.get("d_goal_truth"),
               est_err_m=ar.get("est_err"), d_goal_truth_settled=st.get("d_goal_truth"), d_goal_tf_settled=st.get("d_goal_tf"),
               nav_wall_s=(c or {}).get("nav_wall_s"), nav_sim_s=(c or {}).get("nav_sim_s"), rtf_nav=(c or {}).get("rtf_nav"),
               rtf_total=(c or {}).get("rtf_total"), wall_to_ready_s=(c or {}).get("wall_to_ready_s"),
               truth_path_len_m=(c or {}).get("truth_path_len_m"), wall_episode_s=time.time() - t_ep, attempts=attempts,
               loadavg=os.getloadavg()[0])
    open(f"{OUT}/episodes.jsonl", "a").write(json.dumps(row) + "\n")
    print(json.dumps({k: row[k] for k in ("id", "S", "nav2_result", "retries", "d_goal_truth", "d_goal_tf_est", "wall_episode_s")}), flush=True)
