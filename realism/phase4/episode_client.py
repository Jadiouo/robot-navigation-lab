#!/usr/bin/env python3
"""One attempt of one episode against an already-launched stack.  Writes a JSON result.

episode_client.py OUT.json SCENARIO.json SIM_TIMEOUT_S WALL_TIMEOUT_S
Robust start: waits for /clock, /scan, ground truth, odom, all three action servers, AMCL active + /amcl_pose + map->base TF,
and a few seconds of sim time, and only then sends the goal.
"""
import json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rclpy
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose, FollowPath, NavigateToPose
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import tf2_ros
from common import Monitor, spin_until, spin_for, start_bg

out, scen_f, SIM_TMO, WALL_TMO = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
scen = json.load(open(scen_f))
SX, SY, SYAW = scen["start"]; GX, GY = scen["goal"]; GYAW = scen["goal_yaw"]
res = {"stage": "init", "ok_start": False}


def dump():
    json.dump(res, open(out, "w"), indent=1)


def yaw_q(y):
    return math.sin(y / 2), math.cos(y / 2)


rclpy.init()
mon = Monitor()
start_bg(rclpy, mon)
nav = BasicNavigator()
nav.set_parameters([Parameter("use_sim_time", value=True)])
buf = tf2_ros.Buffer(); tl = tf2_ros.TransformListener(buf, mon)
t_start = time.time()
import threading
def _wd(limit, need_ready):
    def f():
        if need_ready and res.get("ok_start"): return
        res["error"] = f"watchdog_{limit}s_stage_{res.get('stage')}"; dump(); os._exit(0)
    t = threading.Timer(limit, f); t.daemon = True; t.start()
_wd(180, True)      # Phase-4 pilot: one attempt hung 18 min inside waitUntilNav2Active; start-up normally takes ~15 s
_wd(WALL_TMO + 240, False)


def tf_est():
    try:
        t = buf.lookup_transform("map", "base_footprint", rclpy.time.Time())
        return (t.transform.translation.x, t.transform.translation.y)
    except Exception:
        return None


def d(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1]) if p and q else None


try:
    res["stage"] = "wait_sensors"
    if not spin_until(rclpy, mon, lambda: mon.clock is not None and mon.scan_n > 0 and mon.truth and mon.odom, 180):
        raise RuntimeError("no clock/scan/truth/odom")
    res["stage"] = "wait_actions"
    clients = [ActionClient(mon, ComputePathToPose, "compute_path_to_pose"), ActionClient(mon, FollowPath, "follow_path"),
               ActionClient(mon, NavigateToPose, "navigate_to_pose")]
    for c in clients:
        if not c.wait_for_server(timeout_sec=120):
            raise RuntimeError("action server not ready: " + str(c._action_name))
    res["stage"] = "wait_nav2"
    ip = PoseStamped(); ip.header.frame_id = "map"; ip.header.stamp = nav.get_clock().now().to_msg()
    ip.pose.position.x, ip.pose.position.y = SX, SY
    ip.pose.orientation.z, ip.pose.orientation.w = yaw_q(SYAW)
    nav.setInitialPose(ip)
    nav.waitUntilNav2Active()
    res["stage"] = "wait_amcl"
    if not spin_until(rclpy, mon, lambda: mon.amcl is not None and tf_est() is not None, 90):
        raise RuntimeError("no amcl_pose / map->base tf")
    s0 = mon.clock
    spin_until(rclpy, mon, lambda: mon.clock - s0 >= 5.0, 120)           # let AMCL / costmaps settle
    res["wall_to_ready_s"] = time.time() - t_start
    res["ok_start"] = True
    res["truth_at_ready"] = mon.truth; res["tf_at_ready"] = tf_est(); res["amcl_at_ready"] = mon.amcl
    res["stage"] = "navigate"
    # --- send goal
    g = PoseStamped(); g.header.frame_id = "map"; g.header.stamp = nav.get_clock().now().to_msg()
    g.pose.position.x, g.pose.position.y = GX, GY
    g.pose.orientation.z, g.pose.orientation.w = yaw_q(GYAW)
    w0, c0, tr0, od0 = time.time(), mon.clock, mon.truth, mon.odom
    nav.goToPose(g)
    trace = []; timed_out = None; last_tr = 0.0; tp = tr0; tlen = 0.0
    while not nav.isTaskComplete():
        time.sleep(0.02)
        if mon.truth:
            tlen += d(tp, mon.truth); tp = mon.truth
        if time.time() - last_tr > 2.0:
            last_tr = time.time()
            trace.append([round(mon.clock, 1), mon.truth, tf_est(), mon.amcl, mon.odom, mon.cmd])
        if mon.clock - c0 > SIM_TMO: timed_out = "sim"
        elif time.time() - w0 > WALL_TMO: timed_out = "wall"
        if timed_out:
            nav.cancelTask(); spin_for(rclpy, mon, 1.0); break
    w1, c1 = time.time(), mon.clock
    r = nav.getResult()
    res["nav2_result"] = "TIMEOUT_" + timed_out if timed_out else TaskResult(r).name
    try:
        res["nav2_feedback_msg"] = str(nav.getFeedback())[:200]
    except Exception:
        pass
    res["nav_wall_s"] = w1 - w0; res["nav_sim_s"] = c1 - c0; res["rtf_nav"] = (c1 - c0) / (w1 - w0)
    res["truth_path_len_m"] = tlen
    spin_for(rclpy, mon, 0.2)
    G = (GX, GY)
    res["at_result"] = dict(truth=mon.truth, amcl_pose=mon.amcl, tf_map_base=tf_est(), odom=mon.odom,
                            d_goal_truth=d(mon.truth, G), d_goal_amcl=d(mon.amcl, G), d_goal_tf=d(tf_est(), G),
                            est_err=d(mon.truth, tf_est()))
    # settle (sim 3 s): by then Nav2 has commanded zero
    s2 = mon.clock; spin_until(rclpy, mon, lambda: mon.clock - s2 >= 3.0, 60)
    res["settled"] = dict(truth=mon.truth, tf_map_base=tf_est(), d_goal_truth=d(mon.truth, G), d_goal_tf=d(tf_est(), G),
                          est_err=d(mon.truth, tf_est()))
    res["sim_total_s"] = mon.clock; res["wall_total_s"] = time.time() - t_start
    res["rtf_total"] = mon.clock / (time.time() - t_start)
    res["trace"] = trace
    res["stage"] = "done"
except Exception as e:
    res["error"] = repr(e)
dump()
print("CLIENT", res.get("stage"), res.get("nav2_result"), res.get("error"), flush=True)
os._exit(0)
