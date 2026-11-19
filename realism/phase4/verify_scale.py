#!/usr/bin/env python3
"""Verify the scale injector: drive straight 5 m (truth) at 0.2 m/s, then turn ~90 deg, for several S.

Run under ROS env (run_verify.sh).  Writes JSON lines to OUT.  Uses the same full stack as the episodes (Nav2 idle).
"""
import json, math, os, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hall_world, make_config, stack
import rclpy
from geometry_msgs.msg import Twist
from common import Monitor, spin_until, spin_for

out, Ss = sys.argv[1], [float(s) for s in sys.argv[2].split(",")]
work = tempfile.mkdtemp(prefix="p4verify_")
scen = hall_world.build(900000, work); make_config.main(work)
res = []
for i, S in enumerate(Ss):
    tag = f"gzB-verify-{i:03d}"; part = f"gzB_{tag}"
    st = scen["start"]
    procs, env = stack.launch_stack(tag, 85, work, work, scen, S, work)   # full stock stack (same as the episodes)
    os.environ.update(ROS_DOMAIN_ID="85", ROS_LOCALHOST_ONLY="1", GZ_PARTITION=part)
    r = {"S": S}
    try:
        rclpy.init(); n = Monitor("verify")
        pub = n.create_publisher(Twist, "/cmd_vel", 10)
        ok = spin_until(rclpy, n, lambda: n.truth and n.odom and n.odom_clean_last and n.scan_n > 0, 240); spin_for(rclpy, n, 40)
        r["stack_up"] = bool(ok)
        if ok:
            spin_for(rclpy, n, 3)
            t0 = n.truth; o0 = n.odom; c0 = n.odom_clean_last; sim0 = n.clock; w0 = time.time()
            n.odom_pairs.clear()
            tw = Twist(); tw.linear.x = 0.2
            while math.hypot(n.truth[0] - t0[0], n.truth[1] - t0[1]) < 5.0 and time.time() - w0 < 400:
                pub.publish(tw); rclpy.spin_once(n, timeout_sec=0.05)
            pub.publish(Twist()); spin_for(rclpy, n, 3)
            t1, o1, c1 = n.truth, n.odom, n.odom_clean_last
            dt = math.hypot(t1[0] - t0[0], t1[1] - t0[1]); do = math.hypot(o1[0] - o0[0], o1[1] - o0[1]); dc = math.hypot(c1[0] - c0[0], c1[1] - c0[1])
            r.update(truth_disp=dt, odom_disp=do, clean_odom_disp=dc, ratio_biased_over_truth=do / dt, ratio_clean_over_truth=dc / dt,
                     ratio_biased_over_clean=do / dc, sim_s=n.clock - sim0, wall_s=time.time() - w0)
            pos = [math.hypot(b[0] - c[0], b[1] - c[1]) for b, c in n.odom_pairs]
            r["pairs_n"] = len(pos)
            if S == 1.0 and pos:
                r["S1_max_pos_diff_biased_vs_clean_m"] = max(pos)
                r["S1_final_pos_diff_m"] = math.hypot(o1[0] - c1[0], o1[1] - c1[1]) - 0.0
            # rotate ~90 deg (yaw rate untouched by S)
            ty0, oy0 = t1[2], o1[2]; tw = Twist(); tw.angular.z = 0.5; w1 = time.time()
            wrap = lambda a: math.atan2(math.sin(a), math.cos(a))
            while abs(wrap(n.truth[2] - ty0)) < math.pi / 2 and time.time() - w1 < 120:
                pub.publish(tw); rclpy.spin_once(n, timeout_sec=0.05)
            pub.publish(Twist()); spin_for(rclpy, n, 3)
            r["yaw_truth_deg"] = math.degrees(wrap(n.truth[2] - ty0)); r["yaw_odom_deg"] = math.degrees(wrap(n.odom[2] - oy0))
        n.destroy_node()
    except Exception as e:
        r["error"] = repr(e)
    finally:
        try:
            if rclpy.ok(): rclpy.shutdown()
        except Exception: pass
        stack.stop(procs, part)
    print(json.dumps(r), flush=True); res.append(r)
    open(out, "a").write(json.dumps(r) + "\n")
