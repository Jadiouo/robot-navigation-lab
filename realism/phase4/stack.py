"""Process-tree helpers for one Gazebo + Nav2 stack (own GZ_PARTITION / ROS_DOMAIN_ID)."""
import os, signal, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROS_SETUP = "/opt/ros/jazzy/setup.bash"


def env_for(tag: str, domain: int) -> dict:
    e = dict(os.environ)
    e.update(ROS_DOMAIN_ID=str(domain), ROS_LOCALHOST_ONLY="1", GZ_PARTITION=f"gzB_{tag}")
    e.pop("LIBGL_ALWAYS_SOFTWARE", None)
    return e


def kill_by_partition(part: str):
    """SIGKILL every process whose environment carries our GZ_PARTITION (precise: never touches other sessions)."""
    me = os.getpid()
    key = f"GZ_PARTITION={part}".encode()
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            if key in open(f"/proc/{d}/environ", "rb").read().split(b"\0"):
                os.kill(int(d), signal.SIGKILL)
        except Exception:
            pass


def sh(cmd: str, env: dict, log: str, new_group=True) -> subprocess.Popen:
    return subprocess.Popen(["bash", "-c", f"source {ROS_SETUP}; exec {cmd}"], env=env, stdout=open(log, "w"),
                            stderr=subprocess.STDOUT, start_new_session=new_group)


def stop(procs, part: str):
    for p in procs:
        try:
            os.killpg(p.pid, signal.SIGINT)
        except Exception:
            pass
    t0 = time.time()
    while time.time() - t0 < 8 and any(p.poll() is None for p in procs):
        time.sleep(0.5)
    for p in procs:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass
    kill_by_partition(part)


def launch_stack(tag, domain, wdir, cfgdir, scen, S, logdir, nav=True, timeout=900):
    """Start gz+bridge+(nav2)+odom injector. Returns list of Popen (each its own process group)."""
    env = env_for(tag, domain)
    st = scen["start"]
    procs = []
    base = (f"timeout {timeout} ros2 launch nav2_bringup tb3_simulation_launch.py headless:=True use_rviz:=False "
            f"world:={wdir}/hall.sdf.xacro map:={wdir}/hall.yaml params_file:={cfgdir}/nav2_params_dwb.yaml "
            f"robot_sdf:={cfgdir}/gz_waffle_biased.sdf.xacro x_pose:={st[0]} y_pose:={st[1]} z_pose:=0.01 yaw:={st[2]}")
    procs.append(sh(base, env, f"{logdir}/launch.log"))
    procs.append(sh("ros2 run ros_gz_bridge parameter_bridge /odom_clean@nav_msgs/msg/Odometry[gz.msgs.Odometry "
                    "/truth_odom@nav_msgs/msg/Odometry[gz.msgs.Odometry --ros-args -p use_sim_time:=true",
                    env, f"{logdir}/bridge2.log"))
    procs.append(sh(f"python3 {HERE}/odom_scale_node.py --ros-args -p scale:={S} -p use_sim_time:=true", env, f"{logdir}/odom_scale.log"))
    return procs, env
