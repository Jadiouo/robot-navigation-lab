"""Shared rclpy helpers: sim clock, ground truth (gz OdometryPublisher /truth_odom), odom, TF, amcl_pose."""
import math, time
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Twist

MODEL = "turtlebot3_waffle"


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class Monitor(Node):
    def __init__(self, name="p4_mon"):
        super().__init__(name, parameter_overrides=[])
        self.set_parameters([__import__("rclpy").parameter.Parameter("use_sim_time", value=True)])
        self.clock = None
        self.scan_n = 0
        self.truth = None          # (x, y, yaw)
        self.odom = None           # (x, y, yaw)   biased /odom
        self.odom_clean = {}       # stamp(ns) -> (x, y, yaw)
        self.odom_clean_last = None
        self.odom_pairs = []       # (biased, clean) same-stamp pairs
        self.amcl = None
        self.cmd = None
        self.create_subscription(Twist, "/cmd_vel", lambda m: setattr(self, "cmd", (round(m.linear.x, 3), round(m.angular.z, 3))), 10)
        self.create_subscription(Clock, "/clock", self._on_clock, 10)
        self.create_subscription(LaserScan, "/scan", lambda m: setattr(self, "scan_n", self.scan_n + 1), 10)
        self.create_subscription(Odometry, "/truth_odom", self._truth, 50)
        self.create_subscription(Odometry, "/odom", self._odom, 50)
        self.create_subscription(Odometry, "/odom_clean", self._odom_clean, 50)
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self._amcl,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))

    def _on_clock(self, m):
        self.clock = m.clock.sec + m.clock.nanosec * 1e-9

    def _truth(self, m):
        self.truth = self._pose(m)   # gz OdometryPublisher: world-frame pose of the model, never biased

    @staticmethod
    def _pose(m):
        p = m.pose.pose
        return (p.position.x, p.position.y, yaw_of(p.orientation))

    def _odom(self, m):
        self.odom = self._pose(m)
        k = m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
        c = self.odom_clean.pop(k, None)
        if c is not None:
            self.odom_pairs.append((self.odom, c))

    def _odom_clean(self, m):
        k = m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
        self.odom_clean_last = self._pose(m)
        self.odom_clean[k] = self.odom_clean_last
        if len(self.odom_clean) > 200:
            self.odom_clean.pop(next(iter(self.odom_clean)))

    def _amcl(self, m):
        self.amcl = self._pose(m)


def start_bg(rclpy, node):
    """Spin `node` in its own executor thread so it keeps receiving while BasicNavigator blocks in the global executor."""
    import threading
    from rclpy.executors import SingleThreadedExecutor
    ex = SingleThreadedExecutor(); ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    node.bg = True
    return ex


def spin_until(rclpy, node, cond, timeout, step=0.1):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        if getattr(node, "bg", False):
            time.sleep(0.02)
        else:
            rclpy.spin_once(node, timeout_sec=step)
    return cond()


def spin_for(rclpy, node, sec):
    e = time.time() + sec
    while time.time() < e:
        if getattr(node, "bg", False):
            time.sleep(0.02)
        else:
            rclpy.spin_once(node, timeout_sec=0.05)
