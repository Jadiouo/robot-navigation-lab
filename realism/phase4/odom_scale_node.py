#!/usr/bin/env python3
"""Odometry scale-bias injector.

Subscribes to the clean Gazebo DiffDrive odometry (/odom_clean, bridged from gz topic /odom_clean), multiplies the body-frame
LINEAR velocity by S (yaw rate untouched), re-integrates (midpoint rule, sim-time stamps) and publishes
  * /odom                      nav_msgs/Odometry  (odom -> base_footprint), twist scaled by S
  * TF odom -> base_footprint
The gz DiffDrive plugin's own tf/odom topics are NOT bridged to ROS, so AMCL / Nav2 only ever see the biased odometry.
Ground truth comes from the gz OdometryPublisher plugin (/truth_odom) and is never touched.
With S == 1.0 this is a pure re-integration of the clean twist (equivalence is measured in verify_scale.py).
"""
import math, sys
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage


class OdomScale(Node):
    def __init__(self):
        super().__init__("odom_scale")
        self.S = float(self.declare_parameter("scale", 1.0).value)
        self.x = self.y = self.th = 0.0
        self.last = None
        self.pub = self.create_publisher(Odometry, "/odom", 10)
        self.tfpub = self.create_publisher(TFMessage, "/tf", 100)
        self.create_subscription(Odometry, "/odom_clean", self.cb, 50)
        self.get_logger().info(f"odom_scale S={self.S}")

    def cb(self, m: Odometry):
        t = Time.from_msg(m.header.stamp).nanoseconds * 1e-9
        v = m.twist.twist.linear.x * self.S
        w = m.twist.twist.angular.z
        if self.last is not None:
            dt = t - self.last
            if dt <= 0.0:
                return
            self.x += v * dt * math.cos(self.th + 0.5 * w * dt)
            self.y += v * dt * math.sin(self.th + 0.5 * w * dt)
            self.th += w * dt
        self.last = t
        qz, qw = math.sin(0.5 * self.th), math.cos(0.5 * self.th)
        o = Odometry()
        o.header.stamp = m.header.stamp
        o.header.frame_id = "odom"
        o.child_frame_id = "base_footprint"
        o.pose.pose.position.x, o.pose.pose.position.y = self.x, self.y
        o.pose.pose.orientation.z, o.pose.pose.orientation.w = qz, qw
        o.pose.covariance = m.pose.covariance
        o.twist.twist.linear.x = v
        o.twist.twist.angular.z = w
        o.twist.covariance = m.twist.covariance
        self.pub.publish(o)
        tf = TransformStamped()
        tf.header = o.header
        tf.child_frame_id = "base_footprint"
        tf.transform.translation.x, tf.transform.translation.y = self.x, self.y
        tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
        self.tfpub.publish(TFMessage(transforms=[tf]))


def main():
    rclpy.init(args=sys.argv)
    n = OdomScale()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
