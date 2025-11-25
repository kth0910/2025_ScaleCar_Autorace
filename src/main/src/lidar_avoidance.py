#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Optional, Tuple

import numpy as np
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker, MarkerArray


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class LidarAvoidancePlanner:
    """
    LaserScan 기반의 장애물 감지 및 단순 로컬 플래너.
    전방 FOV에서 가장 안전한 방향을 선택해 가깝게 Ackermann 및
    직접 PWM/서보 명령을 동시에 발행하고, RViz 시각화를 위해
    Marker/Path 토픽을 제공한다.
    """

    def __init__(self) -> None:
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.forward_fov = math.radians(rospy.get_param("~forward_fov_deg", 210.0))
        self.max_range = rospy.get_param("~max_range", 8.0)
        self.safe_distance = rospy.get_param("~safe_distance", 1.2)
        self.hard_stop_distance = rospy.get_param("~hard_stop_distance", 0.35)
        self.inflation_margin = rospy.get_param("~inflation_margin", 0.15)
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 2.5)
        self.obstacle_threshold = rospy.get_param("~obstacle_threshold", 1.8)
        self.heading_weight = rospy.get_param("~heading_weight", 0.35)
        self.clearance_weight = rospy.get_param("~clearance_weight", 0.65)
        self.smoothing_window = max(1, int(rospy.get_param("~smoothing_window", 7)))
        self.path_points = max(2, int(rospy.get_param("~path_points", 10)))

        # Speed / steering output 설정
        self.publish_ackermann = rospy.get_param("~publish_ackermann", True)
        self.publish_direct_controls = rospy.get_param("~publish_direct_controls", True)
        self.ackermann_topic = rospy.get_param("~ackermann_topic", "/ackermann_cmd")
        self.max_drive_speed = rospy.get_param("~max_drive_speed", 1.5)  # 회피주행: 안전 속도로 제한
        self.min_drive_speed = rospy.get_param("~min_drive_speed", 0.4)
        self.max_pwm = rospy.get_param("~max_pwm", 1500.0)  # 기존 2000에서 안전 속도로 감소
        self.min_pwm = rospy.get_param("~min_pwm", 900.0)   # 기존 1100에서 감소
        self.servo_center = rospy.get_param("~servo_center", 0.53)
        self.servo_per_rad = rospy.get_param("~servo_per_rad", 0.28)
        self.min_servo = rospy.get_param("~min_servo", 0.05)
        self.max_servo = rospy.get_param("~max_servo", 0.95)
        self.max_steering_angle = math.radians(
            rospy.get_param("~max_steering_angle_deg", 30.0)
        )

        # ROS I/O
        rospy.Subscriber(
            self.scan_topic,
            LaserScan,
            self.scan_callback,
            queue_size=1,
        )
        self.speed_pub = rospy.Publisher(
            "/commands/motor/speed", Float64, queue_size=1
        ) if self.publish_direct_controls else None
        self.steering_pub = rospy.Publisher(
            "/commands/servo/position", Float64, queue_size=1
        ) if self.publish_direct_controls else None
        self.ackermann_pub = rospy.Publisher(
            self.ackermann_topic, AckermannDriveStamped, queue_size=1
        ) if self.publish_ackermann else None
        self.marker_pub = rospy.Publisher(
            "lidar_avoidance/obstacles", MarkerArray, queue_size=1
        )
        self.target_marker_pub = rospy.Publisher(
            "lidar_avoidance/target", Marker, queue_size=1
        )
        self.path_pub = rospy.Publisher("lidar_avoidance/path", Path, queue_size=1)
        self.clearance_pub = rospy.Publisher(
            "lidar_avoidance/closest_obstacle", Float64, queue_size=1
        )

        self.last_scan_time = rospy.get_time()
        self.scan_timeout = rospy.get_param("~scan_timeout", 1.0)  # seconds
        
        rospy.loginfo(
            "LidarAvoidancePlanner ready. Subscribing to %s (hardware only, no simulation)", 
            self.scan_topic
        )

    def scan_callback(self, scan: LaserScan) -> None:
        self.last_scan_time = rospy.get_time()
        prepared = self._prepare_scan(scan)
        if prepared is None:
            self._publish_stop(scan.header, reason="invalid_scan")
            return
        ranges, angles = prepared
        if not ranges.size:
            self._publish_stop(scan.header, reason="empty_scan")
            return

        closest = float(np.min(ranges))
        self.clearance_pub.publish(closest)

        obstacle_points = self._collect_obstacle_points(ranges, angles)
        self._publish_obstacle_markers(scan.header, obstacle_points)

        (
            target_angle,
            target_distance,
            selected_score,
        ) = self._select_target(ranges, angles)

        emergency_stop = closest < self.hard_stop_distance
        if target_angle is None:
            rospy.logwarn_throttle(1.0, "No feasible gap. Stopping vehicle.")
            self._publish_stop(scan.header, reason="no_gap")
            return

        drive_speed, pwm = self._compute_speed_profile(
            target_distance, selected_score, emergency_stop
        )
        steering_angle = clamp(target_angle, -self.max_steering_angle, self.max_steering_angle)
        servo_cmd = self.servo_center + self.servo_per_rad * steering_angle
        servo_cmd = clamp(servo_cmd, self.min_servo, self.max_servo)

        self._publish_path(scan.header, steering_angle, target_distance)
        self._publish_target_marker(scan.header, steering_angle, target_distance)
        self._publish_motion_commands(
            scan.header,
            steering_angle,
            drive_speed,
            pwm,
            servo_cmd,
            emergency_stop,
        )

    def _prepare_scan(self, scan: LaserScan) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        ranges = np.array(scan.ranges, dtype=np.float32)
        angles = scan.angle_min + np.arange(len(ranges), dtype=np.float32) * scan.angle_increment

        valid = np.isfinite(ranges)
        ranges = ranges[valid]
        angles = angles[valid]
        if not ranges.size:
            return None

        fov_half = self.forward_fov * 0.5
        within_fov = np.abs(angles) <= fov_half
        ranges = ranges[within_fov]
        angles = angles[within_fov]
        if not ranges.size:
            return None

        ranges = np.clip(ranges, scan.range_min, self.max_range)

        if self.smoothing_window > 1:
            kernel = np.ones(self.smoothing_window, dtype=np.float32) / float(
                self.smoothing_window
            )
            ranges = np.convolve(ranges, kernel, mode="same")
        return ranges, angles

    def _collect_obstacle_points(
        self, ranges: np.ndarray, angles: np.ndarray
    ) -> np.ndarray:
        mask = ranges < self.obstacle_threshold
        if not np.any(mask):
            return np.zeros((0, 2), dtype=np.float32)
        selected = np.stack((ranges[mask], angles[mask]), axis=1)
        xy = np.zeros_like(selected)
        xy[:, 0] = selected[:, 0] * np.cos(selected[:, 1])
        xy[:, 1] = selected[:, 0] * np.sin(selected[:, 1])
        in_front = xy[:, 0] > 0.0
        return xy[in_front]

    def _select_target(
        self, ranges: np.ndarray, angles: np.ndarray
    ) -> Tuple[Optional[float], float, float]:
        clearance = np.clip(ranges - self.inflation_margin, 0.0, self.max_range)
        safe_mask = clearance > self.safe_distance
        if not np.any(safe_mask):
            return None, 0.0, 0.0

        norm_clearance = clearance / self.max_range
        heading_pref = 1.0 - (np.abs(angles) / (self.forward_fov * 0.5))
        heading_pref = np.clip(heading_pref, 0.0, 1.0)
        scores = (
            self.clearance_weight * norm_clearance
            + self.heading_weight * heading_pref
        )
        scores[~safe_mask] = -np.inf
        idx = int(np.argmax(scores))
        if not np.isfinite(scores[idx]):
            return None, 0.0, 0.0

        target_angle = float(angles[idx])
        target_distance = float(min(ranges[idx], self.lookahead_distance))
        return target_angle, target_distance, float(scores[idx])

    def _compute_speed_profile(
        self, lookahead: float, score: float, emergency_stop: bool
    ) -> Tuple[float, float]:
        if emergency_stop:
            return 0.0, 0.0

        score = clamp(score, 0.0, 1.0)
        distance_ratio = clamp(lookahead / self.lookahead_distance, 0.0, 1.0)
        blended = 0.5 * score + 0.5 * distance_ratio
        drive_speed = (
            self.min_drive_speed
            + (self.max_drive_speed - self.min_drive_speed) * blended
        )
        pwm = self.min_pwm + (self.max_pwm - self.min_pwm) * blended
        return drive_speed, pwm

    def _publish_motion_commands(
        self,
        header,
        steering_angle: float,
        drive_speed: float,
        pwm: float,
        servo_cmd: float,
        emergency_stop: bool,
    ) -> None:
        if self.publish_ackermann and self.ackermann_pub is not None:
            ack_msg = AckermannDriveStamped()
            ack_msg.header = header
            ack_msg.drive.steering_angle = steering_angle
            ack_msg.drive.speed = 0.0 if emergency_stop else drive_speed
            self.ackermann_pub.publish(ack_msg)

        if self.publish_direct_controls and self.speed_pub and self.steering_pub:
            speed_msg = Float64()
            speed_msg.data = 0.0 if emergency_stop else pwm
            servo_msg = Float64()
            servo_msg.data = servo_cmd if not emergency_stop else self.servo_center
            self.speed_pub.publish(speed_msg)
            self.steering_pub.publish(servo_msg)

    def _publish_obstacle_markers(self, header, points: np.ndarray) -> None:
        marker = Marker()
        marker.header = header
        marker.ns = "lidar_obstacles"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.08
        marker.scale.y = 0.08
        marker.color.r = 1.0
        marker.color.g = 0.3
        marker.color.b = 0.3
        marker.color.a = 0.9
        marker.lifetime = rospy.Duration(0.2)
        marker.points = []
        for x, y in points:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            marker.points.append(p)

        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)

    def _publish_target_marker(
        self, header, steering_angle: float, distance: float
    ) -> None:
        marker = Marker()
        marker.header = header
        marker.ns = "lidar_target"
        marker.id = 1
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = max(distance, 0.1)
        marker.scale.y = 0.08
        marker.scale.z = 0.08
        marker.color.r = 0.1
        marker.color.g = 0.8
        marker.color.b = 1.0
        marker.color.a = 0.9
        marker.pose.orientation.z = math.sin(steering_angle * 0.5)
        marker.pose.orientation.w = math.cos(steering_angle * 0.5)
        marker.lifetime = rospy.Duration(0.2)
        self.target_marker_pub.publish(marker)

    def _publish_path(self, header, steering_angle: float, distance: float) -> None:
        path_msg = Path()
        path_msg.header = header
        if distance <= 0.0:
            self.path_pub.publish(path_msg)
            return

        for step in range(1, self.path_points + 1):
            ratio = step / float(self.path_points)
            travel = distance * ratio
            pose = PoseStamped()
            pose.header = header
            pose.pose.position.x = travel * math.cos(steering_angle)
            pose.pose.position.y = travel * math.sin(steering_angle)
            pose.pose.orientation.z = math.sin(steering_angle * 0.5)
            pose.pose.orientation.w = math.cos(steering_angle * 0.5)
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

    def _publish_stop(self, header, reason: str) -> None:
        if self.publish_ackermann and self.ackermann_pub is not None:
            msg = AckermannDriveStamped()
            msg.header = header
            msg.drive.speed = 0.0
            msg.drive.steering_angle = 0.0
            self.ackermann_pub.publish(msg)
        if self.publish_direct_controls and self.speed_pub and self.steering_pub:
            speed_msg = Float64()
            speed_msg.data = 0.0
            servo_msg = Float64()
            servo_msg.data = self.servo_center
            self.speed_pub.publish(speed_msg)
            self.steering_pub.publish(servo_msg)
        rospy.logdebug_throttle(2.0, "Stop command issued (%s)", reason)


def run():
    rospy.init_node("lidar_avoidance_planner")
    LidarAvoidancePlanner()
    rospy.spin()


if __name__ == "__main__":
    run()

