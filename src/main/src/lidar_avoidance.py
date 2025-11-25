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
        self.safe_distance = rospy.get_param("~safe_distance", 0.8)
        self.hard_stop_distance = rospy.get_param("~hard_stop_distance", 0.30)  # 30cm에서 완전 정지
        self.inflation_margin = rospy.get_param("~inflation_margin", 0.20)  # 차폭 반경 20cm
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 2.5)
        self.obstacle_threshold = rospy.get_param("~obstacle_threshold", 0.60)  # 60cm부터 장애물 인식
        self.speed_reduction_start = rospy.get_param("~speed_reduction_start", 0.60)  # 60cm부터 속도 감소 시작
        self.front_obstacle_angle = math.radians(rospy.get_param("~front_obstacle_angle_deg", 30.0))  # 전방 30도 이내 장애물 감지 각도
        self.min_obstacle_points = rospy.get_param("~min_obstacle_points", 3)  # 최소 연속 포인트 수 (노이즈 필터링)
        self.obstacle_cluster_threshold = rospy.get_param("~obstacle_cluster_threshold", 0.15)  # 클러스터링 거리 임계값
        self.heading_weight = rospy.get_param("~heading_weight", 0.35)
        self.clearance_weight = rospy.get_param("~clearance_weight", 0.65)
        self.smoothing_window = max(1, int(rospy.get_param("~smoothing_window", 7)))
        self.path_points = max(2, int(rospy.get_param("~path_points", 10)))

        # Speed / steering output 설정
        self.publish_ackermann = rospy.get_param("~publish_ackermann", True)
        self.publish_direct_controls = rospy.get_param("~publish_direct_controls", True)
        self.ackermann_topic = rospy.get_param("~ackermann_topic", "/ackermann_cmd")
        self.max_drive_speed = rospy.get_param("~max_drive_speed", 0.7)
        self.min_drive_speed = rospy.get_param("~min_drive_speed", 0.3)
        self.max_pwm = rospy.get_param("~max_pwm", 1500.0)
        self.min_pwm = rospy.get_param("~min_pwm", 900.0)
        # 후진 회피 설정
        self.enable_reverse_escape = rospy.get_param("~enable_reverse_escape", True)
        self.reverse_speed = rospy.get_param("~reverse_speed", -0.4)  # 후진 속도 (음수)
        self.reverse_pwm = rospy.get_param("~reverse_pwm", -800.0)  # 후진 PWM (음수)
        self.reverse_fov = math.radians(rospy.get_param("~reverse_fov_deg", 120.0))  # 후방 FOV
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

        # 전방 30도 이내 장애물 감지 확인
        front_obstacle_detected = self._check_front_obstacle(ranges, angles)
        
        if front_obstacle_detected:
            # 전방 30도 이내 장애물 감지 시 멈추고 후진 기동
            rospy.logwarn_throttle(1.0, "Obstacle detected in front 30deg. Stopping and reversing.")
            reverse_angle, reverse_distance, reverse_score = self._select_reverse_target(ranges, angles)
            if reverse_angle is not None:
                steering_angle = clamp(reverse_angle, -self.max_steering_angle, self.max_steering_angle)
                servo_cmd = self.servo_center + self.servo_per_rad * steering_angle
                servo_cmd = clamp(servo_cmd, self.min_servo, self.max_servo)
                
                self._publish_path(scan.header, steering_angle, reverse_distance)
                self._publish_target_marker(scan.header, steering_angle, reverse_distance)
                self._publish_motion_commands(
                    scan.header,
                    steering_angle,
                    self.reverse_speed,
                    self.reverse_pwm,
                    servo_cmd,
                    False,  # 후진 중에는 emergency_stop 아님
                )
                return
            else:
                # 후진 경로도 없으면 정지
                self._publish_stop(scan.header, reason="front_obstacle_no_reverse")
                return

        (
            target_angle,
            target_distance,
            selected_score,
        ) = self._select_target(ranges, angles)

        # 전방 30도 이내 장애물이 없으면 회피 기동 계속 (멈추지 않음)
        emergency_stop = False  # 전방 30도 이내 장애물이 없으면 정지하지 않음
        if target_angle is None:
            # 전방에 경로가 없을 때 후진 회피 시도
            if self.enable_reverse_escape:
                reverse_angle, reverse_distance, reverse_score = self._select_reverse_target(ranges, angles)
                if reverse_angle is not None:
                    rospy.logwarn_throttle(1.0, "No forward path. Reversing to escape. Angle: %.2f deg", 
                                         math.degrees(reverse_angle))
                    steering_angle = clamp(reverse_angle, -self.max_steering_angle, self.max_steering_angle)
                    servo_cmd = self.servo_center + self.servo_per_rad * steering_angle
                    servo_cmd = clamp(servo_cmd, self.min_servo, self.max_servo)
                    
                    self._publish_path(scan.header, steering_angle, reverse_distance)
                    self._publish_target_marker(scan.header, steering_angle, reverse_distance)
                    self._publish_motion_commands(
                        scan.header,
                        steering_angle,
                        self.reverse_speed,
                        self.reverse_pwm,
                        servo_cmd,
                        False,  # 후진 중에는 emergency_stop 아님
                    )
                    return
            
            rospy.logwarn_throttle(1.0, "No feasible gap (forward or reverse). Stopping vehicle.")
            self._publish_stop(scan.header, reason="no_gap")
            return

        # 거리 기반 속도 감소 적용 (60cm부터 점진적으로 감소, 30cm에서 정지)
        drive_speed, pwm = self._compute_speed_profile(
            target_distance, selected_score, emergency_stop, closest
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
        # 1단계: 거리 기준으로 장애물 후보 필터링
        mask = ranges < self.obstacle_threshold
        if not np.any(mask):
            return np.zeros((0, 2), dtype=np.float32)
        
        selected = np.stack((ranges[mask], angles[mask]), axis=1)
        xy = np.zeros_like(selected)
        xy[:, 0] = selected[:, 0] * np.cos(selected[:, 1])
        xy[:, 1] = selected[:, 0] * np.sin(selected[:, 1])
        
        # 2단계: 전방 포인트만 선택
        in_front = xy[:, 0] > 0.0
        front_points = xy[in_front]
        
        if len(front_points) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        
        # 3단계: 클러스터링으로 연속된 포인트만 장애물로 인식 (노이즈 제거)
        clustered_points = self._cluster_obstacle_points(front_points)
        
        return clustered_points
    
    def _cluster_obstacle_points(self, points: np.ndarray) -> np.ndarray:
        """
        간단한 거리 기반 클러스터링으로 연속된 포인트만 장애물로 인식.
        최소 포인트 수 조건을 만족하는 클러스터만 반환.
        """
        if len(points) < self.min_obstacle_points:
            return np.zeros((0, 2), dtype=np.float32)
        
        # 각 포인트 간 거리 계산
        # 간단한 방법: 각도 순으로 정렬 후 인접 포인트 거리 확인
        # 또는 더 정교한 DBSCAN 스타일 클러스터링
        
        # 간단한 방법: 각 포인트에서 가까운 포인트가 있는지 확인
        valid_points = []
        for i, p1 in enumerate(points):
            # 주변에 최소 포인트 수만큼 가까운 포인트가 있는지 확인
            distances = np.sqrt(np.sum((points - p1) ** 2, axis=1))
            nearby_count = np.sum(distances < self.obstacle_cluster_threshold)
            
            if nearby_count >= self.min_obstacle_points:
                valid_points.append(p1)
        
        if len(valid_points) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        
        return np.array(valid_points, dtype=np.float32)

    def _check_front_obstacle(
        self, ranges: np.ndarray, angles: np.ndarray
    ) -> bool:
        """
        전방 30도 각도 이내에 장애물(빨간색으로 표시되는 장애물)이 있는지 확인.
        """
        # 전방 30도 이내 각도 필터링 (±15도)
        front_angle_half = self.front_obstacle_angle * 0.5
        front_mask = np.abs(angles) <= front_angle_half
        
        if not np.any(front_mask):
            return False
        
        front_ranges = ranges[front_mask]
        
        # 장애물 임계값 이내에 포인트가 있는지 확인
        obstacle_mask = front_ranges < self.obstacle_threshold
        
        if not np.any(obstacle_mask):
            return False
        
        # 최소 포인트 수 확인 (노이즈 필터링)
        obstacle_count = np.sum(obstacle_mask)
        return obstacle_count >= self.min_obstacle_points

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

    def _select_reverse_target(
        self, ranges: np.ndarray, angles: np.ndarray
    ) -> Tuple[Optional[float], float, float]:
        """
        후방 방향에서 안전한 경로를 찾습니다.
        후방은 각도가 ±90도 ~ ±180도 범위입니다.
        """
        # 후방 각도 범위: 90도 ~ 180도, -90도 ~ -180도
        reverse_mask = np.abs(angles) > math.pi / 2.0  # |angle| > 90도
        if not np.any(reverse_mask):
            return None, 0.0, 0.0
        
        reverse_ranges = ranges[reverse_mask]
        reverse_angles = angles[reverse_mask]
        
        # 후방 FOV 제한 (예: ±60도 범위)
        reverse_fov_half = self.reverse_fov * 0.5
        # 후방 중심은 180도 또는 -180도
        # 각도를 180도 기준으로 정규화 (0~90도 범위)
        normalized_angles = np.abs(np.abs(reverse_angles) - math.pi)
        within_reverse_fov = normalized_angles <= reverse_fov_half
        
        if not np.any(within_reverse_fov):
            return None, 0.0, 0.0
        
        reverse_ranges = reverse_ranges[within_reverse_fov]
        reverse_angles = reverse_angles[within_reverse_fov]
        normalized_angles = normalized_angles[within_reverse_fov]
        
        clearance = np.clip(reverse_ranges - self.inflation_margin, 0.0, self.max_range)
        safe_mask = clearance > self.safe_distance
        
        if not np.any(safe_mask):
            return None, 0.0, 0.0
        
        norm_clearance = clearance / self.max_range
        # 후방에서는 중앙(180도)을 선호
        center_pref = 1.0 - (normalized_angles / reverse_fov_half)
        center_pref = np.clip(center_pref, 0.0, 1.0)
        
        scores = (
            self.clearance_weight * norm_clearance
            + self.heading_weight * center_pref
        )
        scores[~safe_mask] = -np.inf
        
        idx = int(np.argmax(scores))
        if not np.isfinite(scores[idx]):
            return None, 0.0, 0.0
        
        target_angle = float(reverse_angles[idx])
        target_distance = float(min(reverse_ranges[idx], self.lookahead_distance))
        return target_angle, target_distance, float(scores[idx])

    def _compute_speed_profile(
        self, lookahead: float, score: float, emergency_stop: bool, closest_distance: float
    ) -> Tuple[float, float]:
        if emergency_stop:
            return 0.0, 0.0

        # 거리 기반 속도 감소: 60cm부터 점진적으로 감소, 30cm에서 완전 정지
        # 60cm 이상: 정상 속도
        # 60cm ~ 30cm: 선형 감소
        # 30cm 이하: 정지 (emergency_stop으로 이미 처리됨)
        speed_reduction_factor = 1.0
        if closest_distance < self.speed_reduction_start:
            # 60cm ~ 30cm 구간에서 선형 감소
            reduction_range = self.speed_reduction_start - self.hard_stop_distance  # 0.6 - 0.3 = 0.3m
            if reduction_range > 0:
                distance_in_range = closest_distance - self.hard_stop_distance
                speed_reduction_factor = clamp(distance_in_range / reduction_range, 0.0, 1.0)
        
        score = clamp(score, 0.0, 1.0)
        distance_ratio = clamp(lookahead / self.lookahead_distance, 0.0, 1.0)
        blended = 0.5 * score + 0.5 * distance_ratio
        
        # 기본 속도 계산
        base_drive_speed = (
            self.min_drive_speed
            + (self.max_drive_speed - self.min_drive_speed) * blended
        )
        base_pwm = self.min_pwm + (self.max_pwm - self.min_pwm) * blended
        
        # 거리 기반 속도 감소 적용
        drive_speed = base_drive_speed * speed_reduction_factor
        pwm = self.min_pwm + (base_pwm - self.min_pwm) * speed_reduction_factor
        
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

