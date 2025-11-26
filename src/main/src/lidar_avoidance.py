#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Optional, Tuple

import cv2
import numpy as np
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import Float64
from tf2_ros import Buffer, TransformListener
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
        self.safe_distance = rospy.get_param("~safe_distance", 0.35)  # 35cm 안전 거리
        self.hard_stop_distance = rospy.get_param("~hard_stop_distance", 0.15)  # 15cm에서 완전 정지
        self.inflation_margin = rospy.get_param("~inflation_margin", 0.30)  # 차폭 반경 15cm + 추가 여유 15cm = 30cm
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 1.5)
        self.obstacle_threshold = rospy.get_param("~obstacle_threshold", 1.0)  # 1m부터 장애물 인식
        self.max_drive_speed = rospy.get_param("~max_drive_speed", 0.3)  # m/s (장애물 회피 시 속도)
        self.front_obstacle_angle = math.radians(rospy.get_param("~front_obstacle_angle_deg", 60.0))  # 전방 60도 이내 장애물 감지 각도
        self.min_obstacle_points = rospy.get_param("~min_obstacle_points", 3)  # 최소 연속 포인트 수 (노이즈 필터링)
        self.obstacle_cluster_threshold = rospy.get_param("~obstacle_cluster_threshold", 0.15)  # 클러스터링 거리 임계값
        self.heading_weight = rospy.get_param("~heading_weight", 0.35)
        self.clearance_weight = rospy.get_param("~clearance_weight", 0.65)
        self.smoothing_window = max(1, int(rospy.get_param("~smoothing_window", 7)))
        self.path_points = max(2, int(rospy.get_param("~path_points", 10)))

        # Speed / steering output 설정
        self.publish_ackermann = rospy.get_param("~publish_ackermann", False)
        self.publish_direct_controls = rospy.get_param("~publish_direct_controls", False)
        self.ackermann_topic = rospy.get_param("~ackermann_topic", "/ackermann_cmd")
        # 속도 제어는 main_run.py에서 담당하므로 제거
        self.servo_center = rospy.get_param("~servo_center", 0.53)
        self.servo_per_rad = rospy.get_param("~servo_per_rad", 0.95)  # 라디안 당 서보 변화량 (0.28 -> 0.95로 대폭 상향하여 조향각 확보)
        self.min_servo = rospy.get_param("~min_servo", 0.0)
        self.max_servo = rospy.get_param("~max_servo", 1.0)  # 서보 값은 항상 0~1 범위
        self.max_steering_angle = math.radians(
            rospy.get_param("~max_steering_angle_deg", 60.0)  # 최대 조향각 60도로 추가 확장
        )
        # 화살표 표시 각도 스케일 (서보 각도보다 작게 표시)
        self.arrow_angle_scale = rospy.get_param("~arrow_angle_scale", 0.7)  # 서보 각도의 70%로 표시

        # PID 제어 파라미터 (재적용)
        # target_angle(헤딩 에러)을 0으로 만들기 위한 제어
        self.pid_kp = rospy.get_param("~lidar_pid_kp", 1.3)  # P이득 감소 (급격한 조향 방지)
        self.pid_ki = rospy.get_param("~lidar_pid_ki", 0.1)  # I이득 추가 (지속적인 오차 보정)
        self.pid_kd = rospy.get_param("~lidar_pid_kd", 2.5)  # D이득 증가 (진동 억제 및 부드러움)
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.prev_time = rospy.get_time()
        
        # 조향 관성 (Smoothing) 파라미터
        self.steering_smoothing = rospy.get_param("~lidar_steering_smoothing", 0.7)  # 0.0~1.0 (클수록 관성 큼)
        self.prev_servo_cmd = self.servo_center

        # 카메라-라이다 퓨전 설정 (제거됨)
        self.enable_camera_fusion = False
        
        # ROS I/O

        # ROS I/O
        rospy.Subscriber(
            self.scan_topic,
            LaserScan,
            self.scan_callback,
            queue_size=1,
        )
        # 카메라 구독 제거됨
        # 속도 제어는 main_run.py에서 담당하므로 제거
        # 조향만 제어
        self.steering_pub = None
        self.ackermann_pub = None
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
        self.steering_command_pub = rospy.Publisher(
            "lidar_avoidance/steering_cmd", Float64, queue_size=1
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
        
        # 30cm 이하일 때는 정지 (15cm에서 완전 정지)
        hard_stop_distance = 0.15  # 15cm
        speed_reduction_start = 0.30  # 30cm
        
        if closest <= hard_stop_distance:
            # 15cm 이하는 완전 정지
            rospy.logwarn_throttle(1.0, "Obstacle too close (%.2fm <= %.2fm). Emergency stop.", closest, hard_stop_distance)
            self._publish_stop(scan.header, reason="too_close")
        elif closest < speed_reduction_start:
            # 15cm ~ 30cm: 감속 경고만 (속도 제어는 main_run.py에서 수행)
            rospy.logwarn_throttle(1.0, "Obstacle very close (%.2fm). Speed reduction active.", closest)

        # 카메라 퓨전 적용
        # obstacle_points: 회피 로직용 (60cm 이내만)
        # all_points: 마커 표시용 (모든 포인트, LaserScan과 동일)
        obstacle_points, all_points, lidar_confidence = self._collect_obstacle_points(ranges, angles, scan.header)
                    # if len(camera_obstacles) > 0:
                    #     obstacle_points = np.vstack([obstacle_points, camera_obstacles]) if len(obstacle_points) > 0 else camera_obstacles
        
        # 마커는 모든 포인트를 표시하되, 60cm 이내인 것만 빨간색으로 표시
        self._publish_obstacle_markers(scan.header, all_points, obstacle_points)

        # 전방 30도 이내 장애물 감지 확인 (경고만, 정지하지 않음)
        front_obstacle_detected = self._check_front_obstacle(ranges, angles)
        if front_obstacle_detected:
            rospy.logwarn_throttle(1.0, "Obstacle detected in front 30deg. Attempting to avoid by turning left/right.")

        # 장애물 포인트를 각도로 변환 (회피를 위해)
        # 원본 좌표계를 사용하므로 각도도 원본 좌표계 기준
        obstacle_angles = None
        if len(obstacle_points) > 0:
            rospy.loginfo_throttle(1.0, "Detected %d obstacle points", len(obstacle_points))
            # 장애물 포인트의 각도 계산 (원본 라이다 좌표계 기준)
            obstacle_angles = np.arctan2(obstacle_points[:, 1], obstacle_points[:, 0])
            # 각도는 이미 원본 좌표계이므로 추가 변환 불필요
        else:
            rospy.logdebug_throttle(2.0, "No obstacle points detected")

        # 1m 이내일 때 장애물 회피 기동 수행
        # (속도 감속은 main_run.py에서 수행하지만, 조향은 여기서 계산해야 함)
        speed_reduction_start = 1.0  # 1m
        
        # 항상 장애물 회피 경로 계산 (거리에 상관없이 최적의 경로 찾기)
        (
            target_angle,
            target_distance,
            selected_score,
        ) = self._select_target(ranges, angles, front_obstacle_detected, obstacle_points)
        
        if target_angle is None:
            # 전방에 경로가 없을 때
            if closest < speed_reduction_start:
                # 너무 가까운데 경로도 없으면 정지 (조향은 유지)
                rospy.logwarn_throttle(1.0, "No feasible gap and obstacle close. Stopping vehicle.")
                self._publish_stop(scan.header, reason="no_gap")
                # 조향은 이전 값 유지하거나 중앙으로? 일단 중앙으로
                target_angle = 0.0
            else:
                # 멀리 있는데 경로가 없으면(다 막힘?) 일단 직진
                target_angle = 0.0

        # 전방 30도 이내 장애물이 없으면 회피 기동 계속
        emergency_stop = False

        # PID 제어 적용 (Heading Error Correction)
        # target_angle은 현재 차량 헤딩과 목표 경로 사이의 각도 차이(에러)임.
        # 이 에러를 0으로 줄이는 것이 목표.
        current_time = rospy.get_time()
        dt = current_time - self.prev_time
        if dt <= 0 or dt > 1.0:
            dt = 0.033
        self.prev_time = current_time

        error = target_angle
        
        # 적분항
        self.integral_error += error * dt
        self.integral_error = clamp(self.integral_error, -0.5, 0.5)  # Windup 방지 (엄격하게 제한)
        
        # 미분항
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        
        # PID 출력 (목표 조향각)
        pid_steering_angle = (self.pid_kp * error) + (self.pid_ki * self.integral_error) + (self.pid_kd * derivative)
        
        # 조향각 제한
        steering_angle = clamp(pid_steering_angle, -self.max_steering_angle, self.max_steering_angle)
        
        # [중요] 조향 방향 반전 유지: PID 출력값을 서보 명령으로 변환할 때 부호 반전
        # (Left Turn이 Positive Angle -> 서보 값 감소)
        target_servo = self.servo_center - self.servo_per_rad * steering_angle
        target_servo = clamp(target_servo, self.min_servo, self.max_servo)
        
        # 조향 관성 적용 (Low-Pass Filter)
        # prev_servo_cmd와 target_servo 사이를 부드럽게 보간
        servo_cmd = (self.steering_smoothing * self.prev_servo_cmd) + ((1.0 - self.steering_smoothing) * target_servo)
        self.prev_servo_cmd = servo_cmd

        self._publish_path(scan.header, steering_angle, target_distance)
        self._publish_target_marker(scan.header, steering_angle, target_distance)
        
        # 장애물이 감지된 경우(빨간 마커)에만 조향 명령 발행
        if len(obstacle_points) > 0:
            self.steering_command_pub.publish(Float64(servo_cmd))
            
        self._publish_motion_commands(
            scan.header,
            steering_angle,
            servo_cmd,
            emergency_stop,
        )

    def _prepare_scan(self, scan: LaserScan) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        ranges = np.array(scan.ranges, dtype=np.float32)
        angles = scan.angle_min + np.arange(len(ranges), dtype=np.float32) * scan.angle_increment

        # 라이다가 180도 돌아가 있다고 가정하고 각도를 보정 (0도가 뒤쪽 -> 180도가 앞쪽)
        angles = angles + math.pi
        # 각도를 [-π, π] 범위로 정규화
        angles = np.arctan2(np.sin(angles), np.cos(angles))

        valid = np.isfinite(ranges)
        ranges = ranges[valid]
        angles = angles[valid]
        if not ranges.size:
            return None

        # 전방 FOV만 포함
        fov_half = self.forward_fov * 0.5
        # 전방 FOV: |angle| <= forward_fov/2
        forward_mask = np.abs(angles) <= fov_half
        
        # 전방 FOV 내 포인트만 선택
        ranges = ranges[forward_mask]
        angles = angles[forward_mask]
        if not ranges.size:
            return None

        ranges = np.clip(ranges, scan.range_min, self.max_range)

        if self.smoothing_window > 1:
            kernel = np.ones(self.smoothing_window, dtype=np.float32) / float(
                self.smoothing_window
            )
            ranges = np.convolve(ranges, kernel, mode="same")
        return ranges, angles

    # image_callback, camera_info_callback 제거됨

    def _collect_obstacle_points(
        self, ranges: np.ndarray, angles: np.ndarray, header=None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        # 1단계: 라이다 좌표계에서 카테시안 좌표로 변환 (RViz LaserScan과 동일한 좌표계)
        # LaserScan 표준 좌표계: x = r * cos(angle), y = r * sin(angle)
        # RViz에서 표시되는 좌표계와 동일하게 변환
        xy = np.zeros((len(ranges), 2), dtype=np.float32)
        xy[:, 0] = ranges * np.cos(angles)  # 원본 좌표계
        xy[:, 1] = ranges * np.sin(angles)  # 원본 좌표계
        
        # 2단계: 전방 포인트 필터링 (라이다가 180도 회전되어 있다면 전방이 음수 x)
        # detect_obstacles.py 참고: x_forward = -x, y_lateral = y
        # 전방 포인트: x < 0 (또는 라이다 설치 방향에 따라 다를 수 있음)
        # 일단 모든 포인트를 사용 (전방 FOV는 이미 _prepare_scan에서 필터링됨)
        
        # all_points: 마커 표시용 (모든 포인트, LaserScan과 동일하게 표시)
        all_points = xy
        
        # 3단계: 전방 20도, 1m 이내 장애물만 인식 (회피 로직용)
        # 3-1. 전방 20도 필터링 (±10도)
        front_angle_limit = math.radians(20.0) * 0.5  # ±10도
        front_angle_mask = np.abs(angles) <= front_angle_limit
        
        # 3-2. 거리 필터링 (1m 이내)
        obstacle_detection_range = 1.0  # 1m
        distances = np.linalg.norm(xy, axis=1)
        distance_mask = distances < obstacle_detection_range
        
        # 3-3. 두 조건 모두 만족하는 포인트만 장애물로 인식
        obstacle_mask = front_angle_mask & distance_mask
        obstacle_points = xy[obstacle_mask]
        front_count = len(obstacle_points)
        
        if len(obstacle_points) == 0:
            rospy.logdebug_throttle(2.0, "No obstacle points found (front 20deg, < %.2fm)", obstacle_detection_range)
            return np.zeros((0, 2), dtype=np.float32), all_points, 0.0
        
        min_dist = float(np.min(distances[obstacle_mask]))
        rospy.loginfo_throttle(1.0, "Obstacle detection: %d points (front 20deg, < %.2fm), min_dist=%.2fm, total_points=%d", 
                               front_count, obstacle_detection_range, min_dist, len(xy))
        
        # 4단계: 클러스터링으로 노이즈 제거 (선택적)
        # 가까운 장애물이 많으면 클러스터링 적용, 적으면 그대로 사용
        original_count = len(obstacle_points)
        if len(obstacle_points) > 10:  # 포인트가 많을 때만 클러스터링
            clustered_points = self._cluster_obstacle_points(obstacle_points)
            if len(clustered_points) > 0:
                obstacle_points = clustered_points
                rospy.logdebug_throttle(2.0, "Clustered: %d -> %d points", original_count, len(clustered_points))
            else:
                # 클러스터링 실패 시 원본 포인트 유지 (벽 등 큰 장애물은 클러스터링 없이도 인식)
                rospy.logwarn_throttle(1.0, "Clustering failed (0 points), keeping original %d points", original_count)
        else:
            # 포인트가 적으면 클러스터링 없이 모두 사용
            rospy.logdebug_throttle(2.0, "Skipping clustering for %d points (< 10)", original_count)
        
        # 라이다 신뢰도 계산 (거리 기반: 가까울수록 높은 신뢰도)
        if len(obstacle_points) > 0:
            distances = np.linalg.norm(obstacle_points, axis=1)
            min_dist = float(np.min(distances))
            # 0.15m 이내: 신뢰도 1.0, 0.3m: 신뢰도 0.5, 그 이상: 신뢰도 감소
            lidar_confidence = max(0.0, min(1.0, 1.0 - (min_dist - 0.15) / 0.15))
            rospy.logdebug_throttle(2.0, "Obstacle detection: %d points, closest: %.2fm, confidence: %.2f", 
                                    len(obstacle_points), min_dist, lidar_confidence)
        else:
            lidar_confidence = 0.0
        
        return obstacle_points, all_points, lidar_confidence
    
    def _cluster_obstacle_points(self, points: np.ndarray) -> np.ndarray:
        """
        간단한 거리 기반 클러스터링으로 연속된 포인트만 장애물로 인식.
        최소 포인트 수 조건을 만족하는 클러스터만 반환.
        """
        if len(points) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        
        # 포인트가 적으면 클러스터링 없이 모두 반환 (가까운 장애물은 작은 포인트 수로도 인식)
        if len(points) < self.min_obstacle_points:
            # 포인트가 적어도 1m 이내 장애물이면 모두 인식 (벽 등 큰 장애물도 인식)
            distances = np.linalg.norm(points, axis=1)
            min_dist = float(np.min(distances))
            obstacle_detection_range = 1.0  # 1m
            if min_dist < obstacle_detection_range:  # 1m 이내는 클러스터링 없이도 인식
                rospy.logdebug_throttle(2.0, "Keeping %d points without clustering (min_dist=%.2fm)", len(points), min_dist)
                return points
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
            
            # 최소 포인트 수 조건 완화: 가까운 장애물은 더 적은 포인트로도 인식
            required_points = self.min_obstacle_points
            dist_to_origin = np.linalg.norm(p1)
            if dist_to_origin < 0.4:  # 40cm 이내는 최소 포인트 수를 1로 완화
                required_points = 1
            elif dist_to_origin < 0.5:  # 50cm 이내는 최소 포인트 수를 2로 완화
                required_points = max(1, self.min_obstacle_points - 1)
            elif dist_to_origin < 0.6:  # 60cm 이내는 최소 포인트 수를 2로 완화
                required_points = max(1, self.min_obstacle_points - 1)
            
            if nearby_count >= required_points:
                valid_points.append(p1)
        
        if len(valid_points) == 0:
            # 클러스터링 결과가 없어도 벽 등 큰 장애물은 인식해야 함
            # 원본 포인트의 최소 거리를 확인
            distances = np.linalg.norm(points, axis=1)
            min_dist = float(np.min(distances))
            obstacle_detection_range = 1.0  # 1m
            if min_dist < obstacle_detection_range:
                # 1m 이내 장애물이면 클러스터링 없이도 모두 반환
                rospy.logwarn_throttle(1.0, "Clustering filtered all points, but min_dist=%.2fm < %.2fm. Returning all points.", 
                                      min_dist, obstacle_detection_range)
                return points
            return np.zeros((0, 2), dtype=np.float32)
        
        return np.array(valid_points, dtype=np.float32)

    def _check_front_obstacle(
        self, ranges: np.ndarray, angles: np.ndarray
    ) -> bool:
        """
        전방 30도 각도 이내에 장애물(빨간색으로 표시되는 장애물)이 있는지 확인.
        라이다 좌표계 180도 회전을 고려하여 각도에 π를 더한 위치를 확인.
        """
        # 전방 30도 이내 각도 필터링 (±15도)
        # 라이다 좌표계 180도 회전: 각도에 π를 더한 위치가 전방
        front_angle_half = self.front_obstacle_angle * 0.5
        rotated_angles = angles + math.pi
        # 각도를 [-π, π] 범위로 정규화
        rotated_angles = np.where(rotated_angles > math.pi, rotated_angles - 2 * math.pi, rotated_angles)
        rotated_angles = np.where(rotated_angles < -math.pi, rotated_angles + 2 * math.pi, rotated_angles)
        front_mask = np.abs(rotated_angles) <= front_angle_half
        
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

    # 카메라 관련 메서드 제거됨
    # image_callback, camera_info_callback, _project_lidar_to_camera, _detect_camera_obstacles 제거

    def _select_target(
        self, ranges: np.ndarray, angles: np.ndarray, front_obstacle_detected: bool = False,
        obstacle_points: Optional[np.ndarray] = None
    ) -> Tuple[Optional[float], float, float]:
        """
        장애물 마커(obstacle_points)를 기반으로 최적의 경로를 선택하는 로직.
        반경 1m 내의 장애물 밀도를 평가하고, 10cm 이내 접근 시 강력한 페널티를 부여하여
        장애물 밀도가 낮은 곳으로 주행하도록 유도함.
        """
        # 후보 경로 생성 (정밀도를 위해 샘플 수 증가: 50 -> 90)
        candidate_angles = np.linspace(-self.max_steering_angle, self.max_steering_angle, 90)
        
        best_score = -float('inf')
        best_angle = 0.0
        best_dist = 0.0
        found_valid_path = False

        # 장애물이 없으면 직진
        if obstacle_points is None or len(obstacle_points) == 0:
            return 0.0, self.lookahead_distance, 1.0

        # 벡터화 연산을 위한 준비
        ox = obstacle_points[:, 0]
        oy = obstacle_points[:, 1]
        
        # 사용자 요청: 반경 1m 내에서 탐색
        check_distance = 1.0

        for angle in candidate_angles:
            # 경로 벡터 (단위 벡터)
            path_x = math.cos(angle)
            path_y = math.sin(angle)
            
            # 1. 경로상 투영 거리 (전방 거리)
            dots = ox * path_x + oy * path_y
            
            # 2. 경로 수직 거리 (측면 거리)
            crosses = np.abs(path_x * oy - path_y * ox)
            
            # 유효한 장애물 필터링: 전방에 있고(dots > 0), 1m 이내에 있는 것(check_distance)
            valid_mask = (dots > 0) & (dots < check_distance)
            
            # 충돌 여부 확인 (차폭 이내)
            collision_mask = valid_mask & (crosses < self.inflation_margin)
            
            if np.any(collision_mask):
                # 충돌 발생: 가장 가까운 충돌 지점까지의 거리
                min_dist = np.min(dots[collision_mask])
                # 충돌 경로는 매우 낮은 점수 (-10000)
                score = -10000.0 + min_dist
            else:
                # 충돌 없음
                min_dist = check_distance
                
                # 1. 근접 페널티 (10cm 이내 접근 시 대폭 강화)
                # inflation_margin + 10cm 이내의 장애물에 대해 강력한 페널티 부여
                danger_threshold = self.inflation_margin + 0.10
                danger_mask = valid_mask & (crosses < danger_threshold)
                
                proximity_penalty = 0.0
                if np.any(danger_mask):
                    # 차폭 경계로부터의 거리 (0 ~ 0.1m)
                    dists_from_edge = crosses[danger_mask] - self.inflation_margin
                    # 거리가 가까울수록 페널티가 기하급수적으로 증가
                    # 100.0 가중치로 대폭 강화 (충돌 직전 상황 회피)
                    proximity_penalty = np.sum(100.0 / (dists_from_edge + 0.01))
                
                # 2. 밀도 페널티 (주변 장애물 개수)
                # 차폭보다 조금 더 넓은 범위(예: 차폭 + 40cm) 내의 장애물 개수를 세어 밀도가 낮은 곳 선호
                density_threshold = self.inflation_margin + 0.40
                density_mask = valid_mask & (crosses < density_threshold)
                density_count = np.sum(density_mask)
                
                # 점수 계산
                # - 주행 가능 거리 (클수록 좋음)
                # - 조향 각도 (작을수록 좋음 - 직진 선호)
                # - 근접 페널티 (작을수록 좋음 - 매우 큼)
                # - 밀도 (작을수록 좋음 - 장애물 적은 곳 선호)
                score = (5.0 * min_dist) \
                        - (self.heading_weight * abs(angle)) \
                        - (1.0 * proximity_penalty) \
                        - (0.5 * density_count)
                
                found_valid_path = True

            if score > best_score:
                best_score = score
                best_angle = angle
                best_dist = min_dist
        
        if not found_valid_path and best_score < -5000:
             # 모든 경로가 충돌 위험
             pass

        rospy.logdebug_throttle(0.5, "Best Angle: %.2f deg, Score: %.2f", math.degrees(best_angle), best_score)
        
        return best_angle, best_dist, best_score

    # 속도 제어는 main_run.py에서 담당하므로 제거됨

    def _publish_motion_commands(
        self,
        header,
        steering_angle: float,
        servo_cmd: float,
        emergency_stop: bool,
    ) -> None:
        # 조향만 제어 (속도는 main_run.py에서 제어)
        if emergency_stop:
            # 비상 정지 시 조향각 중앙으로
            servo_cmd = self.servo_center
        
        # 명령 발행
        if self.publish_ackermann and self.ackermann_pub is not None:
            ack_msg = AckermannDriveStamped()
            ack_msg.header = header
            ack_msg.drive.steering_angle = steering_angle
            ack_msg.drive.speed = 0.0  # 속도는 main_run.py에서 제어하므로 0으로 설정
            self.ackermann_pub.publish(ack_msg)

        if self.publish_direct_controls and self.steering_pub:
            servo_msg = Float64()
            servo_msg.data = servo_cmd
            self.steering_pub.publish(servo_msg)

    def _publish_obstacle_markers(self, header, all_points: np.ndarray, obstacle_points: np.ndarray) -> None:
        """
        모든 포인트를 표시하되, 60cm 이내인 것만 빨간색으로 표시.
        all_points: LaserScan의 모든 포인트 (흰 마커와 동일한 위치)
        obstacle_points: 60cm 이내 장애물 포인트 (빨간색으로 표시)
        """
        marker_array = MarkerArray()
        
        # 전방 20도, 0.3m 이내 장애물만 빨간색으로 표시
        if len(obstacle_points) > 0 and len(all_points) > 0:
            # 모든 포인트의 거리 계산
            all_distances = np.linalg.norm(all_points, axis=1)
            obstacle_detection_range = 1.0  # 1m
            close_mask = all_distances < obstacle_detection_range  # 0.3m 이내
            
            # 60cm 이내인 포인트만 빨간색 마커로 표시
            red_marker = Marker()
            red_marker.header = header
            red_marker.ns = "lidar_obstacles"
            red_marker.id = 0
            red_marker.type = Marker.POINTS
            red_marker.action = Marker.ADD
            red_marker.scale.x = 0.08
            red_marker.scale.y = 0.08
            red_marker.color.r = 1.0
            red_marker.color.g = 0.3
            red_marker.color.b = 0.3
            red_marker.color.a = 0.9
            red_marker.lifetime = rospy.Duration(0.2)
            red_marker.points = []
            
            # 60cm 이내인 포인트만 선택 (LaserScan과 동일한 위치)
            close_points = all_points[close_mask]
            for x, y in close_points:
                p = Point()
                p.x = float(x)
                p.y = float(y)
                red_marker.points.append(p)
            
            if len(red_marker.points) > 0:
                marker_array.markers.append(red_marker)
        
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
        marker.color.a = 0.9
        # 라이다 좌표계 180도 회전 후 화살표 방향 보정: 각도 반전 및 서보 각도보다 작게 스케일링
        display_angle = -steering_angle * self.arrow_angle_scale
        
        # 전진 시 파란색으로 표시
        marker.color.r = 0.1
        marker.color.g = 0.8
        marker.color.b = 1.0
        
        marker.pose.orientation.z = math.sin(display_angle * 0.5)
        marker.pose.orientation.w = math.cos(display_angle * 0.5)
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
            # 라이다 좌표계 180도 회전 후 경로 방향 보정: 각도 반전 및 서보 각도보다 작게 스케일링
            display_angle = -steering_angle * self.arrow_angle_scale
            pose.pose.position.x = travel * math.cos(display_angle)
            pose.pose.position.y = travel * math.sin(display_angle)
            pose.pose.orientation.z = math.sin(display_angle * 0.5)
            pose.pose.orientation.w = math.cos(display_angle * 0.5)
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

    def _publish_stop(self, header, reason: str) -> None:
        # 조향만 중앙으로 설정 (속도는 main_run.py에서 제어)
        if self.publish_ackermann and self.ackermann_pub is not None:
            msg = AckermannDriveStamped()
            msg.header = header
            msg.drive.speed = 0.0  # 속도는 main_run.py에서 제어
            msg.drive.steering_angle = 0.0
            self.ackermann_pub.publish(msg)
        if self.publish_direct_controls and self.steering_pub:
            servo_msg = Float64()
            servo_msg.data = self.servo_center
            self.steering_pub.publish(servo_msg)
        rospy.logdebug_throttle(2.0, "Stop command issued (%s)", reason)


def run():
    rospy.init_node("lidar_avoidance_planner")
    LidarAvoidancePlanner()
    rospy.spin()


if __name__ == "__main__":
    run()

