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
        self.safe_distance = rospy.get_param("~safe_distance", 0.40)  # 35cm 안전 거리
        self.hard_stop_distance = rospy.get_param("~hard_stop_distance", 0.15)  # 15cm에서 완전 정지
        self.inflation_margin = rospy.get_param("~inflation_margin", 0.30)  # 차폭 반경 15cm + 추가 여유 15cm = 30cm
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 1.5)
        self.obstacle_threshold = rospy.get_param("~obstacle_threshold", 1.0)  # 1m부터 장애물 인식
        self.max_drive_speed = rospy.get_param("~max_drive_speed", 0.3)  # m/s (장애물 회피 시 속도)
        self.front_obstacle_angle = math.radians(rospy.get_param("~front_obstacle_angle_deg", 90.0))  # 장애물 감지 FOV 180도
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
            rospy.get_param("~max_steering_angle_deg", 45.0)  # 회피 조향각 60도로 축소
        )
        # 화살표 표시 각도 스케일 (서보 각도보다 작게 표시)
        self.arrow_angle_scale = rospy.get_param("~arrow_angle_scale", 0.7)  # 서보 각도의 70%로 표시

        # PID 제어 파라미터 (재적용)
        # target_angle(헤딩 에러)을 0으로 만들기 위한 제어
        self.pid_kp = rospy.get_param("~lidar_pid_kp", 2.3)  # P이득 감소 (급격한 조향 방지)
        self.pid_ki = rospy.get_param("~lidar_pid_ki", 0.2)  # I이득 추가 (지속적인 오차 보정)
        self.pid_kd = rospy.get_param("~lidar_pid_kd", 3.0)  # D이득 증가 (진동 억제 및 부드러움)
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.prev_time = rospy.get_time()
        
        # 조향 관성 (Smoothing) 파라미터
        self.steering_smoothing = rospy.get_param("~lidar_steering_smoothing", 0.8)  # 0.0~1.0 (클수록 관성 큼)
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
        
        # 항상 장애물 회피 경로 계산 (FGM 알고리즘 적용)
        (
            target_angle,
            target_distance,
            selected_score,
        ) = self._select_target(ranges, angles)
        
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
        self, ranges: np.ndarray, angles: np.ndarray
    ) -> Tuple[Optional[float], float, float]:
        """
        Follow-the-Gap Method (FGM) 강화 버전
        1. 전처리 (Smoothing)
        2. Safety Bubble 적용
        3. Max Gap 찾기
        4. Goal Point 선택 (단순 중앙이 아닌, 더 깊은 곳으로 유도)
        """
        # 1. 전처리
        proc_ranges = np.array(ranges, copy=True)
        proc_ranges[np.isinf(proc_ranges)] = self.max_range
        proc_ranges[np.isnan(proc_ranges)] = 0.0
        
        # 노이즈 제거를 위한 Smoothing (Sliding Window)
        # 5개 포인트 평균
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        proc_ranges = np.convolve(proc_ranges, kernel, mode='same')
        
        # 2. Safety Bubble 적용
        # 가장 가까운 포인트 찾기
        min_idx = np.argmin(proc_ranges)
        min_dist = proc_ranges[min_idx]
        
        # Bubble 반경 (차폭 + 여유)
        # inflation_margin은 현재 0.3m로 설정됨
        bubble_radius = self.inflation_margin + 0.15  # 여유를 더 둠 (0.45m)
        
        if min_dist < self.max_range:
            angle_increment = angles[1] - angles[0] if len(angles) > 1 else 0.01
            # Bubble이 차지하는 각도 범위 계산
            if min_dist < 0.01: min_dist = 0.01
            bubble_angle = math.atan2(bubble_radius, min_dist)
            bubble_idx_half = int(bubble_angle / angle_increment)
            
            # 인덱스 범위 클램핑
            start_idx = max(0, min_idx - bubble_idx_half)
            end_idx = min(len(proc_ranges), min_idx + bubble_idx_half + 1)
            
            # Bubble 영역을 0으로 설정 (장애물로 간주하여 Gap에서 제외)
            proc_ranges[start_idx:end_idx] = 0.0
            
        # 3. Max Gap 찾기
        # 주행 가능한 최소 거리 (Threshold)
        gap_threshold = 1.2  # 1.2m 이상 열린 공간 (더 엄격하게)
        
        mask = proc_ranges > gap_threshold
        
        # Gap이 없으면 Threshold를 낮춰서 다시 시도 (Fallback)
        if not np.any(mask):
            gap_threshold = 0.6
            mask = proc_ranges > gap_threshold
            
        # 연속된 True 구간(Gap) 찾기
        padded_mask = np.concatenate(([False], mask, [False]))
        diff = np.diff(padded_mask.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        if len(starts) == 0:
            # 여전히 Gap이 없으면 가장 먼 곳으로
            max_idx = np.argmax(proc_ranges)
            if proc_ranges[max_idx] < 0.3:
                return None, 0.0, 0.0
            return angles[max_idx], proc_ranges[max_idx], 0.5
            
        # 가장 긴 Gap 선택
        gap_lengths = ends - starts
        best_gap_idx = np.argmax(gap_lengths)
        
        best_start = starts[best_gap_idx]
        best_end = ends[best_gap_idx]
        
        # 4. Goal Point 선택 (강화된 로직)
        # 단순히 중앙을 선택하는 것이 아니라, Gap 내에서 가장 깊은(멀리 있는) 지점을 향해 가중치를 둠
        # 이는 장애물 구간 끝에서 탈출할 때 유리함
        gap_indices = np.arange(best_start, best_end)
        gap_ranges = proc_ranges[best_start:best_end]
        
        # Gap 내에서 가장 거리가 먼 지점 찾기 (Max Depth)
        max_depth_idx_in_gap = np.argmax(gap_ranges)
        max_depth_idx = best_start + max_depth_idx_in_gap
        
        # 중앙 인덱스
        center_idx = (best_start + best_end) // 2
        
        # 최종 목표: 중앙과 가장 깊은 지점의 중간 지점 (안정성 + 탈출성)
        # 장애물 구간 끝에서는 깊은 지점이 한쪽 끝에 쏠려 있을 가능성이 높음
        best_idx = int((center_idx + max_depth_idx) / 2)
        
        target_angle = angles[best_idx]
        target_distance = proc_ranges[best_idx]
        
        rospy.logdebug_throttle(0.5, "FGM Enhanced: Gap [%d:%d], Target Angle %.1f deg", 
                               best_start, best_end, math.degrees(target_angle))
        
        return target_angle, target_distance, 1.0

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

