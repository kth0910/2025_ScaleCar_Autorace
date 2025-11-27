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
    LaserScan 기반의 장애물 감지 및 로컬 플래너.
    전방 FOV에서 가장 안전한 경로(Gap)를 찾아 조향 명령(steering_cmd)을 발행한다.
    속도 제어는 main_run.py에 위임하며, 비상 정지 상황 등에서만 ackermann_cmd를 발행할 수 있다.
    RViz 시각화를 위한 Marker/Path 토픽을 제공한다.
    """

    def __init__(self) -> None:
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.forward_fov = math.radians(rospy.get_param("~forward_fov_deg", 210.0))
        self.max_range = rospy.get_param("~max_range", 8.0)
        self.safe_distance = rospy.get_param("~safe_distance", 0.50)  # 50cm 안전 거리
        self.hard_stop_distance = rospy.get_param("~hard_stop_distance", 0.15)  # 15cm에서 완전 정지
        self.inflation_margin = rospy.get_param("~inflation_margin", 0.035)  # 장애물 확장 마진 (0.035m) - 30% 축소
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 1.5)
        self.obstacle_threshold = rospy.get_param("~obstacle_threshold", 1.0)  # 0.7m 이내를 장애물로 인식
        self.max_drive_speed = rospy.get_param("~max_drive_speed", 0.15)  # m/s (장애물 회피 시 속도)
        self.front_obstacle_angle = math.radians(rospy.get_param("~front_obstacle_angle_deg", 90.0))  # 장애물 감지 FOV
        self.min_obstacle_points = rospy.get_param("~min_obstacle_points", 6)  # 최소 연속 포인트 수 (노이즈 필터링 강화)
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
        self.servo_per_rad = rospy.get_param("~servo_per_rad", 0.95)  # 라디안 당 서보 변화량
        self.min_servo = rospy.get_param("~min_servo", 0.0)
        self.max_servo = rospy.get_param("~max_servo", 1.0)  # 서보 값은 항상 0~1 범위
        self.max_steering_angle = math.radians(
            rospy.get_param("~max_steering_angle_deg", 45.0)  # 최대 회피 조향각 45도
        )
        # 화살표 표시 각도 스케일 (시각화용)
        self.arrow_angle_scale = rospy.get_param("~arrow_angle_scale", 0.7)

        # PID 제어 파라미터
        # target_angle(헤딩 에러)을 0으로 만들기 위한 제어
        self.pid_kp = rospy.get_param("~lidar_pid_kp", 2.0)
        self.pid_ki = rospy.get_param("~lidar_pid_ki", 0.05)
        self.pid_kd = rospy.get_param("~lidar_pid_kd", 0.8)
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.prev_time = rospy.get_time()
        
        # 조향 관성 (Smoothing) 파라미터
        self.steering_smoothing = rospy.get_param("~lidar_steering_smoothing", 0.6)  # 0.0~1.0 (클수록 관성 큼)
        self.prev_servo_cmd = self.servo_center

        # 카메라-라이다 퓨전 설정 (현재 미사용)
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
        # 속도 제어는 main_run.py에서 담당하므로 이 노드는 조향 명령만 발행
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
        self.gap_marker_pub = rospy.Publisher(
            "lidar_avoidance/fgm_debug", MarkerArray, queue_size=1
        )
        self.processed_scan_pub = rospy.Publisher(
            "lidar_avoidance/fgm_processed_scan", LaserScan, queue_size=1
        )

        self.last_scan_time = rospy.get_time()
        self.scan_timeout = rospy.get_param("~scan_timeout", 1.0)  # seconds
        self.avoidance_deadline = 0.0  # 회피 모드 유지 시간 (Persistence)
        
        rospy.Subscriber(
            "/commands/servo/position",
            Float64,
            self.servo_callback,
            queue_size=1
        )
        self.current_servo = self.servo_center

        rospy.loginfo(
            "LidarAvoidancePlanner ready. Subscribing to %s (hardware only, no simulation)", 
            self.scan_topic
        )

    def servo_callback(self, msg: Float64) -> None:
        self.current_servo = msg.data

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

        # 속도 제어용 FOV: 전방 90도 (±45도)
        speed_fov_mask = np.abs(angles) <= math.radians(60.0)
        if np.any(speed_fov_mask):
            closest = float(np.min(ranges[speed_fov_mask]))
        else:
            closest = float('inf')
            
        self.clearance_pub.publish(closest)
        
        # 30cm 이하일 때는 정지 (15cm에서 완전 정지)
        hard_stop_distance = 0.20  # 15cm
        speed_reduction_start = 0.30  # 30cm
        
        if closest <= hard_stop_distance:
            # 15cm 이하는 완전 정지
            rospy.logwarn_throttle(1.0, "Obstacle too close (%.2fm <= %.2fm). Emergency stop.", closest, hard_stop_distance)
            self._publish_stop(scan.header, reason="too_close")
        elif closest < speed_reduction_start:
            # 15cm ~ 30cm: 감속 경고만 (속도 제어는 main_run.py에서 수행)
            rospy.logwarn_throttle(1.0, "Obstacle very close (%.2fm). Speed reduction active.", closest)

        # 장애물 포인트 수집
        # obstacle_points: 회피 로직용 (설정된 범위 이내)
        # all_points: 마커 표시용 (모든 포인트)
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
        ) = self._select_target(ranges, angles, scan.header)
        
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
        
        # [중요] 조향 방향 수정: 데이터는 좌우 반전(Mirror) 상태이므로, 조향도 반대로(빼기) 해야 올바른 방향으로 감
        target_servo = self.servo_center - self.servo_per_rad * steering_angle
        target_servo = clamp(target_servo, self.min_servo, self.max_servo)
        
        # 조향 관성 적용 (Low-Pass Filter)
        # prev_servo_cmd와 target_servo 사이를 부드럽게 보간
        servo_cmd = (self.steering_smoothing * self.prev_servo_cmd) + ((1.0 - self.steering_smoothing) * target_servo)
        self.prev_servo_cmd = servo_cmd

        self._publish_path(scan.header, steering_angle, target_distance)
        self._publish_target_marker(scan.header, steering_angle, target_distance)
        
        # [검증 2] 회피 상태 유지 (Persistence)
        # 장애물이 감지되면 데드라인 연장
        if len(obstacle_points) > 0:
            self.avoidance_deadline = rospy.get_time() + 1.0  # 1초간 유지
            
        # 장애물이 감지되었거나, 유지 시간 내라면 조향 명령 발행
        if rospy.get_time() < self.avoidance_deadline:
            self.steering_command_pub.publish(Float64(servo_cmd))
            rospy.loginfo_throttle(0.5, "Avoidance Active! Servo: %.2f, Angle: %.1f deg", servo_cmd, math.degrees(steering_angle))
            
        self._publish_motion_commands(
            scan.header,
            steering_angle,
            servo_cmd,
            emergency_stop,
        )

    def _prepare_scan(self, scan: LaserScan) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        ranges = np.array(scan.ranges, dtype=np.float32)
        angles = scan.angle_min + np.arange(len(ranges), dtype=np.float32) * scan.angle_increment

        # 라이다가 180도 돌아가 있어서 0도가 후방임 -> 180도 회전하여 0도를 전방으로 보정
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
        xy[:, 0] = ranges * np.cos(angles)  # x = r * cos(theta)
        xy[:, 1] = ranges * np.sin(angles)  # y = r * sin(theta) (정방향 시각화)
        
        # 2단계: 전방 포인트 필터링 (라이다가 180도 회전되어 있다면 전방이 음수 x)
        # detect_obstacles.py 참고: x_forward = -x, y_lateral = y
        # 전방 포인트: x < 0 (또는 라이다 설치 방향에 따라 다를 수 있음)
        # 일단 모든 포인트를 사용 (전방 FOV는 이미 _prepare_scan에서 필터링됨)
        
        # all_points: 마커 표시용 (모든 포인트, LaserScan과 동일하게 표시)
        all_points = xy
        
        # 3단계: 장애물 검출 - 전방 44도(±22도), 0.75m 이내 무조건 인식
        # 회전 시 측면으로 빠지는 장애물도 놓치지 않도록 광각 감지 -> 요청에 따라 44도로 변경
        
        # 3-1. 전방 각도 필터링 (±65도) - 전방 130도
        check_fov = math.radians(130.0) 
        half_fov = check_fov * 0.5
        
        # 각도 차이 계산 (0도 기준)
        angle_mask = np.abs(angles) <= half_fov
        
        # 3-2. 거리 필터링 (0.60m) - 60cm 이상은 장애물로 판단하지 않음
        obstacle_detection_range = 0.60
        distances = np.linalg.norm(xy, axis=1)
        distance_mask = distances < obstacle_detection_range
        
        # 3-3. 최종 마스크
        obstacle_mask = angle_mask & distance_mask
        
        obstacle_points = xy[obstacle_mask]
        front_count = len(obstacle_points)
        
        if len(obstacle_points) < self.min_obstacle_points:
            rospy.logdebug_throttle(2.0, "Ignored noise (count=%d < %d)", len(obstacle_points), self.min_obstacle_points)
            return np.zeros((0, 2), dtype=np.float32), all_points, 0.0
        
        min_dist = float(np.min(distances[obstacle_mask]))
        rospy.loginfo_throttle(0.5, "!! OBSTACLE DETECTED !! Count: %d, MinDist: %.2fm", front_count, min_dist)
        
        # 4단계: 클러스터링 (생략 - 일단 다 잡음)
        
        # 라이다 신뢰도 계산
        lidar_confidence = 1.0
        
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
        # 전방 각도 필터링 (이미 0도가 전방으로 정렬됨)
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

    # 카메라 관련 메서드 제거됨
    # image_callback, camera_info_callback, _project_lidar_to_camera, _detect_camera_obstacles 제거

    def _select_target(
        self, ranges: np.ndarray, angles: np.ndarray, header=None
    ) -> Tuple[Optional[float], float, float]:
        """
        Follow-the-Gap Method (FGM) 기반 경로 선택
        1. 전처리 (Smoothing)
        2. Safety Bubble 적용 (장애물 부풀리기)
        3. Max Gap 찾기
        4. Goal Point 선택 (Gap 내에서 가장 깊은 지점과 중앙의 절충)
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
        
        # 2. Safety Bubble 적용 (모든 장애물에 대해)
        # 기존: 가장 가까운 점 하나만 처리 -> 수정: 일정 거리 이내 모든 점 처리하여 라바콘 사이를 벽으로 인식
        
        obstacle_dist_limit = 1.5  # 1.5m 이내의 장애물만 버블 처리 (너무 멀리 있는 것은 무시)
        # 유효한 장애물 인덱스 추출
        obs_indices = np.where((proc_ranges < obstacle_dist_limit) & (proc_ranges > 0.01))[0]
        
        # 마스킹을 위한 배열 (True면 장애물 영역)
        mask_indices = np.zeros(len(proc_ranges), dtype=bool)
        
        angle_increment = angles[1] - angles[0] if len(angles) > 1 else 0.01
        
        # 차폭 반경(0.1m) + 여유(0.05m) = 0.15m 반경 (50cm 차선 대응)
        bubble_radius = self.inflation_margin + 0.10
        
        for i in obs_indices:
            dist = proc_ranges[i]
            # 거리에 따른 버블 크기 (각도) 계산: theta = atan(r / d)
            if dist < 0.01: dist = 0.01
            bubble_angle = math.atan2(bubble_radius, dist)
            bubble_idx_count = int(bubble_angle / angle_increment)
            
            start = max(0, i - bubble_idx_count)
            end = min(len(proc_ranges), i + bubble_idx_count + 1)
            mask_indices[start:end] = True
            
        # Bubble 영역을 0으로 설정 (장애물로 간주하여 Gap에서 제외)
        proc_ranges[mask_indices] = 0.0
            
        # 3. Max Gap 찾기
        # 주행 가능한 최소 거리 (Threshold)
        gap_threshold = 1.2  # 1.2m 이상 열린 공간 (더 엄격하게)
        
        mask = proc_ranges > gap_threshold
        
        # Gap이 없으면 Threshold를 낮춰서 다시 시도 (Fallback)
        if not np.any(mask):
            gap_threshold = 0.4
            mask = proc_ranges > gap_threshold
            
        # 연속된 True 구간(Gap) 찾기
        padded_mask = np.concatenate(([False], mask, [False]))
        diff = np.diff(padded_mask.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        if len(starts) == 0:
            # Gap이 없으면 가장 깊은(먼) 곳을 향해 조향 (직진 대신 탈출 시도)
            max_idx = np.argmax(proc_ranges)
            rospy.logwarn_throttle(0.5, "FGM: No Gap found! Heading to max depth (%.2fm, %.1f deg)", 
                                   proc_ranges[max_idx], math.degrees(angles[max_idx]))
            
            self._publish_fgm_debug(header, proc_ranges, angles, None, None, max_idx)
            return angles[max_idx], proc_ranges[max_idx], 0.5
            
        # 가장 긴 Gap 선택
        gap_lengths = ends - starts
        best_gap_idx = np.argmax(gap_lengths)
        
        best_start = starts[best_gap_idx]
        best_end = ends[best_gap_idx]
        
        # [검증] Gap 정보 로그 출력
        best_gap_len = best_end - best_start
        angle_res = angles[1] - angles[0] if len(angles) > 1 else 0.01
        gap_width_deg = math.degrees(best_gap_len * angle_res)
        # 대략적인 미터 단위 폭 (중심 거리 기준)
        center_dist = proc_ranges[(best_start + best_end) // 2]
        gap_width_m = center_dist * math.radians(gap_width_deg)
        
        rospy.loginfo_throttle(0.5, "FGM Status: Found %d gaps. Best Gap: %.1f deg (approx %.2fm)", 
                               len(starts), gap_width_deg, gap_width_m)
        
        # 4. Goal Point 선택
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
        
        # [검증] 처리된 스캔 데이터 발행 (Safety Bubble이 적용된 모습 확인용)
        if self.processed_scan_pub.get_num_connections() > 0 and header is not None:
            debug_scan = LaserScan()
            debug_scan.header = header
            debug_scan.header.frame_id = "laser" # 강제 설정 (필요시)
            # 라이다 좌우 반전 보정: 스캔 데이터 순서를 역순으로 해석하도록 설정
            # Index 0이 원래 Right(-)였으나 실제로는 Left(+)를 가리킴
            debug_scan.angle_min = angles[-1]  # Positive (Left)
            debug_scan.angle_max = angles[0]   # Negative (Right)
            debug_scan.angle_increment = -angle_res # 감소 방향
            debug_scan.range_min = 0.0
            debug_scan.range_max = self.max_range
            debug_scan.ranges = proc_ranges.tolist()
            self.processed_scan_pub.publish(debug_scan)
        
        rospy.logdebug_throttle(0.5, "FGM Enhanced: Gap [%d:%d], Target Angle %.1f deg", 
                               best_start, best_end, math.degrees(target_angle))
        
        self._publish_fgm_debug(header, proc_ranges, angles, best_start, best_end, best_idx)
        
        return target_angle, target_distance, 1.0

    def _publish_fgm_debug(self, header, ranges, angles, start_idx, end_idx, goal_idx):
        if header is None: return
        marker_array = MarkerArray()
        
        # Gap Marker (Green Points)
        gap_marker = Marker()
        gap_marker.header = header
        gap_marker.ns = "fgm_gap"
        gap_marker.id = 0
        if start_idx is not None and end_idx is not None:
            gap_marker.type = Marker.POINTS
            gap_marker.action = Marker.ADD
            gap_marker.scale.x = 0.05
            gap_marker.scale.y = 0.05
            gap_marker.color.r = 0.0
            gap_marker.color.g = 1.0
            gap_marker.color.b = 0.0
            gap_marker.color.a = 0.8
            
            g_ranges = ranges[start_idx:end_idx]
            g_angles = angles[start_idx:end_idx]
            
            for r, a in zip(g_ranges, g_angles):
                if r > 0.1:  # 유효한 거리만 표시
                    p = Point()
                    p.x = float(r * math.cos(a))
                    p.y = float(r * math.sin(a)) # 정방향 시각화
                    gap_marker.points.append(p)
        else:
            gap_marker.action = Marker.DELETE
            
        marker_array.markers.append(gap_marker)

        # Goal Marker (Cyan Sphere)
        goal_marker = Marker()
        goal_marker.header = header
        goal_marker.ns = "fgm_goal"
        goal_marker.id = 1
        if goal_idx is not None:
            goal_marker.type = Marker.SPHERE
            goal_marker.action = Marker.ADD
            goal_marker.scale.x = 0.2
            goal_marker.scale.y = 0.2
            goal_marker.scale.z = 0.2
            goal_marker.color.r = 0.0
            goal_marker.color.g = 1.0
            goal_marker.color.b = 1.0
            goal_marker.color.a = 1.0
            
            r = ranges[goal_idx]
            a = angles[goal_idx]
            goal_marker.pose.position.x = float(r * math.cos(a))
            goal_marker.pose.position.y = float(r * math.sin(a)) # 정방향 시각화
            goal_marker.pose.orientation.w = 1.0
        else:
            goal_marker.action = Marker.DELETE
            
        marker_array.markers.append(goal_marker)
        
        self.gap_marker_pub.publish(marker_array)

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
        모든 포인트를 표시하되, 설정된 거리 이내인 것만 빨간색으로 표시.
        all_points: LaserScan의 모든 포인트 (흰 마커)
        obstacle_points: 장애물로 인식된 포인트 (빨간색 마커)
        """
        marker_array = MarkerArray()
        
        # 전방 30도, 0.7m 이내 장애물만 빨간색으로 표시
        if len(obstacle_points) > 0 and len(all_points) > 0:
            # 모든 포인트의 거리 및 각도 계산
            all_distances = np.linalg.norm(all_points, axis=1)
            all_angles = np.arctan2(all_points[:, 1], all_points[:, 0])
            
            obstacle_detection_range = 0.60  # 0.60m
            
            # 전방 130도(±65도) 필터링
            front_angle_limit = math.radians(130.0) * 0.5
            angle_mask = np.abs(all_angles) <= front_angle_limit
            
            # 거리 및 각도 필터링
            dist_mask = all_distances < obstacle_detection_range
            
            close_mask = dist_mask & angle_mask
            
            # 조건에 맞는 포인트만 빨간색 마커로 표시
            close_points = all_points[close_mask]
            
            # 마커 초기화 (이전 단계에서 삭제된 부분 복구)
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
        # 화살표 방향 보정: 정방향
        display_angle = steering_angle * self.arrow_angle_scale
        
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
            # 경로 방향 보정: 정방향
            display_angle = steering_angle * self.arrow_angle_scale
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

