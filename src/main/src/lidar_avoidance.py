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
        self.inflation_margin = rospy.get_param("~inflation_margin", 0.15)  # 차폭 반경 15cm
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 1.5)
        self.obstacle_threshold = rospy.get_param("~obstacle_threshold", 0.30)  # 30cm부터 장애물 인식
        self.max_drive_speed = rospy.get_param("~max_drive_speed", 0.6)  # m/s (Main과 동일하게 설정)
        self.front_obstacle_angle = math.radians(rospy.get_param("~front_obstacle_angle_deg", 30.0))  # 전방 30도 이내 장애물 감지 각도
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
        self.servo_per_rad = rospy.get_param("~servo_per_rad", 0.28)
        self.min_servo = rospy.get_param("~min_servo", 0.0)
        self.max_servo = rospy.get_param("~max_servo", 1.0)  # 서보 값은 항상 0~1 범위
        self.max_steering_angle = math.radians(
            rospy.get_param("~max_steering_angle_deg", 30.0)
        )
        # 화살표 표시 각도 스케일 (서보 각도보다 작게 표시)
        self.arrow_angle_scale = rospy.get_param("~arrow_angle_scale", 0.7)  # 서보 각도의 70%로 표시

        # 카메라-라이다 퓨전 설정
        self.enable_camera_fusion = rospy.get_param("~enable_camera_fusion", True)
        self.camera_topic = rospy.get_param("~camera_topic", "/usb_cam/image_rect_color")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/usb_cam/camera_info")
        self.lidar_frame = rospy.get_param("~lidar_frame", "laser")
        self.camera_frame = rospy.get_param("~camera_frame", "usb_cam")
        self.lidar_weight = rospy.get_param("~lidar_weight", 0.7)  # 라이다 신뢰도 가중치
        self.camera_weight = rospy.get_param("~camera_weight", 0.3)  # 카메라 신뢰도 가중치
        
        # 카메라 장애물 감지 설정
        self.camera_obstacle_threshold = rospy.get_param("~camera_obstacle_threshold", 0.3)  # 이미지에서 장애물로 판단할 임계값
        self.camera_roi_bottom = rospy.get_param("~camera_roi_bottom", 0.5)  # 이미지 하단 ROI 비율
        self.camera_roi_top = rospy.get_param("~camera_roi_top", 0.8)  # 이미지 상단 ROI 비율
        
        # 카메라 관련 초기화
        if self.enable_camera_fusion:
            self.bridge = CvBridge()
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer)
            self.camera_matrix = None
            self.dist_coeffs = None
            self.camera_info_received = False
            self.latest_image = None
            self.latest_image_time = 0.0
            self.image_timeout = 0.5  # 0.5초 타임아웃

        # ROS I/O
        rospy.Subscriber(
            self.scan_topic,
            LaserScan,
            self.scan_callback,
            queue_size=1,
        )
        if self.enable_camera_fusion:
            rospy.Subscriber(
                self.camera_topic,
                Image,
                self.image_callback,
                queue_size=1,
            )
            rospy.Subscriber(
                self.camera_info_topic,
                CameraInfo,
                self.camera_info_callback,
                queue_size=1,
            )
        # 속도 제어는 main_run.py에서 담당하므로 제거
        # 조향만 제어
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
        camera_confidence = 0.0
        if self.enable_camera_fusion and self.camera_info_received and self.latest_image is not None:
            camera_obstacles, camera_confidence = self._detect_camera_obstacles(
                obstacle_points, scan.header
            )
            # 신뢰도 기반 퓨전
            if camera_confidence > 0.0 or lidar_confidence > 0.0:
                total_confidence = (self.lidar_weight * lidar_confidence + 
                                  self.camera_weight * camera_confidence)
                if total_confidence > 0.5:  # 퓨전 신뢰도 임계값
                    # 카메라에서 추가로 감지한 장애물이 있으면 추가
                    # 사용자 요청: 충돌 시 라이다 우선 -> 카메라 단독 감지 장애물은 무시 (라이다가 못 본 것은 없는 것으로 간주)
                    pass
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

        # 30cm 이상일 때만 장애물이 가장 적은 곳으로 회피 주행
        speed_reduction_start = 0.30  # 30cm
        if closest >= speed_reduction_start:
            # 30cm 이상: 장애물이 가장 적은 곳으로 회피
            (
                target_angle,
                target_distance,
                selected_score,
            ) = self._select_target(ranges, angles, front_obstacle_detected, obstacle_angles)
            
            if target_angle is None:
                # 전방에 경로가 없을 때 정지
                rospy.logwarn_throttle(1.0, "No feasible gap. Stopping vehicle.")
                self._publish_stop(scan.header, reason="no_gap")
                return
        else:
            # 30cm 미만: 속도 감소 중이므로 조향은 중앙 유지 (속도는 main_run.py에서 제어)
            rospy.logwarn_throttle(1.0, "Obstacle too close (%.2fm < %.2fm). Maintaining center steering.", closest, speed_reduction_start)
            target_angle = 0.0  # 중앙 유지
            target_distance = 0.0
            selected_score = 0.0

        # 전방 30도 이내 장애물이 없으면 회피 기동 계속
        emergency_stop = False

        # 조향각 계산 (속도는 main_run.py에서 제어)
        steering_angle = clamp(target_angle, -self.max_steering_angle, self.max_steering_angle)
        servo_cmd = self.servo_center + self.servo_per_rad * steering_angle
        servo_cmd = clamp(servo_cmd, self.min_servo, self.max_servo)

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

    def image_callback(self, msg: Image) -> None:
        """카메라 이미지 콜백"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image_time = rospy.get_time()
        except Exception as exc:
            rospy.logwarn_throttle(1.0, f"Failed to convert camera image: {exc}")

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """카메라 캘리브레이션 정보 콜백"""
        if not self.camera_info_received:
            self.camera_matrix = np.array(msg.K, dtype=np.float64).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D, dtype=np.float64)
            self.camera_info_received = True
            rospy.loginfo("Camera calibration info received")

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
        
        # 3-2. 거리 필터링 (30cm 이내)
        obstacle_detection_range = 0.3  # 30cm
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
            # 포인트가 적어도 0.3m 이내 장애물이면 모두 인식 (벽 등 큰 장애물도 인식)
            distances = np.linalg.norm(points, axis=1)
            min_dist = float(np.min(distances))
            obstacle_detection_range = 0.3  # 0.3m
            if min_dist < obstacle_detection_range:  # 0.3m 이내는 클러스터링 없이도 인식
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
            obstacle_detection_range = 0.3  # 0.3m
            if min_dist < obstacle_detection_range:
                # 0.3m 이내 장애물이면 클러스터링 없이도 모두 반환
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

    def _project_lidar_to_camera(self, lidar_points: np.ndarray, header) -> Optional[np.ndarray]:
        """
        라이다 포인트를 카메라 이미지 평면에 투영.
        Returns: (N, 2) 형태의 이미지 픽셀 좌표 배열, None if failed
        """
        if not self.camera_info_received or self.latest_image is None:
            return None
        
        try:
            # 라이다 좌표계를 카메라 좌표계로 변환
            transform = self.tf_buffer.lookup_transform(
                self.camera_frame,
                self.lidar_frame,
                header.stamp,
                rospy.Duration(0.1)
            )
            
            # 변환 행렬 구성
            t = transform.transform.translation
            q = transform.transform.rotation
            # 쿼터니언을 회전 행렬로 변환
            qx, qy, qz, qw = q.x, q.y, q.z, q.w
            R = np.array([
                [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
                [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
                [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
            ])
            T = np.array([t.x, t.y, t.z])
            
            # 라이다 포인트를 카메라 좌표계로 변환
            # lidar_points는 (N, 2) 형태 [x, y], z=0 가정
            if len(lidar_points) == 0:
                return None
            
            lidar_3d = np.zeros((len(lidar_points), 3))
            lidar_3d[:, 0] = lidar_points[:, 0]  # x
            lidar_3d[:, 1] = lidar_points[:, 1]  # y
            lidar_3d[:, 2] = 0.0  # z (지면 높이)
            
            # 좌표 변환
            camera_points = (R @ lidar_3d.T).T + T
            
            # 카메라 앞쪽(z > 0)인 포인트만 선택
            valid = camera_points[:, 2] > 0.1
            if not np.any(valid):
                return None
            
            camera_points = camera_points[valid]
            
            # 카메라 내부 파라미터로 이미지 평면에 투영
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]
            
            # 정규화된 좌표
            x_norm = camera_points[:, 0] / camera_points[:, 2]
            y_norm = camera_points[:, 1] / camera_points[:, 2]
            
            # 이미지 픽셀 좌표
            u = fx * x_norm + cx
            v = fy * y_norm + cy
            
            image_points = np.stack([u, v], axis=1)
            
            # 이미지 범위 내 포인트만 반환
            h, w = self.latest_image.shape[:2]
            in_bounds = (image_points[:, 0] >= 0) & (image_points[:, 0] < w) & \
                       (image_points[:, 1] >= 0) & (image_points[:, 1] < h)
            
            if not np.any(in_bounds):
                return None
            
            return image_points[in_bounds]
            
        except Exception as exc:
            rospy.logwarn_throttle(1.0, f"Failed to project lidar to camera: {exc}")
            return None

    def _detect_camera_obstacles(self, lidar_points: np.ndarray, header) -> Tuple[np.ndarray, float]:
        """
        카메라 이미지에서 장애물 감지 및 라이다 포인트와 매칭.
        Returns: (추가 장애물 포인트, 카메라 신뢰도)
        """
        if self.latest_image is None or not self.camera_info_received:
            return np.zeros((0, 2), dtype=np.float32), 0.0
        
        # 이미지 타임아웃 확인
        if rospy.get_time() - self.latest_image_time > self.image_timeout:
            return np.zeros((0, 2), dtype=np.float32), 0.0
        
        try:
            h, w = self.latest_image.shape[:2]
            
            # ROI 설정 (하단 부분만)
            roi_y_start = int(h * self.camera_roi_bottom)
            roi_y_end = int(h * self.camera_roi_top)
            roi = self.latest_image[roi_y_start:roi_y_end, :]
            
            # 간단한 장애물 감지: 어두운 영역 또는 특정 색상 감지
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # 어두운 영역 감지 (장애물 후보)
            _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
            
            # 모폴로지 연산으로 노이즈 제거
            kernel = np.ones((5, 5), np.uint8)
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 라이다 포인트를 카메라 이미지에 투영
            projected_points = self._project_lidar_to_camera(lidar_points, header)
            
            camera_obstacles = []
            confidence_sum = 0.0
            valid_detections = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100:  # 너무 작은 영역은 무시
                    continue
                
                # 컨투어 중심
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"]) + roi_y_start  # 전체 이미지 좌표로 변환
                
                # 라이다 포인트와 매칭 확인
                matched = False
                if projected_points is not None:
                    distances = np.linalg.norm(projected_points - np.array([cx, cy]), axis=1)
                    if np.any(distances < 50):  # 50픽셀 이내면 매칭됨
                        matched = True
                
                # 매칭되지 않은 새로운 장애물 감지
                if not matched:
                    # 이미지 좌표를 라이다 좌표계로 역투영 (간단한 근사)
                    # 실제로는 카메라-라이다 변환의 역변환이 필요하지만, 여기서는 간단히 처리
                    # ROI 하단 중심부의 장애물을 전방으로 가정
                    obstacle_x = -0.5  # 전방 0.5m 가정
                    obstacle_y = (cx - w/2) * 0.001  # 픽셀을 미터로 변환 (근사)
                    camera_obstacles.append([obstacle_x, obstacle_y])
                    
                    # 신뢰도 계산 (면적 기반)
                    confidence = min(1.0, area / 5000.0)  # 최대 5000 픽셀 면적에서 신뢰도 1.0
                    confidence_sum += confidence
                    valid_detections += 1
            
            # 평균 신뢰도
            camera_confidence = confidence_sum / max(1, valid_detections) if valid_detections > 0 else 0.0
            
            if len(camera_obstacles) > 0:
                return np.array(camera_obstacles, dtype=np.float32), camera_confidence
            else:
                return np.zeros((0, 2), dtype=np.float32), camera_confidence
                
        except Exception as exc:
            rospy.logwarn_throttle(1.0, f"Camera obstacle detection failed: {exc}")
            return np.zeros((0, 2), dtype=np.float32), 0.0

    def _select_target(
        self, ranges: np.ndarray, angles: np.ndarray, front_obstacle_detected: bool = False,
        obstacle_angles: Optional[np.ndarray] = None
    ) -> Tuple[Optional[float], float, float]:
        clearance = np.clip(ranges - self.inflation_margin, 0.0, self.max_range)
        safe_mask = clearance > self.safe_distance
        if not np.any(safe_mask):
            return None, 0.0, 0.0

        norm_clearance = clearance / self.max_range
        
        # 각 방향의 장애물 밀도 계산 (장애물이 가장 적은 방향 선택)
        obstacle_density = np.zeros_like(angles, dtype=np.float32)
        if obstacle_angles is not None and len(obstacle_angles) > 0:
            # 각 경로 각도에 대해 주변 장애물 개수 계산
            angle_window = math.radians(15.0)  # 각 방향에서 ±15도 범위의 장애물 개수 계산
            
            for i, angle in enumerate(angles):
                # 각도 차이 계산
                angle_diffs = np.abs(obstacle_angles - angle)
                # 각도 차이를 [-π, π] 범위로 정규화
                angle_diffs = np.minimum(angle_diffs, 2 * math.pi - angle_diffs)
                # 주변 각도 범위 내 장애물 개수
                nearby_obstacles = np.sum(angle_diffs < angle_window)
                # 장애물 밀도: 장애물이 많을수록 값이 큼 (0~1 정규화)
                # 최대 장애물 개수를 10개로 가정하고 정규화
                obstacle_density[i] = min(1.0, nearby_obstacles / 10.0)
        else:
            # 장애물이 없으면 모든 방향의 밀도가 0
            obstacle_density[:] = 0.0
        
        # 장애물이 가장 적은 방향을 선호 (밀도가 낮을수록 점수 높음)
        # obstacle_density를 역으로 변환: 밀도가 낮을수록 높은 점수
        obstacle_avoidance_score = 1.0 - obstacle_density  # 장애물이 적을수록 1에 가까움
        
        # 전방 장애물이 감지되면 장애물 회피를 최우선으로
        if front_obstacle_detected:
            # 장애물 회피를 최우선으로 하되, 안전 거리도 고려
            effective_clearance_weight = 0.3  # 안전 거리 가중치 감소
            effective_heading_weight = 0.1   # 방향 선호도 감소
            effective_obstacle_weight = 0.6  # 장애물 회피 가중치 증가
        else:
            effective_clearance_weight = self.clearance_weight
            effective_heading_weight = self.heading_weight
            effective_obstacle_weight = 0.3  # 장애물 회피 가중치
        
        heading_pref = 1.0 - (np.abs(angles) / (self.forward_fov * 0.5))
        heading_pref = np.clip(heading_pref, 0.0, 1.0)
        
        # 점수 계산: 장애물이 적은 방향을 최우선으로 선택
        scores = (
            effective_clearance_weight * norm_clearance
            + effective_heading_weight * heading_pref
            + effective_obstacle_weight * obstacle_avoidance_score  # 장애물이 적을수록 높은 점수
        )
        
        scores[~safe_mask] = -np.inf
        idx = int(np.argmax(scores))
        if not np.isfinite(scores[idx]):
            return None, 0.0, 0.0

        target_angle = float(angles[idx])
        # 라이다 좌표계 180도 회전: 각도에 π 더하기
        target_angle = target_angle + math.pi
        # 각도를 [-π, π] 범위로 정규화
        if target_angle > math.pi:
            target_angle -= 2 * math.pi
        elif target_angle < -math.pi:
            target_angle += 2 * math.pi
        target_distance = float(min(ranges[idx], self.lookahead_distance))
        
        # 선택된 방향의 장애물 밀도 로그
        selected_density = obstacle_density[idx]
        rospy.logdebug_throttle(1.0, "Selected angle: %.1f deg, obstacle_density: %.2f, clearance: %.2f", 
                               math.degrees(target_angle), selected_density, clearance[idx])
        
        return target_angle, target_distance, float(scores[idx])

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
            obstacle_detection_range = 0.3  # 0.3m
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

