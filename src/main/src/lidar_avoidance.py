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
        self.safe_distance = rospy.get_param("~safe_distance", 0.4)  # 50cm 안전 거리
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
        
        # 속도 스무딩 설정
        self.current_speed = 0.0  # 현재 속도 (부드러운 전환용)
        self.current_pwm = 0.0  # 현재 PWM (부드러운 전환용)
        self.speed_smoothing_rate = rospy.get_param("~speed_smoothing_rate", 2.0)  # 속도 변화율 (m/s^2)
        self.pwm_smoothing_rate = rospy.get_param("~pwm_smoothing_rate", 500.0)  # PWM 변화율 (per second)
        
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

        # 카메라 퓨전 적용
        obstacle_points, lidar_confidence = self._collect_obstacle_points(ranges, angles, scan.header)
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
                    if len(camera_obstacles) > 0:
                        obstacle_points = np.vstack([obstacle_points, camera_obstacles]) if len(obstacle_points) > 0 else camera_obstacles
        
        self._publish_obstacle_markers(scan.header, obstacle_points)

        # 전방 30도 이내 장애물 감지 확인 (경고만, 정지하지 않음)
        front_obstacle_detected = self._check_front_obstacle(ranges, angles)
        if front_obstacle_detected:
            rospy.logwarn_throttle(1.0, "Obstacle detected in front 30deg. Attempting to avoid by turning left/right.")

        # 장애물 포인트를 각도로 변환 (회피를 위해)
        obstacle_angles = None
        if len(obstacle_points) > 0:
            rospy.loginfo_throttle(1.0, "Detected %d obstacle points", len(obstacle_points))
            # 장애물 포인트의 각도 계산 (라이다 좌표계 기준)
            obstacle_angles = np.arctan2(obstacle_points[:, 1], obstacle_points[:, 0])
            # 라이다 좌표계 180도 회전 보정: 각도에서 π 빼기
            obstacle_angles = obstacle_angles - math.pi
            # 각도를 [-π, π] 범위로 정규화
            obstacle_angles = np.where(obstacle_angles > math.pi, obstacle_angles - 2 * math.pi, obstacle_angles)
            obstacle_angles = np.where(obstacle_angles < -math.pi, obstacle_angles + 2 * math.pi, obstacle_angles)
        else:
            rospy.logdebug_throttle(2.0, "No obstacle points detected")

        # 전방 장애물이 있어도 좌우로 회피 경로를 찾음
        (
            target_angle,
            target_distance,
            selected_score,
        ) = self._select_target(ranges, angles, front_obstacle_detected, obstacle_angles)

        # 전방 30도 이내 장애물이 없으면 회피 기동 계속
        emergency_stop = False
        if target_angle is None:
            # 전방에 경로가 없을 때 정지
            rospy.logwarn_throttle(1.0, "No feasible gap. Stopping vehicle.")
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
    ) -> Tuple[np.ndarray, float]:
        # 1단계: 라이다 좌표계에서 카테시안 좌표로 변환
        # 라이다 좌표계가 180도 회전되어 있다고 가정:
        # - 라이다의 0도가 실제로는 후방을 가리킴
        # - 따라서 각도에 π를 더하거나 좌표를 반전해야 함
        # 좌표 변환: x = -r * cos(angle), y = -r * sin(angle)
        # 이렇게 하면 라이다의 0도 방향이 실제 전방(x < 0)이 됨
        xy = np.zeros((len(ranges), 2), dtype=np.float32)
        xy[:, 0] = -ranges * np.cos(angles)  # 전방이 음수 x
        xy[:, 1] = -ranges * np.sin(angles)  # 좌측이 음수 y
        
        # 2단계: 전방 포인트만 선택 (x < 0이 전방, 즉 라이다의 0도 방향)
        # 하지만 실제로는 라이다의 각도 범위를 확인해야 함
        # 일단 모든 포인트를 사용하되, 전방 FOV 내의 포인트만 장애물로 인식
        # 전방 FOV는 이미 _prepare_scan에서 필터링되었으므로 여기서는 모든 포인트가 전방
        
        # 3단계: 전방의 모든 포인트를 장애물로 인식 (거리 제한 없음)
        # 벽 등 멀리 있는 장애물도 인식하기 위해 거리 제한 제거
        obstacle_points = xy  # 모든 포인트를 장애물로 인식
        front_count = len(obstacle_points)
        
        if len(obstacle_points) == 0:
            rospy.logdebug_throttle(2.0, "No obstacle points found")
            return np.zeros((0, 2), dtype=np.float32), 0.0
        
        # 4단계: 가까운 장애물 우선 인식 (선택적 필터링)
        # 가까운 장애물이 있으면 그것을 우선하되, 멀리 있는 장애물도 포함
        distances = np.linalg.norm(obstacle_points, axis=1)
        min_dist = float(np.min(distances))
        
        # 가까운 장애물이 있으면 로그 출력
        if min_dist < self.obstacle_threshold:
            close_count = np.sum(distances < self.obstacle_threshold)
            rospy.logdebug_throttle(2.0, "Found %d close obstacles (< %.2fm) out of %d total points, min_dist=%.2fm", 
                                    close_count, self.obstacle_threshold, front_count, min_dist)
        else:
            rospy.logdebug_throttle(2.0, "No close obstacles, but %d front points detected as obstacles, min_dist=%.2fm", 
                                    front_count, min_dist)
        
        # 4단계: 클러스터링으로 노이즈 제거 (선택적)
        # 가까운 장애물이 많으면 클러스터링 적용, 적으면 그대로 사용
        if len(obstacle_points) > 10:  # 포인트가 많을 때만 클러스터링
            clustered_points = self._cluster_obstacle_points(obstacle_points)
            if len(clustered_points) > 0:
                obstacle_points = clustered_points
                rospy.logdebug_throttle(2.0, "Clustered to %d points", len(clustered_points))
        
        # 라이다 신뢰도 계산 (거리 기반: 가까울수록 높은 신뢰도)
        if len(obstacle_points) > 0:
            distances = np.linalg.norm(obstacle_points, axis=1)
            min_dist = float(np.min(distances))
            # 0.3m 이내: 신뢰도 1.0, 0.6m: 신뢰도 0.5, 그 이상: 신뢰도 감소
            lidar_confidence = max(0.0, min(1.0, 1.0 - (min_dist - 0.3) / 0.3))
            rospy.logdebug_throttle(2.0, "Obstacle detection: %d points, closest: %.2fm, confidence: %.2f", 
                                    len(obstacle_points), min_dist, lidar_confidence)
        else:
            lidar_confidence = 0.0
        
        return obstacle_points, lidar_confidence
    
    def _cluster_obstacle_points(self, points: np.ndarray) -> np.ndarray:
        """
        간단한 거리 기반 클러스터링으로 연속된 포인트만 장애물로 인식.
        최소 포인트 수 조건을 만족하는 클러스터만 반환.
        """
        if len(points) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        
        # 포인트가 적으면 클러스터링 없이 모두 반환 (가까운 장애물은 작은 포인트 수로도 인식)
        if len(points) < self.min_obstacle_points:
            # 포인트가 적어도 매우 가까운 장애물이면 인식
            distances = np.linalg.norm(points, axis=1)
            min_dist = float(np.min(distances))
            if min_dist < 0.5:  # 50cm 이내는 클러스터링 없이도 인식
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
            
            if nearby_count >= required_points:
                valid_points.append(p1)
        
        if len(valid_points) == 0:
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
        
        # 전방 장애물이 감지되면 좌우 회피를 더 선호하도록 가중치 조정
        if front_obstacle_detected:
            # 좌우 회피를 위해 clearance_weight를 높이고 heading_weight를 낮춤
            effective_clearance_weight = 0.8  # 장애물 회피 우선
            effective_heading_weight = 0.2   # 방향 선호도 감소
        else:
            effective_clearance_weight = self.clearance_weight
            effective_heading_weight = self.heading_weight
        
        heading_pref = 1.0 - (np.abs(angles) / (self.forward_fov * 0.5))
        heading_pref = np.clip(heading_pref, 0.0, 1.0)
        
        # 장애물 회피 페널티 계산
        obstacle_penalty = np.ones_like(angles, dtype=np.float32)
        if obstacle_angles is not None and len(obstacle_angles) > 0:
            # 각 각도에 대해 가장 가까운 장애물까지의 각도 거리 계산
            for i, angle in enumerate(angles):
                # 각도 차이 계산 (최소 각도 차이)
                angle_diffs = np.abs(obstacle_angles - angle)
                # 각도 차이를 [-π, π] 범위로 정규화
                angle_diffs = np.minimum(angle_diffs, 2 * math.pi - angle_diffs)
                min_obstacle_angle_diff = float(np.min(angle_diffs))
                
                # 장애물로부터의 각도 거리에 따라 페널티 적용
                # 30도 이내: 강한 페널티 (점수 크게 감소)
                # 30도 ~ 60도: 중간 페널티
                # 60도 이상: 페널티 없음
                if min_obstacle_angle_diff < math.radians(30.0):
                    # 30도 이내: 점수를 0.1로 감소
                    obstacle_penalty[i] = 0.1
                elif min_obstacle_angle_diff < math.radians(60.0):
                    # 30도 ~ 60도: 점수를 0.3으로 감소
                    obstacle_penalty[i] = 0.3
                else:
                    # 60도 이상: 페널티 없음
                    obstacle_penalty[i] = 1.0
        
        scores = (
            effective_clearance_weight * norm_clearance
            + effective_heading_weight * heading_pref
        ) * obstacle_penalty  # 장애물 회피 페널티 적용
        
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
        current_time = rospy.get_time()
        dt = max(0.01, min(0.1, current_time - self.last_scan_time))  # 10ms ~ 100ms
        
        # 속도 결정 (전진만)
        if emergency_stop:
            target_speed = 0.0
            target_pwm = 0.0
        else:
            # 음수 속도는 허용하지 않음 (후진 제거)
            target_speed = max(0.0, drive_speed)
            target_pwm = max(0.0, pwm)
        
        # 점진적 속도 변화
        max_speed_change = self.speed_smoothing_rate * dt
        speed_diff = target_speed - self.current_speed
        if abs(speed_diff) > max_speed_change:
            self.current_speed += math.copysign(max_speed_change, speed_diff)
        else:
            self.current_speed = target_speed
        
        # 점진적 PWM 변화
        max_pwm_change = self.pwm_smoothing_rate * dt
        pwm_diff = target_pwm - self.current_pwm
        if abs(pwm_diff) > max_pwm_change:
            self.current_pwm += math.copysign(max_pwm_change, pwm_diff)
        else:
            self.current_pwm = target_pwm
        
        # 명령 발행
        if self.publish_ackermann and self.ackermann_pub is not None:
            ack_msg = AckermannDriveStamped()
            ack_msg.header = header
            ack_msg.drive.steering_angle = steering_angle
            ack_msg.drive.speed = self.current_speed
            self.ackermann_pub.publish(ack_msg)

        if self.publish_direct_controls and self.speed_pub and self.steering_pub:
            speed_msg = Float64()
            speed_msg.data = self.current_pwm
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

