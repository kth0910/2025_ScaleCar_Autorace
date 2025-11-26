#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import os
import math

import rospy
import cv2
import numpy as np

from std_msgs.msg import Float64, Bool
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge

from warper import Warper
from slidewindow import SlideWindow


class LaneFollower:
    """
    카메라 영상에서 차선을 감지하고 조향/속도를 직접 퍼블리시하는 단순 주행 노드
    """

    TRACKBAR_WINDOW = "Lane Threshold Tuner"

    def __init__(self):
        self.bridge = CvBridge()
        self.warper = Warper()
        self.slidewindow = SlideWindow()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # 파라미터
        self.desired_center = rospy.get_param("~desired_center", 280.0)
        self.pid_kp = rospy.get_param("~steering_kp", -0.0025)
        self.pid_ki = rospy.get_param("~steering_ki", -0.00004)
        self.pid_kd = rospy.get_param("~steering_kd", -0.00045)
        self.steering_gain = self.pid_kp  # backward compatibility
        self.steering_offset = rospy.get_param("~steering_offset", 0.50)  # 중앙 정렬 (0.60 → 0.50)
        self.steering_smoothing = rospy.get_param("~steering_smoothing", 0.55)
        self.steering_smoothing_left = rospy.get_param("~steering_smoothing_left", 0.40)  # 좌측 조향 시 더 빠른 반응
        self.steering_smoothing_right = rospy.get_param("~steering_smoothing_right", 0.55)  # 우측 조향 시 기존 유지
        self.min_servo = rospy.get_param("~min_servo", 0.05)
        self.max_servo = rospy.get_param("~max_servo", 0.95)
        self.center_smoothing = rospy.get_param("~center_smoothing", 0.4)
        self.max_center_step = rospy.get_param("~max_center_step", 25.0)
        self.bias_correction_gain = rospy.get_param("~bias_correction_gain", 1e-4)
        self.max_error_bias = rospy.get_param("~max_error_bias", 120.0)
        self.error_bias = rospy.get_param("~initial_error_bias", 0.0)
        self.max_servo_delta = rospy.get_param("~max_servo_delta", 0.035)
        self.max_servo_delta_left = rospy.get_param("~max_servo_delta_left", 0.05)  # 좌측 조향 시 더 큰 변화 허용
        self.max_servo_delta_right = rospy.get_param("~max_servo_delta_right", 0.035)  # 우측 조향 시 기존 유지
        self.min_mask_pixels = rospy.get_param("~min_mask_pixels", 600)
        self.integral_limit = rospy.get_param("~steering_integral_limit", 500.0)
        self.single_left_ratio = rospy.get_param("~single_left_ratio", 0.40)  # 좌측 조향 개선 (0.35 → 0.40)
        self.single_right_ratio = rospy.get_param("~single_right_ratio", 0.65)
        # 노란 차선 검출 파라미터 (밝기 기반 필터링 없이 색상만 사용)
        self.use_yellow_lane_detection = rospy.get_param("~use_yellow_lane_detection", True)
        self.yellow_hsv_low = np.array(
            rospy.get_param("~yellow_hsv_low", [15, 50, 50]),
            dtype=np.uint8,
        )
        self.yellow_hsv_high = np.array(
            rospy.get_param("~yellow_hsv_high", [40, 255, 255]),
            dtype=np.uint8,
        )
        self.yellow_kernel_size = rospy.get_param("~yellow_kernel_size", 5)

        # 출력 토픽 (기본: 보간기 입력 토픽으로 발행해 단일 퍼블리셔로 정리)
        self.use_unsmoothed_topics = rospy.get_param("~use_unsmoothed_topics", False)
        default_speed_topic = (
            "/commands/motor/unsmoothed_speed"
            if self.use_unsmoothed_topics
            else "/commands/motor/speed"
        )
        default_servo_topic = (
            "/commands/servo/unsmoothed_position"
            if self.use_unsmoothed_topics
            else "/commands/servo/position"
        )
        self.speed_topic = rospy.get_param("~motor_topic", default_speed_topic)
        self.servo_topic = rospy.get_param("~servo_topic", default_servo_topic)
        if self.use_unsmoothed_topics:
            rospy.loginfo(
                "LaneFollower using unsmoothed outputs (%s, %s) to avoid conflicting motor publishers.",
                self.speed_topic,
                self.servo_topic,
            )

        # 퍼블리셔
        self.speed_pub = rospy.Publisher(self.speed_topic, Float64, queue_size=1)
        self.steering_pub = rospy.Publisher(self.servo_topic, Float64, queue_size=1)
        # 보간기가 없을 때를 위한 자동 폴백 (직접 토픽도 동시에 준비)
        self.speed_fallback_topic = rospy.get_param(
            "~fallback_motor_topic", "/commands/motor/speed"
        )
        self.servo_fallback_topic = rospy.get_param(
            "~fallback_servo_topic", "/commands/servo/position"
        )
        self.speed_fallback_pub = None
        self.servo_fallback_pub = None
        if self.speed_topic != self.speed_fallback_topic:
            self.speed_fallback_pub = rospy.Publisher(
                self.speed_fallback_topic, Float64, queue_size=1
            )
        if self.servo_topic != self.servo_fallback_topic:
            self.servo_fallback_pub = rospy.Publisher(
                self.servo_fallback_topic, Float64, queue_size=1
            )
        self._speed_fallback_warned = False
        self._servo_fallback_warned = False
        self.center_pub = rospy.Publisher("/lane_center_x", Float64, queue_size=1)
        self.error_pub = rospy.Publisher("/lane_error", Float64, queue_size=1)

        rospy.Subscriber("usb_cam/image_rect_color", Image, self.image_callback, queue_size=1)
        
        # 라이다 회피 우선: lidar_avoidance가 제어 중인지 확인
        # lidar_avoidance가 제어 중인지 나타내는 토픽 구독
        self.lidar_controlling = False
        self.last_lidar_servo_time = 0.0
        self.lidar_timeout = 0.5  # 0.5초 동안 라이다 제어가 없으면 카메라 제어 사용
        self.last_servo_publish_time = 0.0  # 자신이 발행한 시간 추적

        self.enable_viz = rospy.get_param("~enable_viz", True)
        if not self.enable_viz:
            rospy.logwarn("Visualization disabled via parameter.")

        self.auto_threshold = rospy.get_param("~auto_threshold", not self.enable_viz)
        if self.auto_threshold:
            rospy.loginfo("LaneFollower using auto color thresholds.")
        elif not self.enable_viz:
            rospy.logwarn("Manual thresholds requested but visualization disabled; using ROS params.")

        self.prev_servo = self.steering_offset
        self.current_center = self.desired_center
        self.prev_center = self.desired_center
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.prev_time = rospy.get_time()
        self.initialized = False
        
        # 라이다 제어 감지: lidar_avoidance가 /ackermann_cmd를 발행하면 라이다 제어 중
        rospy.Subscriber("/ackermann_cmd", AckermannDriveStamped, self._lidar_ackermann_callback, queue_size=1)
        
        # 속도 제어 파라미터
        self.max_drive_speed = rospy.get_param("~max_drive_speed", 0.6)  # m/s
        self.min_drive_speed = rospy.get_param("~min_drive_speed", 0.15)  # m/s
        self.max_pwm = rospy.get_param("~max_pwm", 3000.0)  # ERPM (3000 ~ 0.65m/s)
        self.min_pwm = rospy.get_param("~min_pwm", 0.0)  # ERPM (0 = Stop)
        self.speed_reduction_start = rospy.get_param("~speed_reduction_start", 0.30)  # 30cm부터 속도 감소
        self.hard_stop_distance = rospy.get_param("~hard_stop_distance", 0.15)  # 15cm에서 완전 정지
        speed_pwm_param = rospy.get_param("~speed", 2000.0)
        self.speed_smoothing_rate = rospy.get_param("~speed_smoothing_rate", 100.0)  # PWM 변화율 (부드러운 변화를 위해 감소)
        self.speed_smoothing_factor = rospy.get_param("~speed_smoothing_factor", 0.3)  # 지수적 스무딩 계수 (0.0~1.0, 작을수록 더 부드러움)

        # 라이다 속도 명령 구독 (통합 제어용)
        self.lidar_target_speed = self.max_drive_speed  # 초기값은 최대 속도
        self.last_lidar_command_time = 0.0
        rospy.Subscriber("lidar_avoidance/target_speed", Float64, self._lidar_speed_callback, queue_size=1)
        # 색상 기반 속도 제어 파라미터
        self.neutral_lane_speed = rospy.get_param(
            "~neutral_lane_speed",
            self._pwm_to_drive_speed(speed_pwm_param)
        )
        self.red_lane_speed = rospy.get_param("~red_lane_speed", 0.2)
        self.blue_lane_speed = rospy.get_param("~blue_lane_speed", 0.6)
        self.color_roi_height_ratio = rospy.get_param("~color_roi_height_ratio", 0.20)  # 하단 20%만 사용
        self.color_detection_ratio = rospy.get_param("~color_detection_ratio", 0.02)  # 검출 임계값 (2%로 낮춤)
        self.current_color_speed_mps = self._clamp(
            self.neutral_lane_speed, self.min_drive_speed, self.max_drive_speed
        )
        base_color_pwm = self._drive_speed_to_pwm(self.current_color_speed_mps)
        self.speed_value = base_color_pwm
        self.current_speed_pwm = base_color_pwm
        self.last_detected_color = "none"

        self.smoothed_servo = self.steering_offset
        self.last_image_time = 0.0

        # 제어 루프 타이머 (30Hz)
        self.timer = rospy.Timer(rospy.Duration(0.033), self.timer_callback)

        rospy.on_shutdown(self._cleanup)
        rospy.loginfo("LaneFollower initialized.")

    def timer_callback(self, event):
        """주기적인 제어 명령 발행 (카메라 수신 여부와 무관하게 동작)"""
        # 1. 속도 제어 (통합)
        target_pwm = self._compute_integrated_speed()
        self._publish_speed_command(target_pwm)

        # 2. 조향 제어 (라이다 우선권 확인)
        current_time = rospy.get_time()
        is_lidar_controlling = (current_time - self.last_lidar_servo_time < self.lidar_timeout)

        if not is_lidar_controlling:
            # 라이다가 제어하지 않을 때만 카메라 차선 추종 제어 발행
            # 단, 카메라 데이터가 최신일 때만 (0.5초 이내)
            if (current_time - self.last_image_time) < 0.5:
                self.last_servo_publish_time = current_time
                self._publish_servo_command(self.smoothed_servo)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            rospy.logwarn(f"Failed to convert image: {exc}")
            return

        if self.enable_viz and not self.initialized:
            self._setup_trackbars()
            self.initialized = True

        lane_mask = self._create_lane_mask(frame)
        slide_img, center_x, has_lane = self._run_slidewindow(lane_mask)

        if center_x is not None:
            blended = (
                self.center_smoothing * self.prev_center
                + (1.0 - self.center_smoothing) * center_x
            )
            delta = blended - self.prev_center
            delta = self._clamp(delta, -self.max_center_step, self.max_center_step)
            filtered_center = self.prev_center + delta
        else:
            filtered_center = self.prev_center
        self.prev_center = filtered_center
        self.current_center = filtered_center
        detected_color = self._detect_bottom_lane_color(frame)
        self._update_speed_profile_from_color(detected_color)

        left_exists = getattr(self.slidewindow, "left_exists", False)
        right_exists = getattr(self.slidewindow, "right_exists", False)
        left_x = getattr(self.slidewindow, "left_x", None)
        right_x = getattr(self.slidewindow, "right_x", None)
        frame_width = getattr(self.slidewindow, "frame_width", self.desired_center * 2)
        lane_detected = left_exists or right_exists

        self.center_pub.publish(Float64(self.current_center))
        if left_exists and right_exists:
            raw_error = self.desired_center - self.current_center
            self.error_bias += self.bias_correction_gain * raw_error
            self.error_bias = self._clamp(
                self.error_bias, -self.max_error_bias, self.max_error_bias
            )
            error = raw_error - self.error_bias
        elif left_exists:
            target = self.single_left_ratio * frame_width
            raw_error = target - left_x if left_x is not None else 0.0
            error = raw_error
        elif right_exists:
            target = self.single_right_ratio * frame_width
            raw_error = target - right_x if right_x is not None else 0.0
            error = raw_error
        else:
            raw_error = 0.0
            error = 0.0
        self.error_pub.publish(Float64(raw_error))

        current_time = rospy.get_time()
        dt = max(1e-3, current_time - self.prev_time)
        self.prev_time = current_time

        if lane_detected:
            self.integral_error += error * dt
            self.integral_error = self._clamp(self.integral_error, -self.integral_limit, self.integral_limit)
            derivative = (error - self.prev_error) / dt
            control = (
                self.pid_kp * error +
                self.pid_ki * self.integral_error +
                self.pid_kd * derivative
            )
            desired_servo = self.steering_offset + control
        else:
            self.integral_error *= 0.9
            derivative = 0.0
            desired_servo = self.steering_offset

        self.prev_error = error
        desired_servo = self._clamp(desired_servo, self.min_servo, self.max_servo)

        if lane_detected:
            delta_servo = desired_servo - self.prev_servo
            # 좌측 조향(양수)과 우측 조향(음수)에 대해 비대칭 제한 적용
            if delta_servo > 0:
                # 좌측 조향: 더 큰 변화 허용
                delta_servo = self._clamp(delta_servo, 0, self.max_servo_delta_left)
            else:
                # 우측 조향: 기존 제한 유지
                delta_servo = self._clamp(delta_servo, -self.max_servo_delta_right, 0)
            limited_target = self.prev_servo + delta_servo
        else:
            # 차선이 검출되지 않으면 기존 방향 유지
            limited_target = self.prev_servo

        # 좌측 조향과 우측 조향에 대해 비대칭 smoothing 적용
        if limited_target > self.prev_servo:
            # 좌측 조향: 더 빠른 반응 (smoothing 값이 작을수록 빠름)
            smoothing_factor = self.steering_smoothing_left
        else:
            # 우측 조향: 기존 smoothing 유지
            smoothing_factor = self.steering_smoothing_right
        
        smoothed_servo = (
            smoothing_factor * self.prev_servo
            + (1.0 - smoothing_factor) * limited_target
        )
        self.smoothed_servo = smoothed_servo
        self.prev_servo = smoothed_servo
        self.last_image_time = rospy.get_time()

        if self.enable_viz:
            cv2.imshow("Lane Frame", frame)
            cv2.imshow("Lane Mask", lane_mask)
            if slide_img is not None:
                cv2.imshow("Sliding Window", slide_img)
            cv2.waitKey(1)

    def _create_lane_mask(self, frame):
        if self.use_yellow_lane_detection:
            return self._yellow_lane_mask(frame)
        if self.auto_threshold and not self.enable_viz:
            return self._auto_lane_mask(frame)
        return self._manual_lane_mask(frame)

    def _manual_lane_mask(self, frame):
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        if self.enable_viz:
            low_H = cv2.getTrackbarPos("low_H", self.TRACKBAR_WINDOW)
            low_L = cv2.getTrackbarPos("low_L", self.TRACKBAR_WINDOW)
            low_S = cv2.getTrackbarPos("low_S", self.TRACKBAR_WINDOW)
            high_H = cv2.getTrackbarPos("high_H", self.TRACKBAR_WINDOW)
            high_L = cv2.getTrackbarPos("high_L", self.TRACKBAR_WINDOW)
            high_S = cv2.getTrackbarPos("high_S", self.TRACKBAR_WINDOW)
        else:
            low_H = rospy.get_param("~low_H", 120)
            low_L = rospy.get_param("~low_L", 90)
            low_S = rospy.get_param("~low_S", 80)
            high_H = rospy.get_param("~high_H", 360)
            high_L = rospy.get_param("~high_L", 255)
            high_S = rospy.get_param("~high_S", 255)

        lower_lane = np.array([low_H, low_L, low_S], dtype=np.uint16)
        upper_lane = np.array([high_H, high_L, high_S], dtype=np.uint16)
        color_mask = cv2.inRange(hls, lower_lane, upper_lane)
        return self._apply_roi(color_mask)

    def _auto_lane_mask(self, frame):
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
        lower_yellow = np.array([15, 60, 80])
        upper_yellow = np.array([40, 255, 255])
        lower_white_hls = np.array([0, 190, 0])
        upper_white_hls = np.array([255, 255, 120])
        mask_yellow = cv2.inRange(hls, lower_yellow, upper_yellow)
        mask_white_hls = cv2.inRange(hls, lower_white_hls, upper_white_hls)
        L_channel = lab[:, :, 0]
        _, mask_white_lab = cv2.threshold(L_channel, 205, 255, cv2.THRESH_BINARY)
        color_mask = cv2.bitwise_or(mask_yellow, mask_white_hls)
        color_mask = cv2.bitwise_or(color_mask, mask_white_lab)

        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        clahe_img = self.clahe.apply(gray)
        sobelx = cv2.Sobel(clahe_img, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(clahe_img, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(sobelx, sobely)
        if np.max(grad_mag) > 0:
            grad_mag = grad_mag / np.max(grad_mag)
        sobel_mask = np.uint8(grad_mag * 255)
        _, sobel_mask = cv2.threshold(sobel_mask, 80, 255, cv2.THRESH_BINARY)

        canny_mask = cv2.Canny(clahe_img, 70, 180)

        lane_mask = cv2.bitwise_or(color_mask, sobel_mask)
        lane_mask = cv2.bitwise_or(lane_mask, canny_mask)

        kernel = np.ones((5, 5), np.uint8)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return self._apply_roi(lane_mask)

    def _yellow_lane_mask(self, frame):
        """노란색 차선만 검출 (밝기 기반 필터링 없이 색상만 사용)"""
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        # HSV 색상 범위만 사용하여 노란색 검출 (밝기 필터링 없음)
        mask = cv2.inRange(hsv, self.yellow_hsv_low, self.yellow_hsv_high)

        # 노이즈 제거를 위한 모폴로지 연산
        kernel_size = max(3, int(round(self.yellow_kernel_size)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return self._apply_roi(mask)

    @staticmethod
    def _apply_roi(mask):
        h, w = mask.shape[:2]
        polygon = np.array([[
            (int(0.05 * w), h),
            (int(0.05 * w), int(0.65 * h)),
            (int(0.95 * w), int(0.65 * h)),
            (int(0.95 * w), h),
        ]], dtype=np.int32)
        roi_mask = np.zeros_like(mask)
        cv2.fillPoly(roi_mask, polygon, 255)
        return cv2.bitwise_and(mask, roi_mask)

    def _run_slidewindow(self, lane_mask):
        mask_pixels = int(cv2.countNonZero(lane_mask))
        if mask_pixels < self.min_mask_pixels:
            self._reset_line_state()
            return None, self.prev_center, False
        try:
            blur_img = cv2.GaussianBlur(lane_mask, (5, 5), 0)
            warped = self.warper.warp(blur_img)
            slide_img, center_x, _ = self.slidewindow.slidewindow(warped)
            if center_x is None or (isinstance(center_x, float) and math.isnan(center_x)):
                raise ValueError("Invalid center from slidewindow")
            return slide_img, center_x, True
        except Exception as exc:
            rospy.logwarn(f"Slide window failed: {exc}")
            self._reset_line_state()
            fallback_center = self._center_from_mask(lane_mask, self.prev_center)
            return None, fallback_center, mask_pixels >= self.min_mask_pixels

    @staticmethod
    def _center_from_mask(mask, default_value):
        if mask is None:
            return default_value
        moments = cv2.moments(mask, binaryImage=True)
        if moments["m00"] > 0:
            return moments["m10"] / moments["m00"]
        return default_value

    def _setup_trackbars(self):
        cv2.namedWindow(self.TRACKBAR_WINDOW, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("low_H", self.TRACKBAR_WINDOW, 128, 360, self._nothing)
        cv2.createTrackbar("low_L", self.TRACKBAR_WINDOW, 134, 255, self._nothing)
        cv2.createTrackbar("low_S", self.TRACKBAR_WINDOW, 87, 255, self._nothing)
        cv2.createTrackbar("high_H", self.TRACKBAR_WINDOW, 334, 360, self._nothing)
        cv2.createTrackbar("high_L", self.TRACKBAR_WINDOW, 255, 255, self._nothing)
        cv2.createTrackbar("high_S", self.TRACKBAR_WINDOW, 251, 255, self._nothing)

    @staticmethod
    def _clamp(value, low, high):
        return max(low, min(high, value))

    def _lidar_ackermann_callback(self, msg):
        """라이다가 ackermann 명령을 발행하면 호출됨 (라이다 제어 중임을 표시)"""
        self.last_lidar_servo_time = rospy.get_time()
        self.lidar_controlling = True
    
    def _lidar_speed_callback(self, msg):
        """라이다에서 계산된 목표 속도 수신"""
        self.lidar_target_speed = msg.data
        self.last_lidar_command_time = rospy.get_time()
        # 라이다 노드가 살아있음을 확인 (Ackermann 메시지 없이도 조향 권한 양보를 위해)
        self.last_lidar_servo_time = rospy.get_time()

    def _compute_integrated_speed(self) -> float:
        """여러 소스의 속도 명령을 통합하여 최종 속도 계산"""
        current_time = rospy.get_time()
        dt = max(0.001, min(0.1, current_time - self.prev_time))
        self.prev_time = current_time  # 시간 업데이트
        
        # 1. 카메라 기반 기본 속도
        camera_speed_mps = self._clamp(
            self.current_color_speed_mps, self.min_drive_speed, self.max_drive_speed
        )
        
        # 2. 라이다 기반 속도 명령 확인 (타임아웃 체크)
        lidar_speed_mps = self.max_drive_speed
        if (current_time - self.last_lidar_command_time) < 0.5:
            lidar_speed_mps = self.lidar_target_speed
            
        # 3. 통합: 가장 보수적인(느린) 속도 선택
        final_speed_mps = min(camera_speed_mps, lidar_speed_mps)
        
        # 4. PWM 변환
        target_pwm = self._drive_speed_to_pwm(final_speed_mps)
        
        # target_pwm을 범위 내로 확실히 제한 (역회전 방지)
        target_pwm = self._clamp(target_pwm, self.min_pwm, self.max_pwm)
        
        # 이중 스무딩: 선형 제한 + 지수적 스무딩
        # 1단계: 최대 변화량 제한 (급격한 변화 방지)
        max_pwm_change = self.speed_smoothing_rate * dt
        pwm_diff = target_pwm - self.current_speed_pwm
        if abs(pwm_diff) > max_pwm_change:
            limited_target = self.current_speed_pwm + math.copysign(max_pwm_change, pwm_diff)
        else:
            limited_target = target_pwm
        
        # limited_target도 범위 내로 제한 (0.0 미만 절대 불가)
        limited_target = max(0.0, self._clamp(limited_target, self.min_pwm, self.max_pwm))
        
        # 2단계: 지수적 스무딩 (더 부드러운 변화)
        smoothed_pwm = (
            (1.0 - self.speed_smoothing_factor) * self.current_speed_pwm
            + self.speed_smoothing_factor * limited_target
        )
        
        # smoothed_pwm을 범위 내로 확실히 제한 (역회전 방지)
        smoothed_pwm = max(0.0, self._clamp(smoothed_pwm, self.min_pwm, self.max_pwm))
        
        # 최소 변화량 이하는 무시 (미세한 진동 방지)
        if abs(smoothed_pwm - self.current_speed_pwm) < 1.0:
            smoothed_pwm = self.current_speed_pwm
        
        # 최종 값도 범위 확인 (이중 안전장치 - 절대 음수 불가)
        self.current_speed_pwm = max(0.0, self._clamp(smoothed_pwm, self.min_pwm, self.max_pwm))
        
        return self.current_speed_pwm

    def _detect_bottom_lane_color(self, frame):
        """하단 20% 영역에서 빨간색/파란색 검출"""
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        # 하단 20%만 ROI로 사용
        roi_top = int(0.8 * h)  # 상단 80%부터 시작
        roi_bottom = h  # 하단까지
        roi_left = 0  # 전체 너비 사용
        roi_right = w
        
        # 하단 20% 전체 영역에서 색상 검출
        roi = frame[roi_top:roi_bottom, roi_left:roi_right]
        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # 빨간색 검출 범위 (HSV에서 빨간색은 0도와 180도 근처에 있음)
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        # 파란색 검출 범위
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])

        red_mask = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask = cv2.bitwise_or(red_mask, cv2.inRange(hsv, red_lower2, red_upper2))
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        roi_area = roi.shape[0] * roi.shape[1]
        if roi_area == 0:
            return None
        detection_ratio = self._clamp(self.color_detection_ratio, 0.0, 1.0)
        pixel_threshold = max(1, int(roi_area * detection_ratio))

        red_pixels = cv2.countNonZero(red_mask)
        blue_pixels = cv2.countNonZero(blue_mask)

        candidates = []
        if red_pixels >= pixel_threshold:
            candidates.append(("red", red_pixels))
        if blue_pixels >= pixel_threshold:
            candidates.append(("blue", blue_pixels))

        if not candidates:
            return None
        return max(candidates, key=lambda item: item[1])[0]

    def _update_speed_profile_from_color(self, detected_color):
        if detected_color == "red":
            target_speed = self.red_lane_speed
        elif detected_color == "blue":
            target_speed = self.blue_lane_speed
        else:
            target_speed = self.neutral_lane_speed

        target_speed = self._clamp(target_speed, self.min_drive_speed, self.max_drive_speed)
        if abs(target_speed - self.current_color_speed_mps) < 1e-3:
            return

        self.current_color_speed_mps = target_speed
        new_pwm = self._drive_speed_to_pwm(target_speed)
        self.speed_value = new_pwm
        if self.current_speed_pwm is None:
            self.current_speed_pwm = new_pwm

        new_color = detected_color or "none"
        if new_color != self.last_detected_color:
            rospy.loginfo(f"Lane color detected: {new_color} -> {target_speed:.2f} m/s")
            self.last_detected_color = new_color

    def _drive_speed_to_pwm(self, speed_mps: float) -> float:
        if self.max_drive_speed <= self.min_drive_speed:
            return self.min_pwm
        clamped_speed = self._clamp(speed_mps, self.min_drive_speed, self.max_drive_speed)
        normalized = (clamped_speed - self.min_drive_speed) / (self.max_drive_speed - self.min_drive_speed)
        return self.min_pwm + normalized * (self.max_pwm - self.min_pwm)

    def _pwm_to_drive_speed(self, pwm_value: float) -> float:
        if self.max_pwm <= self.min_pwm or self.max_drive_speed <= self.min_drive_speed:
            return self.min_drive_speed
        pwm_value = self._clamp(pwm_value, self.min_pwm, self.max_pwm)
        normalized = (pwm_value - self.min_pwm) / (self.max_pwm - self.min_pwm)
        return self.min_drive_speed + normalized * (self.max_drive_speed - self.min_drive_speed)

    def _publish_speed_command(self, value: float) -> None:
        msg = Float64(value)
        self.speed_pub.publish(msg)
        if self.speed_fallback_pub and self.speed_pub.get_num_connections() == 0:
            if not self._speed_fallback_warned:
                rospy.logwarn(
                    "No subscribers on %s. Falling back to %s.",
                    self.speed_topic,
                    self.speed_fallback_topic,
                )
                self._speed_fallback_warned = True
            self.speed_fallback_pub.publish(Float64(value))
        elif self._speed_fallback_warned and self.speed_pub.get_num_connections() > 0:
            rospy.loginfo(
                "Primary speed topic %s has subscribers again. Stopping fallback.",
                self.speed_topic,
            )
            self._speed_fallback_warned = False

    def _publish_servo_command(self, value: float) -> None:
        msg = Float64(value)
        self.steering_pub.publish(msg)
        if self.servo_fallback_pub and self.steering_pub.get_num_connections() == 0:
            if not self._servo_fallback_warned:
                rospy.logwarn(
                    "No subscribers on %s. Falling back to %s.",
                    self.servo_topic,
                    self.servo_fallback_topic,
                )
                self._servo_fallback_warned = True
            self.servo_fallback_pub.publish(Float64(value))
        elif self._servo_fallback_warned and self.steering_pub.get_num_connections() > 0:
            rospy.loginfo(
                "Primary servo topic %s has subscribers again. Stopping fallback.",
                self.servo_topic,
            )
            self._servo_fallback_warned = False

    @staticmethod
    def _nothing(_):
        pass

    @staticmethod
    def _cleanup():
        if cv2 is None:
            return
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def _reset_line_state(self):
        self.slidewindow.left_exists = False
        self.slidewindow.right_exists = False
        self.slidewindow.left_x = None
        self.slidewindow.right_x = None
        self.slidewindow.lane_width_px = None


def run():
    rospy.init_node("lane_follower")
    LaneFollower()
    rospy.spin()


if __name__ == "__main__":
    run()
