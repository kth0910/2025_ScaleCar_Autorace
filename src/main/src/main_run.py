#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import os
import math

import rospy
import cv2
import numpy as np

from std_msgs.msg import Float64, Bool, String
from sensor_msgs.msg import Image, Joy
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge

from warper import Warper
from slidewindow import SlideWindow


class LaneFollower:
    """
    카메라 기반 차선 추종을 기본으로 하며, 
    라이다 장애물 회피, 횡단보도 정지, 표지판 인식 기능을 통합한 주행 노드
    """

    TRACKBAR_WINDOW = "Lane Threshold Tuner"

    def __init__(self):
        self.bridge = CvBridge()
        self.warper = Warper()
        self.slidewindow = SlideWindow()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # 파라미터
        self.desired_center = rospy.get_param("~desired_center", 305.0)
        self.pid_kp = rospy.get_param("~steering_kp", -0.0045)
        self.pid_ki = rospy.get_param("~steering_ki", -0.0001)
        self.pid_kd = rospy.get_param("~steering_kd", -0.005)
        self.steering_gain = self.pid_kp  # backward compatibility
        self.steering_offset = rospy.get_param("~steering_offset", 0.50)  # 조향 중앙 정렬 오프셋
        self.steering_smoothing = rospy.get_param("~steering_smoothing", 0.55)
        self.steering_smoothing_left = rospy.get_param("~steering_smoothing_left", 0.40)  # 좌측 조향 시 반응성 향상
        self.steering_smoothing_right = rospy.get_param("~steering_smoothing_right", 0.40)  # 우측 조향 시 반응성 향상
        self.min_servo = rospy.get_param("~min_servo", 0.0)
        self.max_servo = rospy.get_param("~max_servo", 1.0)
        self.center_smoothing = rospy.get_param("~center_smoothing", 0.4)
        self.max_center_step = rospy.get_param("~max_center_step", 25.0)
        self.bias_correction_gain = rospy.get_param("~bias_correction_gain", 1e-4)
        self.max_error_bias = rospy.get_param("~max_error_bias", 120.0)
        self.error_bias = rospy.get_param("~initial_error_bias", 0.0)
        self.max_servo_delta = rospy.get_param("~max_servo_delta", 0.035)
        self.max_servo_delta_left = rospy.get_param("~max_servo_delta_left", 0.05)  # 좌측 조향 시 더 큰 변화 허용
        self.max_servo_delta_right = rospy.get_param("~max_servo_delta_right", 0.05)  # 우측 조향 시 기존 유지
        self.min_mask_pixels = rospy.get_param("~min_mask_pixels", 600)
        self.integral_limit = rospy.get_param("~steering_integral_limit", 500.0)
        self.single_left_ratio = rospy.get_param("~single_left_ratio", 0.65)  # 한쪽 차선만 보일 때의 조향 비율 (좌측)
        self.single_right_ratio = rospy.get_param("~single_right_ratio", 0.65)  # 한쪽 차선만 보일 때의 조향 비율 (우측)
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
        self.roi_color_pub = rospy.Publisher("/camera/roi_color", Image, queue_size=1)
        self.detected_color_pub = rospy.Publisher("/camera/detected_color", String, queue_size=1)

        rospy.Subscriber("usb_cam/image_rect_color", Image, self.image_callback, queue_size=1)
        
        # 라이다 회피 우선: lidar_avoidance 노드가 제어 중인지 확인
        self.lidar_controlling = False
        self.last_lidar_servo_time = 0.0
        self.lidar_timeout = 0.5  # 0.5초 동안 라이다 제어 신호가 없으면 카메라 제어로 복귀
        self.last_servo_publish_time = 0.0  # 마지막으로 서보 명령을 발행한 시간

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
        
        # 라이다 제어 감지: lidar_avoidance가 /ackermann_cmd를 발행하면 라이다 제어 중으로 판단
        rospy.Subscriber("/ackermann_cmd", AckermannDriveStamped, self._lidar_ackermann_callback, queue_size=1)
        
        # 속도 제어 파라미터
        self.max_drive_speed = rospy.get_param("~max_drive_speed", 0.75)  # m/s
        self.min_drive_speed = rospy.get_param("~min_drive_speed", 0.15)  # m/s
        self.max_pwm = rospy.get_param("~max_pwm", 3000.0)  # ERPM
        self.min_pwm = rospy.get_param("~min_pwm", 0.0)  # ERPM (0 = Stop)
        self.min_moving_pwm = rospy.get_param("~min_moving_pwm", 900.0)  # ERPM (최소 구동 토크)
        self.speed_reduction_start = rospy.get_param("~speed_reduction_start", 1.0)  # 장애물 1m 전방부터 감속 시작
        self.hard_stop_distance = rospy.get_param("~hard_stop_distance", 0.15)  # 15cm 이내 진입 시 완전 정지
        speed_pwm_param = rospy.get_param("~speed", 1500.0)
        self.speed_smoothing_rate = rospy.get_param("~speed_smoothing_rate", 5000.0)  # PWM 변화율 제한
        self.speed_smoothing_factor = rospy.get_param("~speed_smoothing_factor", 0.8)  # 지수적 스무딩 계수

        # 라이다 장애물 거리 구독 (속도 제어용)
        self.closest_obstacle = None
        self.last_obstacle_time = 0.0
        rospy.Subscriber("lidar_avoidance/closest_obstacle", Float64, self._obstacle_distance_callback, queue_size=1)
        
        # 라이다 조향 명령 구독 (장애물 회피용)
        self.lidar_steering_cmd = self.steering_offset
        self.last_lidar_steering_time = 0.0
        rospy.Subscriber("lidar_avoidance/steering_cmd", Float64, self._lidar_steering_callback, queue_size=1)
        # 색상 기반 속도 제어 파라미터
        self.neutral_lane_speed = rospy.get_param(
            "~neutral_lane_speed", 0.3
        )
        self.red_lane_speed = rospy.get_param("~red_lane_speed", 0.15)
        self.blue_lane_speed = rospy.get_param("~blue_lane_speed", 0.5)
        self.color_roi_height_ratio = rospy.get_param("~color_roi_height_ratio", 0.20)  # 이미지 하단 20% 영역 사용
        self.color_detection_ratio = rospy.get_param("~color_detection_ratio", 0.02)  # 색상 검출 임계값 (전체 픽셀의 2%)
        self.current_color_speed_mps = self._clamp(
            self.neutral_lane_speed, self.min_drive_speed, self.max_drive_speed
        )
        base_color_pwm = self._drive_speed_to_pwm(self.current_color_speed_mps)
        self.speed_value = base_color_pwm
        self.current_speed_pwm = base_color_pwm
        self.last_detected_color = "none"
        self.current_detected_color = "none"

        self.smoothed_servo = self.steering_offset
        self.last_image_time = 0.0

        # 제어 루프 타이머 (30Hz)
        self.timer = rospy.Timer(rospy.Duration(0.033), self.timer_callback)

        # 횡단보도 감지 및 정지 관련 변수
        # 횡단보도 감지 및 정지 관련 변수
        self.crosswalk_state = "IDLE"  # IDLE, WAITING, STOPPING, DONE
        self.stop_line_seen = False
        self.last_stop_line_time = 0.0  # 정지선 감지 시간 기억 (늦은 횡단보도 감지 대응)
        self.crosswalk_stop_start_time = 0.0
        self.crosswalk_detected_time = 0.0
        self.crosswalk_stop_duration = 7.0       # 7초 정지
        
        # 표지판 제어 변수
        self.sign_state = "IDLE"  # IDLE, APPROACHING, STOPPING, TURNING
        self.sign_direction = "Unknown"
        self.sign_timer = 0.0
        self.sign_stop_line_seen = False
        self.sign_stop_duration = 0.5
        self.sign_turn_duration = 0.5
        self.turn_servo_offset = 0.35  # 30도에 해당하는 서보 오프셋 (약 0.35)

        # 조이스틱 수동 제어 관련 변수
        self.manual_mode = False
        self.joy_lb_idx = rospy.get_param("~joy_lb_idx", 4)  # LB 버튼 인덱스 (기본: 4)
        self.joy_axis_throttle = rospy.get_param("~joy_axis_throttle", 1)  # 왼쪽 스틱 상하 (기본: 1)
        self.joy_axis_steering = rospy.get_param("~joy_axis_steering", 3)  # 오른쪽 스틱 좌우 (기본: 3)
        self.joy_max_speed_ratio = rospy.get_param("~joy_max_speed_ratio", 1.0) # 수동 모드 최대 속도 비율
        
        rospy.Subscriber("/joy", Joy, self.joy_callback, queue_size=1)

        rospy.on_shutdown(self._cleanup)
        rospy.loginfo("LaneFollower initialized.")

    def timer_callback(self, event):
        """주기적인 제어 명령 발행 (카메라 수신 여부와 무관하게 동작)"""
        # 수동 모드일 경우 자율주행 로직 스킵
        if self.manual_mode:
            return

        # 1. 속도 제어 (통합)
        target_pwm = self._compute_integrated_speed()
        self._publish_speed_command(target_pwm)

        # 2. 조향 제어 우선순위 로직
        # 우선순위: 1. 장애물 회피 (Lidar) > 2. 표지판 회전 (Sign) > 3. 차선 추종 (Camera)
        current_time = rospy.get_time()
        
        use_lidar_steering = False
        
        # 라이다 조향 명령이 최근(0.5초 이내)에 수신되었다면 라이다 제어 우선
        if (current_time - self.last_lidar_steering_time) < 0.5:
            use_lidar_steering = True
        
        if use_lidar_steering:
            self.last_servo_publish_time = current_time
            self._publish_servo_command(self.lidar_steering_cmd)
        elif self.sign_state == "TURNING":
            # 표지판 인식 후 회전 동작 수행 (강제 조향)
            self.last_servo_publish_time = current_time
            if self.sign_direction == "LEFT":
                # 왼쪽 회전: 서보 값 증가 (+)
                target_servo = self.steering_offset + self.turn_servo_offset
            else:
                # 오른쪽 회전: 서보 값 감소 (-)
                target_servo = self.steering_offset - self.turn_servo_offset
            self._publish_servo_command(target_servo)
        else:
            # 그 외의 경우 (평상시) 카메라 기반 차선 추종 수행
            # 단, 카메라 데이터가 최신일 때만 (0.5초 이내) 유효
            if (current_time - self.last_image_time) < 0.5:
                self.last_servo_publish_time = current_time
                self._publish_servo_command(self.smoothed_servo)

    def joy_callback(self, msg):
        """조이스틱 입력 처리"""
        # LB 버튼 상태 확인 (누르고 있으면 수동 모드)
        if msg.buttons[self.joy_lb_idx] == 1:
            if not self.manual_mode:
                rospy.loginfo("Manual Mode Engaged (Joystick)")
                self.manual_mode = True
            
            # 1. 조향 제어 (오른쪽 스틱 좌우)
            # Axis 3: Left(+1.0) ~ Right(-1.0)
            # Servo: Left(>0.5) ~ Right(<0.5) (steering_offset 기준)
            # 매핑: steering_offset + (axis * scale)
            steering_axis = msg.axes[self.joy_axis_steering]
            # 스틱을 왼쪽으로 밀면(+), 서보값도 증가해야 함(좌회전)
            # 스틱을 오른쪽으로 밀면(-), 서보값은 감소해야 함(우회전)
            # 최대 0.35 정도의 변위 허용
            target_servo = self.steering_offset + (steering_axis * 0.35)
            target_servo = self._clamp(target_servo, self.min_servo, self.max_servo)
            self._publish_servo_command(target_servo)
            
            # 2. 속도 제어 (왼쪽 스틱 상하)
            # Axis 1: Up(+1.0) ~ Down(-1.0)
            throttle_axis = msg.axes[self.joy_axis_throttle]
            
            if throttle_axis >= 0:
                # 전진: min_moving_pwm ~ max_pwm
                # throttle 0~1을 PWM 범위로 매핑
                pwm_range = self.max_pwm - self.min_moving_pwm
                target_pwm = self.min_moving_pwm + (throttle_axis * pwm_range * self.joy_max_speed_ratio)
                if throttle_axis < 0.05: # 데드존
                    target_pwm = 0.0
            else:
                # 후진 (현재 설정상 0으로 처리하거나 후진 로직이 있다면 적용)
                # 여기서는 정지로 처리
                target_pwm = 0.0
                
            target_pwm = self._clamp(target_pwm, self.min_pwm, self.max_pwm)
            self._publish_speed_command(target_pwm)
            
            # 수동 제어 중에는 내부 상태 초기화 (적분 오차 등)
            self.integral_error = 0.0
            
        else:
            if self.manual_mode:
                rospy.loginfo("Autonomous Mode Engaged (Joystick Released)")
                self.manual_mode = False
                # 복귀 시 안전을 위해 속도 0으로 초기화하지 않고 자연스럽게 제어권 넘김

    def image_callback(self, msg):
        try:
            if msg.encoding == "yuyv" or msg.encoding == "yuv422":
                # Use passthrough and manual conversion to avoid swscaler warnings
                frame_yuv = self.bridge.imgmsg_to_cv2(msg, "passthrough")
                frame = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR_YUYV)
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            # Suppress conversion errors/warnings
            return

        if not self.initialized:
            self._setup_trackbars()
            self.initialized = True

        # 0. 정지선 상시 감지 (상태와 무관하게 감지하여 메모리)
        has_stop_line = self._detect_stop_line(frame)
        if has_stop_line:
            self.last_stop_line_time = rospy.get_time()

        # 표지판 감지 및 시각화 (디버깅을 위해 항상 수행)
        # 상태 변경은 내부에서 sign_state == "IDLE"일 때만 발생
        self._detect_traffic_sign(frame)
        
        if self.sign_state == "APPROACHING":
            # 정지선 감지 (위에서 계산한 has_stop_line 사용)
            if has_stop_line:
                self.sign_stop_line_seen = True
            
            # 정지선을 보았다가 사라지면 정지
            if self.sign_stop_line_seen and not has_stop_line:
                self.sign_state = "STOPPING"
                self.sign_timer = rospy.get_time()
                rospy.loginfo(f"Stop line passed. Stopping for sign ({self.sign_direction}).")

        # 횡단보도 로직 상태 머신
        if self.crosswalk_state == "IDLE":
             if self._detect_crosswalk(frame):
                 self.crosswalk_state = "WAITING"
                 self.crosswalk_detected_time = rospy.get_time()
                 rospy.loginfo("Crosswalk detected! Waiting 1s before stop.")
        
        elif self.crosswalk_state == "WAITING":
             if rospy.get_time() - self.crosswalk_detected_time > 1.0:
                 self.crosswalk_state = "STOPPING"
                 self.crosswalk_stop_start_time = rospy.get_time()
                 rospy.loginfo("1s passed. Stopping now.")
        
        # APPROACHING 상태 제거됨 (즉시 정지)
        
        elif self.crosswalk_state == "DONE":
            if self.enable_viz:
                try:
                    cv2.destroyWindow("Crosswalk Debug")
                    cv2.destroyWindow("Stop Line Debug")
                except Exception:
                    pass

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
        # ROI 색상 검출 및 시각화 (빨강/파랑/검정/흰색)
        detected_color = self._process_and_publish_roi(frame)
        self.current_detected_color = detected_color
        self._update_speed_profile_from_color(detected_color)

        left_exists = getattr(self.slidewindow, "left_exists", False)
        right_exists = getattr(self.slidewindow, "right_exists", False)
        left_x = getattr(self.slidewindow, "left_x", None)
        right_x = getattr(self.slidewindow, "right_x", None)
        frame_width = getattr(self.slidewindow, "frame_width", self.desired_center * 2)
        lane_detected = left_exists or right_exists

        self.center_pub.publish(Float64(self.current_center))
        
        if lane_detected:
            raw_error = self.desired_center - self.current_center
            
            # 두 차선이 모두 보일 때만 오차 편향(bias) 학습
            if left_exists and right_exists:
                self.error_bias += self.bias_correction_gain * raw_error
                self.error_bias = self._clamp(
                    self.error_bias, -self.max_error_bias, self.max_error_bias
                )
            
            # 학습된 편향을 적용하여 오차 보정
            error = raw_error - self.error_bias
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
                # 우측 조향: 상대적으로 작은 변화 제한
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
            # 우측 조향
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
            (int(0.05 * w), int(0.75 * h)),
            (int(0.95 * w), int(0.75 * h)),
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
        """라이다 노드가 ackermann 명령을 발행하면 호출됨 (라이다 제어 활성화 상태 감지)"""
        self.last_lidar_servo_time = rospy.get_time()
        self.lidar_controlling = True
    
    def _obstacle_distance_callback(self, msg):
        """라이다에서 가장 가까운 장애물 거리 수신"""
        self.closest_obstacle = msg.data
        self.last_obstacle_time = rospy.get_time()
        
    def _lidar_steering_callback(self, msg):
        """라이다 노드에서 계산된 회피 조향 명령 수신"""
        self.lidar_steering_cmd = msg.data
        self.last_lidar_steering_time = rospy.get_time()

    def _compute_integrated_speed(self) -> float:
        """여러 소스의 속도 명령을 통합하여 최종 속도 계산"""
        current_time = rospy.get_time()
        dt = max(0.001, min(0.1, current_time - self.prev_time))
        self.prev_time = current_time  # 시간 업데이트
        
        # 1. 카메라 기반 기본 속도
        camera_speed_mps = self._clamp(
            self.current_color_speed_mps, self.min_drive_speed, self.max_drive_speed
        )
        
        # 2. 라이다 장애물 거리 기반 안전 속도 계산
        lidar_safe_speed_mps = self.max_drive_speed
        
        if self.closest_obstacle is not None and (current_time - self.last_obstacle_time) < 0.5:
            closest = self.closest_obstacle
            
            # 거리 기반 속도 감소: 1m부터 점진적으로 감소하지만 절대 정지하지 않음 (회피 기동 위해)
            if closest < self.speed_reduction_start:
                # 1m 이내 진입 시 min_drive_speed ~ 0.3m/s 사이로 감속
                # 거리가 0에 가까워져도 최소 이동 속도(min_drive_speed)는 유지
                ratio = self._clamp(closest / self.speed_reduction_start, 0.0, 1.0)
                
                min_avoid_speed = self.min_drive_speed  # 0.15 m/s
                max_avoid_speed = 0.2   # 0.3 m/s
                
                # 거리가 가까울수록 min_avoid_speed에 가까워짐
                lidar_safe_speed_mps = min_avoid_speed + (max_avoid_speed - min_avoid_speed) * ratio
            
        # 3. 통합: 기본적으로 가장 보수적인(느린) 속도 선택
        final_speed_mps = min(camera_speed_mps, lidar_safe_speed_mps)
        
        # [Override] Red/Blue 색상 검출 시 속도 고정 (라이다 안전 속도 무시)
        if self.current_detected_color in ["red", "blue"]:
            rospy.loginfo_throttle(1.0, f"Color Override: {self.current_detected_color} detected. Forcing speed to {self.current_color_speed_mps:.2f} m/s")
            final_speed_mps = self.current_color_speed_mps
        
        # [Override] 횡단보도 정지 로직
        if self.crosswalk_state == "STOPPING":
            elapsed = current_time - self.crosswalk_stop_start_time
            if elapsed < self.crosswalk_stop_duration:
                final_speed_mps = 0.0
                rospy.loginfo_throttle(1.0, f"Crosswalk Stopping... ({elapsed:.1f}/{self.crosswalk_stop_duration}s)")
            else:
                self.crosswalk_state = "DONE"
                rospy.loginfo("Crosswalk stop finished. Logic disabled permanently.")

        # [Override] 표지판 제어 로직 (정지 -> 회전)
        if self.sign_state == "STOPPING":
            elapsed = current_time - self.sign_timer
            if elapsed < self.sign_stop_duration:
                final_speed_mps = 0.0
            else:
                self.sign_state = "TURNING"
                self.sign_timer = current_time
                rospy.loginfo(f"Sign Stop done. Turning {self.sign_direction}...")
        
        if self.sign_state == "TURNING":
            elapsed = current_time - self.sign_timer
            if elapsed < self.sign_turn_duration:
                # 회전 시 속도 (너무 빠르지 않게)
                final_speed_mps = self.neutral_lane_speed
            else:
                self.sign_state = "IDLE"
                self.sign_direction = "Unknown"
                self.sign_stop_line_seen = False
                rospy.loginfo("Sign Turn done. Back to IDLE.")

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

    def _process_and_publish_roi(self, frame):
        """하단 20% 영역에서 빨간색/파란색/검정색/흰색 검출 및 시각화"""
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        # 하단 20%만 ROI로 사용
        roi_top = int(0.65 * h)
        roi_bottom = h
        roi_left = 0
        roi_right = w
        
        roi = frame[roi_top:roi_bottom, roi_left:roi_right].copy()
        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 1. 빨간색
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
        
        # 2. 파란색 (범위 확장: Hue 90~140, Sat 40~255, Val 40~255)
        blue_lower = np.array([90, 40, 40])
        blue_upper = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        # 3. 흰색 (낮은 채도, 높은 명도)
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        # 4. 검정색 (매우 낮은 명도)
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 255, 50])
        black_mask = cv2.inRange(hsv, black_lower, black_upper)

        # 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)

        # 결과 시각화
        debug_roi = roi.copy()
        
        # 컨투어 그리고 라벨링
        contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_white, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_black, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.drawContours(debug_roi, contours_red, -1, (0, 0, 255), 2)
        cv2.drawContours(debug_roi, contours_blue, -1, (255, 0, 0), 2)
        cv2.drawContours(debug_roi, contours_white, -1, (255, 255, 255), 2)
        cv2.drawContours(debug_roi, contours_black, -1, (0, 0, 0), 2)
        
        # 픽셀 수 계산 및 가장 지배적인 색상 결정
        roi_area = roi.shape[0] * roi.shape[1]
        pixel_threshold = int(roi_area * 0.02)  # 2% 이상
        
        red_pixels = cv2.countNonZero(red_mask)
        blue_pixels = cv2.countNonZero(blue_mask)
        white_pixels = cv2.countNonZero(white_mask)
        black_pixels = cv2.countNonZero(black_mask)
        
        candidates = []
        if red_pixels >= pixel_threshold: candidates.append(("red", red_pixels))
        if blue_pixels >= pixel_threshold: candidates.append(("blue", blue_pixels))
        if white_pixels >= pixel_threshold: candidates.append(("white", white_pixels))
        if black_pixels >= pixel_threshold: candidates.append(("black", black_pixels))
        
        detected = "none"
        if candidates:
            detected = max(candidates, key=lambda x: x[1])[0]
            
        # 텍스트 표시
        cv2.putText(debug_roi, f"Detected: {detected}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_roi, f"R:{red_pixels} B:{blue_pixels} W:{white_pixels} K:{black_pixels}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 이미지 발행
        try:
            msg = self.bridge.cv2_to_imgmsg(debug_roi, "bgr8")
            self.roi_color_pub.publish(msg)
        except Exception:
            pass
            
        # 검출된 색상 문자열 발행
        self.detected_color_pub.publish(String(detected))
            
        # 화면 출력
        if self.enable_viz:
            cv2.imshow("ROI Colors", debug_roi)
            
        return detected

    def _update_speed_profile_from_color(self, detected_color):
        if detected_color == "red":
            target_speed = self.red_lane_speed
        elif detected_color == "blue":
            target_speed = self.blue_lane_speed
        elif detected_color == "black":
            target_speed = 0.4  # Black일 때는 0.4m/s 고정
        else:
            target_speed = self.neutral_lane_speed

        target_speed = max(self.min_drive_speed, target_speed)
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
        if speed_mps < 0.01:
            return 0.0
            
        if self.max_drive_speed <= self.min_drive_speed:
            return self.min_moving_pwm
            
        # 유효 주행 속도 범위로 클램핑
        clamped_speed = self._clamp(speed_mps, self.min_drive_speed, self.max_drive_speed)
        
        # [min_drive, max_drive] -> [min_moving, max_pwm] 매핑
        normalized = (clamped_speed - self.min_drive_speed) / (self.max_drive_speed - self.min_drive_speed)
        return self.min_moving_pwm + normalized * (self.max_pwm - self.min_moving_pwm)

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

    def _detect_crosswalk(self, frame):
        # (기존 코드 유지) ...
        # 여기서는 _detect_crosswalk 구현 내용을 건드리지 않으므로
        # 실제 파일의 내용을 그대로 두거나, 이 도구의 특성상
        # _detect_crosswalk의 끝부분 뒤에 새 메소드를 추가하는 방식을 사용해야 합니다.
        # 하지만 multi_replace는 기존 내용을 대체하므로, 
        # _detect_crosswalk 전체를 다시 쓰는 것보다
        # 파일의 맨 끝(마지막 메소드 뒤)에 추가하는 것이 안전합니다.
        # 따라서 이 청크는 취소하고, 아래에 새로운 청크로 파일 끝에 추가하겠습니다.
        pass

    def _detect_traffic_sign(self, frame):
        """
        파란색 원형 표지판 감지 및 화살표 방향 판정 (Pixel Sum 방식)
        - ROI: 화면 중단 20% (0.4h ~ 0.6h)만 사용
        - HSV 파란색 마스크로 표지판 영역 검출 (임계값 완화)
        """
        if frame is None:
            return

        h, w = frame.shape[:2]
        # ROI: 중단 20% (가로/세로 모두)
        roi_top = int(h * 0.4)
        roi_bottom = int(h * 0.6)
        roi_left = int(w * 0.4)
        roi_right = int(w * 0.6)
        
        roi = frame[roi_top:roi_bottom, roi_left:roi_right]
        
        if roi.size == 0:
            return

        # 1. HSV 변환 및 파란색 마스크
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 파란색 범위 대폭 완화
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

        # 2. 컨투어 검출
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_sign = None
        max_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:  # 면적 기준 완화
                continue
                
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
                
            # 원형도(Circularity) 계산
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # 원형 표지판이면 circularity가 1에 가까움 (0.6 이상 허용)
            if circularity > 0.6:
                if area > max_area:
                    max_area = area
                    detected_sign = cnt
        
        if detected_sign is not None:
            x, y, w, h_rect = cv2.boundingRect(detected_sign)
            # ROI 좌표를 원본 좌표로 변환
            x += roi_left
            y += roi_top
            
            # ROI 추출 (약간의 여백을 두고 자르기)
            margin = 2
            # 원본 프레임에서 추출해야 하므로 y 좌표는 이미 보정됨
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h_rect + margin)
            
            roi_sign = frame[y1:y2, x1:x2]
            if roi_sign.size == 0:
                return

            # 3. 화살표(흰색) 추출
            roi_hsv = cv2.cvtColor(roi_sign, cv2.COLOR_BGR2HSV)
            # 흰색: 채도(S) 낮음, 명도(V) 높음
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 60, 255])
            white_mask = cv2.inRange(roi_hsv, lower_white, upper_white)
            
            # 4. 픽셀 합 기반 방향 판정
            # ROI를 4분할하여 하단 좌/우 영역의 흰색 픽셀 수 계산
            rh, rw = white_mask.shape[:2]
            cy, cx = rh // 2, rw // 2
            
            # 하단 왼쪽 (Bottom-Left)
            # y: cy ~ rh, x: 0 ~ cx
            bottom_left_roi = white_mask[cy:rh, 0:cx]
            left_sum = cv2.countNonZero(bottom_left_roi)
            
            # 하단 오른쪽 (Bottom-Right)
            # y: cy ~ rh, x: cx ~ rw
            bottom_right_roi = white_mask[cy:rh, cx:rw]
            right_sum = cv2.countNonZero(bottom_right_roi)
            
            direction = "Unknown"
            
            # 판정 로직
            # 왼쪽 아래 픽셀이 더 많으면 -> 스템이 왼쪽에 있음 -> 우회전 표지판 (Right Arrow)
            if left_sum > right_sum * 1.1:
                direction = "RIGHT"
            # 오른쪽 아래 픽셀이 더 많으면 -> 스템이 오른쪽에 있음 -> 좌회전 표지판 (Left Arrow)
            elif right_sum > left_sum * 1.1:
                direction = "LEFT"
            
            # 상태 업데이트 (IDLE일 때만)
            if direction != "Unknown" and self.sign_state == "IDLE":
                self.sign_state = "APPROACHING"
                self.sign_direction = direction
                self.sign_stop_line_seen = False
                rospy.loginfo(f"Traffic Sign Detected: {direction}. Waiting for stop line.")

            # 디버깅 로그
            rospy.loginfo_throttle(0.5, f"Sign Pixel Sums -> L(Bottom-Left): {left_sum}, R(Bottom-Right): {right_sum} => {direction}")
            
            # 5. 결과 시각화
            color = (0, 255, 0) if direction != "Unknown" else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h_rect), color, 2)
            label = f"{direction}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 디버그 창
            if self.enable_viz:
                # 시각화를 위해 분할선 그리기
                debug_mask = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
                cv2.line(debug_mask, (0, cy), (rw, cy), (0, 0, 255), 1) # 가로선
                cv2.line(debug_mask, (cx, cy), (cx, rh), (0, 0, 255), 1) # 세로선 (하단만)
                cv2.imshow("Sign Arrow Mask (Split)", debug_mask)

        if self.enable_viz:
            cv2.imshow("Sign Blue Mask", blue_mask)
    def _detect_crosswalk(self, frame):
        """
        Detect Crosswalk: Vertical rectangles aligned horizontally (Zebra crossing)
        Improved Logic:
        1. Detect Vertical Lines.
        2. Cluster segments by X-coordinate (merge broken lines).
        3. Check for a sequence of these vertical edges with appropriate spacing.
        """
        # 1. BEV Warp
        try:
            warped = self.warper.warp(frame)
        except Exception:
            return False

        # 2. Edge Detection
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 120) # Lower thresholds

        # 3. Hough Lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=20,          # Lower threshold
            minLineLength=15,      # Shorter lines accepted
            maxLineGap=20          # Larger gap to connect segments
        )

        if lines is None:
            if self.enable_viz:
                cv2.imshow("Crosswalk Debug", warped)
                cv2.imshow("Crosswalk Edges", edges)
            return False

        # 4. Filter Vertical Lines
        vertical_segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Vertical check: near +/- 90 degrees
            if 60 < abs(angle) < 120: # Relaxed angle
                vertical_segments.append(line[0])

        if not vertical_segments:
            return False

        # 5. Cluster by X-coordinate
        # Group segments that are vertically aligned (similar X)
        vertical_segments.sort(key=lambda l: (l[0] + l[2]) / 2) # Sort by avg X
        
        clusters = []
        current_cluster = []
        
        # X tolerance to consider segments as part of the same vertical edge
        x_tolerance = 15 
        
        for seg in vertical_segments:
            seg_x = (seg[0] + seg[2]) / 2
            
            if not current_cluster:
                current_cluster.append(seg)
                continue
                
            last_seg = current_cluster[-1]
            last_x = (last_seg[0] + last_seg[2]) / 2
            
            if abs(seg_x - last_x) <= x_tolerance:
                current_cluster.append(seg)
            else:
                clusters.append(current_cluster)
                current_cluster = [seg]
        
        if current_cluster:
            clusters.append(current_cluster)
            
        # 6. Analyze Clusters (Valid Vertical Edges)
        valid_edges = []
        for cl in clusters:
            # Calculate total vertical span
            ys = [min(s[1], s[3]) for s in cl]
            ye = [max(s[1], s[3]) for s in cl]
            min_y = min(ys)
            max_y = max(ye)
            span = max_y - min_y
            
            # Avg X
            xs = [(s[0] + s[2]) / 2 for s in cl]
            avg_x = sum(xs) / len(xs)
            
            # X-coordinate filtering (Ignore left side noise)
            if avg_x < 200:
                continue
            
            # Must have significant vertical span
            if span > 40: 
                valid_edges.append({'x': avg_x, 'min_y': min_y, 'max_y': max_y})

        # 7. Check for Horizontal Pattern (Sequence of Edges)
        # We need at least 4 edges (approx 2 stripes)
        if len(valid_edges) < 4:
            if self.enable_viz:
                debug_img = warped.copy()
                for edge in valid_edges:
                    cv2.line(debug_img, (int(edge['x']), int(edge['min_y'])), (int(edge['x']), int(edge['max_y'])), (0, 255, 255), 2)
                cv2.imshow("Crosswalk Debug", debug_img)
            return False
            
        # Check spacing between edges
        # Stripes have width, spaces have width. Both create gaps between edges.
        # We look for a consistent chain of edges.
        
        consecutive_count = 1
        max_consecutive = 1
        
        for i in range(1, len(valid_edges)):
            dx = valid_edges[i]['x'] - valid_edges[i-1]['x']
            
            # Expected gap: 20px ~ 150px
            if 20 <= dx <= 150:
                consecutive_count += 1
            else:
                max_consecutive = max(max_consecutive, consecutive_count)
                consecutive_count = 1
                
        max_consecutive = max(max_consecutive, consecutive_count)
        
        is_crosswalk = max_consecutive >= 4
        
        if self.enable_viz:
            debug_img = warped.copy()
            for edge in valid_edges:
                cv2.line(debug_img, (int(edge['x']), int(edge['min_y'])), (int(edge['x']), int(edge['max_y'])), (0, 255, 0), 2)
            
            if is_crosswalk:
                cv2.putText(debug_img, f"CROSSWALK! (Count: {max_consecutive})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(debug_img, f"Edges: {max_consecutive}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
            cv2.imshow("Crosswalk Debug", debug_img)
            cv2.imshow("Crosswalk Edges", edges)

        return is_crosswalk

    def _detect_stop_line(self, frame):
        """
        BEV 변환 후 하단부의 가로로 긴 흰색 막대(정지선) 검출
        """
        # 1. BEV 변환
        try:
            warped = self.warper.warp(frame)
        except Exception:
            return False
            
        h, w = warped.shape[:2]
        
        # 2. ROI 설정 (하단 40% 영역)
        roi_top = int(h * 0.6)
        roi = warped[roi_top:, :]
        
        # 3. 흰색 영역 추출
        blur = cv2.GaussianBlur(roi, (5, 5), 0)
        hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
        
        lower_white = np.array([0, 130, 0]) # 밝기 기준 완화 (150 -> 130)
        upper_white = np.array([180, 255, 255])
        mask = cv2.inRange(hls, lower_white, upper_white)
        
        # 4. Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 5. 윤곽선 검출
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        has_stop_line = False
        debug_roi = roi.copy() if self.enable_viz else None
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200: continue # 면적 기준 완화 (500 -> 200)
            
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            # 조건: 가로로 긴 직사각형 (bw > bh * 2.0)
            # 그리고 너비가 화면의 일정 비율 이상이어야 함 (예: 15% 이상)
            if bw > bh * 2.0 and bw > w * 0.15: # 비율 기준 대폭 완화
                has_stop_line = True
                if self.enable_viz:
                    cv2.rectangle(debug_roi, (x, y), (x+bw, y+bh), (0, 0, 255), 2)
                    cv2.putText(debug_roi, "Stop Line", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        if self.enable_viz:
            cv2.imshow("Stop Line Debug", debug_roi)
            
        return has_stop_line

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
