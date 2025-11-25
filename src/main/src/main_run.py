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
        self.steering_offset = rospy.get_param("~steering_offset", 0.60)
        self.steering_smoothing = rospy.get_param("~steering_smoothing", 0.55)
        self.min_servo = rospy.get_param("~min_servo", 0.05)
        self.max_servo = rospy.get_param("~max_servo", 0.95)
        self.speed_value = rospy.get_param("~speed", 2000.0)
        self.center_smoothing = rospy.get_param("~center_smoothing", 0.4)
        self.max_center_step = rospy.get_param("~max_center_step", 25.0)
        self.bias_correction_gain = rospy.get_param("~bias_correction_gain", 1e-4)
        self.max_error_bias = rospy.get_param("~max_error_bias", 120.0)
        self.error_bias = rospy.get_param("~initial_error_bias", 0.0)
        self.max_servo_delta = rospy.get_param("~max_servo_delta", 0.035)
        self.min_mask_pixels = rospy.get_param("~min_mask_pixels", 600)
        self.integral_limit = rospy.get_param("~steering_integral_limit", 500.0)
        self.single_left_ratio = rospy.get_param("~single_left_ratio", 0.35)
        self.single_right_ratio = rospy.get_param("~single_right_ratio", 0.65)

        # 퍼블리셔
        self.speed_pub = rospy.Publisher("/commands/motor/speed", Float64, queue_size=1)
        self.steering_pub = rospy.Publisher("/commands/servo/position", Float64, queue_size=1)
        self.center_pub = rospy.Publisher("/lane_center_x", Float64, queue_size=1)
        self.error_pub = rospy.Publisher("/lane_error", Float64, queue_size=1)

        rospy.Subscriber("usb_cam/image_rect_color", Image, self.image_callback, queue_size=1)
        
        # 라이다 회피 우선: lidar_avoidance가 제어 중인지 확인
        # lidar_avoidance가 제어 중인지 나타내는 토픽 구독
        self.lidar_controlling = False
        self.last_lidar_servo_time = 0.0
        self.lidar_timeout = 0.5  # 0.5초 동안 라이다 제어가 없으면 카메라 제어 사용
        self.last_servo_publish_time = 0.0  # 자신이 발행한 시간 추적

        self.enable_viz = rospy.get_param(
            "~enable_viz",
            bool(os.environ.get("DISPLAY"))
        )
        if not self.enable_viz:
            rospy.logwarn("DISPLAY not found. Visualization disabled.")

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
        
        # 라이다 장애물 거리 구독 (속도 제어용) - 초기화 먼저
        self.closest_obstacle = None
        self.last_obstacle_time = 0.0
        rospy.Subscriber("lidar_avoidance/closest_obstacle", Float64, self._obstacle_distance_callback, queue_size=1)
        
        # 속도 제어 파라미터
        self.max_drive_speed = rospy.get_param("~max_drive_speed", 0.4)  # m/s
        self.min_drive_speed = rospy.get_param("~min_drive_speed", 0.15)  # m/s
        self.max_pwm = rospy.get_param("~max_pwm", 1500.0)
        self.min_pwm = rospy.get_param("~min_pwm", 900.0)
        self.speed_reduction_start = rospy.get_param("~speed_reduction_start", 0.30)  # 30cm부터 속도 감소
        self.hard_stop_distance = rospy.get_param("~hard_stop_distance", 0.20)  # 20cm에서 완전 정지
        self.current_speed_pwm = self.max_pwm
        self.speed_smoothing_rate = rospy.get_param("~speed_smoothing_rate", 100.0)  # PWM 변화율 (부드러운 변화를 위해 감소)
        self.speed_smoothing_factor = rospy.get_param("~speed_smoothing_factor", 0.3)  # 지수적 스무딩 계수 (0.0~1.0, 작을수록 더 부드러움)

        rospy.on_shutdown(self._cleanup)
        rospy.loginfo("LaneFollower initialized.")

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
            delta_servo = self._clamp(delta_servo, -self.max_servo_delta, self.max_servo_delta)
            limited_target = self.prev_servo + delta_servo
        else:
            # 차선이 검출되지 않으면 기존 방향 유지
            limited_target = self.prev_servo

        smoothed_servo = (
            self.steering_smoothing * self.prev_servo
            + (1.0 - self.steering_smoothing) * limited_target
        )
        self.prev_servo = smoothed_servo

        # 라이다 우선: lidar_avoidance가 제어 중인지 확인
        # lidar_avoidance는 /ackermann_cmd를 발행하므로, 이를 확인하거나
        # 간단히 타임아웃 기반으로 처리 (lidar_avoidance가 활성화되어 있으면 주기적으로 제어)
        current_time = rospy.get_time()
        is_lidar_controlling = (current_time - self.last_lidar_servo_time < self.lidar_timeout)
        
        # 속도는 항상 main_run.py에서 제어 (라이다 제어 중이어도)
        # 라이다가 제어 중일 때는 조향만 라이다가 제어하고, 속도는 main_run.py가 제어
        target_speed_pwm = self._compute_speed_with_obstacle()
        self.speed_pub.publish(Float64(target_speed_pwm))
        
        # 조향은 라이다가 제어 중이면 발행하지 않음
        if not is_lidar_controlling:
            # 라이다가 제어하지 않을 때만 카메라 차선 추종 제어 발행
            self.last_servo_publish_time = current_time
            self.steering_pub.publish(Float64(smoothed_servo))

        if self.enable_viz:
            cv2.imshow("Lane Frame", frame)
            cv2.imshow("Lane Mask", lane_mask)
            if slide_img is not None:
                cv2.imshow("Sliding Window", slide_img)
            cv2.waitKey(1)

    def _create_lane_mask(self, frame):
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
    
    def _obstacle_distance_callback(self, msg):
        """라이다에서 가장 가까운 장애물 거리 수신"""
        self.closest_obstacle = msg.data
        self.last_obstacle_time = rospy.get_time()

    def _compute_speed_with_obstacle(self) -> float:
        """장애물 거리에 따라 속도 계산"""
        current_time = rospy.get_time()
        dt = max(0.001, min(0.1, current_time - self.prev_time))
        self.prev_time = current_time  # 시간 업데이트
        
        # 기본 속도 (PWM) - 범위 제한 (역회전 방지)
        base_pwm = self._clamp(self.speed_value, self.min_pwm, self.max_pwm)
        
        # 장애물이 감지되면 속도 감소
        if self.closest_obstacle is not None and (current_time - self.last_obstacle_time) < 0.5:
            closest = self.closest_obstacle
            
            # 거리 기반 속도 감소: 30cm부터 점진적으로 감소, 20cm에서 완전 정지
            speed_reduction_factor = 1.0
            if closest < self.speed_reduction_start:
                # 30cm ~ 20cm 구간에서 선형 감소
                reduction_range = self.speed_reduction_start - self.hard_stop_distance  # 0.3 - 0.2 = 0.1m
                if reduction_range > 0:
                    distance_in_range = closest - self.hard_stop_distance
                    speed_reduction_factor = self._clamp(distance_in_range / reduction_range, 0.0, 1.0)
            elif closest <= self.hard_stop_distance:
                # 20cm 이하는 완전 정지 (min_pwm으로 설정, 역회전 방지)
                speed_reduction_factor = 0.0
            
            # 속도 계산 (PWM) - min_pwm 이상으로 유지 (역회전 방지)
            speed_range = self.max_pwm - self.min_pwm
            target_pwm = self.min_pwm + speed_range * speed_reduction_factor
        else:
            # 장애물이 없으면 기본 속도 (범위 제한)
            target_pwm = base_pwm
        
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
        
        # limited_target도 범위 내로 제한
        limited_target = self._clamp(limited_target, self.min_pwm, self.max_pwm)
        
        # 2단계: 지수적 스무딩 (더 부드러운 변화)
        smoothed_pwm = (
            (1.0 - self.speed_smoothing_factor) * self.current_speed_pwm
            + self.speed_smoothing_factor * limited_target
        )
        
        # smoothed_pwm을 범위 내로 확실히 제한 (역회전 방지)
        smoothed_pwm = self._clamp(smoothed_pwm, self.min_pwm, self.max_pwm)
        
        # 최소 변화량 이하는 무시 (미세한 진동 방지)
        if abs(smoothed_pwm - self.current_speed_pwm) < 1.0:
            smoothed_pwm = self.current_speed_pwm
        
        # 최종 값도 범위 확인 (이중 안전장치)
        self.current_speed_pwm = self._clamp(smoothed_pwm, self.min_pwm, self.max_pwm)
        
        return self.current_speed_pwm

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
