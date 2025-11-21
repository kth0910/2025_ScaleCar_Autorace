#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import os
import math

import rospy
import cv2
import numpy as np

from std_msgs.msg import Float64
from sensor_msgs.msg import Image
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

        # 파라미터
        self.desired_center = rospy.get_param("~desired_center", 280.0)
        self.steering_gain = rospy.get_param("~steering_gain", -0.003)
        self.steering_offset = rospy.get_param("~steering_offset", 0.57)
        self.steering_smoothing = rospy.get_param("~steering_smoothing", 0.7)
        self.min_servo = rospy.get_param("~min_servo", 0.1)
        self.max_servo = rospy.get_param("~max_servo", 0.95)
        self.speed_value = rospy.get_param("~speed", 2000.0)
        self.center_smoothing = rospy.get_param("~center_smoothing", 0.5)
        self.max_center_step = rospy.get_param("~max_center_step", 25.0)
        self.bias_correction_gain = rospy.get_param("~bias_correction_gain", 1e-4)
        self.max_error_bias = rospy.get_param("~max_error_bias", 120.0)
        self.error_bias = rospy.get_param("~initial_error_bias", 0.0)
        self.max_servo_delta = rospy.get_param("~max_servo_delta", 0.03)

        # 퍼블리셔
        self.speed_pub = rospy.Publisher("/commands/motor/speed", Float64, queue_size=1)
        self.steering_pub = rospy.Publisher("/commands/servo/position", Float64, queue_size=1)
        self.center_pub = rospy.Publisher("/lane_center_x", Float64, queue_size=1)
        self.error_pub = rospy.Publisher("/lane_error", Float64, queue_size=1)

        rospy.Subscriber("usb_cam/image_rect_color", Image, self.image_callback, queue_size=1)

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
        self.initialized = False

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
        slide_img, center_x = self._run_slidewindow(lane_mask)

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

        self.center_pub.publish(Float64(self.current_center))
        raw_error = self.desired_center - self.current_center
        self.error_bias += self.bias_correction_gain * raw_error
        self.error_bias = self._clamp(
            self.error_bias, -self.max_error_bias, self.max_error_bias
        )
        error = raw_error - self.error_bias
        self.error_pub.publish(Float64(error))

        target_servo = self.steering_offset + self.steering_gain * error
        target_servo = self._clamp(target_servo, self.min_servo, self.max_servo)
        delta_servo = target_servo - self.prev_servo
        delta_servo = self._clamp(delta_servo, -self.max_servo_delta, self.max_servo_delta)
        limited_target = self.prev_servo + delta_servo
        smoothed_servo = (
            self.steering_smoothing * self.prev_servo
            + (1.0 - self.steering_smoothing) * limited_target
        )
        self.prev_servo = smoothed_servo

        self.speed_pub.publish(Float64(self.speed_value))
        self.steering_pub.publish(Float64(smoothed_servo))

        if self.enable_viz:
            cv2.imshow("Lane Frame", frame)
            cv2.imshow("Lane Mask", lane_mask)
            if slide_img is not None:
                cv2.imshow("Sliding Window", slide_img)
            cv2.waitKey(1)

    def _create_lane_mask(self, frame):
        if self.auto_threshold:
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
            low_H = rospy.get_param("~low_H", 128)
            low_L = rospy.get_param("~low_L", 134)
            low_S = rospy.get_param("~low_S", 87)
            high_H = rospy.get_param("~high_H", 334)
            high_L = rospy.get_param("~high_L", 255)
            high_S = rospy.get_param("~high_S", 251)

        lower_lane = np.array([low_H, low_L, low_S], dtype=np.uint8)
        upper_lane = np.array([high_H, high_L, high_S], dtype=np.uint8)
        lane_mask = cv2.inRange(hls, lower_lane, upper_lane)
        return lane_mask

    def _auto_lane_mask(self, frame):
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        lower_yellow = np.array([15, 60, 60])
        upper_yellow = np.array([40, 255, 255])
        lower_white = np.array([0, 200, 0])
        upper_white = np.array([255, 255, 90])
        mask_yellow = cv2.inRange(hls, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(hls, lower_white, upper_white)
        lane_mask = cv2.bitwise_or(mask_yellow, mask_white)
        kernel = np.ones((5, 5), np.uint8)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return lane_mask

    def _run_slidewindow(self, lane_mask):
        try:
            blur_img = cv2.GaussianBlur(lane_mask, (5, 5), 0)
            warped = self.warper.warp(blur_img)
            slide_img, center_x, _ = self.slidewindow.slidewindow(warped)
            return slide_img, center_x
        except Exception as exc:
            rospy.logwarn(f"Slide window failed: {exc}")
            fallback_center = self._center_from_mask(lane_mask)
            return None, fallback_center

    @staticmethod
    def _center_from_mask(mask):
        moments = cv2.moments(mask, binaryImage=True)
        if moments["m00"] > 0:
            return moments["m10"] / moments["m00"]
        return None

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


def run():
    rospy.init_node("lane_follower")
    LaneFollower()
    rospy.spin()


if __name__ == "__main__":
    run()
