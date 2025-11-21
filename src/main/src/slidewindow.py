import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import *
from matplotlib.pyplot import *

import rospy
from std_msgs.msg import Int32, String, Float32, Float64
from sensor_msgs.msg import Image

TOTAL_CNT = 50

class SlideWindow:

    def __init__(self):
        self.current_line = "LEFT"
        rospy.Subscriber("currentLane", Int32, self.voidCallback)

    def voidCallback(self, _data):
        self.current_line = _data.data

    def slidewindow(self, binary_warped):
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        height, width = binary_warped.shape[:2]

        histogram = np.sum(binary_warped[height // 2:, :], axis=0)
        midpoint = width // 2
        leftx_base = np.argmax(histogram[:midpoint]) if np.any(histogram[:midpoint]) else int(width * 0.25)
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint if np.any(histogram[midpoint:]) else int(width * 0.75)

        nwindows = 20
        window_height = int(height / nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        if nonzerox.size == 0:
            return out_img, width // 2, "UNKNOWN"

        margin = 50
        minpix = 30
        leftx_current = leftx_base
        rightx_current = rightx_base
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 1)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (255, 0, 0), 1)

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        left_detected = left_lane_inds.size > 150
        right_detected = right_lane_inds.size > 150

        lane_offset = int(width * 0.135)
        y_eval = height - 1

        def eval_poly(fit, y):
            return fit[0] * y ** 2 + fit[1] * y + fit[2]

        if left_detected:
            left_fit = np.polyfit(nonzeroy[left_lane_inds], nonzerox[left_lane_inds], 2)
            left_x = eval_poly(left_fit, y_eval)
        else:
            left_x = leftx_base

        if right_detected:
            right_fit = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 2)
            right_x = eval_poly(right_fit, y_eval)
        else:
            right_x = rightx_base

        if left_detected and right_detected:
            x_location = int((left_x + right_x) / 2)
            current_line = "BOTH"
        elif left_detected:
            x_location = int(left_x + lane_offset)
            current_line = "LEFT"
        elif right_detected:
            x_location = int(right_x - lane_offset)
            current_line = "RIGHT"
        else:
            x_location = width // 2
            current_line = "UNKNOWN"

        x_location = max(0, min(width - 1, x_location))
        return out_img, x_location, current_line
