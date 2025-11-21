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
        self.left_fit = None
        self.right_fit = None
        self.leftx = None
        self.rightx = None
        self.left_cnt = 25
        self.right_cnt = 25
        self.current_lane = 1
        rospy.Subscriber("currentLane", Int32, self.voidCallback)

    def voidCallback(self, _data):
        self.current_lane = _data.data

    def _safe_mean(self, values, fallback):
        if values.size == 0:
            return fallback
        return int(np.mean(values))

    def _histogram_bases(self, img):
        height, width = img.shape[:2]
        histogram = np.sum(img[height//2:, :], axis=0)
        midpoint = width // 2
        left_hist = histogram[:midpoint]
        right_hist = histogram[midpoint:]

        default_left = int(width * 0.25)
        default_right = int(width * 0.75)

        left_base = int(np.argmax(left_hist)) if np.any(left_hist) else default_left
        right_base = int(np.argmax(right_hist)) + midpoint if np.any(right_hist) else default_right
        return left_base, right_base

    def slidewindow(self, img):
        x_location = 280

        # 이미지 초기화 및 크기 설정       
        out_img = np.dstack((img, img, img)) * 255 # 입력 이미지를 기반으로 출력 이미지 시각화
        height = img.shape[0]
        width = img.shape[1]

        # 슬라이딩 윈도우 관련 설정
        window_height = 7 # 슬라이딩 윈도우 높이
        nwindows = 30     # 슬라이딩 윈도우 개수
        
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        if nonzerox.size == 0:
            return out_img, x_location, self.current_line
        
        # 차선 박스의 초기 위치와 크기 설정
        margin = 20 
        minpix = 10
        left_lane_inds = []
        right_lane_inds = []

        # 윈도우 박스 및 라인 위치 초기화
        win_h1 = 400
        win_h2 = 480
        win_l_w_l = 140
        win_l_w_r = 240
        win_r_w_l = 310
        win_r_w_r = 440
        
        # 왼쪽 차선 박스 그리기
        pts_left = np.array([[win_l_w_l, win_h2], [win_l_w_l, win_h1], [win_l_w_r, win_h1], [win_l_w_r, win_h2]], np.int32)
        cv2.polylines(out_img, [pts_left], False, (0,255,0), 1)

        # 오른쪽 차선 박스 그리기
        pts_right = np.array([[win_r_w_l, win_h2], [win_r_w_l, win_h1], [win_r_w_r, win_h1], [win_r_w_r, win_h2]], np.int32)
        cv2.polylines(out_img, [pts_right], False, (255,0,0), 1)
        
        # 가운데 차선 박스 그리기
        pts_catch = np.array([[0, 340], [width, 340]], np.int32)
        cv2.polylines(out_img, [pts_catch], False, (0,120,120), 1)


        # 초기 차선 인덱스 설정
        good_left_inds = ((nonzerox >= win_l_w_l) & (nonzeroy <= win_h2) & (nonzeroy > win_h1) & (nonzerox <= win_l_w_r)).nonzero()[0]
        good_right_inds = ((nonzerox >= win_r_w_l) & (nonzeroy <= win_h2) & (nonzeroy > win_h1) & (nonzerox <= win_r_w_r)).nonzero()[0]

        left_base, right_base = self._histogram_bases(img)

        if self.current_lane == 1:
            self.current_line = "LEFT"
            line_flag = 1
            x_current = self._safe_mean(nonzerox[good_left_inds], left_base)
            y_current = self._safe_mean(nonzeroy[good_left_inds], height - 1)
        elif self.current_lane == 2:
            self.current_line = "RIGHT"
            line_flag = 2
            x_current = self._safe_mean(nonzerox[good_right_inds], right_base)
            y_current = self._safe_mean(nonzeroy[good_right_inds], height - 1)
        else:
            self.current_line = "MID"
            line_flag = 3
            center_inds = ((nonzeroy >= win_h1) & (nonzeroy <= win_h2)).nonzero()[0]
            x_current = self._safe_mean(nonzerox[center_inds], width // 2)
            y_current = self._safe_mean(nonzeroy[center_inds], height - 1)

        if x_current is None:
            x_current = width // 2
        if y_current is None:
            y_current = height - 1

        if line_flag != 3:
            for window in range(nwindows):
                win_y_low = y_current - (window + 1) * window_height
                win_y_high = y_current - window * window_height
                win_x_low = x_current - margin
                win_x_high = x_current + margin

                win_y_low = max(win_y_low, 0)
                win_y_high = min(win_y_high, height)

                cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high),
                              (0, 255, 0) if line_flag == 1 else (255, 0, 0), 1)

                good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                             (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

                if line_flag == 1:
                    left_lane_inds.extend(good_inds)
                else:
                    right_lane_inds.extend(good_inds)

                if good_inds.size > minpix:
                    x_current = int(np.mean(nonzerox[good_inds]))
                else:
                    x_current = x_current if line_flag == 1 else x_current

                if win_y_low >= 338 and win_y_low < 344:
                    offset = int(width * 0.135)
                    if line_flag == 1:
                        x_location = x_current + offset
                    else:
                        x_location = x_current - offset

        if line_flag == 3:
            if center_inds.size > 0:
                x_location = int(np.mean(nonzerox[center_inds]))
            else:
                x_location = width // 2
        
        return out_img, x_location, self.current_line
