import cv2
import numpy as np


class SlideWindow:

    def __init__(self):
        self.lane_width_ratio = 0.27
        self.min_lane_pixels = 150
        self.current_line = "MID"
        self.left_exists = False
        self.right_exists = False
        self.left_x = None
        self.right_x = None
        self.lane_width_px = None
        self.frame_width = None

    def slidewindow(self, img):
        out_img = np.dstack((img, img, img)) * 255
        height, width = img.shape[:2]
        self.frame_width = width

        histogram = np.sum(img[height // 2:, :], axis=0)
        midpoint = width // 2
        left_hist = histogram[:midpoint]
        right_hist = histogram[midpoint:]
        leftx_base = np.argmax(left_hist) if np.any(left_hist) else int(width * 0.25)
        rightx_base = np.argmax(right_hist) + midpoint if np.any(right_hist) else int(width * 0.75)

        nwindows = 12
        # 상위 윈도우는 곡률이 심할 때 너무 미리 반응하게 하므로,
        # 조향 계산에는 하단 윈도우만 우선적으로 사용합니다.
        # 12개 중 하단 5개(약 40%)만 사용하여 "turns too early" 문제를 완화합니다.
        calc_nwindows = 5
        
        window_height = height // nwindows
        margin = 35
        minpix = 20

        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base
        left_lane_inds = []
        right_lane_inds = []
        
        # 조향 계산용 인덱스 (하단 윈도우만 포함)
        left_lane_inds_calc = []
        right_lane_inds_calc = []

        if nonzerox.size == 0:
            self.left_exists = False
            self.right_exists = False
            self.left_x = None
            self.right_x = None
            self.lane_width_px = int(width * self.lane_width_ratio)
            self.current_line = "MID"
            return out_img, width // 2, self.current_line

        for window in range(nwindows):
            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height
            win_y_low = max(win_y_low, 0)

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

            left_lane_inds.extend(good_left_inds)
            right_lane_inds.extend(good_right_inds)
            
            # 하단 윈도우인 경우 계산용 리스트에도 추가
            if window < calc_nwindows:
                left_lane_inds_calc.extend(good_left_inds)
                right_lane_inds_calc.extend(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_exists = len(left_lane_inds) > self.min_lane_pixels
        right_exists = len(right_lane_inds) > self.min_lane_pixels
        lane_width_px = int(width * self.lane_width_ratio)

        if left_exists:
            # 계산용 픽셀이 충분하면 그것만 사용 (가까운 곳 우선)
            if len(left_lane_inds_calc) > minpix * 2:
                left_x = int(np.mean(nonzerox[left_lane_inds_calc]))
            else:
                # 가까운 곳에 픽셀이 부족하면 전체 사용
                left_x = int(np.mean(nonzerox[left_lane_inds]))
        else:
            left_x = None

        if right_exists:
            if len(right_lane_inds_calc) > minpix * 2:
                right_x = int(np.mean(nonzerox[right_lane_inds_calc]))
            else:
                right_x = int(np.mean(nonzerox[right_lane_inds]))
        else:
            right_x = None

        self.left_exists = left_exists
        self.right_exists = right_exists
        self.left_x = left_x
        self.right_x = right_x
        self.lane_width_px = lane_width_px

        if left_exists and right_exists:
            center_x = (left_x + right_x) // 2
            self.current_line = "MID"
        elif left_exists:
            center_x = left_x + lane_width_px
            self.current_line = "LEFT"
        elif right_exists:
            center_x = right_x - lane_width_px
            self.current_line = "RIGHT"
        else:
            center_x = width // 2
            self.current_line = "MID"

        return out_img, center_x, self.current_line
