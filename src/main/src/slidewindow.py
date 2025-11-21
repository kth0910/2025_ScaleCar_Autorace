import cv2
import numpy as np


class SlideWindow:

    def __init__(self):
        self.lane_width_ratio = 0.27
        self.min_lane_pixels = 150
        self.current_line = "MID"
        self.last_lane_width = None
        self.lane_width_smoothing = 0.4

    def slidewindow(self, img):
        out_img = np.dstack((img, img, img)) * 255
        height, width = img.shape[:2]

        histogram = np.sum(img[height // 2:, :], axis=0)
        midpoint = width // 2
        leftx_base = np.argmax(histogram[:midpoint]) if np.any(histogram[:midpoint]) else int(width * 0.25)
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint if np.any(histogram[midpoint:]) else int(width * 0.75)

        nwindows = 12
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

            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_exists = len(left_lane_inds) > self.min_lane_pixels
        right_exists = len(right_lane_inds) > self.min_lane_pixels

        lane_width_default = width * self.lane_width_ratio
        lane_width_px = self.last_lane_width if self.last_lane_width is not None else lane_width_default

        left_x = int(np.mean(nonzerox[left_lane_inds])) if left_exists else None
        right_x = int(np.mean(nonzerox[right_lane_inds])) if right_exists else None

        if left_exists and right_exists:
            measured_width = max(1, right_x - left_x)
            if 0.5 * lane_width_default < measured_width < 1.5 * lane_width_default:
                if self.last_lane_width is None:
                    lane_width_px = measured_width
                else:
                    lane_width_px = (
                        self.lane_width_smoothing * self.last_lane_width +
                        (1 - self.lane_width_smoothing) * measured_width
                    )
                self.last_lane_width = lane_width_px
            center_x = (left_x + right_x) // 2
            self.current_line = "BOTH"
        elif left_exists:
            center_x = int(left_x + lane_width_px / 2.0)
            self.current_line = "LEFT_ONLY"
        elif right_exists:
            center_x = int(right_x - lane_width_px / 2.0)
            self.current_line = "RIGHT_ONLY"
        else:
            center_x = width // 2
            self.current_line = "NONE"

        return out_img, center_x, self.current_line
