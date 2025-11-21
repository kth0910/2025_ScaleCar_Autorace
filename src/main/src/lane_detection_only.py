#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from std_msgs.msg import Float64, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from warper import Warper
from slidewindow import SlideWindow

class LaneDetectionOnly:
    
    def __init__(self):
        self.warper = Warper()
        self.slidewindow = SlideWindow()
        self.bridge = CvBridge()
        
        # 차선 중심 위치 퍼블리시
        self.lane_center_pub = rospy.Publisher("/lane_center_x", Float64, queue_size=1)
        self.current_lane_pub = rospy.Publisher("/current_lane", String, queue_size=1)
        
        # 카메라 이미지 구독
        rospy.Subscriber("usb_cam/image_rect_color", Image, self.image_callback)
        
        # 초기화 플래그
        self.initialized = False
        
        # 차선 중심 위치
        self.slide_x_location = 280  # 기본값 (이미지 중앙)
        self.current_lane_window = "MID"
        
        rospy.loginfo("Lane Detection Only Node Started")
    
    def image_callback(self, _data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(_data, "bgr8")
            
            # 트랙바 초기화 (한 번만)
            if not self.initialized:
                cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)
                cv2.createTrackbar('low_H', 'Lane Detection', 128, 360, self.nothing)
                cv2.createTrackbar('low_L', 'Lane Detection', 134, 255, self.nothing)
                cv2.createTrackbar('low_S', 'Lane Detection', 87, 255, self.nothing)
                cv2.createTrackbar('high_H', 'Lane Detection', 334, 360, self.nothing)
                cv2.createTrackbar('high_L', 'Lane Detection', 255, 255, self.nothing)
                cv2.createTrackbar('high_S', 'Lane Detection', 251, 255, self.nothing)
                self.initialized = True
            
            # 트랙바 값 읽기
            low_H = cv2.getTrackbarPos('low_H', 'Lane Detection')
            low_L = cv2.getTrackbarPos('low_L', 'Lane Detection')
            low_S = cv2.getTrackbarPos('low_S', 'Lane Detection')
            high_H = cv2.getTrackbarPos('high_H', 'Lane Detection')
            high_L = cv2.getTrackbarPos('high_L', 'Lane Detection')
            high_S = cv2.getTrackbarPos('high_S', 'Lane Detection')
            
            # HLS 변환
            hls_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HLS)
            
            # 차선 색상 필터링
            lower_lane = np.array([low_H, low_L, low_S])
            upper_lane = np.array([high_H, high_L, high_S])
            lane_image = cv2.inRange(hls_image, lower_lane, upper_lane)
            
            # 차선 감지 수행
            self.laneDetection(lane_image)
            
            # 결과 시각화
            cv2.imshow("Original", cv_image)
            cv2.imshow("Lane Binary", lane_image)
            cv2.waitKey(1)
            
        except Exception as e:
            rospy.logerr(f"Error in image callback: {e}")
    
    def laneDetection(self, lane_image):
        try:
            # 가우시안 블러
            kernel_size = 5
            blur_img = cv2.GaussianBlur(lane_image, (kernel_size, kernel_size), 0)
            
            # 투시 변환
            warped_img = self.warper.warp(blur_img)
            
            # 슬라이딩 윈도우로 차선 감지
            self.slide_img, self.slide_x_location, self.current_lane_window = \
                self.slidewindow.slidewindow(warped_img)
            
            # 결과 시각화
            cv2.imshow("Warped", warped_img)
            cv2.imshow("Slide Window Result", self.slide_img)
            
            # 차선 중심 위치 퍼블리시
            center_msg = Float64()
            center_msg.data = float(self.slide_x_location)
            self.lane_center_pub.publish(center_msg)
            
            # 현재 차선 정보 로그
            rospy.loginfo(f"Lane Center X: {self.slide_x_location}, Current Lane: {self.current_lane_window}")
            
        except Exception as e:
            rospy.logerr(f"Error in lane detection: {e}")
    
    def nothing(self, x):
        pass

def run():
    rospy.init_node("lane_detection_only")
    detector = LaneDetectionOnly()
    rospy.spin()

if __name__ == '__main__':
    run()

