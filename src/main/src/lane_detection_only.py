#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from std_msgs.msg import Float64, String
from cv_bridge import CvBridge

from warper import Warper
from slidewindow import SlideWindow

class LaneDetectionOnly:
    
    def __init__(self):
        self.warper = Warper()
        self.slidewindow = SlideWindow()
        self.bridge = CvBridge()
        
        self.initialized = False
        
        # slide window return variable initialization
        self.slide_img = None 
        self.slide_x_location = 0
        self.current_lane_window = ""
        
        # Publisher: 차선 중심 위치 퍼블리시
        self.lane_center_pub = rospy.Publisher("/lane_center_x", Float64, queue_size=1)
        self.lane_info_pub = rospy.Publisher("/lane_info", String, queue_size=1)
        
        # Subscriber: 보정된 카메라 이미지 구독
        rospy.Subscriber("usb_cam/image_rect_color", Image, self.laneCallback)
        
        rospy.loginfo("Lane Detection Only Node Started")
    
    def laneCallback(self, _data):
        # detect lane
        if self.initialized == False:
            cv2.namedWindow("Lane_Detection", cv2.WINDOW_NORMAL) 
            cv2.createTrackbar('low_H', 'Lane_Detection', 128, 360, nothing)
            cv2.createTrackbar('low_L', 'Lane_Detection', 134, 255, nothing)
            cv2.createTrackbar('low_S', 'Lane_Detection', 87, 255, nothing)
            cv2.createTrackbar('high_H', 'Lane_Detection', 334, 360, nothing)
            cv2.createTrackbar('high_L', 'Lane_Detection', 255, 255, nothing)
            cv2.createTrackbar('high_S', 'Lane_Detection', 251, 255, nothing)
            self.initialized = True
        
        cv2_image = self.bridge.imgmsg_to_cv2(_data)
        originalImg = cv2_image.copy()
        
        # HLS 색공간 변환 및 차선 추출
        low_H = cv2.getTrackbarPos('low_H', 'Lane_Detection')
        low_L = cv2.getTrackbarPos('low_L', 'Lane_Detection')
        low_S = cv2.getTrackbarPos('low_S', 'Lane_Detection')
        high_H = cv2.getTrackbarPos('high_H', 'Lane_Detection')
        high_L = cv2.getTrackbarPos('high_L', 'Lane_Detection')
        high_S = cv2.getTrackbarPos('high_S', 'Lane_Detection')

        cv2.cvtColor(cv2_image, cv2.COLOR_BGR2HLS) # BGR to HLS

        lower_lane = np.array([low_H, low_L, low_S]) 
        upper_lane = np.array([high_H, high_L, high_S])

        lane_image = cv2.inRange(cv2_image, lower_lane, upper_lane)

        cv2.imshow("Lane Image", lane_image)
        self.laneDetection(lane_image)
        
        # 차선 정보 퍼블리시
        lane_center_msg = Float64()
        lane_center_msg.data = float(self.slide_x_location)
        self.lane_center_pub.publish(lane_center_msg)
        
        rospy.loginfo(f"Lane Center X: {self.slide_x_location}, Lane: {self.current_lane_window}")

        cv2.waitKey(1)
    
    def laneDetection(self, lane_image):
        kernel_size = 5
        blur_img = cv2.GaussianBlur(lane_image, (kernel_size, kernel_size), 0)
        warped_img = self.warper.warp(blur_img)
        self.slide_img, self.slide_x_location, self.current_lane_window = self.slidewindow.slidewindow(warped_img)
        cv2.imshow("slide_img", self.slide_img)

def nothing(x):
    pass

def run():
    rospy.init_node("lane_detection_only")
    detector = LaneDetectionOnly()
    rospy.spin()

if __name__ == '__main__':
    run()

