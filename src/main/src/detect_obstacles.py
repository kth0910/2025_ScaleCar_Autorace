#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import rospy
import math
import time
from std_msgs.msg import String, Int32, Float32
from sensor_msgs.msg import LaserScan

class LidarReceiver():

    def __init__(self):
        # subscriber: LaserScan(/scan) 기반으로 장애물을 인식함
        rospy.Subscriber("/scan", LaserScan, self.lidar_callback)
        # publisher: 장애물 인식 상태를 전달함
        self.warning_pub = rospy.Publisher("lidar_warning", String, queue_size=5)
        self.object_pub = rospy.Publisher("object_condition", Float32, queue_size=5)

        self.count_flag = 0
        self.flag_flag = 0
        self.count_t1 = 0
        self.x1 = 0
        self.x2 = 0


    def lidar_callback(self, scan):
        # ROI: 차량 전방 좁은 폭의 박스 (라이다 좌표계 기준)
        # 주의: 실제 프레임 정의에 따라 x,y 정의가 다를 수 있음
        left_y = -0.2   # m
        right_y = 0.2   # m
        near_x = -0.3   # m (차량 바로 앞)
        far_x = -1.3    # m (더 먼 앞)
        WARNING_CNT = 1

        angle = scan.angle_min
        self.point_cnt = 0
        close_lats = []

        for r in scan.ranges:
            if math.isinf(r) or math.isnan(r):
                angle += scan.angle_increment
                continue
            # 폴라 -> 카테시안 (라이다 좌표계: x=forward, y=left 가정)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            angle += scan.angle_increment

            # 기존 로직과 부호 일관성 유지를 위해 음의 x를 전방으로 간주
            # 필요 시 실제 설치 방향에 맞게 부호 조정
            x_forward = -x
            y_lateral = y

            # object_condition: 아주 근거리 전방의 y 편차
            if left_y < y_lateral < right_y and near_x < x_forward < 0.0:
                close_lats.append(y_lateral)

            # WARNING 판정용 영역
            if left_y < y_lateral < right_y and far_x < x_forward < 0.0:
                self.point_cnt += 1

        if close_lats:
            # 대표값(평균) 발행
            self.object_pub.publish(sum(close_lats) / float(len(close_lats)))

        if self.point_cnt >= WARNING_CNT:
            self.warning_pub.publish("WARNING")
        else:
            self.warning_pub.publish("safe")


def run():
    rospy.init_node("lidar_example")
    new_class = LidarReceiver()
    rospy.spin()


if __name__ == '__main__':
    run()