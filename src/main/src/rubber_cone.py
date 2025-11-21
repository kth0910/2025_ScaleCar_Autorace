#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import rospy
import os

from std_msgs.msg import Int32, String, Float32
from sensor_msgs.msg import Image 
from sensor_msgs.msg import LaserScan

class ClusterLidar :

    def __init__(self) :
        # subscriber: /scan 기반. 현재는 미구현이므로 placeholder로 동작
        rospy.Subscriber("/scan", LaserScan, self.rubber_callback)
        # publisher: 조향각을 전달함
        self.rabacon_pub = rospy.Publisher("rubber_cone", Float32, queue_size = 5)
        self.angle = 0.0 


    def rubber_callback(self, _data):
        # 아직 라이다 스캔 기반 원뿔 감지는 미구현 -> 미션 비활성 유지용 상수 발행
        self.rabacon_pub.publish(1000.0)


def run():
    rospy.init_node("rubber_drive")
    cluster = ClusterLidar()
    rospy.spin()


if __name__ == '__main__':
    run()