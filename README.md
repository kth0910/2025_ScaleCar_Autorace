# 스케일카 기반 자율주행 경진대회

2023년 하계 혁신융합대학 스케일카 기반 자율주행 경진대회

[대회 공지 링크](https://coss.kookmin.ac.kr/fvedu/community/notice.do?mode=view&articleNo=5904366&article.offset=10&articleLimit=10)

## 📺 시연 영상

추후 업로드

## 소개

- 도로 주행
    - **Sliding Window**
    - slidewindow.py
- 어린이 보호 구역
    - [**Aruco Marker Detector** - fiducial](https://github.com/UbiquityRobotics/fiducials/)
    - 
- Rubber cone 주행
    - [Obstacle Detector](https://github.com/tysik/obstacle_detector)
- 정적 장애물
    - [Obstacle Detector](https://github.com/tysik/obstacle_detector)
- 동적 장애물
    - OpenCV

```bash
git clone https://github.com/kmu-kobot/2023_ScaleCar_Autorace.git
catkin_make
roslaunch main main.launch
```

## LiDAR 회피 주행 파이프라인

- `main/src/lidar_avoidance.py` 노드는 `LaserScan(/scan)`을 받아 장애물 좌표를 마커로 시각화하고, 가장 안전한 gap을 따라 Ackermann 조향각과 `/commands/{motor,servo}` PWM을 동시에 출력합니다.
- 새로운 `main/launch/lidar_avoidance.launch` 는 아래 구성요소를 한 번에 올립니다.
  - `rplidar_ros` 드라이버 (포트/baud 인자 제공)
  - 선택적 `vesc_driver`, `ackermann_to_vesc` 변환 및 VESC 파라미터(`racecar/racecar/config/racecar-v2/vesc.yaml`)
  - RViz 설정(`main/rviz/lidar_avoidance.rviz`) : 장애물 MarkerArray, 목표 벡터, 플래닝 Path
- 실행 예시

```bash
# 기본 실행 (RPLIDAR S1 + VESC, 회피주행)
# VESC 드라이버는 기본적으로 활성화되어 있습니다
roslaunch main lidar_avoidance.launch \
  serial_port:=/dev/ttyUSB0 \
  serial_baudrate:=256000 \
  vesc_port:=/dev/ttyVesc

# VESC 포트가 다른 경우
roslaunch main lidar_avoidance.launch \
  serial_port:=/dev/ttyUSB0 \
  vesc_port:=/dev/ttyUSB1  # 실제 VESC 포트에 맞게 변경

# 다른 RPLIDAR 모델 사용 시 baudrate 변경
# A1/A2: serial_baudrate:=115200
# A3: serial_baudrate:=256000 (S1과 동일)
# S2: serial_baudrate:=1000000
```

- **하드웨어 연결 문제 해결**:
  - `RESULT_OPERATION_TIMEOUT` 오류 발생 시:
    1. 시리얼 포트 확인: `ls -l /dev/ttyUSB*` 또는 `ls -l /dev/ttyACM*`
    2. 권한 확인: `sudo chmod 666 /dev/ttyUSB0` (포트에 맞게 변경)
    3. 올바른 포트 지정: `serial_port:=/dev/ttyUSB0` (실제 포트에 맞게)
    4. **RPLIDAR 모델별 baudrate**:
       - **S1 (기본값)**: 256000
       - A3: 256000
       - A1/A2: 115200
       - S2: 1000000
    5. 하드웨어 연결 확인: USB 케이블 및 전원 연결 상태 확인
  - 노드는 `respawn="true"`로 설정되어 있어 연결 실패 시 자동 재시도합니다.
  - **RPLIDAR S1 사용 시**: 기본 baudrate가 256000으로 설정되어 있습니다. 다른 모델 사용 시 `serial_baudrate` 인자를 변경하세요.
  - **VESC 연결 확인**:
    1. VESC 포트 확인: `ls -l /dev/ttyVesc*` 또는 `ls -l /dev/ttyUSB*` (VESC가 연결된 포트)
    2. 권한 설정: `sudo chmod 666 /dev/ttyVesc` (또는 실제 VESC 포트)
    3. VESC 드라이버는 기본적으로 활성화되어 있습니다 (`use_vesc_driver:=true`)
    4. VESC가 다른 포트에 연결된 경우: `vesc_port:=/dev/ttyUSB1` (실제 포트에 맞게)

- `publish_ackermann` 또는 `publish_direct_controls` 인자를 조정하면 기존 Ackermann 파이프라인이나 직접 PWM 제어 중 원하는 경로만 사용할 수 있습니다.

## 시스템 구성 및 아키텍처

![rosgraph](https://github.com/kmu-kobot/2023_ScaleCar_Autorace/assets/84698896/40a653a7-ce15-47c8-a24b-b4c1ff280f5d)

## 💻 개발 환경 및 개발 언어

- 운영체제: Ubuntu 20.04, ROS noetic
- IDE: Visual Studio Code
- 개발 언어: Python 3.10.4
- 협업 툴: Github, Notion

## 팀 정보

국민대학교 소프트웨어융합대학 임베디드 소프트웨어 동아리 **KOBOT** 12기 ROBOT 팀

| 이름 | 이메일 | 담당 |
| --- | --- | --- |
| [안선영](https://github.com/SeoooooNyeong) | bm9024@kookmin.ac.kr | TM, 시뮬레이션 환경 구축 |
| [안지한](https://github.com/Anjihan) | jihan5575@kookmin.ac.kr | ROS, HW 담당 |
| [이세현](https://github.com/sehyeon518) | lifethis21@kookmin.ac.kr | 알고리즘, 주행 제어 |
| [차예찬](https://github.com/ChaNeeeeeee) | 3004yechan@kookmin.ac.kr | 영상처리, 알고리즘 |
