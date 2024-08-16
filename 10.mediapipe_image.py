import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt

# mediapipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 이미지 파일 경로
image_path = "오른다리올림.png"
# 이미지 읽기
image = cv2.imread(image_path)
# BGR에서 RGB로 변환
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 각도 계산 함수
def calculate_angle(a, b, c):
    a = np.array(a)  # 첫 번째 지점
    b = np.array(b)  # 두 번째 지점
    c = np.array(c)  # 세 번째 지점
    print("a: ",a)
    print("b: ",b)
    print("c: ",c)
    print("arctan1: ",np.arctan2(c[1]-b[1], c[0]-b[0]))
    print("arctan2: ",np.arctan2(a[1]-b[1], a[0]-b[0]))
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    print("radians: ",radians)
    angle = np.abs(radians * 180.0 / np.pi)
    print("angle: ",angle)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# 포즈 추정
with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
    results = pose.process(image_rgb)
    image_rgb.flags.writeable = True
    image_output = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # 포즈 랜드마크 그리기
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image_output, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 예를 들어 왼쪽 다리 각도 계산
        left_hip = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image.shape[1],
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image.shape[0]]
        left_knee = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * image.shape[1],
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * image.shape[0]]
        left_ankle = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * image.shape[1],
                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * image.shape[0]]
        left_angle = calculate_angle(left_hip, left_knee, left_ankle)
        print(f"왼쪽 무릎 각도: {left_angle:.2f}도")

# 결과 이미지 표시
cv2.imshow("Pose Estimation", image_output)
cv2.waitKey(0)
cv2.destroyAllWindows()