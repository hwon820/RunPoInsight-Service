import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import matplotlib

# 유니코드 깨짐현상 해결
matplotlib.rcParams['axes.unicode_minus'] = False
# 나눔고딕 폰트 적용
plt.rcParams["font.family"] = 'NanumGothic'
# 그래프 (-) 기호 표시
matplotlib.rc("axes", unicode_minus = False)

# mediapipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# VideoCapture 초기화
video_path = "track12_1.mp4"
cap = cv2.VideoCapture(video_path)

# 각도 계산 함수
def calculate_angle(a, b, c):
    a = np.array(a)  # 첫 번째 지점
    b = np.array(b)  # 두 번째 지점
    c = np.array(c)  # 세 번째 지점
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def get_video_duration(filename):
    with VideoFileClip(filename) as video:
        return video.duration  # 이 값은 초 단위로 반환됩니다.
    
# 각도 저장을 위한 리스트
left_angles_per_frame = []
right_angles_per_frame = []
# slope 값을 저장할 리스트 초기화
slopes = []
# 지면 레벨을 계산하기 위해 프레임 높이 설정
# _, sample_frame = cap.read()
# ground_level = get_ground_level(sample_frame.shape[0])
# 각도가 증가한 후 감소한 횟수 카운트
count = 0

# 프레임당 각도 계산
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_height, frame_width, _ = frame.shape
        # BGR에서 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        # 포즈 감지
        results = pose.process(frame_rgb)
        # 다시 RGB에서 BGR로 변환
        frame_rgb.flags.writeable = True
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        # 왼다리 각도 계산
        if results.pose_landmarks:
            left_hip = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y]
            left_knee = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x,
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y]
            left_ankle = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y]
            left_angle = calculate_angle(left_hip, left_knee, left_ankle)
            left_angles_per_frame.append(left_angle)
            # 각도가 증가한 후 감소한 횟수 카운트
            if len(left_angles_per_frame) >= 5:
                # 리스트에서 최근 5개의 각도를 참조하여 조건을 확인
                i = 11.5
                if left_angles_per_frame[-5] - left_angles_per_frame[-3] >= i and left_angles_per_frame[-1] - left_angles_per_frame[-3] >= i:
                    count += 1
        if results.pose_landmarks:
            right_hip = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * frame_width,
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * frame_height]
            right_knee = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * frame_width,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * frame_height]
            right_ankle = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * frame_width,
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * frame_height]
            right_angle = calculate_angle(right_hip, right_knee, right_ankle)
            right_angles_per_frame.append(right_angle)
        # 어깨와 골반 landmark 인덱스
        right_shoulder_index = 11
        left_shoulder_index = 12
        left_hip_index = 23
        right_hip_index = 24
        # 어깨와 골반의 중간점 계산
        if results.pose_landmarks:
            right_shoulder_landmark = results.pose_landmarks.landmark[right_shoulder_index]
            left_shoulder_landmark = results.pose_landmarks.landmark[left_shoulder_index]
            right_hip_landmark = results.pose_landmarks.landmark[right_hip_index]
            left_hip_landmark = results.pose_landmarks.landmark[left_hip_index]
            center_1_x = int((right_shoulder_landmark.x + left_shoulder_landmark.x) / 2 * frame.shape[1])
            center_1_y = int((right_shoulder_landmark.y + left_shoulder_landmark.y) / 2 * frame.shape[0])
            center_2_x = int((right_hip_landmark.x + left_hip_landmark.x) / 2 * frame.shape[1])
            center_2_y = int((right_hip_landmark.y + left_hip_landmark.y) / 2 * frame.shape[0])
            # 선분의 기울기 계산
            if center_2_y != center_1_y:
                slope = (center_2_x - center_1_x) / (center_2_y - center_1_y)
                slopes.append(slope)
            else:
                slopes.append(np.inf)  # y값이 같을 경우 기울기를 무한대로 처리
            # 선분 그리기
            cv2.line(frame_bgr, (center_1_x, center_1_y), (center_2_x, center_2_y), (0, 255, 0), 2)
        # 화면에 포즈 그리기
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Runner Pose Analysis', frame_bgr)
        frame_number += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 비디오 종료
cap.release()
cv2.destroyAllWindows()

# 극솟값 개수 출력
print("극소값 개수:", count)

# 각도 리스트 그래프로 표시
plt.figure(figsize=(10, 5))
plt.plot(left_angles_per_frame)
plt.title('영상 프레임에 따른 왼쪽 무릎 각도 변화')
plt.xlabel('프레임')
plt.ylabel('각도 (degrees)')
plt.show()

duration = get_video_duration(video_path)
print(f"이 영상의 길이: {duration} 초")
result = 60.0 / duration
cadence = result * count * 2
print(f"cadence: {cadence:.1f}")

# 최소 및 최대 각도 찾기
left_min_angle = min(left_angles_per_frame)
left_max_angle = max(left_angles_per_frame)
right_min_angle = min(right_angles_per_frame)
right_max_angle = max(right_angles_per_frame)

# 각도의 최소 및 최대 값의 인덱스 찾기
left_min_index = left_angles_per_frame.index(left_min_angle)
left_max_index = left_angles_per_frame.index(left_max_angle)
right_min_index = right_angles_per_frame.index(right_min_angle)
right_max_index = right_angles_per_frame.index(right_max_angle)

# 그래프 그리기
plt.figure(figsize=(10, 5))
plt.plot(left_angles_per_frame, label='Left Knee Angle', linestyle='-', color='skyblue')
plt.plot(right_angles_per_frame, label='Right Knee Angle', linestyle='-', color='pink')

# 최소 및 최대 각도 표시
plt.scatter(left_min_index, left_min_angle, color='yellow', s=100, label='Left Min')
plt.scatter(left_max_index, left_max_angle, color='yellow', s=100, label='Left Max')
plt.scatter(right_min_index, right_min_angle, color='yellow', s=100, label='Right Min')
plt.scatter(right_max_index, right_max_angle, color='yellow', s=100, label='Right Max')
plt.title('프레임에 따른 무릎 각도 변화')
plt.xlabel('프레임')
plt.ylabel('각 (Degrees)')
plt.legend()
plt.grid(True)
plt.show()

# 최소 및 최대 각도 출력
print(f"왼쪽 무릎의 최소 각도: {left_min_angle:.2f}도    \t 왼쪽 무릎의 최대 각도: {left_max_angle:.2f}도")
print(f"오른쪽 무릎의 최소 각도: {right_min_angle:.2f}도 \t 오른쪽 무릎의 최대 각도: {right_max_angle:.2f}도")

# slopes 리스트를 numpy 배열로 변환
slopes_np = np.array(slopes)
# 최댓값과 최솟값 출력
max_slope = np.max(slopes_np)
min_slope = np.min(slopes_np)
plt.figure(figsize=(10, 5))
plt.plot(slopes_np, linestyle='-')
plt.title('프레임에 따른 상체 기울기')
plt.xlabel('프레임')
plt.ylabel('기울기')
plt.grid(True)
plt.show()
print("최대 기울기: {:.2f}".format(max_slope))
print("최소 기울기: {:.2f}".format(min_slope))