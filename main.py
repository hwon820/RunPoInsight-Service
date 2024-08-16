import time
import os
import main_person
import main_num
import main_mediapipe
import main_mediapipe2
import server_upload
import mp4change

# 이전 상태를 저장할 딕셔너리 초기화
prev_state = {}

# 감시할 폴더 경로
folder_path = '/media/piai/SAMSUNG/video_file/realtime'
output_path = '/media/piai/SAMSUNG/video_file/detect'

def get_folder_state(folder):
    """
    주어진 폴더의 상태를 반환하는 함수.
    파일명과 수정 시간의 딕셔너리를 반환함.
    """
    folder_state = {}
    for root, _, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            # 파일의 수정 시간 가져오기
            modified_time = os.path.getmtime(file_path)
            folder_state[file_path] = modified_time
    return folder_state

def check_for_updates():
    """
    이전 상태와 현재 상태를 비교하여 추가된 파일을 확인하는 함수.
    """
    global prev_state
    current_state = get_folder_state(folder_path)
    
    # 파일이 추가되었는지 확인하고 추가된 파일 출력
    for file_path, _ in current_state.items():
        if file_path not in prev_state:
            main_person.run_yolov8_tracking(file_path)
            # 폴더 내의 모든 파일 경로 출력
            for filename in os.listdir(output_path):
                file_path2 = os.path.join(output_path, filename)
                most_num = main_num.extract_numbers_from_video(file_path2)
                # 파일명에서 'track_' 제거하여 숫자 부분만 추출
                track_number = os.path.splitext(filename)[0].replace('track_', '')
                # 새로운 파일명 생성
                new_file_name = f"{most_num}.mp4"
                # 새로운 파일 경로 생성
                new_file_path = os.path.join(os.path.dirname(file_path2), new_file_name)
                # 파일 이름 변경
                os.rename(file_path2, new_file_path)
                result = main_mediapipe.analyze_video(new_file_path)
                print(result)
                main_mediapipe2.select_video(new_file_path)
                mp4change.convert_video('output_video.mp4','output_video3.mp4')
                server_upload.upload_video_analysis('https://946d-141-223-140-89.ngrok-free.app', 1, 2, '/home/piai/바탕화면/ai_project/yolov8_detect/output_video3.mp4', '/home/piai/바탕화면/ai_project/yolov8_detect/knee_angle_changes.png', {
        "l1_mes": "케이던스가 낮습니다. 보폭을 줄여주세요.",
        "l2_mes": "상체가 앞으로 기울었습니다. 상체를 당겨주세요",
        "l3_mes": "무릎을 조금 더 굽혀주세요.",
    })
    
    # 이전 상태를 전체 현재 상태로 업데이트
    prev_state = current_state

# 프로그램 시작 시 초기 상태 설정
prev_state = get_folder_state(folder_path)

# 주기적으로 업데이트 확인
while True:
    check_for_updates()
    time.sleep(5)  # 5초마다 확인

# /media/piai/SAMSUNG/video_file/detect/track_1.mp4