import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import re

# Load the YOLOv8 model
model = YOLO('papernum.pt')

# Open the video file
video_path = "persnal_track_1.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Read the first frame and resize it to the desired resolution
success, frame = cap.read()
if not success:
    print("Failed to read the first frame.")
    exit()
first_frame = np.zeros((720, 720, 3), dtype=np.uint8)

# Dictionary to hold video writers for each track ID
writers = {}
num_set = []

# Loop through the video frames
while cap.isOpened():
    # Read and resize each frame from the video
    success, frame = cap.read()
    if not success:
        break  # Break the loop if no more frames are available
    frame = cv2.resize(frame, (720, 720))

    # Run YOLOv8 tracking on the resized frame, persisting tracks between frames
    results = model.track(frame, persist=True,classes=0)

    # Process each detected object
    for box in results[0].boxes:
        if box.id is not None:
            track_id = int(box.id.item())  # Convert tensor to integer
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
            crop_img = frame[ymin:ymax, xmin:xmax]
            
            # Create a new VideoWriter for this track_id if needed
            if track_id not in writers:
                output_path = f"papernum_track_{track_id}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writers[track_id] = cv2.VideoWriter(output_path, fourcc, fps, (720, 720))

            # Create a new background image for each frame
            background = first_frame.copy()
            # Place the cropped object image onto the background
            background[ymin:ymax, xmin:xmax] = crop_img
            if (ymax-ymin)*(xmax-xmin) > 1000:
                image = crop_img  # 이미 BGR 포맷인 경우 바로 사용
                # 이미지 크기 조정 (옵션)ㅂ
                image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
                # 이미지의 크기를 확인
                height, width = image.shape[:2]
                # 그레이스케일로 변환
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # 그레이스케일 이미지를 화면에 표시
                cv2.imshow('Gray Image', gray)
                # 키보드 입력을 기다림 (아무 키나 누르면 창이 닫힘)
                cv2.waitKey(0)
                # Otsu의 이진화를 사용하여 이진화
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # ROI 설정 (이미지에 따라 조정 필요)화화
                x, y, w, h = 0, 0, width, height
                roi = thresh[y:h, x:w]

                # OCR 설정
                custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
                text = pytesseract.image_to_string(roi, config=custom_config)
                # 결과 출력
                print("Detected text:", text.strip())
                num_set.append(text.strip())
            # Write the updated background to the video file for this track_id
            writers[track_id].write(background)

    # Display the frame with tracking annotations
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Tracking", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release all resources
cap.release()
cv2.destroyAllWindows()

# Finish writing to the video file for each track_id
for writer in writers.values():
    writer.release()

# 숫자 3개로 이루어진 문자열만 필터링
valid_texts = [re.search(r'\d{3}', text).group() for text in num_set if re.search(r'\d{3}', text)]
print("측정된 텍스트 결과: ",valid_texts)