from ultralytics import YOLO
import torch
from datetime import datetime  # Import the datetime class from the datetime module

model = YOLO('./yolov8n.pt')
model.train(data='person.yaml',epochs=100,imgsz=720)
# 현재 시간을 "년월일-시분초" 형식으로 가져옴
current_time = datetime.now().strftime('%Y%m%d-%H%M%S')

# 파일 이름 설정
filename = f'person.pt'
# filename = f'yolov8_{current_time}.pt'
# 모델 상태 저장
torch.save(model.state_dict(), filename)