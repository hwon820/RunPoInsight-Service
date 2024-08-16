from ultralytics import YOLO

model = YOLO(model='person.pt')

model.predict('run_test.mp4',save=False,show=True,imgsz=720)