import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model
model = YOLO('papernum.pt')

# Open the video file
video_path = "persnal_track_2.mp4"
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
