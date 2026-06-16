import os

from ultralytics import YOLO
import cv2

# root_dir = '/yolo8/object_detection/'
root_dir = r'H:\SOFTWARE_DEVELOPMENT\MACHINE_LEARNING_PROJECT\ai-engineer-projects\yolo8\object_detection'
VIDEOS_DIR = os.path.join(root_dir, 'videos')

print(VIDEOS_DIR)

video_path = os.path.join(VIDEOS_DIR, 'alpaca1.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join(root_dir, 'runs', 'detect', 'train-2', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

# threshold = 0.5

# Lower the threshold temporarily to see if weak detections exist
threshold = 0.25

while True:
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # --- NEW: Display the video on screen ---
    # Resize the frame slightly just in case your video is larger than your monitor
    display_frame = cv2.resize(frame, (W // 2, H // 2))
    cv2.imshow('YOLO Detection - Press Q to Quit', display_frame)

    # Wait 1ms for a key press. If 'q' is pressed, break the loop early.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # ----------------------------------------

    # Write the frame to the saved output video
    out.write(frame)

    ret, frame = cap.read()
    if not ret:
        break

cap.release()
out.release()
cv2.destroyAllWindows()