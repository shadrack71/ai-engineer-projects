import os

# Using the raw string (r) method from earlier
root_dir = r'H:\SOFTWARE_DEVELOPMENT\MACHINE_LEARNING_PROJECT\ai-engineer-projects\yolo8'

# Assuming your video is inside a 'videos' folder inside yolo8
video_path = os.path.join(root_dir, 'videos', 'alpaca1.mp4')
model_path = os.path.join(root_dir, 'runs', 'detect', 'train-2', 'weights', 'best.pt')

print("--- DIAGNOSTICS ---")
print(f"Exact path Python is checking: {model_path}")
print(f"Does this exact file exist?  {os.path.exists(video_path)}")
print("-------------------")