from ultralytics import YOLO

model = YOLO("yolo8.yaml")

result = model.train(data="config.yaml", epochs=100, imgsz=640)