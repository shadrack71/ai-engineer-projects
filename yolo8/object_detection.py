from ultralytics import YOLO

# model = YOLO("yolo26n.yaml")
#
# result = model.train(data="config.yaml", epochs=1)



# Load a model
model = YOLO("yolo26n.yaml")  # build a new model from YAML
model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo26n.yaml").load("yolo26n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="config.yaml", epochs=4, imgsz=640)