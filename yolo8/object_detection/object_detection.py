from ultralytics import YOLO


if __name__ == '__main__':

    # Load the pretrained model (Best for custom datasets)
    model = YOLO("yolo26n.pt")
    # Train the model
    result = model.train(
        data="config.yaml",
        epochs=100,
        imgsz=640,
        patience=50,
        device=0,

        # --- THE FIX ---
        workers=2,  # Limits background processes to prevent RAM overflow.
        batch=8
    )
