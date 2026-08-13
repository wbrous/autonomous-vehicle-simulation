from ultralytics import YOLO

def main():
    # Load YOLOv12-small
    model = YOLO("yolo12s.yaml")
    
    # Train the model in a separate folder 'actual_lisa_signs_model'
    results = model.train(
        data="lisa_dataset/data.yaml",
        epochs=3, # keeping epochs small to ensure script completes during this session, can be increased later
        imgsz=640,
        project="actual_lisa_signs_model",
        name="run",
        device="0",
        batch=16
    )
    
if __name__ == "__main__":
    main()
