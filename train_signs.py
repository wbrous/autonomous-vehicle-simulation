from ultralytics import YOLO

def main():
    model = YOLO("yolo12s.yaml")
    
    results = model.train(
        data="lisa_signs.yaml",
        epochs=3,
        imgsz=640,
        project="lisa_signs_model_dir", # separate folder
        name="run",
        device="0",
        batch=16
    )
    
if __name__ == "__main__":
    main()
