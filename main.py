"""
YOLOv8 Nano Training Script with Roboflow Dataset
================================================
Train a YOLOv8 nano model on a custom Roboflow dataset.

Quick start:
    pip install -r requirements.txt
    python main.py

GPU Backends
------------
- NVIDIA: standard PyTorch CUDA build
- AMD   : install ROCm PyTorch (see README.md)
- Apple : MPS (Metal Performance Shaders)
"""

import os

# ROCm: hide integrated GPU (Raphael) before PyTorch initializes CUDA/HIP
if os.environ.get("HIP_VISIBLE_DEVICES") is None:
    os.environ["HIP_VISIBLE_DEVICES"] = "0"

import torch

# ============================================================
# CUSTOMIZABLE CONFIGURATION VARIABLES
# ============================================================

# Roboflow credentials and project info
ROBOFLOW_API_KEY = "guWp7Ac1Rjum0j2Y1nM2"
ROBOFLOW_WORKSPACE = "asdasd-t7l4k"
ROBOFLOW_PROJECT = "car-bcsfh"
ROBOFLOW_VERSION = 1
ROBOFLOW_FORMAT = "yolov8"

# Model configuration
MODEL_TYPE = "yolov8x.pt"  # yolov8x = extra large (most accurate, 68M params)

# Training hyperparameters
EPOCHS = 50  # 50 epochs sufficient for single-class fine-tune; 100 overfits
IMG_SIZE = 640
BATCH_SIZE = 16  # yolov8x uses ~6x more VRAM than nano; 64→16 to fit 20GB
LEARNING_RATE = 0.01
PATIENCE = 15  # stop early if validation mAP plateaus — saves hours of wasted compute

# Augmentation & training options
SEED = 42

# Output configuration
PROJECT_NAME = "yolov8-car-training"
RUN_NAME = "nano-run-1"

# Post-training validation
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# ============================================================
# AUTO-DETECT BEST AVAILABLE DEVICE
# ============================================================


def detect_device():
    """Return the best available device string for Ultralytics."""
    if torch.cuda.is_available():
        # Works for both NVIDIA CUDA and AMD ROCm (ROCm uses CUDA API compat)
        backend = "ROCm" if torch.version.hip else "CUDA"
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0) if count else "unknown"
        print(f"[System] {backend} detected — {count} device(s): {name}")
        return "0"

    if torch.backends.mps.is_available():
        print("[System] MPS (Apple Silicon) detected")
        return "mps"

    cpu_count = os.cpu_count() or 1
    print(f"[System] No GPU detected — using CPU with {cpu_count} cores")
    return "cpu"

DEVICE = detect_device()

# ROCm requires workers=0 to avoid "Module not initialized" crash in dataloaders
WORKERS = 0 if DEVICE != "cpu" and torch.version.hip else (os.cpu_count() or 1)

# ============================================================
# TRAINING SCRIPT
# ============================================================

from pathlib import Path

from roboflow import Roboflow
from ultralytics import YOLO


def download_roboflow_dataset(
    api_key: str, workspace: str, project: str, version: int, fmt: str
) -> str:
    """Download dataset from Roboflow and return path to data.yaml."""
    print(f"\n[Roboflow] Downloading dataset: {workspace}/{project} v{version} ...")

    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    dataset = project_obj.version(version).download(fmt)

    data_yaml = os.path.join(dataset.location, "data.yaml")
    print(f"[Roboflow] Dataset ready: {data_yaml}")
    return data_yaml


def train_model(data_yaml: str) -> str:
    """Train YOLOv8 nano and return path to best weights."""
    print(f"\n[Ultralytics] Loading model: {MODEL_TYPE}")
    model = YOLO(MODEL_TYPE)

    print(
        f"[Ultralytics] Starting training for {EPOCHS} epochs on device='{DEVICE}' ..."
    )
    model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        lr0=LEARNING_RATE,
        patience=PATIENCE,
        workers=WORKERS,
        device=DEVICE,
        seed=SEED,
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,
    )

    best_weights = str(model.trainer.best)
    print(f"[Ultralytics] Training complete. Best weights: {best_weights}")
    return best_weights


def validate_model(weights_path: str, data_yaml: str):
    """Run validation on the trained model."""
    print(f"\n[Ultralytics] Validating model: {weights_path}")
    model = YOLO(weights_path)

    metrics = model.val(
        data=data_yaml,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=DEVICE,
        split="test",
    )

    print("\n[Results] Validation complete!")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP75:    {metrics.box.map75:.4f}")


def predict_example(weights_path: str, source_dir: str):
    """Run inference on a few images from the dataset for quick visual testing."""
    print(f"\n[Ultralytics] Running sample predictions ...")

    test_images_dir = os.path.join(source_dir, "test", "images")
    if not os.path.isdir(test_images_dir):
        test_images_dir = os.path.join(source_dir, "valid", "images")

    if not os.path.isdir(test_images_dir):
        print("  Could not find test/valid images for sample prediction.")
        return

    image_files = [
        os.path.join(test_images_dir, f)
        for f in os.listdir(test_images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ][:5]

    if not image_files:
        print("  No images found for sample prediction.")
        return

    model = YOLO(weights_path)
    predict_dir = os.path.join(PROJECT_NAME, RUN_NAME, "predictions")
    os.makedirs(predict_dir, exist_ok=True)

    for img_path in image_files:
        model.predict(
            source=img_path,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            save=True,
            project=predict_dir,
            exist_ok=True,
        )
        print(f"  Predicted: {img_path}")

    print(f"  Predictions saved to: {predict_dir}")


def main():
    if DEVICE == "cpu":
        cpu_count = os.cpu_count() or 1
        torch.set_num_threads(cpu_count)
        print(f"[System] CPU threads set to {cpu_count}")

    # 1. Download dataset
    data_yaml = download_roboflow_dataset(
        ROBOFLOW_API_KEY,
        ROBOFLOW_WORKSPACE,
        ROBOFLOW_PROJECT,
        ROBOFLOW_VERSION,
        ROBOFLOW_FORMAT,
    )

    # 2. Train model
    best_weights = train_model(data_yaml)

    # 3. Validate model
    validate_model(best_weights, data_yaml)

    # 4. Sample predictions
    dataset_root = os.path.dirname(data_yaml)
    predict_example(best_weights, dataset_root)

    print("\n✅ All done!")
    print(f"   Best weights: {os.path.abspath(best_weights)}")
    print(f"   Training logs: {os.path.abspath(os.path.join(PROJECT_NAME, RUN_NAME))}")


if __name__ == "__main__":
    main()
