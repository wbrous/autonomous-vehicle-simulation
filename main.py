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

MODEL_TYPE = "yolov8n.pt"  # yolov8x = extra large (most accurate, 68M params)

# Dataset size cap (set to None to use all images)
MAX_IMAGES = 2000

# Training hyperparameters
EPOCHS = 50  # 50 epochs sufficient for single-class fine-tune; 100 overfits
IMG_SIZE = 640
BATCH_SIZE = 64  # yolov8x uses ~6x more VRAM than nano; 64→16 to fit 20GB
LEARNING_RATE = 0.05
PATIENCE = 15  # stop early if validation mAP plateaus — saves hours of wasted compute

# Augmentation & training options
SEED = 42

# Output configuration
PROJECT_NAME = "yolov8x-vehicle-training"
RUN_NAME = "vehicle-run-1"

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

import csv
from pathlib import Path

import matplotlib
from roboflow import Roboflow
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import subprocess
import tarfile
import time

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


def subset_dataset(data_yaml: str, max_images: int) -> str:
    """Create a subset of the dataset with at most max_images total."""
    import random

    yaml_dir = Path(data_yaml).parent
    with open(data_yaml) as f:
        lines = f.readlines()

    # Parse splits
    splits = {}
    for line in lines:
        line = line.strip()
        if line.startswith("train:"):
            splits["train"] = line.split(":", 1)[1].strip()
        elif line.startswith("val:"):
            splits["val"] = line.split(":", 1)[1].strip()
        elif line.startswith("test:"):
            splits["test"] = line.split(":", 1)[1].strip()

    all_images = []
    for split, rel_path in splits.items():
        img_dir = yaml_dir / rel_path
        if img_dir.exists():
            all_images.extend(
                (split, p)
                for p in img_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            )

    if len(all_images) <= max_images:
        print(f"[Subset] Dataset has {len(all_images)} images, no need to subset.")
        return data_yaml

    random.seed(SEED)
    random.shuffle(all_images)
    selected = all_images[:max_images]

    # Build per-split counts
    split_counts = {}
    for split, _ in selected:
        split_counts[split] = split_counts.get(split, 0) + 1

    print(f"[Subset] Reducing dataset to {max_images} images: {split_counts}")

    # Subset each split
    for split, rel_path in splits.items():
        img_dir = yaml_dir / rel_path
        lbl_dir = img_dir.parent / "labels"
        if not img_dir.exists():
            continue

        # Collect files to keep
        keep_stems = {p.stem for s, p in selected if s == split}

        # Remove images not in keep_stems
        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
            }:
                continue
            if img_file.stem not in keep_stems:
                lbl_file = lbl_dir / (img_file.stem + ".txt")
                img_file.unlink(missing_ok=True)
                lbl_file.unlink(missing_ok=True)

    return data_yaml


def plot_training_graphs(save_dir: str):
    """Read results.csv and generate training visualizations."""
    csv_path = os.path.join(save_dir, "results.csv")
    if not os.path.isfile(csv_path):
        print(f"[Graphs] Warning: {csv_path} not found, skipping plots.")
        return

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v for k, v in row.items()})

    if not rows:
        print(f"[Graphs] Warning: {csv_path} has no data rows, skipping plots.")
        return

    def get_floats(col):
        try:
            return [float(r[col]) for r in rows]
        except (KeyError, ValueError) as exc:
            print(
                f"[Graphs] Warning: missing or invalid column '{col}' ({exc}), skipping plot."
            )
            return None

    epoch = get_floats("epoch")
    if epoch is None:
        return

    train_box = get_floats("train/box_loss")
    train_cls = get_floats("train/cls_loss")
    train_dfl = get_floats("train/dfl_loss")
    val_box = get_floats("val/box_loss")
    val_cls = get_floats("val/cls_loss")
    val_dfl = get_floats("val/dfl_loss")
    precision = get_floats("metrics/precision(B)")
    recall = get_floats("metrics/recall(B)")
    map50 = get_floats("metrics/mAP50(B)")
    map50_95 = get_floats("metrics/mAP50-95(B)")
    lr0 = get_floats("lr/pg0")
    lr1 = get_floats("lr/pg1")
    lr2 = get_floats("lr/pg2")

    # 1. Loss curves
    if all(
        v is not None
        for v in (train_box, train_cls, train_dfl, val_box, val_cls, val_dfl)
    ):
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        ax_top.plot(epoch, train_box, label="box_loss")
        ax_top.plot(epoch, train_cls, label="cls_loss")
        ax_top.plot(epoch, train_dfl, label="dfl_loss")
        ax_top.set_ylabel("Loss")
        ax_top.set_title("Training & Validation Loss")
        ax_top.legend()
        ax_top.grid(True)

        ax_bot.plot(epoch, val_box, label="box_loss")
        ax_bot.plot(epoch, val_cls, label="cls_loss")
        ax_bot.plot(epoch, val_dfl, label="dfl_loss")
        ax_bot.set_ylabel("Loss")
        ax_bot.set_xlabel("Epoch")
        ax_bot.legend()
        ax_bot.grid(True)

        fig.savefig(
            os.path.join(save_dir, "loss_curves.png"), dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    # 2. Precision & Recall
    if all(v is not None for v in (precision, recall)):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epoch, precision, label="Precision")
        ax.plot(epoch, recall, label="Recall")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.set_title("Precision & Recall")
        ax.legend()
        ax.grid(True)
        fig.savefig(
            os.path.join(save_dir, "precision_recall.png"), dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    # 3. mAP
    if all(v is not None for v in (map50, map50_95)):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epoch, map50, label="mAP@50")
        ax.plot(epoch, map50_95, label="mAP@50-95")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.set_title("Mean Average Precision")
        ax.legend()
        ax.grid(True)
        fig.savefig(os.path.join(save_dir, "mAP.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 4. Learning rate
    if all(v is not None for v in (lr0, lr1, lr2)):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epoch, lr0, label="pg0")
        ax.plot(epoch, lr1, label="pg1")
        ax.plot(epoch, lr2, label="pg2")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.legend()
        ax.grid(True)
        fig.savefig(
            os.path.join(save_dir, "learning_rate.png"), dpi=150, bbox_inches="tight"
        )
        plt.close(fig)




def train_model(data_yaml: str) -> tuple[str, str]:
    """Train YOLOv8 nano and return paths to best weights (.pt) and ONNX export."""
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
    run_dir = str(model.trainer.save_dir)
    plot_training_graphs(run_dir)
    print(
        f"[Graphs] Saved to {run_dir}: loss_curves.png, precision_recall.png, mAP.png, learning_rate.png"
    )
    print(f"[Ultralytics] Training complete. Best weights: {best_weights}")

    # Export best model to ONNX for faster ROCm inference
    onnx_path = os.path.join(run_dir, "best.onnx")
    if not os.path.isfile(onnx_path):
        print(f"[ONNX] Exporting best model to ONNX ...")
        export_model = YOLO(best_weights)
        export_model.export(format="onnx", imgsz=IMG_SIZE, half=(DEVICE != "cpu"))
        print(f"[ONNX] Export complete: {onnx_path}")
    else:
        print(f"[ONNX] Already exists: {onnx_path}")

    return best_weights, onnx_path


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

    # 1. Use combined local dataset
    data_yaml = "combined-vehicle-dataset/data.yaml"
    if not os.path.isfile(data_yaml):
        print(f"[Error] Combined dataset not found: {data_yaml}")
        print("  Run: python build_dataset.py")
        raise SystemExit(1)
    print(f"[Dataset] Using combined vehicle dataset: {data_yaml}")
    print(f"  (Run 'python build_dataset.py' to rebuild if needed)")

    # 2. Optionally cap dataset size
    if MAX_IMAGES is not None:
        data_yaml = subset_dataset(data_yaml, MAX_IMAGES)

    # 3. Train model
    best_weights, onnx_path = train_model(data_yaml)

    # 4. Validate model using ONNX (faster on ROCm)
    validate_model(onnx_path, data_yaml)

    # 5. Sample predictions using ONNX
    dataset_root = os.path.dirname(data_yaml)
    predict_example(onnx_path, dataset_root)


if __name__ == "__main__":
    main()
