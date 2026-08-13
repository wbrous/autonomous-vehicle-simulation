import argparse
import os
import shutil
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ROCm: hide integrated GPU (Raphael) before PyTorch initializes CUDA/HIP
if os.environ.get("HIP_VISIBLE_DEVICES") is None:
    os.environ["HIP_VISIBLE_DEVICES"] = "0"

import torch
from ultralytics import YOLO

# ============================================================
# CUSTOMIZABLE CONFIGURATION VARIABLES
# ============================================================

MODEL_TYPE = "yolo12m.yaml"  # YOLOv12 medium — 8-class vehicle/sign/light detector

# Training hyperparameters
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 32  # m-model is larger; 32 safe on 20GB VRAM
LEARNING_RATE = 0.01
PATIENCE = 20
SEED = 42

# Output configuration
PROJECT_NAME = "vistas_vehicle_model"
RUN_NAME = "yolo12m_vistas_100epochs"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# ============================================================
# AUTO-DETECT BEST AVAILABLE DEVICE
# ============================================================

def detect_device():
    if torch.cuda.is_available():
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
WORKERS = 0 if DEVICE != "cpu" and torch.version.hip else (os.cpu_count() or 1)

# ============================================================
# TRAINING SCRIPT
# ============================================================

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
        return

    def get_floats(col):
        try:
            return [float(r[col]) for r in rows]
        except (KeyError, ValueError):
            return None

    epoch = get_floats("epoch")
    if epoch is None: return

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

    if all(v is not None for v in (train_box, train_cls, train_dfl, val_box, val_cls, val_dfl)):
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        ax_top.plot(epoch, train_box, label="box_loss")
        ax_top.plot(epoch, train_cls, label="cls_loss")
        ax_top.plot(epoch, train_dfl, label="dfl_loss")
        ax_top.set_ylabel("Loss")
        ax_top.legend()
        ax_top.grid(True)
        ax_bot.plot(epoch, val_box, label="box_loss")
        ax_bot.plot(epoch, val_cls, label="cls_loss")
        ax_bot.plot(epoch, val_dfl, label="dfl_loss")
        ax_bot.set_ylabel("Loss")
        ax_bot.set_xlabel("Epoch")
        ax_bot.legend()
        ax_bot.grid(True)
        fig.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    if all(v is not None for v in (precision, recall)):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epoch, precision, label="Precision")
        ax.plot(epoch, recall, label="Recall")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True)
        fig.savefig(os.path.join(save_dir, "precision_recall.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    if all(v is not None for v in (map50, map50_95)):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epoch, map50, label="mAP@50")
        ax.plot(epoch, map50_95, label="mAP@50-95")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True)
        fig.savefig(os.path.join(save_dir, "mAP.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def save_epoch_checkpoint(trainer):
    """Copy last.pt and best.pt into an epoch-specific subfolder after each save."""
    epoch_num = trainer.epoch + 1  # 0-indexed → 1-indexed
    ckpt_dir = Path(trainer.save_dir) / "epoch_checkpoints" / f"epoch_{epoch_num:02d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    last_pt = Path(trainer.save_dir) / "weights" / "last.pt"
    best_pt = Path(trainer.save_dir) / "weights" / "best.pt"

    if last_pt.exists():
        shutil.copy2(last_pt, ckpt_dir / "last.pt")
    if best_pt.exists():
        shutil.copy2(best_pt, ckpt_dir / "best.pt")

    print(f"  [Checkpoint] Saved epoch {epoch_num} → {ckpt_dir}")


def train_model(data_yaml: str, resume: bool = False) -> tuple[str, str]:
    """Train YOLOv12-small and return paths to best weights and run directory."""
    if resume:
        last_ckpt = Path(f"{PROJECT_NAME}/{RUN_NAME}/weights/last.pt")
        if not last_ckpt.exists():
            last_ckpt = Path("runs/detect") / PROJECT_NAME / RUN_NAME / "weights/last.pt"
        if not last_ckpt.exists():
            print("[Ultralytics] --resume requested but no checkpoint found; starting fresh.")
            resume = False
        else:
            print(f"\n[Ultralytics] Resuming from checkpoint: {last_ckpt}")
            model = YOLO(str(last_ckpt))
    if not resume:
        print(f"\n[Ultralytics] Loading fresh model: {MODEL_TYPE}")
        model = YOLO(MODEL_TYPE)

    model.add_callback("on_model_save", save_epoch_checkpoint)

    print(f"[Ultralytics] Starting training for {EPOCHS} epochs on device='{DEVICE}' ...")
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
        cos_lr=True,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.2,
        scale=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.1,
        close_mosaic=20,
        resume=resume,
    )

    best_weights = str(model.trainer.best)
    run_dir = str(model.trainer.save_dir)
    plot_training_graphs(run_dir)
    print(f"[Graphs] Saved to {run_dir}: loss_curves.png, precision_recall.png, mAP.png")
    print(f"[Ultralytics] Training complete. Best weights: {best_weights}")
    print(f"[Checkpoints] Per-epoch snapshots in: {run_dir}/epoch_checkpoints/")

    return best_weights, run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from the last checkpoint in the run directory."
    )
    args = parser.parse_args()

    data_yaml = "vistas_yolo/data.yaml"

    if not os.path.isfile(data_yaml):
        print(f"[Error] Dataset not found: {data_yaml}")
        raise SystemExit(1)

    print(f"[Dataset] Using {data_yaml}")

    best_weights, run_dir = train_model(data_yaml, resume=args.resume)

    ckpt_dir = os.path.join(run_dir, "epoch_checkpoints")
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best weights: {best_weights}")
    print(f"  Run directory: {run_dir}")
    print(f"  Epoch checkpoints: {ckpt_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
