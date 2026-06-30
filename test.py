"""
Batch inference on a test directory using a trained YOLOv8 model.

Usage:
    python test.py --weights runs/detect/.../weights/best.pt --source car-1/test/images
    python test.py                    # uses latest best.pt and car-1/test/images
"""

import argparse
import os
import time

import torch
from ultralytics import YOLO


def detect_device():
    """Return the best available device string for Ultralytics."""
    if torch.cuda.is_available():
        backend = "ROCm" if torch.version.hip else "CUDA"
        name = torch.cuda.get_device_name(0)
        print(f"[System] {backend} detected: {name}")
        return "0"
    if torch.backends.mps.is_available():
        print("[System] MPS (Apple Silicon) detected")
        return "mps"
    print("[System] No GPU detected — using CPU")
    return "cpu"


def find_latest_weights():
    """Search for the most recently modified best.pt under runs/."""
    candidates = []
    runs_dir = "runs/detect"
    if os.path.isdir(runs_dir):
        for root, _dirs, files in os.walk(runs_dir):
            if "best.pt" in files:
                candidates.append(os.path.join(root, "best.pt"))
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 inference on a test image directory."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained weights (best.pt). Defaults to latest run.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="car-1/test/images",
        help="Directory of test images to run inference on.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="test-results",
        help="Output directory for predictions.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="run",
        help="Subdirectory name inside --project.",
    )
    args = parser.parse_args()

    # Resolve weights path
    weights_path = args.weights
    if weights_path is None:
        weights_path = find_latest_weights()
        if weights_path is None:
            print(
                "[Error] No --weights provided and no best.pt found under runs/detect/."
            )
            raise SystemExit(1)
        print(f"[Auto] Using latest weights: {weights_path}")
    else:
        print(f"[Args] Using weights: {weights_path}")

    if not os.path.isfile(weights_path):
        print(f"[Error] Weights file not found: {weights_path}")
        raise SystemExit(1)

    # Resolve source directory
    source_dir = args.source
    if not os.path.isdir(source_dir):
        print(f"[Error] Source directory not found: {source_dir}")
        raise SystemExit(1)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        os.path.join(source_dir, f)
        for f in os.listdir(source_dir)
        if os.path.splitext(f.lower())[1] in image_exts
    ]
    if not image_files:
        print(f"[Error] No images found in {source_dir}")
        raise SystemExit(1)
    print(f"[Info] Found {len(image_files)} images in {source_dir}")

    # Load model
    device = detect_device()
    print(f"[Info] Loading model ...")
    model = YOLO(weights_path)

    # Run batch prediction
    print(f"[Info] Running inference (conf={args.conf}, iou={args.iou}) ...")

    # Force absolute path for project so Ultralytics doesn't prepend runs/detect/
    project_path = os.path.abspath(args.project)

    t0 = time.time()
    results_gen = model.predict(
        source=source_dir,
        conf=args.conf,
        iou=args.iou,
        device=device,
        save=True,
        project=project_path,
        name=args.name,
        exist_ok=True,
        verbose=False,
        stream=True,
    )

    # Stats
    total_detections = 0
    images_with_detections = 0
    avg_conf = 0.0
    conf_count = 0
    num_images = 0

    for r in results_gen:
        num_images += 1
        n_boxes = len(r.boxes)
        total_detections += n_boxes
        if n_boxes > 0:
            images_with_detections += 1

        if r.boxes.conf is not None:
            avg_conf += r.boxes.conf.sum().item()
            conf_count += len(r.boxes.conf)

    elapsed = time.time() - t0
    avg_conf = avg_conf / conf_count if conf_count > 0 else 0.0

    print()
    print("=" * 50)
    print("[Results] Inference complete")
    print(f"  Images processed:     {num_images}")
    print(f"  Images with objects:  {images_with_detections}")
    print(f"  Total detections:     {total_detections}")
    print(
        f"  Avg detections/img:   {total_detections / num_images if num_images > 0 else 0.0:.2f}"
    )
    print(f"  Avg confidence:       {avg_conf:.3f}")
    print(
        f"  Time elapsed:         {elapsed:.1f}s ({num_images / elapsed if elapsed > 0 else 0.0:.1f} img/s)"
    )
    print(
        f"  Output saved to:      {os.path.abspath(os.path.join(args.project, args.name))}"
    )
    print("=" * 50)


if __name__ == "__main__":
    main()
