"""
Download and merge multiple Roboflow vehicle datasets into a single 'vehicle' class.

Usage:
    python build_dataset.py
"""

import os
import shutil
from pathlib import Path

from roboflow import Roboflow

ROBOFLOW_API_KEY = "guWp7Ac1Rjum0j2Y1nM2"

# (workspace, project, version) → download name prefix
DATASETS = [
    ("asdasd-t7l4k", "car-bcsfh", 1, "car"),           # ~11.5K images
    ("vehicle-detection-yob23", "vehicle-yolo-v8", 1, "vehicle_yolo"),  # ~3K
    ("yolov8-m2va2", "detect-vehicles-m5h3j", 1, "detect_vehicles"),   # ~3.5K
    ("yolov8-foyly", "vehicle-dpbrt", 1, "vehicle_basic"),              # ~1K
]

OUTPUT_DIR = "combined-vehicle-dataset"
IMAGES_PER_SPLIT = {"train": 0.80, "valid": 0.15, "test": 0.05}
MAX_IMAGES = 2000  # cap total images for faster training


def download_datasets():
    """Download all datasets in YOLOv8 format, return list of paths."""
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    paths = []
    for workspace, project, version, prefix in DATASETS:
        try:
            proj = rf.workspace(workspace).project(project)
            ds = proj.version(version).download("yolov8", location=f"tmp_{prefix}")
            paths.append((prefix, ds.location))
            print(f"[Downloaded] {prefix}: {ds.location}")
        except Exception as e:
            print(f"[Skip] {prefix}: {e}")
    return paths


def remap_labels(label_path: Path, class_map: dict):
    """Rewrite label file so every box uses class index 0 (vehicle)."""
    lines = label_path.read_text().strip().split("\n")
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        orig_cls = parts[0]
        if orig_cls in class_map:
            parts[0] = "0"  # collapse to vehicle
            new_lines.append(" ".join(parts))
    if new_lines:
        label_path.write_text("\n".join(new_lines) + "\n")
    else:
        label_path.unlink()  # remove empty labels


def collect_all_images_labels(dataset_paths: list):
    """Gather all (image_path, label_path) pairs from downloaded datasets."""


    pairs = []
    for prefix, loc in dataset_paths:
        data_yaml = Path(loc) / "data.yaml"
        if not data_yaml.exists():
            print(f"[Warn] No data.yaml in {loc}, skipping")
            continue

        # Read class names from data.yaml to build remap
        class_map = {"0": "vehicle", "1": "vehicle", "2": "vehicle",
                     "3": "vehicle", "4": "vehicle", "5": "vehicle"}

        for split in ["train", "valid", "test"]:
            split_dir = Path(loc) / split
            if not split_dir.exists():
                continue
            img_dir = split_dir / "images"
            lbl_dir = split_dir / "labels"
            if not img_dir.exists() or not lbl_dir.exists():
                continue

            for img_file in img_dir.iterdir():
                if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                lbl_file = lbl_dir / (img_file.stem + ".txt")
                if not lbl_file.exists():
                    continue
                # Remap labels to single class; file may be removed if empty
                remap_labels(lbl_file, class_map)
                if lbl_file.exists():
                    pairs.append((img_file, lbl_file))
                    if len(pairs) >= MAX_IMAGES:
                        return pairs
    return pairs

def build_split(pairs: list, out_dir: Path):
    """Copy images/labels into train/valid/test with the defined split ratios."""
    import random
    random.seed(42)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * IMAGES_PER_SPLIT["train"])
    n_val = int(n * IMAGES_PER_SPLIT["valid"])
    # rest goes to test

    splits = {
        "train": pairs[:n_train],
        "valid": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    for split_name, split_pairs in splits.items():
        img_out = out_dir / split_name / "images"
        lbl_out = out_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_src, lbl_src in split_pairs:
            if not lbl_src.exists():
                continue  # label removed during remap (empty/no valid boxes)
            img_dst = img_out / img_src.name
            lbl_dst = lbl_out / lbl_src.name
            shutil.copy2(img_src, img_dst)
            shutil.copy2(lbl_src, lbl_dst)

        print(f"  {split_name}: {len(split_pairs)} images")


def write_data_yaml(out_dir: Path):
    """Write unified data.yaml for single-class vehicle detection."""
    content = f"""path: {out_dir.absolute()}
train: train/images
val: valid/images
test: test/images

nc: 1
names:
  0: vehicle
"""
    (out_dir / "data.yaml").write_text(content)


def clean_tmp_dirs():
    """Remove temporary download directories."""
    for workspace, project, version, prefix in DATASETS:
        tmp = Path(f"tmp_{prefix}")
        if tmp.exists():
            shutil.rmtree(tmp)
            print(f"[Cleaned] {tmp}")


def main():
    out = Path(OUTPUT_DIR)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Downloading datasets ===")
    dataset_paths = download_datasets()

    print(f"\n=== Step 2: Collecting & remapping labels ===")
    pairs = collect_all_images_labels(dataset_paths)
    print(f"Total valid image-label pairs: {len(pairs)}")

    print(f"\n=== Step 3: Building splits ===")
    build_split(pairs, out)

    print(f"\n=== Step 4: Writing data.yaml ===")
    write_data_yaml(out)

    print(f"\n=== Step 5: Cleanup ===")
    clean_tmp_dirs()

    print(f"\n✅ Combined dataset ready: {out.absolute()}")
    print(f"   Classes: 1 (vehicle)")
    print(f"   Total images: {len(pairs)}")


if __name__ == "__main__":
    main()
