"""
Download and merge multiple Roboflow vehicle + pedestrian datasets
into a unified 2-class schema: 0=vehicle, 1=pedestrian.

Usage:
    python build_dataset.py
"""

import os
import shutil
from pathlib import Path

from roboflow import Roboflow

def _load_dotenv():
    """Load ROBOFLOW_API_KEY from repo-root .env (keeps the secret out of git)."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")

# Each entry: workspace, project, version, prefix, class_remap
# class_remap maps source class index string → target class index string
DATASETS = [
    # --- Known working vehicle datasets ---
    {"workspace": "asdasd-t7l4k", "project": "car-bcsfh", "version": 1,
     "prefix": "car", "class_remap": {str(i): "0" for i in range(20)}},
    {"workspace": "yolov8-foyly", "project": "vehicle-dpbrt", "version": 1,
     "prefix": "vehicle_basic", "class_remap": {str(i): "0" for i in range(20)}},
    {"workspace": "roboflow-100", "project": "vehicles-q0x2v", "version": 2,
     "prefix": "vehicles_rf100", "class_remap": {str(i): "0" for i in range(20)}},
     
    # --- New valid vehicle datasets (replacing broken ones) ---
    {"workspace": "vehicledataset-oksvx", "project": "vehicle-c1wb2", "version": 2,
     "prefix": "vehicle_22k", "class_remap": {str(i): "0" for i in range(20)}},
    {"workspace": "indian-institute-of-technology-bhubaneswar", "project": "augmented-vehicle-dataset", "version": 8,
     "prefix": "vehicle_iit", "class_remap": {str(i): "0" for i in range(20)}},

    # --- New valid pedestrian dataset (replacing broken ones) ---
    {"workspace": "pedestrian-detection-qqdeh", "project": "pedestrian-detection-mdspp", "version": 1,
     "prefix": "pedestrian_4k", "class_remap": {str(i): "1" for i in range(20)}},
]

# Local datasets: skip Roboflow download, use existing directory
# pedestrian--7 classes: 0='-', 1='Pedestrian', 2='car', 3='pedestrian', 4='train'
LOCAL_DATASETS = [
    {"path": "pedestrian--7", "prefix": "pedestrian_local",
     "class_remap": {"1": "1", "2": "0", "3": "1"}},
]

OUTPUT_DIR = "combined-vehicle-dataset"
IMAGES_PER_SPLIT = {"train": 0.80, "valid": 0.15, "test": 0.05}
MAX_IMAGES = None  # use all available images


def download_datasets():
    """Download all Roboflow datasets, return list of (prefix, location, class_remap)."""
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    paths = []
    for ds in DATASETS:
        workspace = ds["workspace"]
        project = ds["project"]
        version = ds["version"]
        prefix = ds["prefix"]
        class_remap = ds["class_remap"]
        try:
            proj = rf.workspace(workspace).project(project)
            downloaded = proj.version(version).download("yolov8", location=f"tmp_{prefix}")
            paths.append((prefix, downloaded.location, class_remap))
            print(f"[Downloaded] {prefix}: {downloaded.location}")
        except Exception as e:
            print(f"[Skip] {prefix}: {e}")

    # Add local datasets (no download needed)
    for ds in LOCAL_DATASETS:
        loc = ds["path"]
        if Path(loc).exists():
            paths.append((ds["prefix"], loc, ds["class_remap"]))
            print(f"[Local] {ds['prefix']}: {loc}")
        else:
            print(f"[Skip] {ds['prefix']}: path '{loc}' not found")

    return paths


def remap_labels(label_path: Path, class_remap: dict):
    """Rewrite label file using the given class_remap; delete file if no valid boxes remain."""
    lines = label_path.read_text().strip().split("\n")
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        orig_cls = parts[0]
        if orig_cls in class_remap:
            parts[0] = class_remap[orig_cls]
            new_lines.append(" ".join(parts))
    if new_lines:
        label_path.write_text("\n".join(new_lines) + "\n")
    else:
        label_path.unlink()  # remove empty labels


def collect_all_images_labels(dataset_paths: list):
    """Gather all (image_path, label_path) pairs from datasets, remapping labels."""
    pairs = []
    for prefix, loc, class_remap in dataset_paths:
        data_yaml = Path(loc) / "data.yaml"
        if not data_yaml.exists():
            print(f"[Warn] No data.yaml in {loc}, skipping")
            continue

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
                remap_labels(lbl_file, class_remap)
                if lbl_file.exists():
                    pairs.append((img_file, lbl_file))
                    if MAX_IMAGES is not None and len(pairs) >= MAX_IMAGES:
                        return pairs

        print(f"  [{prefix}] collected so far: {len(pairs)}")
    return pairs


def build_split(pairs: list, out_dir: Path):
    """Copy images/labels into train/valid/test with the defined split ratios."""
    import random
    random.seed(42)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * IMAGES_PER_SPLIT["train"])
    n_val = int(n * IMAGES_PER_SPLIT["valid"])

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
    """Write unified data.yaml for 2-class vehicle+pedestrian detection."""
    content = f"""path: {out_dir.absolute()}
train: train/images
val: valid/images
test: test/images

nc: 2
names:
  0: vehicle
  1: pedestrian
"""
    (out_dir / "data.yaml").write_text(content)


def clean_tmp_dirs():
    """Remove temporary download directories."""
    for ds in DATASETS:
        tmp = Path(f"tmp_{ds['prefix']}")
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
    print(f"   Classes: 2 (vehicle, pedestrian)")
    print(f"   Total images: {len(pairs)}")


if __name__ == "__main__":
    main()
