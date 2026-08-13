#!/usr/bin/env python3
"""
Convert Mapillary Traffic Sign Dataset (MTSD) → YOLO format.

Expected MTSD layout (as downloaded from Mapillary):
    <mtsd_root>/
        annotations/
            train/<key>.json
            val/<key>.json
            test/<key>.json          # may lack labels (fully_annotated=False)
        images/
            train/<key>.jpg
            val/<key>.jpg
            test/<key>.jpg
        splits/
            train.txt                # one key per line
            val.txt
            test.txt                 # optional

Output layout:
    <out_root>/
        images/train/   images/val/   images/test/
        labels/train/   labels/val/   labels/test/
        data.yaml

Usage:
    python mtsd_to_yolo.py \\
        --mtsd   /path/to/mtsd \\
        --out    /path/to/mtsd_yolo \\
        --min-instances 10           # drop classes with fewer total instances
        --skip-pano                  # skip 360° panoramic images
        --symlink                    # symlink images instead of copying (saves disk)
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_split_keys(splits_dir: Path, split: str) -> list[str]:
    txt = splits_dir / f"{split}.txt"
    if not txt.exists():
        return []
    return [l.strip() for l in txt.read_text().splitlines() if l.strip()]


def load_annotation(ann_dir: Path, split: str, key: str) -> dict | None:
    p = ann_dir / split / f"{key}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def find_image(images_dir: Path, split: str, key: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        p = images_dir / split / f"{key}{ext}"
        if p.exists():
            return p
    return None


def bbox_to_yolo(bbox: dict, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    xmin = max(0.0, float(bbox["xmin"]))
    ymin = max(0.0, float(bbox["ymin"]))
    xmax = min(float(img_w), float(bbox["xmax"]))
    ymax = min(float(img_h), float(bbox["ymax"]))
    bw = xmax - xmin
    bh = ymax - ymin
    if bw <= 0 or bh <= 0:
        return None
    cx = (xmin + bw / 2) / img_w
    cy = (ymin + bh / 2) / img_h
    return cx, cy, bw / img_w, bh / img_h


# ---------------------------------------------------------------------------
# Pass 1: count instances per class across train+val to build the class list
# ---------------------------------------------------------------------------

def count_instances(ann_dir: Path, splits_dir: Path, splits: list[str],
                    skip_pano: bool) -> Counter:
    counts: Counter = Counter()
    for split in splits:
        for key in load_split_keys(splits_dir, split):
            ann = load_annotation(ann_dir, split, key)
            if ann is None:
                continue
            if skip_pano and ann.get("ispano", False):
                continue
            for obj in ann.get("objects", []):
                label = obj.get("label", "")
                if label:
                    counts[label] += 1
    return counts


# ---------------------------------------------------------------------------
# Pass 2: write YOLO files
# ---------------------------------------------------------------------------

def convert_split(split: str, ann_dir: Path, images_dir: Path, splits_dir: Path,
                  out_root: Path, label_to_id: dict[str, int],
                  skip_pano: bool, symlink: bool) -> dict:
    keys = load_split_keys(splits_dir, split)
    if not keys:
        return {"images": 0, "labels": 0, "skipped": 0}

    out_img_dir = out_root / "images" / split
    out_lbl_dir = out_root / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    n_images = n_labels = n_skipped = 0

    for key in keys:
        ann = load_annotation(ann_dir, split, key)
        if ann is None:
            n_skipped += 1
            continue
        if skip_pano and ann.get("ispano", False):
            n_skipped += 1
            continue

        src_img = find_image(images_dir, split, key)
        if src_img is None:
            n_skipped += 1
            continue

        img_w = ann.get("width", 0)
        img_h = ann.get("height", 0)
        if img_w <= 0 or img_h <= 0:
            n_skipped += 1
            continue

        lines: list[str] = []
        for obj in ann.get("objects", []):
            label = obj.get("label", "")
            if label not in label_to_id:
                continue  # filtered-out class
            bbox = obj.get("bbox")
            if bbox is None:
                continue
            coords = bbox_to_yolo(bbox, img_w, img_h)
            if coords is None:
                continue
            cls_id = label_to_id[label]
            lines.append(f"{cls_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}")

        # Write image (symlink saves ~20 GB for the full dataset)
        dst_img = out_img_dir / src_img.name
        if not dst_img.exists():
            if symlink:
                dst_img.symlink_to(src_img.resolve())
            else:
                shutil.copy2(src_img, dst_img)

        # Write label even if empty (YOLO needs the file to exist for negatives)
        lbl_file = out_lbl_dir / f"{key}.txt"
        lbl_file.write_text("\n".join(lines))

        n_images += 1
        n_labels += len(lines)

    return {"images": n_images, "labels": n_labels, "skipped": n_skipped}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert MTSD → YOLO format")
    parser.add_argument("--mtsd",  required=True, help="Path to MTSD root directory")
    parser.add_argument("--out",   required=True, help="Output directory for YOLO dataset")
    parser.add_argument("--min-instances", type=int, default=10,
                        help="Minimum total instances for a class to be included (default: 10)")
    parser.add_argument("--skip-pano", action="store_true",
                        help="Skip panoramic (360°) images")
    parser.add_argument("--symlink", action="store_true",
                        help="Symlink images instead of copying (saves disk space)")
    parser.add_argument("--include-test", action="store_true",
                        help="Also convert the test split (usually unannotated in MTSD)")
    args = parser.parse_args()

    mtsd_root  = Path(args.mtsd).expanduser().resolve()
    out_root   = Path(args.out).expanduser().resolve()
    ann_dir    = mtsd_root / "annotations"
    images_dir = mtsd_root / "images"
    splits_dir = mtsd_root / "splits"

    for d, name in [(ann_dir, "annotations"), (images_dir, "images"), (splits_dir, "splits")]:
        if not d.exists():
            sys.exit(f"[Error] Expected directory not found: {d}\n"
                     "Make sure --mtsd points to the MTSD root.")

    count_splits = ["train", "val"]
    all_splits   = ["train", "val"] + (["test"] if args.include_test else [])

    # ------------------------------------------------------------------
    # Pass 1: build class list from train+val
    # ------------------------------------------------------------------
    print("[1/3] Counting class instances across train+val ...")
    counts = count_instances(ann_dir, splits_dir, count_splits, args.skip_pano)
    print(f"      Found {len(counts)} unique classes, {sum(counts.values())} total instances")

    kept = {label: n for label, n in counts.items() if n >= args.min_instances}
    dropped = len(counts) - len(kept)
    print(f"      Keeping {len(kept)} classes (≥{args.min_instances} instances), "
          f"dropping {dropped} rare classes")

    if not kept:
        sys.exit("[Error] No classes survive the --min-instances filter. Lower the threshold.")

    # Sort alphabetically for a stable class ordering
    sorted_labels = sorted(kept.keys())
    label_to_id   = {label: i for i, label in enumerate(sorted_labels)}

    # ------------------------------------------------------------------
    # Pass 2: convert each split
    # ------------------------------------------------------------------
    print("\n[2/3] Converting splits ...")
    for split in all_splits:
        print(f"      [{split}] ", end="", flush=True)
        stats = convert_split(split, ann_dir, images_dir, splits_dir,
                              out_root, label_to_id, args.skip_pano, args.symlink)
        print(f"{stats['images']} images, {stats['labels']} label rows, "
              f"{stats['skipped']} skipped")

    # ------------------------------------------------------------------
    # Pass 3: write data.yaml
    # ------------------------------------------------------------------
    print("\n[3/3] Writing data.yaml ...")
    yaml_path = out_root / "data.yaml"

    # YOLO expects absolute paths for train/val/test
    yaml_data = {
        "path":  str(out_root),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(sorted_labels),
        "names": sorted_labels,
    }
    if args.include_test:
        yaml_data["test"] = "images/test"

    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Also write a human-readable class list for reference
    cls_list_path = out_root / "classes.txt"
    cls_list_path.write_text("\n".join(f"{i:3d}  {label}  ({counts.get(label, 0)} instances)"
                                        for i, label in enumerate(sorted_labels)))

    print(f"\nDone.")
    print(f"  Dataset : {out_root}")
    print(f"  Classes : {len(sorted_labels)}  (see {cls_list_path.name})")
    print(f"  YAML    : {yaml_path}")
    print(f"\nTo train:")
    print(f"  python train_lisa.py  # after updating YAML path and class count in the script")
    print(f"  # or directly:")
    print(f"  yolo train model=yolo12s.yaml data={yaml_path} epochs=100 imgsz=640")


if __name__ == "__main__":
    main()
