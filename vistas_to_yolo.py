#!/usr/bin/env python3
"""
Convert Mapillary Vistas Dataset → YOLO format (8 classes).

Reads directly from the zip — no extraction needed.

Classes:
    0  car
    1  truck
    2  bus
    3  motorcycle
    4  bicycle
    5  person
    6  traffic_sign
    7  traffic_light

Usage:
    python vistas_to_yolo.py \
        --zip  mtsddataset.zip \
        --out  vistas_yolo \
        [--symlink]   # N/A for zip source — images are always written
"""

import argparse
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Class mapping: Vistas label → output class id
# ---------------------------------------------------------------------------
CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle",
           "person", "traffic_sign", "traffic_light"]

LABEL_MAP: dict[str, int] = {
    # vehicles
    "object--vehicle--car":             0,
    "object--vehicle--caravan":         1,  # treat as truck
    "object--vehicle--trailer":         1,  # treat as truck
    "object--vehicle--truck":           1,
    "object--vehicle--bus":             2,
    "object--vehicle--on-rails":        2,  # tram/train — bus bucket
    "object--vehicle--motorcycle":      3,
    "object--vehicle--wheeled-slow":    4,  # moped/scooter — bicycle bucket
    "object--vehicle--bicycle":         4,
    # people
    "human--person--individual":        5,
    "human--person--person-group":      5,
    # traffic signs — all facing-front variants; skip backs (not visible to driver)
    "object--traffic-sign--front":              6,
    "object--traffic-sign--direction-front":    6,
    "object--traffic-sign--information-parking":6,
    "object--traffic-sign--temporary-front":    6,
    "object--traffic-sign--ambiguous":          6,
    # traffic lights
    "object--traffic-light--general-upright":   7,
    "object--traffic-light--general-horizontal":7,
    "object--traffic-light--general-single":    7,
    "object--traffic-light--pedestrians":       7,
    "object--traffic-light--cyclists":          7,
    "object--traffic-light--other":             7,
}

# ---------------------------------------------------------------------------
# Polygon → axis-aligned bounding box
# ---------------------------------------------------------------------------
def polygon_to_bbox_yolo(polygon: list, img_w: int, img_h: int):
    """polygon: list of [x, y] pairs (absolute pixels). Returns (cx, cy, w, h) normalized."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    xmin, xmax = max(0.0, min(xs)), min(float(img_w), max(xs))
    ymin, ymax = max(0.0, min(ys)), min(float(img_h), max(ys))
    bw, bh = xmax - xmin, ymax - ymin
    if bw <= 0 or bh <= 0:
        return None
    return (
        (xmin + bw / 2) / img_w,
        (ymin + bh / 2) / img_h,
        bw / img_w,
        bh / img_h,
    )


# ---------------------------------------------------------------------------
# Convert one split
# ---------------------------------------------------------------------------
def convert_split(zf: zipfile.ZipFile, zip_names: list[str],
                  split: str, zip_split: str,
                  out_root: Path) -> dict:
    out_img_dir = out_root / "images" / split
    out_lbl_dir = out_root / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Build lookup: key → zip path for images
    img_lookup: dict[str, str] = {}
    for name in zip_names:
        if name.startswith(f"{zip_split}/images/") and not name.endswith("/"):
            stem = name.split("/")[-1].rsplit(".", 1)[0]
            img_lookup[stem] = name

    # All annotation files
    ann_paths = [n for n in zip_names
                 if n.startswith(f"{zip_split}/v2.0/polygons/") and n.endswith(".json")]

    n_images = n_boxes = n_skipped = 0
    total = len(ann_paths)

    for i, ann_path in enumerate(ann_paths):
        if i % 1000 == 0:
            print(f"  [{split}] {i}/{total} ...", flush=True)

        key = ann_path.split("/")[-1].replace(".json", "")

        img_zip_path = img_lookup.get(key)
        if img_zip_path is None:
            n_skipped += 1
            continue

        # Load annotation
        with zf.open(ann_path) as f:
            ann = json.load(f)

        img_w = ann.get("width", 0)
        img_h = ann.get("height", 0)
        if img_w <= 0 or img_h <= 0:
            n_skipped += 1
            continue

        # Build YOLO label lines
        lines: list[str] = []
        for obj in ann.get("objects", []):
            label = obj.get("label", "")
            cls_id = LABEL_MAP.get(label)
            if cls_id is None:
                continue
            polygon = obj.get("polygon")
            if not polygon:
                continue
            coords = polygon_to_bbox_yolo(polygon, img_w, img_h)
            if coords is None:
                continue
            lines.append(f"{cls_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}")

        # Extract and write image
        ext = img_zip_path.rsplit(".", 1)[-1]
        dst_img = out_img_dir / f"{key}.{ext}"
        if not dst_img.exists():
            with zf.open(img_zip_path) as src, open(dst_img, "wb") as dst:
                shutil.copyfileobj(src, dst)

        # Write label (empty file = negative/background image, YOLO handles it)
        lbl_file = out_lbl_dir / f"{key}.txt"
        lbl_file.write_text("\n".join(lines))

        n_images += 1
        n_boxes += len(lines)

    return {"images": n_images, "boxes": n_boxes, "skipped": n_skipped}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Vistas → YOLO 8-class converter")
    parser.add_argument("--zip", default="mtsddataset.zip",
                        help="Path to Vistas zip (default: mtsddataset.zip)")
    parser.add_argument("--out", default="vistas_yolo",
                        help="Output directory (default: vistas_yolo)")
    args = parser.parse_args()

    zip_path = Path(args.zip).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()

    if not zip_path.exists():
        sys.exit(f"[Error] Zip not found: {zip_path}")

    print(f"[Info] Reading zip index ...")
    with zipfile.ZipFile(zip_path) as zf:
        zip_names = zf.namelist()
        print(f"[Info] {len(zip_names):,} entries in zip")

        splits = [("train", "training"), ("val", "validation")]
        all_stats = {}

        print(f"\n[Converting] Output → {out_root}\n")
        for out_split, zip_split in splits:
            print(f"[{out_split}]")
            stats = convert_split(zf, zip_names, out_split, zip_split, out_root)
            all_stats[out_split] = stats
            print(f"  done: {stats['images']} images, {stats['boxes']} boxes, "
                  f"{stats['skipped']} skipped\n")

    # Write data.yaml
    yaml_path = out_root / "data.yaml"
    yaml_data = {
        "path":  str(out_root),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(CLASSES),
        "names": CLASSES,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print("=" * 50)
    print(f"Done!")
    for split, s in all_stats.items():
        print(f"  {split}: {s['images']} images, {s['boxes']} boxes")
    print(f"  YAML: {yaml_path}")
    print(f"\nTo train:")
    print(f"  python train_lisa.py  # update YAML path + nc=8 at top of script")
    print(f"  # or directly:")
    print(f"  yolo train model=yolo12m.yaml data={yaml_path} epochs=100 imgsz=640")


if __name__ == "__main__":
    main()
