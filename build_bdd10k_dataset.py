#!/usr/bin/env python3
"""
Download BDD100K (10K subset) from the HuggingFace mirror `dgural/bdd100k` and
convert to YOLO format with our 4-class taxonomy for adaptive cruise control:
    0 = sedan       (BDD "car"        — avg 1.5 m tall)
    1 = truck       (BDD "truck"      — avg 3.5 m tall)
    2 = bus         (BDD "bus"        — avg 3.0 m tall)
    3 = pedestrian  (BDD "pedestrian" — avg 1.7 m tall)

Why BDD: dashcam viewpoint matching deployment, so the pinhole camera model's
H_real-per-class actually corresponds to the bbox heights the model sees.

Primary path: pull the whole `dgural/bdd100k` HF repo (10K images + a single
`samples.json` carrying all detections in FiftyOne normalized-bbox format).
No Berkeley DNS dependency, no 6.5 GB zip — total fetch is ~1 GB.

Usage:
    python build_bdd10k_dataset.py                  # default: HF snapshot
    python build_bdd10k_dataset.py --max-per-split 800
    python build_bdd10k_dataset.py --skip-download  # assume raw already pulled
    python build_bdd10k_dataset.py --seed 123

Splits: 70% train / 20% val / 10% test, deterministic by SHA-1 of the
image filepath so the same seed gives the same split every run.
"""

import argparse
import hashlib
import os
# HF_HUB_DISABLE_XET must be set before huggingface_hub.constants is imported,
# otherwise xet is still used and anonymous clients get HTTP 429 mid-stream.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# Be polite to anonymous rate limits — single-thread HF downloads so we don't
# poke the xet-read-token endpoint in parallel.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
import json
import shutil
import zipfile
from pathlib import Path

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
OUTPUT_DIR = Path("combined-bdd10k")
DOWNLOAD_DIR = Path("bdd10k_raw")

# HuggingFace mirror: 10K BDD images (one .jpg each under data/) + labels
# bundled in samples.json (FiftyOne format, normalized [x, y, w, h] bbox).
HF_REPO = "dgural/bdd100k"
SAMPLES_JSON = "samples.json"

# Split fractions for our 10K subset. BDD doesn't ship a 10K-specific split,
# so we make our own — deterministic by hashing the filepath so reruns are
# stable until you change --seed.
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.20, "test": 0.10}

# BDD category → our class id. Categories NOT in this map are dropped.
# (rider/motor/bike/traffic light/sign/train/other.vehicle/trailer intentionally
# dropped so the model stays focused on the four classes ACC cares about.)
BDD_TO_OUR_CLASS = {
    "car":        0,   # sedan (+SUV/van, which BDD folds into "car")
    "truck":      1,
    "bus":        2,
    "pedestrian": 3,
}
CLASS_NAMES = {0: "sedan", 1: "truck", 2: "bus", 3: "pedestrian"}

# Fallback Berkeley URLs (only used if --labels-source berkeley|--labels-zip).
BERKELEY_LABELS = "https://dl.cv.ethz.ch/bdd100k/data/bdd100k_det_20_labels_trainval.zip"


# ------------------------------------------------------------------
# DOWNLOAD — HF snapshot
# ------------------------------------------------------------------
def fetch_hf_snapshot(dest: Path) -> Path:
    """Pull the full dgural/bdd100k repo (10K images + samples.json) into dest.

    HF's new xet CDN rate-limits anonymous clients hard (HTTP 429 mid-stream),
    so we fall back to classic HTTP transfer via HF_HUB_DISABLE_XET (set at
    module top, before huggingface_hub.constants imports) and cap workers to
    dodge tokenless rate limits."""
    from huggingface_hub import snapshot_download
    print(f"[HF] snapshot_download({HF_REPO}) → {dest} "
           "(xet disabled, max_workers=2)")
    out = snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=str(dest),
        ignore_patterns=["dataset_preview.gif"],
        max_workers=2,
    )
    print(f"[HF] done: {out}")
    return Path(out)
def split_for_stem(stem: str, seed: int) -> str:
    """Hash filepath stem + seed → deterministic train/val/test bucket.
    Same input always lands in the same split until you change --seed."""
    h = hashlib.sha1(f"{seed}:{stem}".encode()).hexdigest()
    bucket = (int(h[:8], 16) % 1000) / 1000.0   # in [0, 1)
    acc = 0.0
    for split, frac in SPLIT_FRACTIONS.items():
        acc += frac
        if bucket < acc:
            return split
    return list(SPLIT_FRACTIONS)[-1]   # defensive tail


def bbox_fiftyone_to_yolo(box_rel):
    """FiftyOne: [x, y, w, h] rel, top-left origin. YOLO: [cx, cy, w, h] rel."""
    x, y, w, h = box_rel
    return x + w / 2.0, y + h / 2.0, w, h


def run_conversion(repo_dir: Path, out_root: Path, seed: int = 42,
                   max_per_split: int | None = None) -> dict:
    """Walk samples.json, build YOLO train/val/test layout under out_root."""
    out_root = Path(out_root)
    for split in SPLIT_FRACTIONS:
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "labels").mkdir(parents=True, exist_ok=True)

    samples_path = repo_dir / SAMPLES_JSON
    if not samples_path.exists():
        raise RuntimeError(f"{samples_path} missing — run without --skip-download")

    print(f"[Parse] loading {samples_path} ...")
    with open(samples_path) as f:
        doc = json.load(f)
    samples = doc["samples"] if isinstance(doc, dict) and "samples" in doc else doc
    print(f"[Parse] {len(samples)} samples")

    # Pre-assign splits so we can honour max-per-split deterministically.
    assigned = {"train": [], "val": [], "test": []}
    for s in samples:
        stem = Path(s["filepath"]).stem
        split = split_for_stem(stem, seed)
        assigned[split].append(s)
    for split, items in assigned.items():
        items.sort(key=lambda s: s["filepath"])
        if max_per_split is not None:
            assigned[split] = items[:max_per_split]
        print(f"[Split] {split}: {len(assigned[split])} images")

    written = {"train": 0, "val": 0, "test": 0}
    for split, items in assigned.items():
        for s in items:
            fp = s["filepath"]
            src_img = repo_dir / fp
            if not src_img.exists():
                continue
            stem = Path(fp).stem
            dst_img = out_root / split / "images" / Path(fp).name
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            dets = (s.get("detections") or {}).get("detections", [])
            lines = []
            for d in dets:
                cat = d.get("label")
                if cat not in BDD_TO_OUR_CLASS:
                    continue
                box = d.get("bounding_box")
                if not box or len(box) != 4:
                    continue
                cx, cy, w, h = bbox_fiftyone_to_yolo(box)
                # Drop degenerate / off-frame boxes (FiftyOne stores rel coords).
                if w <= 0 or h <= 0 or cx <= 0 or cx >= 1 or cy <= 0 or cy >= 1:
                    continue
                cls = BDD_TO_OUR_CLASS[cat]
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            dst_lbl = out_root / split / "labels" / (stem + ".txt")
            dst_lbl.write_text("\n".join(lines))
            written[split] += 1
    return written


def write_data_yaml(out_dir: Path):
    names_block = "\n".join(f"  {k}: {v}" for k, v in sorted(CLASS_NAMES.items()))
    content = f"""path: {out_dir.absolute()}
train: train/images
val: val/images
test: test/images

nc: {len(CLASS_NAMES)}
names:
{names_block}
"""
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(content)
    print(f"[YAML] wrote {yaml_path}")
    print(content)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for deterministic train/val/test split")
    ap.add_argument("--max-per-split", type=int, default=None,
                    help="cap each split at this many images (debug / smoke test)")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--raw", type=Path, default=DOWNLOAD_DIR)
    ap.add_argument("--skip-download", action="store_true",
                    help="assume the HF repo snapshot already exists at --raw")
    args = ap.parse_args()

    args.raw.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        repo_dir = fetch_hf_snapshot(args.raw)
    else:
        repo_dir = args.raw
        print(f"[Skip] using existing snapshot at {repo_dir}")

    if not (repo_dir / SAMPLES_JSON).exists():
        print(f"[Error] {repo_dir}/{SAMPLES_JSON} missing. "
              "Re-run without --skip-download.")
        raise SystemExit(1)

    print("\n=== Converting to YOLO format ===")
    written = run_conversion(repo_dir, args.out, seed=args.seed,
                               max_per_split=args.max_per_split)
    for split, n in written.items():
        print(f"  {split}: {n} images converted")

    print("\n=== Writing data.yaml ===")
    write_data_yaml(args.out)
    print("[Done] BDD10K ready at", args.out)


if __name__ == "__main__":
    main()