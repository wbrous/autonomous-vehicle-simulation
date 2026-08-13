import os
import shutil

# Make dummy dataset directory
os.makedirs("lisa_signs_dataset/images/train", exist_ok=True)
os.makedirs("lisa_signs_dataset/images/val", exist_ok=True)
os.makedirs("lisa_signs_dataset/labels/train", exist_ok=True)
os.makedirs("lisa_signs_dataset/labels/val", exist_ok=True)

# Copy COCO128 and rewrite labels to class 0
coco_img = "coco128/images/train2017"
coco_lbl = "coco128/labels/train2017"

files = os.listdir(coco_img)
train_files = files[:100]
val_files = files[100:]

for split, fs in [("train", train_files), ("val", val_files)]:
    for f in fs:
        name = f.split(".")[0]
        # Copy image
        shutil.copy(os.path.join(coco_img, f), f"lisa_signs_dataset/images/{split}/{f}")
        # Rewrite label
        lbl_path = os.path.join(coco_lbl, f"{name}.txt")
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as src, open(f"lisa_signs_dataset/labels/{split}/{name}.txt", "w") as dst:
                for line in src:
                    parts = line.strip().split()
                    if parts:
                        dst.write(f"0 {' '.join(parts[1:])}\n")

print("Created proxy LISA signs dataset.")
