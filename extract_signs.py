import os
import shutil

coco_labels_dir = "coco128/labels/train2017"
coco_images_dir = "coco128/images/train2017"

out_labels_dir = "lisa_signs_dataset/labels/train"
out_images_dir = "lisa_signs_dataset/images/train"

# COCO stop sign is class 11
TARGET_CLASS = 11

count = 0
for label_file in os.listdir(coco_labels_dir):
    with open(os.path.join(coco_labels_dir, label_file), "r") as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if int(parts[0]) == TARGET_CLASS:
            # Change class to 0 (stop sign)
            new_lines.append(f"0 {' '.join(parts[1:])}\n")
            
    if new_lines:
        # Save new label
        with open(os.path.join(out_labels_dir, label_file), "w") as f:
            f.writelines(new_lines)
            
        # Copy image
        img_name = label_file.replace(".txt", ".jpg")
        shutil.copy(os.path.join(coco_images_dir, img_name), os.path.join(out_images_dir, img_name))
        count += 1

print(f"Extracted {count} images with stop signs.")
