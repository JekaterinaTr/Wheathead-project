import os
import json
import yaml
from pathlib import Path
from PIL import Image
from datetime import datetime

# ================= CONFIGURATION =================
yaml_path = r"D:\Projects\Project1_wheatheads\project1_dataset\project1_dataset.yaml"

labels_dir = r"D:\Projects\Project1_wheatheads\project1_dataset\labels\test"
images_dir = r"D:\Projects\Project1_wheatheads\project1_dataset\images\test"

output_json = r"D:\Projects\Project1_wheatheads\project1_dataset\annotations_coco\test_coco.json"

# Image extensions to check
IMG_EXTS = [".jpg", ".jpeg", ".png"]

# ==== LOAD YAML ====
with open(yaml_path, 'r') as f:
    data_yaml = yaml.safe_load(f)

if "names" not in data_yaml:
    raise ValueError("YAML file does not contain 'names' field")

class_names = data_yaml["names"]

coco = {
    "info": {
        "description": "YOLO to COCO converted dataset",
        "version": "1.0",
        "year": datetime.now().year,
        "contributor": "Auto-conversion script",
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    },
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": []
}

# Categories (with supercategory)
for i, name in enumerate(class_names):
    coco["categories"].append({
        "id": i,
        "name": name,
        "supercategory": "NONE"
    })

annotation_id = 1
image_id = 1

# ================= CONVERSION =================
label_files = sorted(Path(labels_dir).glob("*.txt"))

for label_file in label_files:
    stem = label_file.stem

    # Find corresponding image
    image_path = None
    for ext in IMG_EXTS:
        candidate = Path(images_dir) / f"{stem}{ext}"
        if candidate.exists():
            image_path = candidate
            break

    if image_path is None:
        print(f"⚠️ Image not found for label: {label_file.name}")
        continue

    # Load image size
    with Image.open(image_path) as img:
        width, height = img.size

    coco["images"].append({
        "id": image_id,
        "file_name": image_path.name,
        "width": width,
        "height": height
    })

    # Read YOLO label file
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id, xc, yc, bw, bh = map(float, parts)

            # Convert to absolute COCO format
            xc *= width
            yc *= height
            bw *= width
            bh *= height

            x_min = xc - bw / 2
            y_min = yc - bh / 2

            # Clamp to image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            bw = min(bw, width - x_min)
            bh = min(bh, height - y_min)

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(class_id),
                "bbox": [x_min, y_min, bw, bh],
                "area": bw * bh,
                "iscrowd": 0
            })

            annotation_id += 1

    image_id += 1

# ================= SAVE JSON =================
os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, "w") as f:
    json.dump(coco, f, indent=4)

print(f"✅ COCO annotations saved to:\n{output_json}")
        
