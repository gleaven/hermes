#!/usr/bin/env python3
"""Download COCO val2017 and extract a vehicles-only subset in YOLO format.

Usage:
    python prepare_coco_vehicles.py [--output /path/to/output] [--max-images 200]

This script:
  1. Downloads COCO val2017 images + annotations (if not cached)
  2. Filters for vehicle classes: car, motorcycle, bus, truck
  3. Converts bounding boxes to YOLO format (normalized xywh)
  4. Copies matching images + labels to the output directory
  5. Creates metadata.json + dataset.yaml

Source: Microsoft COCO (Common Objects in Context) — US-based, CC BY 4.0
"""
import argparse
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

# COCO class IDs for vehicles
# Full list: https://cocodataset.org/#detection-2017
VEHICLE_CLASSES = {
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}

# Map COCO class IDs to YOLO class indices (0-based)
COCO_TO_YOLO = {3: 0, 4: 1, 6: 2, 8: 3}

COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download_file(url: str, dest: Path, desc: str = ""):
    """Download a file with progress."""
    if dest.exists():
        print(f"  [cached] {dest.name}")
        return

    print(f"  Downloading {desc or url}...")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {pct}% ({mb:.0f}/{total_mb:.0f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=progress_hook)
    print()


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract a ZIP file."""
    import zipfile
    print(f"  Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def main():
    parser = argparse.ArgumentParser(description="Prepare COCO vehicles subset")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: hermes/data/samples/coco-vehicles)")
    parser.add_argument("--cache", type=str, default="/tmp/coco_cache",
                        help="Cache directory for COCO downloads")
    parser.add_argument("--max-images", type=int, default=200,
                        help="Maximum number of images to include")
    args = parser.parse_args()

    # Determine output path
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    output_dir = Path(args.output) if args.output else project_dir / "data" / "samples" / "coco-vehicles"
    cache_dir = Path(args.cache)

    print(f"=== COCO Vehicles Dataset Preparation ===")
    print(f"Output: {output_dir}")
    print(f"Cache:  {cache_dir}")
    print(f"Max images: {args.max_images}")
    print()

    # Step 1: Download COCO data
    print("[1/5] Downloading COCO val2017...")
    images_zip = cache_dir / "val2017.zip"
    annotations_zip = cache_dir / "annotations_trainval2017.zip"

    download_file(COCO_VAL_IMAGES_URL, images_zip, "val2017 images (~1GB)")
    download_file(COCO_ANNOTATIONS_URL, annotations_zip, "annotations (~252MB)")

    # Step 2: Extract
    print("\n[2/5] Extracting archives...")
    coco_images_dir = cache_dir / "val2017"
    coco_annotations_file = cache_dir / "annotations" / "instances_val2017.json"

    if not coco_images_dir.exists():
        extract_zip(images_zip, cache_dir)
    else:
        print("  [cached] val2017/")

    if not coco_annotations_file.exists():
        extract_zip(annotations_zip, cache_dir)
    else:
        print("  [cached] annotations/")

    # Step 3: Parse annotations and filter vehicles
    print("\n[3/5] Filtering vehicle annotations...")
    with open(coco_annotations_file) as f:
        coco = json.load(f)

    # Build image lookup
    image_lookup = {img["id"]: img for img in coco["images"]}

    # Collect vehicle annotations grouped by image
    image_annotations = {}
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in VEHICLE_CLASSES:
            continue
        if ann.get("iscrowd", 0):
            continue

        img_id = ann["image_id"]
        if img_id not in image_annotations:
            image_annotations[img_id] = []

        # COCO bbox: [x, y, width, height] in pixels
        x, y, w, h = ann["bbox"]
        img_info = image_lookup[img_id]
        img_w, img_h = img_info["width"], img_info["height"]

        # Convert to YOLO format: normalized center x, center y, width, height
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h

        # Clamp to [0, 1]
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        nw = max(0, min(1, nw))
        nh = max(0, min(1, nh))

        # Skip tiny boxes
        if nw < 0.01 or nh < 0.01:
            continue

        image_annotations[img_id].append({
            "class_id": COCO_TO_YOLO[cat_id],
            "class_name": VEHICLE_CLASSES[cat_id],
            "cx": cx, "cy": cy, "w": nw, "h": nh,
        })

    # Sort by number of annotations (prefer images with more objects)
    sorted_images = sorted(image_annotations.keys(),
                           key=lambda x: len(image_annotations[x]), reverse=True)

    # Limit to max images
    selected = sorted_images[:args.max_images]
    print(f"  Found {len(image_annotations)} images with vehicles")
    print(f"  Selected {len(selected)} images (max {args.max_images})")

    # Step 4: Copy images and create labels
    print("\n[4/5] Creating dataset...")
    img_out = output_dir / "images"
    lbl_out = output_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    class_counts = {name: 0 for name in VEHICLE_CLASSES.values()}
    total_labels = 0

    for img_id in selected:
        img_info = image_lookup[img_id]
        filename = img_info["file_name"]
        src = coco_images_dir / filename

        if not src.exists():
            continue

        # Copy image
        shutil.copy2(src, img_out / filename)

        # Write YOLO label file
        stem = Path(filename).stem
        lines = []
        for ann in image_annotations[img_id]:
            lines.append(f"{ann['class_id']} {ann['cx']:.6f} {ann['cy']:.6f} {ann['w']:.6f} {ann['h']:.6f}")
            class_counts[ann["class_name"]] += 1
            total_labels += 1

        (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    actual_count = len(list(img_out.glob("*.jpg")))
    print(f"  Images: {actual_count}")
    print(f"  Labels: {total_labels}")
    print(f"  Classes: {class_counts}")

    # Step 5: Create metadata and dataset.yaml
    print("\n[5/5] Writing metadata...")

    classes = [
        {"id": 0, "name": "car", "color": "#00ff88"},
        {"id": 1, "name": "motorcycle", "color": "#00f0ff"},
        {"id": 2, "name": "bus", "color": "#ffaa00"},
        {"id": 3, "name": "truck", "color": "#ff3366"},
    ]

    metadata = {
        "name": "COCO Vehicles",
        "description": "Vehicle detection subset from Microsoft COCO val2017",
        "source": "Microsoft COCO (CC BY 4.0)",
        "image_count": actual_count,
        "label_count": total_labels,
        "classes": classes,
        "class_distribution": class_counts,
        "format": "yolo",
        "preloaded": True,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # dataset.yaml for YOLO training
    yaml_content = f"""path: {output_dir}
train: images
val: images

nc: 4
names: {{0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}}
"""
    (output_dir / "dataset.yaml").write_text(yaml_content)

    print(f"\n=== Done! Dataset ready at {output_dir} ===")
    print(f"  {actual_count} images, {total_labels} labels, 4 classes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
