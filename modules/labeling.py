"""YOLO format label I/O and auto-labeling."""
import json
import logging
from pathlib import Path

logger = logging.getLogger("hermes.labeling")


def read_yolo_labels(labels_dir: str) -> dict:
    """Read all YOLO format label files from a directory."""
    lbl_dir = Path(labels_dir)
    labels = {}
    if not lbl_dir.exists():
        return labels

    for f in lbl_dir.iterdir():
        if f.suffix == ".txt" and f.stat().st_size > 0:
            boxes = []
            for line in f.read_text().strip().split("\n"):
                parts = line.strip().split()
                if len(parts) >= 5:
                    boxes.append({
                        "class_id": int(parts[0]),
                        "x": float(parts[1]),
                        "y": float(parts[2]),
                        "w": float(parts[3]),
                        "h": float(parts[4]),
                    })
            labels[f.stem] = boxes
    return labels


def write_yolo_labels(labels_dir: str, annotations: dict):
    """Write YOLO format label files."""
    lbl_dir = Path(labels_dir)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for image_stem, boxes in annotations.items():
        lines = []
        for box in boxes:
            lines.append(
                f"{box['class_id']} {box['x']:.6f} {box['y']:.6f} {box['w']:.6f} {box['h']:.6f}"
            )
        (lbl_dir / f"{image_stem}.txt").write_text("\n".join(lines) + "\n" if lines else "")


def auto_label_dataset(dataset_dir: str, model_path: str) -> dict:
    """Run YOLO inference on all images and save as draft labels."""
    from ultralytics import YOLO

    ds_path = Path(dataset_dir)
    img_dir = ds_path / "images"
    lbl_dir = ds_path / "labels"
    lbl_dir.mkdir(exist_ok=True)

    model = YOLO(model_path)
    images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])

    labeled = 0
    total_boxes = 0
    class_ids_found = set()

    for img_path in images:
        results = model(str(img_path), verbose=False)[0]
        boxes_data = results.boxes

        if boxes_data is not None and len(boxes_data) > 0:
            lines = []
            for i in range(len(boxes_data)):
                cls_id = int(boxes_data.cls[i].item())
                # Get normalized xywh
                xywhn = boxes_data.xywhn[i]
                x, y, w, h = xywhn[0].item(), xywhn[1].item(), xywhn[2].item(), xywhn[3].item()
                lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                class_ids_found.add(cls_id)

            (lbl_dir / f"{img_path.stem}.txt").write_text("\n".join(lines) + "\n")
            labeled += 1
            total_boxes += len(lines)
        else:
            # Write empty label file
            (lbl_dir / f"{img_path.stem}.txt").write_text("")

    # Build class list from model names
    classes = []
    for cls_id in sorted(class_ids_found):
        name = model.names.get(cls_id, f"class_{cls_id}")
        classes.append({"id": cls_id, "name": name})

    # Update metadata
    meta_file = ds_path / "metadata.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        meta["classes"] = classes
        meta_file.write_text(json.dumps(meta, indent=2))

    logger.info(f"Auto-labeled {labeled}/{len(images)} images with {total_boxes} boxes")
    return {
        "labeled": labeled,
        "total_images": len(images),
        "total_boxes": total_boxes,
        "classes": classes,
    }
