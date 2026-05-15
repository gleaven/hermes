"""Dataset management utilities."""
import json
import shutil
import zipfile
from pathlib import Path


def create_dataset_yaml(dataset_dir: str, classes: list[dict]) -> str:
    """Create a YOLO-format dataset.yaml for training."""
    ds_path = Path(dataset_dir)
    img_dir = ds_path / "images"
    lbl_dir = ds_path / "labels"

    # Split images 80/20 for train/val
    images = sorted([f.name for f in img_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
    split = int(len(images) * 0.8)
    train_images = images[:split]
    val_images = images[split:]

    # Create train/val directories
    train_img_dir = ds_path / "train" / "images"
    train_lbl_dir = ds_path / "train" / "labels"
    val_img_dir = ds_path / "val" / "images"
    val_lbl_dir = ds_path / "val" / "labels"

    # Clean stale train/val splits so remapped labels are always fresh
    for d in [ds_path / "train", ds_path / "val"]:
        if d.exists():
            shutil.rmtree(d)
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Build contiguous class ID mapping (old_id -> new 0-based index)
    sorted_classes = sorted(classes, key=lambda c: c["id"])
    id_remap = {c["id"]: i for i, c in enumerate(sorted_classes)}
    class_names = {i: c["name"] for i, c in enumerate(sorted_classes)}
    nc = len(sorted_classes)

    def _copy_with_remap(img_name, dst_img_dir, dst_lbl_dir):
        src_img = img_dir / img_name
        dst_img = dst_img_dir / img_name
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
        lbl_name = Path(img_name).stem + ".txt"
        src_lbl = lbl_dir / lbl_name
        dst_lbl = dst_lbl_dir / lbl_name
        if src_lbl.exists():
            # Remap class IDs to contiguous 0-based indices
            lines = src_lbl.read_text().strip().splitlines()
            remapped = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        old_id = int(float(parts[0]))
                        new_id = id_remap.get(old_id, old_id)
                        remapped.append(f"{new_id} {' '.join(parts[1:])}")
                    except ValueError:
                        remapped.append(line)
                else:
                    remapped.append(line)
            dst_lbl.write_text("\n".join(remapped) + "\n")

    for img_name in train_images:
        _copy_with_remap(img_name, train_img_dir, train_lbl_dir)
    for img_name in val_images:
        _copy_with_remap(img_name, val_img_dir, val_lbl_dir)

    yaml_content = f"""path: {dataset_dir}
train: train/images
val: val/images

nc: {nc}
names: {json.dumps(class_names)}
"""
    yaml_path = ds_path / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    return str(yaml_path)
