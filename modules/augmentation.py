"""Albumentations-based image augmentation pipeline."""
import base64
import io
import json
import logging
import shutil
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("hermes.augmentation")

# Named augmentation presets
AUGMENTATION_PRESETS = {
    "rain": ("Rain", lambda: A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=1.0)),
    "fog": ("Fog", lambda: A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.5, p=1.0)),
    "blur": ("Motion Blur", lambda: A.MotionBlur(blur_limit=7, p=1.0)),
    "flip_h": ("H-Flip", lambda: A.HorizontalFlip(p=1.0)),
    "rotate": ("Rotate 90", lambda: A.RandomRotate90(p=1.0)),
    "brightness": ("Brightness", lambda: A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0)),
    "color_jitter": ("Color Jitter", lambda: A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1, p=1.0)),
    "noise": ("Noise", lambda: A.GaussNoise(std_range=(0.05, 0.15), p=1.0)),
    "cutout": ("Cutout", lambda: A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(20, 40), hole_width_range=(20, 40), p=1.0)),
}


def _image_to_base64(img: np.ndarray, fmt: str = "JPEG") -> str:
    """Convert numpy image (BGR) to base64 string."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format=fmt, quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def preview_augmentations(img_path: str, aug_names: list) -> dict:
    """Generate augmented previews for a single image."""
    img = cv2.imread(img_path)
    if img is None:
        return {"error": "Could not read image"}

    # Resize for preview (max 640px wide)
    h, w = img.shape[:2]
    if w > 640:
        scale = 640 / w
        img = cv2.resize(img, (640, int(h * scale)))

    original_b64 = _image_to_base64(img)
    variants = []

    for aug_name in aug_names:
        if aug_name not in AUGMENTATION_PRESETS:
            continue

        display_name, aug_fn = AUGMENTATION_PRESETS[aug_name]
        try:
            transform = A.Compose([aug_fn()])
            result = transform(image=img)
            aug_b64 = _image_to_base64(result["image"])
            variants.append({
                "name": aug_name,
                "display_name": display_name,
                "image": aug_b64,
            })
        except Exception as e:
            logger.warning(f"Augmentation {aug_name} failed: {e}")

    return {
        "original": original_b64,
        "variants": variants,
    }


def apply_augmentations(dataset_dir: str, aug_names: list, multiplier: int) -> dict:
    """Apply augmentations to entire dataset, expanding it by multiplier."""
    ds_path = Path(dataset_dir)
    img_dir = ds_path / "images"
    lbl_dir = ds_path / "labels"

    images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
    original_count = len(images)

    if not aug_names:
        return {"error": "No augmentations selected"}

    # Build augmentation pipeline
    aug_fns = []
    for name in aug_names:
        if name in AUGMENTATION_PRESETS:
            _, fn = AUGMENTATION_PRESETS[name]
            aug_fns.append(fn())

    if not aug_fns:
        return {"error": "No valid augmentations"}

    # Create transform with bbox support
    transform = A.Compose(
        [A.OneOf(aug_fns, p=1.0)],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )

    generated = 0
    for rep in range(multiplier - 1):  # -1 because originals count as 1x
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Read labels if they exist
            lbl_file = lbl_dir / f"{img_path.stem}.txt"
            bboxes = []
            class_labels = []
            if lbl_file.exists() and lbl_file.stat().st_size > 0:
                for line in lbl_file.read_text().strip().split("\n"):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        bboxes.append([float(x) for x in parts[1:5]])
                        class_labels.append(int(parts[0]))

            try:
                result = transform(
                    image=img,
                    bboxes=bboxes,
                    class_labels=class_labels,
                )

                # Save augmented image
                aug_name = f"{img_path.stem}_aug{rep + 1}{img_path.suffix}"
                cv2.imwrite(str(img_dir / aug_name), result["image"])

                # Save transformed labels
                aug_lbl_name = f"{img_path.stem}_aug{rep + 1}.txt"
                lines = []
                for bbox, cls_id in zip(result["bboxes"], result["class_labels"]):
                    lines.append(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
                (lbl_dir / aug_lbl_name).write_text("\n".join(lines) + "\n" if lines else "")

                generated += 1
            except Exception as e:
                logger.warning(f"Augmentation failed for {img_path.name}: {e}")

    # Update metadata
    meta_file = ds_path / "metadata.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        new_count = len(list(img_dir.iterdir()))
        meta["image_count"] = new_count
        meta["augmented"] = True
        meta["original_count"] = original_count
        meta_file.write_text(json.dumps(meta, indent=2))

    final_count = original_count + generated
    logger.info(f"Augmented dataset: {original_count} -> {final_count} images ({generated} generated)")
    return {
        "original_count": original_count,
        "generated": generated,
        "final_count": final_count,
        "multiplier": multiplier,
    }
