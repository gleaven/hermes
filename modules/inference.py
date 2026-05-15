"""YOLO inference and Supervision-based visualization."""
import base64
import io
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from PIL import Image

logger = logging.getLogger("hermes.inference")

# Cache loaded models
_model_cache = {}

# Class colors for visualization (tactical palette)
CLASS_COLORS = sv.ColorPalette.from_hex([
    "#00ff88",  # green
    "#00f0ff",  # cyan
    "#ffaa00",  # amber
    "#ff3366",  # red
    "#7b2dff",  # purple
    "#ff00aa",  # magenta
    "#00b4d8",  # teal
    "#76b900",  # lime
])


def _get_model(model_path: str):
    """Get or load a cached YOLO model."""
    global _model_cache
    if model_path not in _model_cache:
        from ultralytics import YOLO
        _model_cache[model_path] = YOLO(model_path)
        logger.info(f"Loaded model: {model_path}")
    return _model_cache[model_path]


def _image_to_base64(img: np.ndarray) -> str:
    """Convert BGR numpy array to base64 JPEG."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def detect_objects(img_path: str, model_path: str, confidence: float = 0.25) -> dict:
    """Run YOLO detection and return structured results."""
    model = _get_model(model_path)

    start = time.time()
    results = model(img_path, conf=confidence, verbose=False)[0]
    inference_ms = round((time.time() - start) * 1000)

    detections = sv.Detections.from_ultralytics(results)

    detection_list = []
    if len(detections) > 0:
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i]
            detection_list.append({
                "class_id": int(detections.class_id[i]),
                "class_name": results.names[int(detections.class_id[i])],
                "confidence": round(float(detections.confidence[i]), 3),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })

    # Class summary
    class_summary = {}
    for d in detection_list:
        name = d["class_name"]
        class_summary[name] = class_summary.get(name, 0) + 1

    return {
        "detections": detection_list,
        "count": len(detection_list),
        "class_summary": class_summary,
        "inference_ms": inference_ms,
        "model": Path(model_path).name,
    }


def detect_and_annotate(
    img_path: str,
    model_path: str,
    confidence: float = 0.25,
    style: str = "box_corner",
) -> dict:
    """Run detection and return Supervision-annotated image as base64."""
    model = _get_model(model_path)

    start = time.time()
    results = model(img_path, conf=confidence, verbose=False)[0]
    inference_ms = round((time.time() - start) * 1000)

    detections = sv.Detections.from_ultralytics(results)
    img = cv2.imread(img_path)

    # Build labels
    labels = []
    if len(detections) > 0:
        for i in range(len(detections)):
            cls_name = results.names[int(detections.class_id[i])]
            conf = float(detections.confidence[i])
            labels.append(f"{cls_name} {conf:.0%}")

    # Apply selected annotation style
    annotated = img.copy()

    if style == "box_corner":
        annotated = sv.BoxCornerAnnotator(
            thickness=2,
            corner_length=15,
            color=CLASS_COLORS,
        ).annotate(scene=annotated, detections=detections)
        annotated = sv.LabelAnnotator(
            text_scale=0.4,
            text_padding=4,
            color=CLASS_COLORS,
        ).annotate(scene=annotated, detections=detections, labels=labels)

    elif style == "box":
        annotated = sv.BoxAnnotator(
            thickness=2,
            color=CLASS_COLORS,
        ).annotate(scene=annotated, detections=detections)
        annotated = sv.LabelAnnotator(
            text_scale=0.4,
            text_padding=4,
            color=CLASS_COLORS,
        ).annotate(scene=annotated, detections=detections, labels=labels)

    elif style == "heatmap":
        annotated = sv.HeatMapAnnotator(
            position=sv.Position.CENTER,
            opacity=0.6,
            radius=40,
        ).annotate(scene=annotated, detections=detections)

    elif style == "circle":
        annotated = sv.CircleAnnotator(
            thickness=2,
            color=CLASS_COLORS,
        ).annotate(scene=annotated, detections=detections)
        annotated = sv.LabelAnnotator(
            text_scale=0.4,
            text_padding=4,
            color=CLASS_COLORS,
        ).annotate(scene=annotated, detections=detections, labels=labels)

    # Detection summary
    class_summary = {}
    for i in range(len(detections)):
        name = results.names[int(detections.class_id[i])]
        class_summary[name] = class_summary.get(name, 0) + 1

    return {
        "image": _image_to_base64(annotated),
        "count": len(detections),
        "class_summary": class_summary,
        "inference_ms": inference_ms,
        "style": style,
    }
