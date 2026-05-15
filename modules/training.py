"""YOLO training wrapper with progress callbacks."""
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger("hermes.training")


def run_training(job_id: str, config: dict, job_dir: str, progress_callback):
    """Run YOLO training with epoch-level progress reporting."""
    from ultralytics import YOLO
    from modules.datasets import create_dataset_yaml

    ds_path = config["dataset_path"]
    model_name = config["model"]
    epochs = config["epochs"]
    imgsz = config["imgsz"]
    batch = config["batch"]

    # Load class info from metadata
    meta_file = Path(ds_path) / "metadata.json"
    classes = []
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        classes = meta.get("classes", [])

    if not classes:
        raise ValueError("No classes defined. Please label your dataset first.")

    # Create dataset.yaml with train/val split
    dataset_yaml = create_dataset_yaml(ds_path, classes)
    logger.info(f"Dataset YAML created at {dataset_yaml}")

    # Initialize model
    model = YOLO(f"{model_name}.pt")

    # Track metrics history
    history = {
        "box_loss": [],
        "cls_loss": [],
        "dfl_loss": [],
        "mAP50": [],
        "mAP50_95": [],
    }
    start_time = time.time()

    def on_train_epoch_end(trainer):
        epoch = trainer.epoch + 1
        metrics = trainer.metrics or {}
        loss = trainer.loss_items if hasattr(trainer, "loss_items") else None

        # Extract loss values
        box_loss = float(metrics.get("train/box_loss", 0))
        cls_loss = float(metrics.get("train/cls_loss", 0))
        dfl_loss = float(metrics.get("train/dfl_loss", 0))

        # If loss items available from trainer directly
        if loss is not None and len(loss) >= 3:
            box_loss = float(loss[0])
            cls_loss = float(loss[1])
            dfl_loss = float(loss[2])

        # Extract mAP
        map50 = float(metrics.get("metrics/mAP50(B)", 0))
        map50_95 = float(metrics.get("metrics/mAP50-95(B)", 0))

        history["box_loss"].append(round(box_loss, 4))
        history["cls_loss"].append(round(cls_loss, 4))
        history["dfl_loss"].append(round(dfl_loss, 4))
        history["mAP50"].append(round(map50, 4))
        history["mAP50_95"].append(round(map50_95, 4))

        elapsed = time.time() - start_time
        epoch_time = elapsed / epoch if epoch > 0 else 0
        remaining = epoch_time * (epochs - epoch)

        progress_callback({
            "job_id": job_id,
            "status": "training",
            "epoch": epoch,
            "total_epochs": epochs,
            "progress": round(epoch / epochs * 100, 1),
            "metrics": {
                "box_loss": round(box_loss, 4),
                "cls_loss": round(cls_loss, 4),
                "dfl_loss": round(dfl_loss, 4),
                "mAP50": round(map50, 4),
                "mAP50_95": round(map50_95, 4),
            },
            "history": history,
            "elapsed_seconds": round(elapsed),
            "eta_seconds": round(remaining),
        })

    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    # Notify starting
    progress_callback({
        "job_id": job_id,
        "status": "training",
        "epoch": 0,
        "total_epochs": epochs,
        "progress": 0,
        "metrics": {},
        "history": history,
    })

    # Run training
    logger.info(f"Starting YOLO training: model={model_name}, epochs={epochs}, imgsz={imgsz}, batch={batch}")
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=job_dir,
        name="train",
        exist_ok=True,
        device=0,
        workers=2,
        verbose=False,
    )

    # Final metrics
    elapsed = time.time() - start_time
    final_metrics = {
        "box_loss": history["box_loss"][-1] if history["box_loss"] else 0,
        "cls_loss": history["cls_loss"][-1] if history["cls_loss"] else 0,
        "dfl_loss": history["dfl_loss"][-1] if history["dfl_loss"] else 0,
        "mAP50": history["mAP50"][-1] if history["mAP50"] else 0,
        "mAP50_95": history["mAP50_95"][-1] if history["mAP50_95"] else 0,
    }

    # Save training metrics for demo mode
    metrics_output = {
        "history": history,
        "final": final_metrics,
        "config": config,
        "elapsed_seconds": round(elapsed),
    }
    metrics_path = Path(job_dir) / "training-metrics.json"
    metrics_path.write_text(json.dumps(metrics_output, indent=2))

    # Find best model path
    best_path = Path(job_dir) / "train" / "weights" / "best.pt"
    model_path = str(best_path) if best_path.exists() else None

    progress_callback({
        "job_id": job_id,
        "status": "completed",
        "epoch": epochs,
        "total_epochs": epochs,
        "progress": 100,
        "metrics": final_metrics,
        "history": history,
        "model_path": model_path,
        "elapsed_seconds": round(elapsed),
    })

    logger.info(f"Training complete: mAP50={final_metrics['mAP50']:.4f}, elapsed={elapsed:.0f}s")
