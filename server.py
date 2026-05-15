"""HERMES - Computer Vision Pipeline Demo."""
import asyncio
import base64
import io
import json
import logging
import math
import os
import shutil
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from PIL import Image

import redis
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="[HERMES] %(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("hermes")

DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
REDIS_URL = os.environ.get("REDIS_URL", "redis://demo-redis:6379/9")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))
STATIC_DIR = Path(__file__).parent / "static"
SAMPLES_DIR = Path(__file__).parent / "data" / "samples"
PRETRAINED_DIR = Path(__file__).parent / "data" / "pretrained"

_redis = None
_ws_clients: dict[str, set] = {}
_yolo_model = None


def get_redis():
    global _redis
    if _redis is None:
        _redis = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis


def _job_key(job_id: str) -> str:
    return f"hermes:job:{job_id}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("HERMES starting...")
    for d in ["uploads", "jobs", "datasets"]:
        (DATA_DIR / d).mkdir(parents=True, exist_ok=True)
    yield
    logger.info("HERMES shutting down")


app = FastAPI(title="HERMES", lifespan=lifespan, docs_url=None, redoc_url=None)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "service": "hermes"}


# ---------------------------------------------------------------------------
# GPU Status
# ---------------------------------------------------------------------------
@app.get("/api/gpu")
async def gpu_status():
    result = {"available": False, "utilization_pct": 0}
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu", "--format=csv,noheader,nounits"],
            timeout=5, text=True,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        name = parts[0] if parts else "GPU"
        util = int(parts[1]) if len(parts) > 1 and parts[1] not in ("[N/A]", "") else 0
        result = {"available": True, "name": name, "utilization_pct": util}
    except Exception:
        pass

    # Add PyTorch memory info if available
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            alloc = torch.cuda.memory_allocated(0)
            total = props.total_memory
            result["available"] = True
            result.setdefault("name", props.name)
            result["allocated_mb"] = round(alloc / 1024**2)
            result["total_mb"] = round(total / 1024**2)
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Dataset Management
# ---------------------------------------------------------------------------
def _scan_datasets():
    """Scan sample and user datasets, return metadata list."""
    datasets = []

    # Pre-loaded samples
    if SAMPLES_DIR.exists():
        for d in sorted(SAMPLES_DIR.iterdir()):
            meta_file = d / "metadata.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                meta["id"] = d.name
                meta["source"] = "sample"
                meta["path"] = str(d)
                datasets.append(meta)

    # User-created datasets
    ds_dir = DATA_DIR / "datasets"
    if ds_dir.exists():
        for d in sorted(ds_dir.iterdir()):
            meta_file = d / "metadata.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                meta["id"] = d.name
                meta["source"] = "user"
                meta["path"] = str(d)
                datasets.append(meta)

    return datasets


@app.get("/api/datasets")
async def list_datasets():
    return _scan_datasets()


@app.get("/api/datasets/{dataset_id}/images")
async def list_dataset_images(dataset_id: str, page: int = 1, per_page: int = 50):
    ds = _find_dataset(dataset_id)
    if not ds:
        return JSONResponse({"error": "Dataset not found"}, 404)

    img_dir = Path(ds["path"]) / "images"
    lbl_dir = Path(ds["path"]) / "labels"
    images = sorted(
        [f.name for f in img_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    )
    total = len(images)
    start = (page - 1) * per_page
    page_images = images[start : start + per_page]

    result = []
    for img_name in page_images:
        stem = Path(img_name).stem
        has_label = (lbl_dir / f"{stem}.txt").exists()
        result.append({"name": img_name, "labeled": has_label})

    return {"images": result, "total": total, "page": page, "per_page": per_page}


@app.get("/api/datasets/{dataset_id}/images/{image_name}")
async def serve_dataset_image(dataset_id: str, image_name: str):
    ds = _find_dataset(dataset_id)
    if not ds:
        return JSONResponse({"error": "Dataset not found"}, 404)
    img_path = Path(ds["path"]) / "images" / image_name
    if not img_path.exists():
        return JSONResponse({"error": "Image not found"}, 404)
    return FileResponse(img_path)


def _find_dataset(dataset_id: str):
    for ds in _scan_datasets():
        if ds["id"] == dataset_id:
            return ds
    return None


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        return JSONResponse({"error": f"File too large ({size_mb:.1f}MB > {MAX_UPLOAD_MB}MB)"}, 413)

    dataset_id = uuid.uuid4().hex[:8]
    ds_dir = DATA_DIR / "datasets" / dataset_id
    img_dir = ds_dir / "images"
    lbl_dir = ds_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    if file.filename.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            img_exts = {".jpg", ".jpeg", ".png", ".webp"}
            extracted = 0
            for name in zf.namelist():
                if Path(name).suffix.lower() in img_exts and not name.startswith("__MACOSX"):
                    data = zf.read(name)
                    (img_dir / Path(name).name).write_bytes(data)
                    extracted += 1
        if extracted == 0:
            shutil.rmtree(ds_dir)
            return JSONResponse({"error": "No images found in ZIP"}, 400)
    else:
        ext = Path(file.filename).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            shutil.rmtree(ds_dir)
            return JSONResponse({"error": f"Unsupported format: {ext}"}, 400)
        (img_dir / file.filename).write_bytes(content)

    image_count = len(list(img_dir.iterdir()))
    meta = {
        "name": Path(file.filename).stem.replace("_", " ").replace("-", " ").title(),
        "classes": [],
        "image_count": image_count,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (ds_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    return {"dataset_id": dataset_id, "image_count": image_count, "name": meta["name"]}


# ---------------------------------------------------------------------------
# Satellite Tile Extraction
# ---------------------------------------------------------------------------
ESRI_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
MAX_TILES = 500
ZOOM_RESOLUTIONS = {15: 4.77, 16: 2.39, 17: 1.19, 18: 0.60, 19: 0.30}


def _lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Convert lat/lon to tile x, y at given zoom level (Web Mercator)."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int(
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)
        / 2.0
        * n
    )
    return x, y


def _compute_output_tiles(cols: int, rows: int, tile_size: int, overlap_pct: int):
    """Compute output tile grid dimensions after re-slicing a composite."""
    pw, ph = cols * 256, rows * 256
    stride = max(1, int(tile_size * (1.0 - overlap_pct / 100.0)))
    out_cols = max(1, math.ceil((pw - tile_size) / stride) + 1) if pw >= tile_size else (1 if pw > 0 else 0)
    out_rows = max(1, math.ceil((ph - tile_size) / stride) + 1) if ph >= tile_size else (1 if ph > 0 else 0)
    return out_cols, out_rows, stride


def _stitch_and_reslice(
    img_dir: Path,
    raw_tiles: list[tuple[int, int, Path]],
    x_min: int, y_min: int, cols: int, rows: int,
    output_tile_size: int, overlap_pct: int,
    region_idx: int,
) -> int:
    """Stitch raw tiles into composite, re-slice with overlap, return output tile count."""
    RAW = 256
    composite = Image.new("RGB", (cols * RAW, rows * RAW), (0, 0, 0))

    for tx, ty, fpath in raw_tiles:
        try:
            tile_img = Image.open(fpath)
            px = (tx - x_min) * RAW
            py = (ty - y_min) * RAW
            composite.paste(tile_img, (px, py))
            tile_img.close()
        except Exception:
            pass
        fpath.unlink(missing_ok=True)

    stride = max(1, int(output_tile_size * (1.0 - overlap_pct / 100.0)))
    cw, ch = composite.size
    output_count = 0
    y = 0
    row_idx = 0

    while y < ch:
        col_idx = 0
        x = 0
        while x < cw:
            box = (x, y, min(x + output_tile_size, cw), min(y + output_tile_size, ch))
            crop = composite.crop(box)
            if crop.size != (output_tile_size, output_tile_size):
                padded = Image.new("RGB", (output_tile_size, output_tile_size), (0, 0, 0))
                padded.paste(crop, (0, 0))
                crop = padded
            fname = f"tile_r{region_idx}_{row_idx:03d}_{col_idx:03d}.jpg"
            crop.save(img_dir / fname, quality=95)
            output_count += 1
            col_idx += 1
            x += stride
        row_idx += 1
        y += stride

    composite.close()
    return output_count


VALID_TILE_SIZES = (256, 512, 640, 1024)


@app.post("/api/datasets/tile-estimate")
async def tile_estimate(body: dict):
    """Estimate tile count for capture regions with optional re-slicing."""
    zoom = int(body.get("zoom", 18))
    output_tile_size = int(body.get("output_tile_size", 640))
    overlap_pct = int(body.get("overlap_pct", 0))

    if zoom < 15 or zoom > 19:
        return JSONResponse({"error": "Zoom must be between 15 and 19"}, 400)
    if output_tile_size not in VALID_TILE_SIZES:
        return JSONResponse({"error": "Invalid output tile size"}, 400)
    if not (0 <= overlap_pct <= 50):
        return JSONResponse({"error": "Overlap must be 0-50%"}, 400)

    # Support both single-region (backward compat) and multi-region
    raw_regions = body.get("regions")
    if raw_regions is None:
        raw_regions = [{"south": body["south"], "west": body["west"],
                        "north": body["north"], "east": body["east"]}]

    total_raw = 0
    total_output = 0
    total_area = 0.0
    region_details = []

    for reg in raw_regions:
        south, west = float(reg["south"]), float(reg["west"])
        north, east = float(reg["north"]), float(reg["east"])

        x_min, y_min = _lat_lon_to_tile(north, west, zoom)
        x_max, y_max = _lat_lon_to_tile(south, east, zoom)
        cols = x_max - x_min + 1
        rows = y_max - y_min + 1
        raw_count = max(cols, 0) * max(rows, 0)

        out_cols, out_rows, _ = _compute_output_tiles(cols, rows, output_tile_size, overlap_pct)
        output_count = out_cols * out_rows

        lat_center = (south + north) / 2
        area = abs(east - west) * 111.32 * math.cos(math.radians(lat_center)) * abs(north - south) * 111.32

        total_raw += raw_count
        total_output += output_count
        total_area += area
        region_details.append({
            "raw_tiles": raw_count, "output_tiles": output_count,
            "cols": cols, "rows": rows, "out_cols": out_cols, "out_rows": out_rows,
        })

    return {
        "raw_tile_count": total_raw,
        "output_tile_count": total_output,
        "region_count": len(raw_regions),
        "regions": region_details,
        "resolution_m_px": ZOOM_RESOLUTIONS.get(zoom, 0),
        "area_km2": round(total_area, 2),
        "exceeds_max": total_raw > MAX_TILES,
        "max_tiles": MAX_TILES,
        "output_tile_size": output_tile_size,
        "overlap_pct": overlap_pct,
        # Legacy compat
        "tile_count": total_output,
        "cols": region_details[0]["cols"] if region_details else 0,
        "rows": region_details[0]["rows"] if region_details else 0,
    }


@app.post("/api/datasets/from-tiles")
async def create_from_tiles(body: dict):
    """Download Esri tiles for one or more regions, stitch and re-slice.

    Streams newline-delimited JSON progress updates in two phases:
    Phase 1 (download): raw tile fetching per region
    Phase 2 (reslice): composite stitching and re-slicing per region
    """
    name = body.get("name", "Satellite Capture")
    zoom = int(body.get("zoom", 18))
    output_tile_size = int(body.get("output_tile_size", 640))
    overlap_pct = int(body.get("overlap_pct", 0))

    if zoom < 15 or zoom > 19:
        return JSONResponse({"error": "Zoom must be between 15 and 19"}, 400)
    if output_tile_size not in VALID_TILE_SIZES:
        return JSONResponse({"error": "Invalid output tile size"}, 400)
    if not (0 <= overlap_pct <= 50):
        return JSONResponse({"error": "Overlap must be 0-50%"}, 400)

    # Support both single-region (backward compat) and multi-region
    raw_regions = body.get("regions")
    if raw_regions is None:
        raw_regions = [{"south": body["south"], "west": body["west"],
                        "north": body["north"], "east": body["east"]}]

    # Validate total raw tiles
    total_raw = 0
    region_specs = []
    for reg in raw_regions:
        south, west = float(reg["south"]), float(reg["west"])
        north, east = float(reg["north"]), float(reg["east"])
        x_min, y_min = _lat_lon_to_tile(north, west, zoom)
        x_max, y_max = _lat_lon_to_tile(south, east, zoom)
        cols = x_max - x_min + 1
        rows = y_max - y_min + 1
        raw_count = cols * rows
        if raw_count <= 0:
            return JSONResponse({"error": "Invalid capture region"}, 400)
        total_raw += raw_count
        region_specs.append((south, west, north, east, x_min, y_min, x_max, y_max, cols, rows, raw_count))

    if total_raw > MAX_TILES:
        return JSONResponse(
            {"error": f"Total {total_raw} raw tiles exceeds max {MAX_TILES}"}, 400
        )

    async def _stream():
        dataset_id = uuid.uuid4().hex[:8]
        ds_dir = DATA_DIR / "datasets" / dataset_id
        img_dir = ds_dir / "images"
        lbl_dir = ds_dir / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        total_downloaded = 0
        total_failed = 0
        total_output = 0
        region_meta = []

        for ridx, (south, west, north, east, x_min, y_min, x_max, y_max, cols, rows, raw_count) in enumerate(region_specs):
            # Phase 1: Download raw tiles for this region
            downloaded = 0
            failed = 0
            raw_tiles = []  # (tx, ty, Path)

            async with httpx.AsyncClient(timeout=15.0) as client:
                processed = 0
                for ty in range(y_min, y_max + 1):
                    for tx in range(x_min, x_max + 1):
                        url = ESRI_TILE_URL.format(z=zoom, y=ty, x=tx)
                        try:
                            resp = await client.get(url)
                            if resp.status_code == 200:
                                # Save as temp raw file
                                raw_path = img_dir / f"_raw_r{ridx}_{tx}_{ty}.jpg"
                                raw_path.write_bytes(resp.content)
                                raw_tiles.append((tx, ty, raw_path))
                                downloaded += 1
                            else:
                                failed += 1
                        except Exception:
                            failed += 1
                        processed += 1
                        await asyncio.sleep(0.05)

                        if processed % 5 == 0 or processed == raw_count:
                            yield json.dumps({
                                "type": "progress",
                                "phase": "download",
                                "region": ridx,
                                "downloaded": downloaded,
                                "failed": failed,
                                "total": raw_count,
                            }) + "\n"

            total_downloaded += downloaded
            total_failed += failed

            if downloaded == 0:
                region_meta.append({
                    "bbox": {"south": south, "west": west, "north": north, "east": east},
                    "raw_tiles": 0, "output_tiles": 0,
                })
                continue

            # Phase 2: Stitch + re-slice
            yield json.dumps({
                "type": "progress",
                "phase": "reslice",
                "region": ridx,
                "status": "stitching",
                "output_tiles": 0,
            }) + "\n"

            output_count = _stitch_and_reslice(
                img_dir, raw_tiles, x_min, y_min, cols, rows,
                output_tile_size, overlap_pct, ridx,
            )
            total_output += output_count

            yield json.dumps({
                "type": "progress",
                "phase": "reslice",
                "region": ridx,
                "status": "complete",
                "output_tiles": output_count,
            }) + "\n"

            region_meta.append({
                "bbox": {"south": south, "west": west, "north": north, "east": east},
                "raw_tiles": downloaded, "output_tiles": output_count,
            })

        if total_output == 0 and total_downloaded == 0:
            shutil.rmtree(ds_dir)
            yield json.dumps({"type": "error", "error": "Failed to download any tiles"}) + "\n"
            return

        resolution = ZOOM_RESOLUTIONS.get(zoom, 0)
        meta = {
            "name": name,
            "classes": [],
            "image_count": total_output,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source_type": "satellite",
            "zoom": zoom,
            "resolution_m_px": resolution,
            "output_tile_size": output_tile_size,
            "overlap_pct": overlap_pct,
            "regions": region_meta,
            "total_raw_tiles": total_downloaded,
            "failed_tiles": total_failed,
        }
        (ds_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        yield json.dumps({
            "type": "complete",
            "dataset_id": dataset_id,
            "image_count": total_output,
            "failed": total_failed,
            "name": name,
            "resolution_m_px": resolution,
            "region_count": len(raw_regions),
            "output_tile_size": output_tile_size,
            "overlap_pct": overlap_pct,
        }) + "\n"

    return StreamingResponse(_stream(), media_type="application/x-ndjson")


@app.post("/api/datasets/{dataset_id}/append-tiles")
async def append_tiles(dataset_id: str, body: dict):
    """Append additional satellite tile regions to an existing dataset.

    Same stitch+reslice pipeline as from-tiles, but appends to existing dataset.
    Region indices continue from the highest existing region index.
    """
    ds_dir = DATA_DIR / "datasets" / dataset_id
    meta_path = ds_dir / "metadata.json"
    if not meta_path.exists():
        return JSONResponse({"error": "Dataset not found"}, 404)

    meta = json.loads(meta_path.read_text())
    zoom = int(body.get("zoom", meta.get("zoom", 18)))
    output_tile_size = int(body.get("output_tile_size", meta.get("output_tile_size", 640)))
    overlap_pct = int(body.get("overlap_pct", meta.get("overlap_pct", 0)))

    if zoom < 15 or zoom > 19:
        return JSONResponse({"error": "Zoom must be between 15 and 19"}, 400)
    if output_tile_size not in VALID_TILE_SIZES:
        return JSONResponse({"error": "Invalid output tile size"}, 400)
    if not (0 <= overlap_pct <= 50):
        return JSONResponse({"error": "Overlap must be 0-50%"}, 400)

    raw_regions = body.get("regions", [])
    if not raw_regions:
        return JSONResponse({"error": "No regions provided"}, 400)

    # Compute starting region index from existing data
    existing_regions = meta.get("regions", [])
    region_start_idx = len(existing_regions)

    # Validate total raw tiles for new regions
    total_raw = 0
    region_specs = []
    for reg in raw_regions:
        south, west = float(reg["south"]), float(reg["west"])
        north, east = float(reg["north"]), float(reg["east"])
        x_min, y_min = _lat_lon_to_tile(north, west, zoom)
        x_max, y_max = _lat_lon_to_tile(south, east, zoom)
        cols = x_max - x_min + 1
        rows = y_max - y_min + 1
        raw_count = cols * rows
        if raw_count <= 0:
            return JSONResponse({"error": "Invalid capture region"}, 400)
        total_raw += raw_count
        region_specs.append((south, west, north, east, x_min, y_min, x_max, y_max, cols, rows, raw_count))

    if total_raw > MAX_TILES:
        return JSONResponse(
            {"error": f"Total {total_raw} raw tiles exceeds max {MAX_TILES}"}, 400
        )

    img_dir = ds_dir / "images"

    async def _stream():
        total_downloaded = 0
        total_failed = 0
        total_output = 0
        new_region_meta = []

        for ridx_offset, (south, west, north, east, x_min, y_min, x_max, y_max, cols, rows, raw_count) in enumerate(region_specs):
            ridx = region_start_idx + ridx_offset

            # Phase 1: Download
            downloaded = 0
            failed = 0
            raw_tiles = []

            async with httpx.AsyncClient(timeout=15.0) as client:
                processed = 0
                for ty in range(y_min, y_max + 1):
                    for tx in range(x_min, x_max + 1):
                        url = ESRI_TILE_URL.format(z=zoom, y=ty, x=tx)
                        try:
                            resp = await client.get(url)
                            if resp.status_code == 200:
                                raw_path = img_dir / f"_raw_r{ridx}_{tx}_{ty}.jpg"
                                raw_path.write_bytes(resp.content)
                                raw_tiles.append((tx, ty, raw_path))
                                downloaded += 1
                            else:
                                failed += 1
                        except Exception:
                            failed += 1
                        processed += 1
                        await asyncio.sleep(0.05)

                        if processed % 5 == 0 or processed == raw_count:
                            yield json.dumps({
                                "type": "progress",
                                "phase": "download",
                                "region": ridx_offset,
                                "downloaded": downloaded,
                                "failed": failed,
                                "total": raw_count,
                            }) + "\n"

            total_downloaded += downloaded
            total_failed += failed

            if downloaded == 0:
                new_region_meta.append({
                    "bbox": {"south": south, "west": west, "north": north, "east": east},
                    "raw_tiles": 0, "output_tiles": 0,
                })
                continue

            # Phase 2: Stitch + re-slice
            yield json.dumps({
                "type": "progress", "phase": "reslice",
                "region": ridx_offset, "status": "stitching", "output_tiles": 0,
            }) + "\n"

            output_count = _stitch_and_reslice(
                img_dir, raw_tiles, x_min, y_min, cols, rows,
                output_tile_size, overlap_pct, ridx,
            )
            total_output += output_count

            yield json.dumps({
                "type": "progress", "phase": "reslice",
                "region": ridx_offset, "status": "complete", "output_tiles": output_count,
            }) + "\n"

            new_region_meta.append({
                "bbox": {"south": south, "west": west, "north": north, "east": east},
                "raw_tiles": downloaded, "output_tiles": output_count,
            })

        # Update metadata
        meta["regions"] = existing_regions + new_region_meta
        meta["image_count"] = meta.get("image_count", 0) + total_output
        meta["total_raw_tiles"] = meta.get("total_raw_tiles", 0) + total_downloaded
        meta["failed_tiles"] = meta.get("failed_tiles", 0) + total_failed
        meta_path.write_text(json.dumps(meta, indent=2))

        yield json.dumps({
            "type": "complete",
            "dataset_id": dataset_id,
            "image_count": meta["image_count"],
            "appended": total_output,
            "failed": total_failed,
            "region_count": len(meta["regions"]),
        }) + "\n"

    return StreamingResponse(_stream(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# Dataset Management (rename / delete)
# ---------------------------------------------------------------------------
@app.delete("/api/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a user-created dataset."""
    ds = _find_dataset(dataset_id)
    if not ds:
        return JSONResponse({"error": "Dataset not found"}, 404)
    if ds.get("source") == "sample":
        return JSONResponse({"error": "Cannot delete sample datasets"}, 403)
    shutil.rmtree(ds["path"])
    return {"ok": True, "deleted": dataset_id}


@app.patch("/api/datasets/{dataset_id}")
async def rename_dataset(dataset_id: str, body: dict):
    """Rename a dataset."""
    ds = _find_dataset(dataset_id)
    if not ds:
        return JSONResponse({"error": "Dataset not found"}, 404)
    new_name = body.get("name", "").strip()
    if not new_name:
        return JSONResponse({"error": "Name is required"}, 400)
    meta_path = Path(ds["path"]) / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["name"] = new_name
    meta_path.write_text(json.dumps(meta, indent=2))
    return {"ok": True, "name": new_name}


# ---------------------------------------------------------------------------
# Label Management
# ---------------------------------------------------------------------------
@app.get("/api/labels/{dataset_id}")
async def get_all_labels(dataset_id: str):
    ds = _find_dataset(dataset_id)
    if not ds:
        return JSONResponse({"error": "Dataset not found"}, 404)
    lbl_dir = Path(ds["path"]) / "labels"
    labels = {}
    if lbl_dir.exists():
        for f in lbl_dir.iterdir():
            if f.suffix == ".txt" and f.stat().st_size > 0:
                boxes = []
                for line in f.read_text().strip().split("\n"):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            boxes.append({
                                "class_id": int(float(parts[0])),
                                "x": float(parts[1]),
                                "y": float(parts[2]),
                                "w": float(parts[3]),
                                "h": float(parts[4]),
                            })
                        except ValueError:
                            continue
                labels[f.stem] = boxes
    return labels


@app.get("/api/labels/{dataset_id}/{image_name}")
async def get_image_labels(dataset_id: str, image_name: str):
    ds = _find_dataset(dataset_id)
    if not ds:
        return JSONResponse({"error": "Dataset not found"}, 404)
    stem = Path(image_name).stem
    lbl_file = Path(ds["path"]) / "labels" / f"{stem}.txt"
    boxes = []
    if lbl_file.exists() and lbl_file.stat().st_size > 0:
        for line in lbl_file.read_text().strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    boxes.append({
                        "class_id": int(float(parts[0])),
                        "x": float(parts[1]),
                        "y": float(parts[2]),
                        "w": float(parts[3]),
                        "h": float(parts[4]),
                    })
                except ValueError:
                    continue
    return {"image": image_name, "boxes": boxes}


@app.post("/api/labels/{dataset_id}")
async def save_labels(dataset_id: str, body: dict):
    ds = _find_dataset(dataset_id)
    if not ds:
        return JSONResponse({"error": "Dataset not found"}, 404)
    lbl_dir = Path(ds["path"]) / "labels"
    lbl_dir.mkdir(exist_ok=True)

    labels = body.get("labels", {})
    classes = body.get("classes", [])
    saved = 0
    for image_stem, boxes in labels.items():
        lines = []
        for box in boxes:
            lines.append(f"{box['class_id']} {box['x']:.6f} {box['y']:.6f} {box['w']:.6f} {box['h']:.6f}")
        (lbl_dir / f"{image_stem}.txt").write_text("\n".join(lines) + "\n" if lines else "")
        saved += 1

    # Update metadata with classes
    meta_file = Path(ds["path"]) / "metadata.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        meta["classes"] = classes
        meta_file.write_text(json.dumps(meta, indent=2))

    return {"saved": saved}


@app.post("/api/labels/{dataset_id}/auto")
async def auto_label(dataset_id: str):
    ds = _find_dataset(dataset_id)
    if not ds:
        return JSONResponse({"error": "Dataset not found"}, 404)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: _run_auto_label(ds))
    return result


def _run_auto_label(ds: dict):
    from modules.labeling import auto_label_dataset

    ds_path = Path(ds["path"])
    # Use pre-trained model if available, else default YOLO11n
    model_path = PRETRAINED_DIR / "yolo11n-vehicles.pt"
    if not model_path.exists():
        model_path = "yolo11n.pt"

    result = auto_label_dataset(str(ds_path), str(model_path))
    return result


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
@app.post("/api/augment/preview")
async def augment_preview(body: dict):
    dataset_id = body.get("dataset_id")
    image_name = body.get("image_name")
    augmentations = body.get("augmentations", [])

    ds = _find_dataset(dataset_id)
    if not ds:
        return JSONResponse({"error": "Dataset not found"}, 404)

    img_path = Path(ds["path"]) / "images" / image_name
    if not img_path.exists():
        return JSONResponse({"error": "Image not found"}, 404)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _preview_augmentations(str(img_path), augmentations),
    )
    return result


def _preview_augmentations(img_path: str, aug_names: list):
    from modules.augmentation import preview_augmentations

    return preview_augmentations(img_path, aug_names)


@app.post("/api/augment/apply")
async def augment_apply(body: dict):
    dataset_id = body.get("dataset_id")
    augmentations = body.get("augmentations", [])
    multiplier = body.get("multiplier", 3)

    ds = _find_dataset(dataset_id)
    if not ds:
        return JSONResponse({"error": "Dataset not found"}, 404)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _apply_augmentations(ds, augmentations, multiplier),
    )
    return result


def _apply_augmentations(ds: dict, aug_names: list, multiplier: int):
    from modules.augmentation import apply_augmentations

    ds_path = Path(ds["path"])
    return apply_augmentations(str(ds_path), aug_names, multiplier)


# ---------------------------------------------------------------------------
# Dataset Stats
# ---------------------------------------------------------------------------
@app.get("/api/datasets/{dataset_id}/stats")
async def dataset_stats(dataset_id: str):
    ds = _find_dataset(dataset_id)
    if not ds:
        return JSONResponse({"error": "Dataset not found"}, 404)

    ds_path = Path(ds["path"])
    img_dir = ds_path / "images"
    lbl_dir = ds_path / "labels"

    images = [f for f in img_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    image_count = len(images)

    class_counts = {}
    labeled_count = 0
    total_boxes = 0
    for img in images:
        lbl_file = lbl_dir / f"{img.stem}.txt"
        if lbl_file.exists() and lbl_file.stat().st_size > 0:
            labeled_count += 1
            for line in lbl_file.read_text().strip().split("\n"):
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cls_id = int(float(parts[0]))
                    except ValueError:
                        continue
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                    total_boxes += 1

    # Load class names from metadata
    meta_file = ds_path / "metadata.json"
    classes = []
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        classes = meta.get("classes", [])

    # Compute balance score (0-100, 100 = perfectly balanced)
    if class_counts:
        counts = list(class_counts.values())
        max_c, min_c = max(counts), min(counts)
        balance = round(min_c / max_c * 100) if max_c > 0 else 100
    else:
        balance = 0

    coverage = round(labeled_count / image_count * 100) if image_count > 0 else 0

    return {
        "image_count": image_count,
        "labeled_count": labeled_count,
        "total_boxes": total_boxes,
        "avg_boxes_per_image": round(total_boxes / labeled_count, 2) if labeled_count > 0 else 0,
        "class_distribution": class_counts,
        "classes": classes,
        "balance_score": balance,
        "coverage_score": coverage,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@app.post("/api/train")
async def start_training(body: dict):
    dataset_id = body.get("dataset_id")
    ds = _find_dataset(dataset_id)
    if not ds:
        return JSONResponse({"error": "Dataset not found"}, 404)

    job_id = uuid.uuid4().hex[:8]
    job_dir = DATA_DIR / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": body.get("model", "yolo11n"),
        "epochs": min(int(body.get("epochs", 50)), 200),
        "imgsz": int(body.get("imgsz", 640)),
        "batch": int(body.get("batch", 16)),
        "dataset_path": ds["path"],
        "dataset_id": dataset_id,
    }

    r = get_redis()
    state = {
        "job_id": job_id,
        "status": "starting",
        "epoch": 0,
        "total_epochs": config["epochs"],
        "progress": 0,
        "metrics": {},
        "history": {"box_loss": [], "cls_loss": [], "dfl_loss": [], "mAP50": [], "mAP50_95": []},
        "config": config,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    r.set(_job_key(job_id), json.dumps(state), ex=86400)

    asyncio.create_task(_run_training_async(job_id, config, str(job_dir)))
    return {"job_id": job_id}


async def _run_training_async(job_id: str, config: dict, job_dir: str):
    loop = asyncio.get_event_loop()

    def progress_callback(update: dict):
        r = get_redis()
        r.set(_job_key(job_id), json.dumps(update), ex=86400)
        loop.call_soon_threadsafe(asyncio.ensure_future, _broadcast_training(job_id, update))

    try:
        await loop.run_in_executor(None, lambda: _do_training(job_id, config, job_dir, progress_callback))
    except Exception as e:
        logger.exception(f"Training failed for job {job_id}")
        r = get_redis()
        state = {"job_id": job_id, "status": "failed", "error": str(e), "progress": 0}
        r.set(_job_key(job_id), json.dumps(state), ex=86400)
        await _broadcast_training(job_id, state)


def _do_training(job_id: str, config: dict, job_dir: str, progress_callback):
    from modules.training import run_training

    run_training(job_id, config, job_dir, progress_callback)


async def _broadcast_training(job_id: str, state: dict):
    clients = _ws_clients.get(job_id, set())
    dead = set()
    for ws in clients:
        try:
            await ws.send_json(state)
        except Exception:
            dead.add(ws)
    clients -= dead


@app.get("/api/train/history")
async def training_history():
    """List completed training runs with saved metrics."""
    jobs_dir = DATA_DIR / "jobs"
    runs = []
    if jobs_dir.exists():
        for jd in sorted(jobs_dir.iterdir(), reverse=True):
            metrics_file = jd / "training-metrics.json"
            best_model = jd / "train" / "weights" / "best.pt"
            if metrics_file.exists():
                try:
                    data = json.loads(metrics_file.read_text())
                    runs.append({
                        "job_id": jd.name,
                        "model_path": str(best_model) if best_model.exists() else None,
                        "config": data.get("config", {}),
                        "final": data.get("final", {}),
                        "history": data.get("history", {}),
                        "elapsed_seconds": data.get("elapsed_seconds", 0),
                    })
                except Exception:
                    pass
    return {"runs": runs}


@app.delete("/api/train/history")
async def clear_training_history():
    """Clear all training job data."""
    jobs_dir = DATA_DIR / "jobs"
    if jobs_dir.exists():
        shutil.rmtree(jobs_dir)
        jobs_dir.mkdir(parents=True, exist_ok=True)
    return {"cleared": True}


@app.get("/api/train/{job_id}")
async def get_training_status(job_id: str):
    r = get_redis()
    data = r.get(_job_key(job_id))
    if not data:
        return JSONResponse({"error": "Job not found"}, 404)
    return json.loads(data)


@app.websocket("/ws/train/{job_id}")
async def ws_training(ws: WebSocket, job_id: str):
    await ws.accept()
    if job_id not in _ws_clients:
        _ws_clients[job_id] = set()
    _ws_clients[job_id].add(ws)
    logger.info(f"Training WS connected: {job_id}")

    # Send current state immediately
    r = get_redis()
    data = r.get(_job_key(job_id))
    if data:
        await ws.send_json(json.loads(data))

    try:
        while True:
            await ws.receive_text()  # Keep alive
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.get(job_id, set()).discard(ws)
        logger.info(f"Training WS disconnected: {job_id}")


# ---------------------------------------------------------------------------
# Inference & Visualization
# ---------------------------------------------------------------------------
@app.post("/api/detect")
async def detect(body: dict):
    dataset_id = body.get("dataset_id")
    image_name = body.get("image_name")
    model_path = body.get("model_path")
    confidence = float(body.get("confidence", 0.25))

    ds = _find_dataset(dataset_id)
    if not ds:
        return JSONResponse({"error": "Dataset not found"}, 404)

    img_path = Path(ds["path"]) / "images" / image_name
    if not img_path.exists():
        return JSONResponse({"error": "Image not found"}, 404)

    if not model_path:
        # Try to find a trained model in jobs, else use pretrained
        model_path = str(PRETRAINED_DIR / "yolo11n-vehicles.pt")
        # Check if there's a recent training job with a model
        jobs_dir = DATA_DIR / "jobs"
        if jobs_dir.exists():
            for jd in sorted(jobs_dir.iterdir(), reverse=True):
                best = jd / "train" / "weights" / "best.pt"
                if best.exists():
                    model_path = str(best)
                    break

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _run_detection(str(img_path), model_path, confidence),
    )
    return result


def _run_detection(img_path: str, model_path: str, confidence: float):
    from modules.inference import detect_objects

    return detect_objects(img_path, model_path, confidence)


@app.post("/api/detect/annotate")
async def detect_annotate(body: dict):
    dataset_id = body.get("dataset_id")
    image_name = body.get("image_name")
    model_path = body.get("model_path")
    confidence = float(body.get("confidence", 0.25))
    style = body.get("style", "box_corner")

    ds = _find_dataset(dataset_id)
    if not ds:
        return JSONResponse({"error": "Dataset not found"}, 404)

    img_path = Path(ds["path"]) / "images" / image_name
    if not img_path.exists():
        return JSONResponse({"error": "Image not found"}, 404)

    if not model_path:
        model_path = str(PRETRAINED_DIR / "yolo11n-vehicles.pt")
        jobs_dir = DATA_DIR / "jobs"
        if jobs_dir.exists():
            for jd in sorted(jobs_dir.iterdir(), reverse=True):
                best = jd / "train" / "weights" / "best.pt"
                if best.exists():
                    model_path = str(best)
                    break

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _run_annotated_detection(str(img_path), model_path, confidence, style),
    )
    return result


def _run_annotated_detection(img_path: str, model_path: str, confidence: float, style: str):
    from modules.inference import detect_and_annotate

    return detect_and_annotate(img_path, model_path, confidence, style)


# ---------------------------------------------------------------------------
# Test Image Upload (for inference on external images)
# ---------------------------------------------------------------------------
@app.post("/api/test-images/upload")
async def upload_test_image(file: UploadFile = File(...)):
    """Upload an external image for model testing."""
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse({"error": "Only image files allowed"}, 400)

    test_dir = DATA_DIR / "test-images"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Save with unique prefix to avoid collisions
    safe_name = file.filename.replace("/", "_").replace("\\", "_")
    dest = test_dir / safe_name
    if dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        dest = test_dir / f"{stem}_{uuid.uuid4().hex[:6]}{suffix}"

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10MB limit per image
        return JSONResponse({"error": "Image too large (max 10MB)"}, 400)

    dest.write_bytes(content)
    return {"name": dest.name, "size": len(content)}


@app.get("/api/test-images")
async def list_test_images():
    """List uploaded test images."""
    test_dir = DATA_DIR / "test-images"
    if not test_dir.exists():
        return {"images": []}
    images = sorted(
        [f.name for f in test_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")],
    )
    return {"images": images}


@app.get("/api/test-images/{image_name}")
async def serve_test_image(image_name: str):
    """Serve an uploaded test image."""
    img_path = DATA_DIR / "test-images" / image_name
    if not img_path.exists():
        return JSONResponse({"error": "Image not found"}, 404)
    return FileResponse(str(img_path))


@app.post("/api/detect/annotate-upload")
async def detect_annotate_upload(body: dict):
    """Run detection on an uploaded test image (not from a dataset)."""
    image_name = body.get("image_name")
    model_path = body.get("model_path")
    confidence = float(body.get("confidence", 0.25))
    style = body.get("style", "box_corner")

    img_path = DATA_DIR / "test-images" / image_name
    if not img_path.exists():
        return JSONResponse({"error": "Test image not found"}, 404)

    if not model_path:
        model_path = str(PRETRAINED_DIR / "yolo11n-vehicles.pt")
        jobs_dir = DATA_DIR / "jobs"
        if jobs_dir.exists():
            for jd in sorted(jobs_dir.iterdir(), reverse=True):
                best = jd / "train" / "weights" / "best.pt"
                if best.exists():
                    model_path = str(best)
                    break

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _run_annotated_detection(str(img_path), model_path, confidence, style),
    )
    return result


# ---------------------------------------------------------------------------
# Model Export
# ---------------------------------------------------------------------------
@app.get("/api/model/export")
async def export_model():
    """Download the trained YOLO model."""
    # Find the most recent trained model
    model_path = None
    jobs_dir = DATA_DIR / "jobs"
    if jobs_dir.exists():
        for jd in sorted(jobs_dir.iterdir(), reverse=True):
            best = jd / "train" / "weights" / "best.pt"
            if best.exists():
                model_path = best
                break

    if not model_path:
        # Fall back to pretrained
        model_path = PRETRAINED_DIR / "yolo11n-vehicles.pt"
        if not model_path.exists():
            return JSONResponse({"error": "No trained model available"}, 404)

    return FileResponse(
        str(model_path),
        media_type="application/octet-stream",
        filename=f"hermes-model-{model_path.stem}.pt",
    )


# ---------------------------------------------------------------------------
# Pre-baked Demo Data
# ---------------------------------------------------------------------------
@app.get("/api/demo/prebaked")
async def get_prebaked():
    result = {"available": False}
    metrics_file = PRETRAINED_DIR / "training-metrics.json"
    if metrics_file.exists():
        result["training_metrics"] = json.loads(metrics_file.read_text())
        result["available"] = True
    model_path = PRETRAINED_DIR / "yolo11n-vehicles.pt"
    result["has_model"] = model_path.exists()
    return result


# ---------------------------------------------------------------------------
# SPA Fallback + Static Files
# ---------------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
