# HERMES — End-to-End Computer Vision Pipeline

> Capture imagery, label it, augment it, train YOLO11 on your local
> GPU, and test the model — all in one browser interface, with no
> external services.

---

## What this demo is

HERMES walks you through the full computer-vision build pipeline as a
**seven-stage workflow** in a single web app:

1. **Data Source** — pick where training imagery comes from (Esri
   satellite tiles or a curated dataset gallery).
2. **Data Collection** — pull a pre-built dataset, upload your own ZIP
   of images, or extract raw imagery for a region of the planet.
3. **Annotation** — draw bounding boxes on an interactive HTML5 canvas,
   or run AUTO-LABEL to bootstrap labels from a pre-trained model.
4. **Augmentation** — preview and apply Albumentations transforms
   (rain, fog, blur, flip, rotate, brightness, color jitter, noise,
   cutout) to expand the training set 2×–5×.
5. **Quality Assurance** — inspect class distribution, balance score,
   coverage score, and a shuffleable sample grid before training.
6. **Model Training** — kick off YOLO11 (n or s) on the GPU with live
   loss curves, mAP@50 / mAP@50-95 charts, and ETA.
7. **Model Testing** — run the trained model on an uploaded test image,
   switch between Supervision annotation styles (corner / box / heatmap
   / circle), adjust the confidence threshold, and download the `.pt`
   checkpoint.

There is also an **auto-play demo mode** (top-right button) that walks
through every stage with the bundled pre-baked training metrics, so you
can present the whole pipeline end-to-end in a few minutes without
actually running training.

The reference scenario is **vehicle detection on satellite tiles**:
draw a box over San Francisco, Dubai, or any other location at zoom
15–19, extract the tiles, label cars and trucks, train YOLO11, and run
detection on a fresh test image — all without leaving the page.

### Stage 1 — Data Source

Two source cards on the landing screen:

- **GEO CAPTURE** — a Leaflet map over **Esri World Imagery**. Draw
  one or more rectangular capture regions, choose a zoom level (Z15
  ≈ 4.77 m/px down to Z19 ≈ 0.30 m/px), and the server fetches the
  raw 256-px tiles, stitches them into a composite, and re-slices the
  composite into uniform output tiles (256, 512, 640, or 1024 px) with
  optional 0–50 % overlap. Hard cap of **500 raw tiles per request**.
  Quick-jump buttons for San Francisco, the Grand Canyon, Dubai, and
  the Statue of Liberty are pre-wired.
- **TRAINING DATASETS** — a dataset gallery card list plus a ZIP
  upload zone. The bundled `scripts/prepare_coco_vehicles.py` and
  `scripts/download_gallery.py` populate the gallery (COCO vehicles
  subset and ten Roboflow Universe domains: PPE, military vehicles,
  bone-fracture X-ray, FLIR thermal, traffic signs, marine species,
  corrosion, aerial livestock, crop pests, trail-camera wildlife —
  all CC BY 4.0). Out of the box `data/samples/` is empty — see
  *Pre-loading the gallery* below.

### Stage 3 — Annotation

A custom HTML5 canvas labeler:

- Click-and-drag to draw a YOLO-format bounding box.
- Right-click a box to delete it.
- Manage the class palette in-line (add / rename / pick a color).
- **AUTO-LABEL** runs `yolo11n.pt` (or `data/pretrained/yolo11n-vehicles.pt`
  if present) over every image in the dataset and saves the boxes as
  draft labels you can refine.
- Labels are written to `<dataset>/labels/<image>.txt` in standard
  YOLO `class_id cx cy w h` (normalized) format.

### Stage 6 — Model Training

The trainer wraps Ultralytics' `YOLO.train(...)` and reports per-epoch
metrics over a WebSocket. The Stage-5 dashboard renders:

- A progress bar with epoch / total and live ETA.
- Loss-curve chart (`box_loss`, `cls_loss`, `dfl_loss`).
- mAP chart (`mAP@50`, `mAP@50-95`).
- Final-metrics summary on completion.

Datasets are auto-split 80 / 20 train / val and a `dataset.yaml` is
generated on the fly. Class IDs are remapped to a contiguous 0-based
range so you can label with any IDs and training still works.

---

## Capabilities (at a glance)

- Seven-stage browser workflow from raw tiles to deployable `.pt`.
- Esri World Imagery satellite ingestion with stitch + re-slice +
  overlap, multi-region capture, and append-to-dataset.
- Interactive bounding-box labeler with AUTO-LABEL bootstrap.
- Albumentations augmentation (9 presets) with bbox-aware transforms.
- Class-distribution / balance / coverage QA dashboard.
- YOLO11n or YOLO11s GPU training with live loss + mAP charts over
  WebSocket.
- Roboflow Supervision annotation in 4 visual styles.
- One-click model export (`.pt` download).
- Auto-play demo mode using pre-baked training metrics.
- Pre-baked COCO vehicles dataset script + 10-domain Roboflow gallery
  downloader.
- Bundled Redis (job state) and optional Caddy HTTPS proxy.

---

## The modules

| Module | Purpose |
|---|---|
| `modules/augmentation.py` | Albumentations preset library and apply / preview pipeline; handles bbox transforms in YOLO format. |
| `modules/datasets.py` | YOLO `dataset.yaml` builder with 80/20 train/val split and contiguous class-ID remapping. |
| `modules/inference.py` | YOLO model cache + Supervision-based annotated detection (BoxCorner, Box, HeatMap, Circle styles). |
| `modules/labeling.py` | YOLO label-file I/O and AUTO-LABEL bootstrap from a pre-trained model. |
| `modules/training.py` | Ultralytics `YOLO.train()` wrapper with per-epoch progress callback (loss + mAP). |

Plus two helper scripts:

| Script | Purpose |
|---|---|
| `scripts/prepare_coco_vehicles.py` | Downloads COCO val2017, filters to car/motorcycle/bus/truck, converts to YOLO format, drops the result into `data/samples/coco-vehicles/`. |
| `scripts/download_gallery.py` | Downloads ten CC BY 4.0 Roboflow Universe datasets (PPE, military, X-ray, thermal, traffic signs, marine, corrosion, livestock, pests, wildlife) into the gallery. Needs `ROBOFLOW_API_KEY`. |

---

## Reference build platform

This demo was built and tested on a **Dell Pro Max GB10** (NVIDIA Grace
Blackwell, **ARM / aarch64** architecture). It will run on standard
x86_64 NVIDIA Linux hosts as well, but the bundled Dockerfile pins
PyTorch 2.9.1 + CUDA 13.0 wheels because that's the only stable
combination on aarch64 with the GB10's `sm_121` compute capability. On
older GPUs you may need to change `CUDA_ARCH` (see Configuration below).

---

## Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| OS | Linux | macOS / Windows lack pass-through GPU support — won't work. |
| Docker | 24.x or newer | With Compose **v2** (`docker compose`, not `docker-compose`). |
| GPU | NVIDIA, ≥ 6 GB VRAM | YOLO11n with batch 16 at 640 px fits comfortably; YOLO11s wants more. |
| GPU driver | Recent enough for your CUDA version | `nvidia-smi` must work on the host. |
| NVIDIA Container Toolkit | Installed and configured for Docker | Required to expose the GPU to the container. |
| Disk | ~5 GB | Image (~3 GB) + Ultralytics weights cache + dataset volume + Redis volume. |
| RAM | 8 GB minimum | 16 GB recommended; Docker memory limit is set to 8 G in the compose file. |
| API key | None | Everything runs locally. Optional `ROBOFLOW_API_KEY` only for the gallery downloader. |

---

## Installation (step-by-step)

These instructions assume a fresh Linux box. If you already have Docker
+ the NVIDIA Container Toolkit working, skip to step 4.

### 1. Verify your GPU is visible to the host

```bash
nvidia-smi
```

You should see a table with your GPU model, driver version, and CUDA
version. If this command fails, **fix your NVIDIA driver before going
further** — the rest will not work.

### 2. Install Docker Engine + Compose v2

Ubuntu / Debian:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"   # let your user run docker without sudo
newgrp docker                      # apply the new group in this shell
docker compose version             # should print "Docker Compose version v2.x.x"
```

If `docker compose version` reports "command not found", install the
plugin:

```bash
sudo apt install docker-compose-plugin
```

### 3. Install the NVIDIA Container Toolkit

Ubuntu / Debian:

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify it works inside Docker:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

You should see the same `nvidia-smi` table you saw on the host. If
this fails, fix it before continuing.

### 4. Clone the repo

```bash
git clone https://github.com/gleaven/hermes.git
cd hermes
```

### 5. Create the environment file

```bash
cp .env.example .env
```

The defaults are sensible. Edit `.env` only if you need to change ports
or raise the upload limit. Common edits:

| Variable | Default | When to change |
|---|---|---|
| `APP_PORT` | `8080` | Port `8080` is already taken on your host. |
| `MAX_UPLOAD_MB` | `200` | You want to upload larger ZIP archives. |
| `REDIS_HOST_PORT` | `6379` | Port `6379` is already taken on your host. |

### 6. Build and start

```bash
docker compose up -d --build
```

The first build takes **5–15 minutes** (downloads CUDA base image
~3 GB, installs PyTorch + Ultralytics + Albumentations + Supervision).
Subsequent starts take ~10 seconds.

### 7. Verify it's healthy

```bash
docker compose ps
# both `demo-hermes` and `demo-redis` should show "healthy" within ~2 min

curl -s http://localhost:8080/health
# {"status":"ok","service":"hermes"}

curl -s http://localhost:8080/api/gpu | python3 -m json.tool
# Should report available=true, your GPU name, and total/allocated MB.
```

### 8. Open the UI

<http://localhost:8080/>

You'll land on the **DATA SOURCE** stage. Click GEO CAPTURE to extract
satellite tiles, or TRAINING DATASETS to pick from the gallery. The
top-right play button starts auto-play demo mode.

### 9. (Optional) Pre-load the gallery

The shipped image has an empty `data/samples/` directory. To populate
it:

```bash
# Bundled COCO vehicles (no key needed):
docker exec -it demo-hermes python3 scripts/prepare_coco_vehicles.py

# Ten-domain Roboflow gallery (free key from https://app.roboflow.com/settings/api):
docker exec -e ROBOFLOW_API_KEY=YOUR_KEY -it demo-hermes \
    python3 scripts/download_gallery.py
```

Datasets appear in the Stage-1 gallery on the next page reload.

### 10. (Optional) Tail the logs

```bash
docker compose logs -f hermes
```

---

## Configuration

All variables can be set in `.env` or exported in your shell.

| Variable | Default | What it controls |
|---|---|---|
| `APP_PORT` | `8080` | Browser-facing port for the UI and API. |
| `MAX_UPLOAD_MB` | `200` | Per-upload size limit for dataset ZIPs (and individual image uploads). |
| `REDIS_HOST_PORT` | `6379` | Where the bundled Redis is exposed on the host. |
| `REDIS_URL` | `redis://demo-redis:6379/9` | Connection string the trainer uses for job state; override when you BYO Redis. |
| `DATA_DIR` | `/app/data` | In-container data root (uploads, jobs, datasets, samples, pretrained). Rarely changed. |
| `DEMO_HOSTNAME` | `localhost` | Hostname Caddy serves under (proxy profile only). |
| `HTTP_PORT` | `8081` | Caddy HTTP port. |
| `HTTPS_PORT` | `8443` | Caddy HTTPS port. |

### Build-time arguments

If you're not on a GB10 / `sm_121` GPU, override the CUDA arch when
building:

```bash
docker compose build --build-arg CUDA_ARCH=8.6   # e.g. RTX 3090
docker compose up -d
```

Common values: `8.0` (A100), `8.6` (RTX 30xx), `8.9` (RTX 40xx),
`9.0` (H100), `12.0`/`12.1` (Grace Blackwell / GB10).

> Note: the compose file also pins `TORCH_CUDA_ARCH_LIST=12.1` as a
> runtime env var. This only affects JIT-compiled CUDA kernels at
> runtime; it does not override the wheels baked into the image.

---

## Live controls (in the browser)

Stage-by-stage:

- **Data Source / GEO CAPTURE:** zoom (Z15–Z19), output tile size
  (256 / 512 / 640 / 1024), overlap (0–50 %), draw multiple regions,
  live capture estimate (raw tiles, output tiles, area km²), append
  more regions to an existing satellite dataset.
- **Data Collection:** dataset gallery cards (rename / delete user
  datasets; sample datasets are read-only), drag-and-drop ZIP upload.
- **Annotation:** class palette (add / rename / color-pick), draw on
  canvas, AUTO-LABEL with bundled `yolo11n.pt`, save labels.
- **Augmentation:** select any subset of the 9 Albumentations presets
  (Rain, Fog, Motion Blur, H-Flip, Rotate 90, Brightness, Color
  Jitter, Noise, Cutout), preview on a sample image, pick a 2×–5×
  multiplier, apply.
- **QA:** class distribution chart, balance score, coverage score,
  shuffleable sample grid with bbox overlays.
- **Training:** model (YOLO11n / YOLO11s), epochs slider (10–200),
  image size (320 / 640), batch size (8 / 16), live loss + mAP charts,
  ETA.
- **Testing:** upload any external image, run detection, switch
  annotator style (CORNER / BOX / HEATMAP / CIRCLE), adjust confidence
  (0.0–1.0), download the trained `.pt` model.

The top-right **demo mode button** auto-plays the entire pipeline,
substituting in the pre-baked training metrics from
`data/pretrained/training-metrics.json` so the training stage finishes
in seconds.

---

## External services (BYO)

If you'd rather use your own Redis (e.g. a managed instance), uncomment
`REDIS_URL` in `.env` and start with the BYO override:

```bash
docker compose -f docker-compose.yml -f docker-compose.byo.yml up -d
```

`docker-compose.byo.yml` scales the bundled `redis` service to zero
and drops the `depends_on` so only the `hermes` container runs locally.

| Variable | Example |
|---|---|
| `REDIS_URL` | `redis://redis.example.com:6379/9` |

Redis is used **only** for live training-job state (per-epoch metrics
broadcast to the WebSocket). Completed runs are also persisted to
`data/jobs/<job_id>/training-metrics.json`, so losing Redis does not
lose your trained models.

---

## Optional HTTPS reverse proxy

Caddy is bundled as an opt-in profile. It auto-provisions Let's
Encrypt certs when `DEMO_HOSTNAME` is a real DNS name pointing at this
host:

```bash
DEMO_HOSTNAME=hermes.example.com docker compose --profile proxy up -d
```

For local testing keep `DEMO_HOSTNAME=localhost` and Caddy will issue
a self-signed cert.

---

## Authentication

HERMES runs **without authentication** by default. For shared
deployments, put one of these in front of it:

- **Caddy basic auth** — add a `basic_auth` block to the Caddyfile.
- **oauth2-proxy in front of Caddy** — for SSO-style auth.
- **Cloudflare Tunnel + Access policies** — easiest if you're already
  on Cloudflare.

---

## Architecture (file map)

| File | Purpose |
|---|---|
| `server.py` | FastAPI app: REST + WebSocket endpoints, satellite tile fetch / stitch / re-slice, dataset CRUD, label CRUD, augmentation, training orchestration, inference, model export. |
| `modules/datasets.py` | YOLO `dataset.yaml` builder with 80/20 split + contiguous class-ID remap. |
| `modules/labeling.py` | YOLO label I/O + AUTO-LABEL pipeline. |
| `modules/augmentation.py` | Albumentations preset registry + bbox-aware preview/apply. |
| `modules/training.py` | Ultralytics `YOLO.train()` wrapper with per-epoch progress callback. |
| `modules/inference.py` | YOLO model cache + Supervision-annotated detection (4 styles). |
| `scripts/prepare_coco_vehicles.py` | Bundled COCO vehicles dataset preparer. |
| `scripts/download_gallery.py` | Bundled 10-domain Roboflow Universe downloader. |
| `static/index.html` | Single-page app with all 7 stage panels. |
| `static/js/satellite.js` | Leaflet map, region drawing, tile-extract progress UI. |
| `static/js/labeling.js` | Canvas-based bounding-box labeler. |
| `static/js/augmentation.js` | Augmentation preset toggles + preview. |
| `static/js/quality.js` | QA stats dashboard. |
| `static/js/training.js` | Training-config form, WebSocket client, live charts. |
| `static/js/assessment.js` | Test-image upload + detection result rendering. |
| `static/js/demo-mode.js` | Auto-play through all stages using pre-baked metrics. |
| `data/samples/` | Pre-loaded sample datasets (empty by default; populate with the scripts). |
| `data/pretrained/` | Optional `yolo11n-vehicles.pt` + `training-metrics.json` for demo mode. |
| `Caddyfile` | Optional reverse-proxy config. |
| `Dockerfile` | CUDA 13.0 base, PyTorch 2.9.1, Python deps. |

---

## Troubleshooting

- **`nvidia-smi` works on host but not in container** — the NVIDIA
  Container Toolkit isn't wired into Docker. Run `sudo nvidia-ctk
  runtime configure --runtime=docker && sudo systemctl restart docker`
  and try the test container in step 3 again.
- **Training stalls or runs on CPU** — confirm `/api/gpu` reports
  `available: true`. If not, the container fell back to CPU; the
  nvidia runtime isn't the default. Re-run step 3.
- **`Out of memory` during training** — drop batch size from 16 to 8
  in the Stage-5 config card, or pick `YOLO11n` instead of `YOLO11s`,
  or shrink image size from 640 to 320.
- **`unsupported gpu architecture` during PyTorch import** — your GPU's
  compute capability isn't in the wheel's arch list. Rebuild with
  `--build-arg CUDA_ARCH=<your arch>` (see Configuration).
- **Upload rejected with 413** — the ZIP is bigger than `MAX_UPLOAD_MB`.
  Raise it in `.env` and `docker compose up -d` to apply.
- **Tile extraction errors with "exceeds max 500 raw tiles"** — your
  drawn region is too large for the chosen zoom. Either zoom out, draw
  a smaller region, or split it into multiple regions (each capped
  separately, but the total still has to fit in 500).
- **AUTO-LABEL produces no boxes** — the bundled fallback is generic
  `yolo11n.pt`, which is COCO-class. If your domain isn't COCO (e.g.
  thermal imagery), drop a domain-specific `.pt` into
  `data/pretrained/yolo11n-vehicles.pt` to override.
- **Empty dataset gallery** — `data/samples/` ships empty. Run
  `scripts/prepare_coco_vehicles.py` and/or `scripts/download_gallery.py`
  inside the container to populate it (see step 9).
- **Port already in use** — change `APP_PORT` (or `REDIS_HOST_PORT`)
  in `.env`.
- **`demo-hermes` health check failing** — give it longer; PyTorch
  takes a moment to import on first start. Check
  `docker compose logs hermes` for stack traces.
- **Trained models disappear after `docker compose down -v`** — `-v`
  removes the named `hermes-data` volume, which holds `data/jobs/`.
  Drop the `-v` to keep them.

---

## FAQ

**Q: Can I use a CPU?** Technically YOLO11n will train on CPU, but the
container is GPU-targeted (`nvidia` runtime + `device=0` in the
trainer). It will fail to find a CUDA device and crash the job. There
is no CPU fallback path.

**Q: Where do trained models live?** `data/jobs/<job_id>/train/weights/best.pt`
inside the `hermes-data` volume. The `/api/model/export` endpoint
serves the most recent one.

**Q: Can I use my own pre-trained model for AUTO-LABEL?** Yes — drop
it at `/app/data/pretrained/yolo11n-vehicles.pt` (path is hard-coded).
The labeler will use it instead of the generic COCO `yolo11n.pt`.

**Q: How do I add more dataset gallery sources?** Drop a directory at
`data/samples/<id>/` containing `images/`, `labels/`, and a
`metadata.json` with `name`, `classes`, and `image_count`. It will
appear in the gallery on next page load.

**Q: Why Esri tiles specifically?** They're high-resolution, free for
moderate use, and don't require an API key. The 500-tile cap exists
to stay polite.

---

## Credits

Built by Andrew Meinecke.
