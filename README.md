# Vehicle Damage Detector — Core API

FastAPI service that detects vehicle damage — **dent, scratch, crack,
broken, tire_flat** — from an uploaded photo using a fine-tuned YOLOv8
instance segmentation model. For each detection it returns the class,
confidence score, bounding box, and a mask polygon that traces the actual
damaged region (not just a box around the whole car).

**Scope of this pass:** detection API + training pipeline + a simple
browser test UI. No auth, billing, database, multi-tenancy, or production
frontend yet (see [Out of scope](#out-of-scope) below).

A fine-tuned model already exists at `models/best.pt` (trained on the
official CarDD dataset, 100 epochs) — see [Quick Start](#quick-start) to
just run it.

## Prerequisites

- Python 3.11+
- Optional but recommended: a CUDA-capable GPU (training on CPU is very slow)
- A Kaggle account — only needed if you want to re-download the dataset or retrain

## Quick Start

If `models/best.pt` already exists (check with `ls models/`), you can skip
straight to running the API:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Then open **http://localhost:8000** in a browser for the upload UI, or see
[Usage](#usage) below for the REST API.

If `models/best.pt` doesn't exist, the API still starts and runs — it falls
back to the base COCO-pretrained `yolov8s-seg.pt`, which won't recognize
damage classes. Follow [Full Setup From Scratch](#full-setup-from-scratch)
to train it yourself.

## Full Setup From Scratch

Use this if you're setting up on a new machine, want to retrain, or
`models/best.pt` doesn't exist yet.

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Set your Kaggle credentials in `.env` (or place a `kaggle.json` under
`~/.kaggle/` instead — `kagglehub` picks up either):

```
KAGGLE_USERNAME=...
KAGGLE_KEY=...
```

`DATASET_SLUG` points at `issamjebnouni/cardd` — the official CarDD
dataset, verified to ship real COCO polygon segmentation masks (not
bounding boxes, not RLE) with a pre-made train/val/test split
(2816/810/374 images). It has 6 raw categories (dent, scratch, crack,
glass_shatter, lamp_broken, tire_flat); this project merges
`glass_shatter` + `lamp_broken` into a single `broken` class, giving the 5
final classes above (see `training/convert_to_yolo.py`'s
`CATEGORY_ALIASES`).

> An earlier candidate dataset (`gabrielfcarvalho/cardd-with-yolo-...`)
> turned out to only have bounding-box labels despite the name — not
> usable for segmentation training. Verify any dataset you swap in ships
> real polygon `segmentation` fields before relying on it.

Also set `DEVICE` in `.env`:
- `DEVICE=cpu` — works everywhere, slow (multiple hours for 100 epochs on ~4000 images)
- `DEVICE=cuda:0` — requires an NVIDIA GPU + CUDA-enabled torch; ~90 minutes for 100 epochs

### 3. Download the dataset

```bash
python scripts/download_dataset.py
```

Downloads the raw dataset into `data/raw/` (~2.8GB), prints a summary of
its contents, and converts it into the YOLO-seg layout training expects
(`data/processed/images/{train,val,test}`,
`data/processed/labels/{train,val,test}`), using the dataset's own
train/val/test split rather than inventing a new one.

If you swap in a different dataset later, the converter
(`training/convert_to_yolo.py`) auto-detects whether the raw data is
already YOLO-shaped or in COCO JSON format, and falls back to a random
85/15 split if no named train/val/test json files are present. **Inspect
`data/raw/`** if the format doesn't match either case — the script fails
with a clear message; extend `convert_to_yolo.py`'s `detect_raw_format()` /
`convert_coco_to_yolo_seg()` to match what you find (e.g. different JSON
key names, or RLE-encoded masks requiring `pycocotools`).

To skip auto-conversion and inspect the raw download first:

```bash
python scripts/download_dataset.py --skip-convert
```

### 4. Train the model

```bash
python training/train.py --epochs 100 --imgsz 640 --batch 16 --device cuda:0
```

Fine-tunes `yolov8s-seg.pt` (COCO-pretrained) on `configs/dataset.yaml`.
Run from the repo root (the dataset yaml's relative `path:` is resolved
against the current working directory).

Key flags (all optional, see `python training/train.py --help`):

| Flag | Default | Notes |
|---|---|---|
| `--epochs` | 100 | |
| `--imgsz` | 640 | |
| `--batch` | 16 | use `-1` for Ultralytics auto-batch |
| `--device` | from `.env` `DEVICE` | `cpu` or `cuda:0` |
| `--weights` | `yolov8s-seg.pt` | starting checkpoint to fine-tune |
| `--patience` | 50 | early-stop after this many epochs with no improvement |

On completion, the best checkpoint is copied to `models/best.pt`, which is
the path the API loads from. Training progress/metrics are also written to
`runs/segment/<name>/results.csv`.

**Results from the included `models/best.pt`** (100 epochs, validated on
810 held-out images):

| Class | Mask mAP50 |
|---|---|
| tire_flat | 0.96 |
| broken | 0.94 |
| dent | 0.59 |
| scratch | 0.55 |
| crack | 0.42 |
| **Overall** | **0.69** |

`crack` is the weakest class — subtle/underrepresented in the training
data. More epochs, more data, or class-balanced sampling would help if you
retrain.

## Running the API

```bash
uvicorn app.main:app --reload --port 8000
```

`GET /health` reports whether `models/best.pt` was found and loaded (vs.
the fallback pretrained weights).

## Usage

### Browser UI

Open **http://localhost:8000** — upload or drag-and-drop a photo, click
"Run Detection", and it shows the annotated image next to a table of
detected classes, confidence, and bounding boxes.

### REST API

**`GET /health`**

```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true,"weights_path":"models/best.pt"}
```

**`POST /api/v1/detect`** — multipart file upload (`file` field), JPEG/PNG/WEBP.

JSON response (default):

```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -F "file=@/path/to/car.jpg"
```

```json
{
  "filename": "car.jpg",
  "image_width": 640,
  "image_height": 480,
  "count": 1,
  "detections": [
    {
      "class_name": "scratch",
      "confidence": 0.92,
      "bbox": {"x1": 268.2, "y1": 246.1, "x2": 400.2, "y2": 383.0},
      "polygon": [[351.0, 246.0], [350.0, 250.0], "..."]
    }
  ]
}
```

Annotated image instead of JSON — add `?annotate=true`:

```bash
curl -X POST "http://localhost:8000/api/v1/detect?annotate=true" \
  -F "file=@/path/to/car.jpg" \
  --output annotated.png
```

The annotated image draws every detection's mask, box, and class label
(no confidence score shown) in a single color, configurable via
`mask_color_rgb` in `app/config.py` (default `#84ff00` / RGB
`132, 255, 0`).

Error responses: `415` unsupported file type, `413` file too large
(`MAX_UPLOAD_SIZE_MB` in `.env`), `422` corrupt/unreadable image.

## Configuration reference

All settings live in `.env` (see `.env.example`), loaded via
`app/config.py`:

| Variable | Default | Purpose |
|---|---|---|
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | — | Kaggle API auth for dataset download |
| `DATASET_SLUG` | `issamjebnouni/cardd` | Kaggle dataset to download |
| `MODEL_WEIGHTS_PATH` | `models/best.pt` | fine-tuned weights the API loads |
| `FALLBACK_WEIGHTS_PATH` | `yolov8s-seg.pt` | used if `MODEL_WEIGHTS_PATH` doesn't exist |
| `CONFIDENCE_THRESHOLD` | `0.25` | minimum detection confidence |
| `IOU_THRESHOLD` | `0.45` | NMS IoU threshold |
| `DEVICE` | `cpu` | inference/training device |
| `MAX_UPLOAD_SIZE_MB` | `10` | upload size limit |
| `ALLOWED_CONTENT_TYPES` | `image/jpeg,image/png,image/webp` | comma-separated allowed MIME types |

## Development

### Run tests

```bash
pytest
```

Tests mock the model layer, so they run without network access, a GPU, or
trained weights. They validate the API contract (response shape, error
handling for bad content-type/corrupt images), not detection accuracy.

### Project structure

```
app/                  FastAPI service (config, model loading, inference shaping, routes)
app/static/index.html Browser test UI
configs/dataset.yaml  Ultralytics data config (class list, train/val/test paths)
scripts/              Dataset download
training/             COCO→YOLO conversion, training script
models/               Trained weights (best.pt) — gitignored
data/                 Raw + processed dataset — gitignored
tests/                pytest suite with a mocked model
```

## Out of scope

- Authentication / API keys
- Per-user usage limits / billing (Stripe, etc.)
- Database / persistence layer
- Production web frontend / dashboard (the current UI is a test tool only)
- Docker / deployment configuration
- Multi-tenancy
