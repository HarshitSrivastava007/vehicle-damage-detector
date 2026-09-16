# Vehicle Damage Detector — Core API

FastAPI service that detects vehicle damage — **dent, scratch, crack,
broken, tire_flat** — from an uploaded photo using a fine-tuned YOLOv8
instance segmentation model. For each detection it returns the class,
confidence score, bounding box, and a mask polygon that traces the actual
damaged region (not just a box around the whole car).

**Scope of this pass:** detection API + training pipeline + a browser test
UI + API-key auth and a database (users, keys, detection history). No
billing, usage limits, multi-tenancy, or production frontend yet (see [Out
of scope](#out-of-scope) below).

A fine-tuned model already exists at `models/best.pt` (trained on the
official CarDD dataset, 100 epochs) — see [Quick Start](#quick-start) to
just run it.

**Production note:** since a trained model is already included, this repo
has no Kaggle account details, credentials, or dataset-download code in
it — that was only ever needed once, to produce `models/best.pt`, and has
been removed to keep the production app free of unnecessary credentials/
dependencies. See [Retraining / adding more data](#retraining--adding-more-data)
if you need to pull fresh data again later.

## Prerequisites

- Python 3.11+
- Optional but recommended: a CUDA-capable GPU (training on CPU is very slow, only relevant if retraining)

## Quick Start

If `models/best.pt` already exists (check with `ls models/`), you can skip
straight to running the API:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Then open **http://localhost:8000** in a browser for the upload UI (click
"Get a key" to register and auto-fill an API key), or see
[Authentication](#authentication) + [Usage](#usage) below for the REST API.

If `models/best.pt` doesn't exist, the API still starts and runs — it falls
back to the base COCO-pretrained `yolov8s-seg.pt`, which won't recognize
damage classes. Follow [Full Setup From Scratch](#full-setup-from-scratch)
to train it yourself.

## Full Setup From Scratch

Use this if you're setting up on a new machine and `models/best.pt`
doesn't exist yet (if you need to retrain with fresh/updated data, see
[Retraining / adding more data](#retraining--adding-more-data) first to get
`data/processed/` populated, then come back here).

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

Set `DEVICE` in `.env` if you plan to train:
- `DEVICE=cpu` — works everywhere, slow (multiple hours for 100 epochs on ~4000 images)
- `DEVICE=cuda:0` — requires an NVIDIA GPU + CUDA-enabled torch; ~90 minutes for 100 epochs

### 3. Train the model

Requires `data/processed/images/{train,val,test}` and
`data/processed/labels/{train,val,test}` to already be populated in
YOLO-seg layout — see [Retraining / adding more
data](#retraining--adding-more-data) if that's not there yet.

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

## Retraining / adding more data

The Kaggle download script (`scripts/download_dataset.py`) and its
supporting config (`kaggle_username`, `kaggle_key`, `dataset_slug`,
`dataset_raw_dir`, `dataset_processed_dir` in `app/config.py`) were
intentionally removed from this repo for production — they were only ever
a one-time step to produce `models/best.pt`, and there's no reason for a
deployed app to hold Kaggle credentials or a dataset-download dependency.
`training/convert_to_yolo.py` (COCO → YOLO-seg conversion) and
`training/train.py` were kept, since they don't depend on Kaggle at all —
they just need `data/raw/` (or already-converted `data/processed/`)
populated by some means.

To pull the original dataset again and retrain from scratch:

1. Recover the original download script from git history (it was last
   present, unchanged, in commit `e35f4f6`):
   ```bash
   mkdir -p scripts
   git show e35f4f6:scripts/download_dataset.py > scripts/download_dataset.py
   ```
2. Reinstall its dependency: `pip install kagglehub` (add `kagglehub` back
   to `requirements.txt` too if keeping this long-term).
3. Add these fields back to the `Settings` class in `app/config.py`
   (they read from `.env`, matching the pattern the other fields use):
   ```python
   kaggle_username: str = ""
   kaggle_key: str = ""
   dataset_slug: str = "issamjebnouni/cardd"
   dataset_raw_dir: Path = Path("data/raw")
   dataset_processed_dir: Path = Path("data/processed")
   ```
4. Add matching entries back to `.env` (get a Kaggle API key from Kaggle →
   Account → "Create New API Token"):
   ```
   KAGGLE_USERNAME=...
   KAGGLE_KEY=...
   DATASET_SLUG=issamjebnouni/cardd
   DATASET_RAW_DIR=data/raw
   DATASET_PROCESSED_DIR=data/processed
   ```
5. Run it: `python scripts/download_dataset.py` — see [Full Setup From
   Scratch](#full-setup-from-scratch) step 3 for what happens next
   (training).

If you're adding a *different* dataset instead of re-pulling CarDD, you
don't need Kaggle at all — just get your images/annotations into
`data/raw/` by any means, then run
`training/convert_to_yolo.py`'s `convert()` (or write a one-off script)
to normalize it into `data/processed/`, matching the class list in
`configs/dataset.yaml`.

## Running the API

```bash
uvicorn app.main:app --reload --port 8000
```

`GET /health` reports whether `models/best.pt` was found and loaded (vs.
the fallback pretrained weights).

## Authentication

`POST /api/v1/detect` and `GET /api/v1/detections` require an API key.
`GET /`, `GET /health`, and the `/api/v1/auth/*` endpoints do not.

Get a key by registering (no login/password — just an email, for now):

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com"}'
# {"user_id":1,"email":"you@example.com","api_key":"vdd_...«shown once»..."}
```

Save the `api_key` — it's only ever shown at creation time (only its hash
is stored). Pass it on every authenticated request:

```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -H "Authorization: Bearer vdd_..." \
  -F "file=@/path/to/car.jpg"
```

Account/key management:

| Method & Path | Auth | Purpose |
|---|---|---|
| `POST /api/v1/auth/register` | none | Create a user + first API key |
| `GET /api/v1/auth/me` | required | Current user's id/email |
| `POST /api/v1/auth/keys` | required | Issue an additional key |
| `GET /api/v1/auth/keys` | required | List your keys (never shows the raw key, only a display prefix) |
| `DELETE /api/v1/auth/keys/{id}` | required | Revoke a key (`404` if not yours, `409` if already revoked) |

There's no password reset, email verification, or admin panel yet — this
is intentionally a minimal foundation (see [Out of scope](#out-of-scope)).

## Usage

### Browser UI

Open **http://localhost:8000** — enter an API key or click "Get a key" to
register one on the spot, then upload or drag-and-drop a photo and click
"Run Detection". Shows the annotated image next to a table of detected
classes, confidence, and bounding boxes. The key is stored in the
browser's `localStorage` only.

### REST API

**`GET /health`**

```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true,"weights_path":"models/best.pt"}
```

**`POST /api/v1/detect`** — multipart file upload (`file` field), JPEG/PNG/WEBP. Requires `Authorization: Bearer <api_key>` (see [Authentication](#authentication)).

JSON response (default):

```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -H "Authorization: Bearer vdd_..." \
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
  -H "Authorization: Bearer vdd_..." \
  -F "file=@/path/to/car.jpg" \
  --output annotated.png
```

The annotated image draws every detection's mask, box, and class label
(no confidence score shown) in a single color, configurable via
`mask_color_rgb` in `app/config.py` (default `#84ff00` / RGB
`132, 255, 0`).

Error responses: `401` missing/invalid/revoked API key, `415` unsupported
file type, `413` file too large (`MAX_UPLOAD_SIZE_MB` in `.env`), `422`
corrupt/unreadable image.

Every successful call (JSON or annotated) also writes a detection-history
row, retrievable via:

**`GET /api/v1/detections?limit=20&offset=0`** — your own detection
history, newest first (`limit` clamped to 1–100).

```bash
curl "http://localhost:8000/api/v1/detections" -H "Authorization: Bearer vdd_..."
```

```json
[
  {"id": 1, "filename": "car.jpg", "detection_count": 1, "class_counts": {"scratch": 1}, "created_at": "2026-09-16T15:13:12"}
]
```

## Configuration reference

All settings live in `.env` (see `.env.example`), loaded via
`app/config.py`:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_WEIGHTS_PATH` | `models/best.pt` | fine-tuned weights the API loads |
| `FALLBACK_WEIGHTS_PATH` | `yolov8s-seg.pt` | used if `MODEL_WEIGHTS_PATH` doesn't exist |
| `CONFIDENCE_THRESHOLD` | `0.25` | minimum detection confidence |
| `IOU_THRESHOLD` | `0.45` | NMS IoU threshold |
| `DEVICE` | `cpu` | inference/training device |
| `MAX_UPLOAD_SIZE_MB` | `10` | upload size limit |
| `ALLOWED_CONTENT_TYPES` | `image/jpeg,image/png,image/webp` | comma-separated allowed MIME types |
| `DATABASE_URL` | `sqlite:///./app.db` | users/API keys/detection history storage; swap to `postgresql+psycopg2://...` (+ `pip install psycopg2-binary`) for Postgres, no code change needed |

## Development

### Run tests

```bash
pytest
```

Tests mock the model layer and use an isolated in-memory SQLite database
per test (via monkeypatching `app.db.engine`/`SessionLocal`), so they run
without network access, a GPU, trained weights, or touching the real
`app.db` file. They validate the API contract (response shape, auth
enforcement, error handling), not detection accuracy.

### Project structure

```
app/                    FastAPI service (config, model loading, inference shaping, routes)
app/db.py               SQLAlchemy engine/session setup
app/db_models.py        User, APIKey, DetectionLog ORM models
app/auth.py             API key generation/hashing + auth dependency
app/routers/auth.py     Register/list-keys/create-key/revoke-key endpoints
app/static/index.html   Browser test UI
configs/dataset.yaml    Ultralytics data config (class list, train/val/test paths)
training/               COCO→YOLO conversion, training script
models/                 Trained weights (best.pt) — gitignored
data/                   Raw + processed dataset — gitignored
tests/                  pytest suite with a mocked model + isolated in-memory DB
```

## Out of scope

- Per-user usage limits / billing (Stripe, etc.)
- Password reset, email verification, admin panel
- Production web frontend / dashboard (the current UI is a test tool only)
- Docker / deployment configuration
- Multi-tenancy beyond basic per-user data isolation (already in place: every user only sees their own keys/detection history)
