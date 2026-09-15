# Vehicle Damage Detector — Core API

FastAPI service that detects vehicle damage (dent, scratch, crack, glass shatter,
lamp broken, tire flat) from an uploaded photo using a YOLOv8 instance
segmentation model, returning class, confidence, bounding box, and mask
polygon per detection.

**Scope of this pass:** detection API + training pipeline only. No auth,
billing, database, multi-tenancy, or frontend UI yet (see "Out of scope"
below).

## Prerequisites

- Python 3.11+
- Optional: a CUDA-capable GPU for reasonable training times (CPU works but is slow)
- A Kaggle account (for downloading the dataset)

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Configure environment

```bash
cp .env.example .env
```

Set your Kaggle credentials in `.env` (or place a `kaggle.json` under
`~/.kaggle/` instead — `kagglehub` will pick either up):

```
KAGGLE_USERNAME=...
KAGGLE_KEY=...
```

**Important:** `DATASET_SLUG` in `.env.example` is a placeholder pointing at
a CarDD-derived Kaggle dataset (6 classes: dent, scratch, crack,
glass_shatter, lamp_broken, tire_flat). It was **not verified** against the
original reference notebook (`engamohammed/car-damage-instance-segmentation`
on Kaggle), which could not be fetched during setup. Before downloading,
open that notebook's "Input" panel on Kaggle and confirm the actual dataset
slug it uses — update `DATASET_SLUG` in `.env` if it differs.

## 3. Download the dataset

```bash
python scripts/download_dataset.py
```

This downloads the raw dataset into `data/raw/`, prints a summary of its
contents, and attempts to convert it into the YOLO-seg layout expected by
training (`data/processed/images/{train,val}`, `data/processed/labels/{train,val}`).

The converter (`training/convert_to_yolo.py`) auto-detects whether the raw
data is already YOLO-shaped or in COCO JSON format. **Inspect `data/raw/`
after downloading** — if the format doesn't match either case, the script
will fail with a clear message; extend `convert_to_yolo.py`'s
`detect_raw_format()`/`convert_coco_to_yolo_seg()` to match what you find
(e.g. different JSON key names, or RLE-encoded masks requiring
`pycocotools`).

To skip auto-conversion and inspect first:

```bash
python scripts/download_dataset.py --skip-convert
```

## 4. Train the model

```bash
python training/train.py --epochs 100 --imgsz 640 --batch 16
```

Fine-tunes `yolov8s-seg.pt` (COCO-pretrained) on `configs/dataset.yaml`.
Defaults to `DEVICE=cpu` from `.env` — pass `--device cuda:0` if you have a
GPU, since CPU training for 100 epochs on a real dataset will be slow.

On completion, the best checkpoint is copied to `models/best.pt`, which is
the path the API loads from.

## 5. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

If `models/best.pt` doesn't exist yet, the API falls back to the base
COCO-pretrained `yolov8s-seg.pt` (won't recognize damage classes, but lets
you verify the plumbing end-to-end before training finishes).

### Endpoints

- `GET /health` — reports service status and which weights are loaded.
- `POST /api/v1/detect` — multipart file upload, returns detections as JSON.
  Add `?annotate=true` to instead get back an annotated PNG.

Example:

```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -F "file=@/path/to/car.jpg"
```

## 6. Run tests

```bash
pytest
```

Tests mock the model layer, so they run without network access, a GPU, or
trained weights. They validate the API contract (response shape, error
handling for bad content-type/corrupt images), not detection accuracy.

## Out of scope for this pass

- Authentication / API keys
- Per-user usage limits / billing (Stripe, etc.)
- Database / persistence layer
- Web frontend (upload UI, dashboard)
- Docker / deployment configuration
- Multi-tenancy
