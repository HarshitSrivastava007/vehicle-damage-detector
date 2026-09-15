# Vehicle Damage Detector — Core API

FastAPI service that detects vehicle damage (dent, scratch, crack, broken,
tire flat) from an uploaded photo using a YOLOv8 instance segmentation
model, returning class, confidence, bounding box, and mask polygon per
detection.

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

`DATASET_SLUG` points at `issamjebnouni/cardd` — the official CarDD
dataset, verified to ship real COCO polygon segmentation masks (not
bounding boxes, not RLE) with a pre-made train/val/test split (2816/810/374
images). It has 6 categories (dent, scratch, crack, glass_shatter,
lamp_broken, tire_flat); this project merges `glass_shatter` +
`lamp_broken` into a single `broken` class, giving 5 final classes: dent,
scratch, crack, broken, tire_flat (see `training/convert_to_yolo.py`'s
`CATEGORY_ALIASES`).

Note: an earlier candidate dataset (`gabrielfcarvalho/cardd-with-yolo-...`)
turned out to only have bounding-box labels despite the name — it's not
usable for segmentation training. `issamjebnouni/cardd` was verified by
downloading its `train.json`/`val.json`/`test.json` and confirming real
multi-point polygons in the `segmentation` field before switching to it.

## 3. Download the dataset

```bash
python scripts/download_dataset.py
```

This downloads the raw dataset into `data/raw/`, prints a summary of its
contents, and converts it into the YOLO-seg layout expected by training
(`data/processed/images/{train,val,test}`, `data/processed/labels/{train,val,test}`),
using the dataset's own `train.json`/`val.json`/`test.json` splits rather
than inventing a new random split.

If you swap in a different dataset later, the converter (`training/convert_to_yolo.py`)
auto-detects whether the raw data is already YOLO-shaped or in COCO JSON
format, and falls back to a random 85/15 split if no named train/val/test
json files are present. **Inspect `data/raw/`** if the format doesn't match
either case — the script will fail with a clear message; extend
`convert_to_yolo.py`'s `detect_raw_format()`/`convert_coco_to_yolo_seg()` to
match what you find (e.g. different JSON key names, or RLE-encoded masks
requiring `pycocotools`).

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
