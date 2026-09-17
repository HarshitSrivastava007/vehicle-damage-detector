# Vehicle Damage Detector — Core API

FastAPI service that detects vehicle damage — **dent, scratch, crack,
broken, tire_flat** — from an uploaded photo using a fine-tuned YOLOv8
instance segmentation model. For each detection it returns the class,
confidence score, bounding box, and a mask polygon that traces the actual
damaged region (not just a box around the whole car).

**Scope of this pass:** detection API + training pipeline + API-key auth
+ a production web dashboard (React SPA) with session login, an admin
role, and per-user API-call cost tracking. No billing/payment
collection, multi-tenancy beyond per-user isolation, or Docker yet (see
[Out of scope](#out-of-scope)).

## Dashboard

`GET /` serves a React SPA (source in `frontend/`) once it's built (see
[Frontend dev workflow](#frontend-dev-workflow)) — log in, see your
usage/cost, manage API keys, try the detector, and (if you're an admin)
manage all users' cost-per-call. The old developer test page still
works at `GET /internal/legacy-test-ui` (paste an API key, upload a
photo, no build step needed) — it's excluded from the API docs
(`include_in_schema=False`) and is a manual-testing tool only, not the
production UI.

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

For the dashboard, build the SPA once (see [Frontend dev
workflow](#frontend-dev-workflow) for the day-to-day dev setup):

```bash
cd frontend && npm install && npm run build && cd ..
```

Then open **http://localhost:8000** and register/log in. (Or, without
building anything, open **http://localhost:8000/internal/legacy-test-ui**
for a quick no-build-step upload test with a pasted API key.)

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

## Frontend dev workflow

The dashboard (`frontend/`) is a Vite + React + TypeScript SPA. Two ways
to run it:

**Day-to-day development** (hot reload, two processes):
```bash
uvicorn app.main:app --reload --port 8000   # terminal 1
cd frontend && npm install && npm run dev   # terminal 2 — http://localhost:5173
```
Vite's dev server proxies `/api/*` to port 8000 (see
`frontend/vite.config.ts`), so the browser only ever talks to one origin
(`:5173`) — this matters because the session cookie is `SameSite=Lax`
and wouldn't be sent on genuinely cross-origin requests.

**Production-like** (single origin, no hot reload):
```bash
cd frontend && npm run build && cd ..
uvicorn app.main:app --port 8000            # no --reload
```
FastAPI serves the built SPA directly (`frontend/dist/`) at `/`, with a
catch-all fallback so client-side routes (`/keys`, `/detect`, `/admin`,
etc.) resolve correctly on a hard refresh. Rebuild (`npm run build`)
after any frontend change — `app/main.py` doesn't rebuild it for you.

## Authentication

Two independent mechanisms, both accepted by `POST /api/v1/detect`,
`GET /api/v1/detections`, and `/api/v1/auth/keys` (list/create/revoke):

- **API key** (`Authorization: Bearer vdd_...`) — for programmatic access
  (scripts, curl, the dashboard's `?annotate=true` calls aren't affected
  either way).
- **Session cookie** — set by logging in; this is what the dashboard uses.
  See [Session login & admin](#session-login--admin).

`GET /health` and `POST /api/v1/auth/register`/`login` don't require
either.

Get a key by registering. `password` is optional — omit it for an
API-key-only account, or set it if you also want to log into the
dashboard later:

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "optional"}'
# {"user_id":1,"email":"you@example.com","api_key":"vdd_...«shown once»..."}
```

Save the `api_key` — it's only ever shown at creation time (only its hash
is stored). Pass it on every request that doesn't already have a session
cookie:

```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -H "Authorization: Bearer vdd_..." \
  -F "file=@/path/to/car.jpg"
```

Account/key management:

| Method & Path | Auth | Purpose |
|---|---|---|
| `POST /api/v1/auth/register` | none | Create a user + first API key |
| `GET /api/v1/auth/me` | API key only | Current user's id/email |
| `POST /api/v1/auth/keys` | API key or session | Issue an additional key |
| `GET /api/v1/auth/keys` | API key or session | List your keys (never shows the raw key, only a display prefix) |
| `DELETE /api/v1/auth/keys/{id}` | API key or session | Revoke a key (`404` if not yours, `409` if already revoked) |

There's no password reset or email verification yet — intentionally a
minimal foundation (see [Out of scope](#out-of-scope)).

## Session login & admin

Separate from API keys: `password`-registered users can log in for a
browser session (httponly cookie, 7-day expiry), which is what the
dashboard uses. An **admin** role can view all users and set each user's
**cost per API call**, in INR (₹) — tracked, not charged (no payment
collection). `cost_per_call`/`cost` values are just plain decimal
numbers server-side (no currency field) — the dashboard formats them as
₹ (see `frontend/src/utils/currency.ts`); the API itself is
currency-agnostic.

```bash
curl -c cookies.txt -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "optional"}'
# sets a `vdd_session` cookie; response is your user info incl. is_admin/cost_per_call

curl -b cookies.txt http://localhost:8000/api/v1/auth/session   # "who am I"
curl -b cookies.txt -X POST http://localhost:8000/api/v1/auth/logout
```

**Every user starts as non-admin, and there's no API to promote anyone**
— that would be a chicken-and-egg problem (an admin-only endpoint can't
create the first admin). Bootstrap the first admin directly:

```bash
python scripts/promote_admin.py you@example.com
```

Admin-only endpoints (`403` for non-admins):

| Method & Path | Purpose |
|---|---|
| `GET /api/v1/admin/users` | List all users with `cost_per_call`, `is_active`, and `total_detections` |
| `POST /api/v1/admin/users` | Create a new user directly (email + password + optional `is_admin`/`cost_per_call`); returns an API key once, same as `/register` |
| `PATCH /api/v1/admin/users/{id}/cost` | Set a user's cost per API call (e.g. `{"cost_per_call": "0.05"}`) |
| `PATCH /api/v1/admin/users/{id}/status` | Activate/deactivate a user (`{"is_active": false}`) |

Deactivating a user takes effect **immediately** — every existing API key
and logged-in session for that user is checked against `is_active` on
every request, so it's not just a block on future logins. An admin can't
deactivate their own account (`400`) — no self-lockout.

Every user (session-authenticated) can see their own usage:

```bash
curl -b cookies.txt http://localhost:8000/api/v1/usage/summary
# {"calls_this_month":3,"cost_this_month":"0.1500","calls_all_time":10,"cost_all_time":"0.5000"}
```

Changing a user's `cost_per_call` only affects *future* calls — each
`DetectionLog` row (see [`GET /api/v1/detections`](#rest-api)) snapshots
the rate at the time of that call, so past costs never retroactively
change.

## Usage

### Dashboard

Open **http://localhost:8000** (after building the SPA — see [Frontend
dev workflow](#frontend-dev-workflow)) and register/log in. Pages:
Dashboard (usage summary + detection history), API Keys (create/revoke),
Try Detector (upload a photo, see the annotated result), and Admin (if
`is_admin` — list all users, create new users, set cost per call,
activate/deactivate).

### Browser UI (developer test page)

Open **http://localhost:8000/internal/legacy-test-ui** — no build step
needed. Enter an API key or click "Get a key" to register one on the
spot, then upload or drag-and-drop a photo and click "Run Detection".
Shows the annotated image next to a table of detected classes,
confidence, and bounding boxes. This is a manual-testing tool, not the
dashboard. The key is stored in the browser's `localStorage` only.

### REST API

**`GET /health`**

```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true,"weights_path":"models/best.pt"}
```

**`POST /api/v1/detect`** — multipart file upload (`file` field), JPEG/PNG/WEBP. Requires an API key or session cookie (see [Authentication](#authentication)).

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
  {"id": 1, "filename": "car.jpg", "detection_count": 1, "class_counts": {"scratch": 1}, "cost": "0.0500", "created_at": "2026-09-16T15:13:12"}
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
| `ENVIRONMENT` | `development` | `production` makes the session login cookie `Secure` (HTTPS only) |

## Development

### Database schema changes

There's no Alembic yet (see `app/db.py`'s `init_db()`) — `create_all()`
only creates tables that don't exist, it never alters an existing one.
If you have a local `app.db` from before a schema change (e.g. the
`is_active` column added for user activation), either delete it
(`rm app.db`, losing local data) or add the column manually:
```bash
sqlite3 app.db "ALTER TABLE users ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT 1"
```
This is a deliberate, revisit-later tradeoff — introduce Alembic once
this is deployed somewhere with real user data to preserve.

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
app/                     FastAPI service (config, model loading, inference shaping, routes)
app/db.py                SQLAlchemy engine/session setup
app/db_models.py         User, APIKey, UserSession, DetectionLog ORM models
app/auth.py              API key generation/hashing + auth dependency
app/session_auth.py      Password hashing + session-cookie auth/admin dependencies
app/current_user.py      Combined dependency: accepts API key OR session cookie
app/routers/auth.py      Register/login/logout/session/keys endpoints
app/routers/admin.py     Admin-only: list users, set cost per call
app/routers/usage.py     Session-authenticated usage/cost summary
app/static/index.html    Developer test page (served at /internal/legacy-test-ui)
scripts/promote_admin.py One-off script to bootstrap the first admin user
frontend/                React + TypeScript SPA (the dashboard) — see Frontend dev workflow
frontend/dist/           Build output FastAPI serves at / — gitignored, run `npm run build`
configs/dataset.yaml     Ultralytics data config (class list, train/val/test paths)
training/                COCO→YOLO conversion, training script
models/                  Trained weights (best.pt) — gitignored
data/                    Raw + processed dataset — gitignored
tests/                   pytest suite with a mocked model + isolated in-memory DB
```

## Out of scope

- Actually charging/collecting payment (Stripe, etc.) — cost per call is
  tracked and reportable, not billed
- Enforcing usage limits/quotas (nothing stops a user from exceeding
  whatever their cost implies)
- Password reset, email verification
- Admin role management via API (bootstrapping is a manual script — see
  [Session login & admin](#session-login--admin))
- Self-serve signup form in the dashboard (registration for an API key
  stays an API-only flow — see [Authentication](#authentication))
- Frontend automated tests (backend has full pytest coverage; the SPA is
  a thin display/mutation layer over already-tested endpoints)
- Docker / deployment configuration
- Multi-tenancy beyond basic per-user data isolation (already in place: every user only sees their own keys/detection history)
