# syntax=docker/dockerfile:1

# ---- Stage 1: build the React SPA ----
FROM node:20-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: backend runtime, serving the built SPA ----
FROM python:3.12-slim AS backend

# opencv-python (pulled in by ultralytics) needs these at runtime even
# though there's no GUI use in this codebase — it still links against
# libGL/libglib on import. ultralytics pins the exact package name
# "opencv-python", so swapping to opencv-python-headless would install
# both side by side (file conflicts in site-packages/cv2) rather than
# replacing it — simpler to just satisfy the real dependency here.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY configs/ configs/
COPY training/ training/
COPY scripts/ scripts/
COPY models/ models/
COPY --from=frontend-build /frontend/dist frontend/dist

# Non-root runtime user. /app/instance is where the default sqlite
# DATABASE_URL points when run via docker-compose (see docker-compose.yml)
# — created and owned by appuser so the app can write the db file there.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/instance \
    && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
