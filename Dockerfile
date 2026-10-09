# syntax=docker/dockerfile:1

##############################################################################
# Legal GraphRAG — container image
#
# Entry point: uvicorn legal_graphrag.api.main:app  (package lives under src/)
# Serves the API + the self-contained web console (frontend/index.html) on :8000
#
# Multi-stage build:
#   1) builder  — compiles/installs all wheels into an isolated venv
#   2) runtime  — slim image with only the venv + OCR/PDF system libs
#
# Heavy ML deps (torch, sentence-transformers, transformers) are installed
# CPU-only to keep the image reasonable and avoid pulling CUDA.
##############################################################################

# ----------------------------------------------------------------------------
# Stage 1: builder
# ----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build toolchain for any packages without prebuilt wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Isolated virtualenv we copy into the runtime image.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

WORKDIR /build
COPY requirements.txt .

# Install torch CPU-only FIRST from the PyTorch CPU index so the (CUDA) PyPI
# wheel is never pulled. requirements.txt then sees torch already satisfied.
# If your torch pin isn't on the CPU index, drop the --index-url line and let
# requirements.txt install it normally.
RUN pip install --no-cache-dir "torch==2.13.0" \
    --index-url https://download.pytorch.org/whl/cpu \
    || echo "[build] CPU torch install skipped/failed — will resolve via requirements.txt"

# Install the rest of the pinned dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------------------------------
# Stage 2: runtime
# ----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# System libraries needed at RUN time:
#   tesseract-ocr   -> pytesseract (scanned-PDF OCR)
#   poppler-utils   -> pdf2image (PDF -> image rasterization)
#   libgomp1        -> OpenMP runtime for torch / scikit-learn
#   libglib2.0-0    -> transitive dep for some image libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Bring in the prebuilt venv.
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    CHROMA_PERSIST_DIR=/app/data/chroma_db

# Non-root user for safety.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

# Copy the application (respect .dockerignore). Adjust if your layout differs:
#   src/legal_graphrag/...   (the package; entry = legal_graphrag.api.main:app)
#   frontend/index.html      (served at / by the API)
#   scripts/, config.json, etc.
COPY . /app

# Writable dirs for data + model cache, owned by the non-root user.
RUN mkdir -p /app/data/chroma_db /app/data/metadata /app/data/uploads \
    /app/.cache/huggingface \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Container healthcheck: the API serves the console at / (200 OK).
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; \
    sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status==200 else 1)" \
    || exit 1

# Production server (no --reload). --app-dir src puts the package on the path.
CMD ["uvicorn", "legal_graphrag.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]