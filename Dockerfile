# CPU-only image. No torch, no CUDA: embeddings run on onnxruntime via fastembed.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/models DATA_DIR=/data \
    HF_HOME=/models HF_HUB_CACHE=/models FASTEMBED_CACHE_PATH=/models SENTENCE_TRANSFORMERS_HOME=/models

WORKDIR /app
COPY requirements.lock ./
RUN pip install -r requirements.lock
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

# Optional: bake the model into the image (docker build --build-arg BAKE_MODEL=1 .)
ARG BAKE_MODEL=0
COPY scripts/download_model.py ./scripts/
RUN if [ "$BAKE_MODEL" = "1" ]; then python scripts/download_model.py; fi

# Non-root runtime user. chown fails under rootless builders that only have a single UID mapping
# (e.g. podman without the setuid newuidmap helper: "chown: Invalid argument"), so fall back to
# world-writable volume dirs there. Run such images with `--user 0` (= the unprivileged host user).
RUN useradd --no-create-home -u 1000 app && mkdir -p /models /data \
    && (chown -R app /models /data 2>/dev/null || chmod -R a+rwX /models /data)
USER 1000
VOLUME ["/models", "/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["uvicorn", "rag.api:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
