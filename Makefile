PY := .venv/bin/python
export MODEL_DIR ?= $(CURDIR)/models
export DATA_DIR ?= $(CURDIR)/data
export HF_HOME := $(MODEL_DIR)
export FASTEMBED_CACHE_PATH := $(MODEL_DIR)
export PYTHONPATH := $(CURDIR)/src

.PHONY: venv install test test-slow download-model run demo docker-build docker-up clean

venv:
	uv venv .venv -p 3.12 || python3 -m venv .venv

install: venv
	uv pip install --python $(PY) -r requirements.lock && uv pip install --python $(PY) -e ".[dev]" \
	  || ($(PY) -m pip install -r requirements.lock && $(PY) -m pip install -e ".[dev]")

test:
	$(PY) -m pytest -q

test-slow: download-model
	HF_HUB_OFFLINE=1 $(PY) -m pytest -q -m slow

download-model:
	$(PY) scripts/download_model.py

run:
	$(PY) -m uvicorn rag.api:app_factory --factory --host 0.0.0.0 --port 8000

demo:  # requires `make run` in another shell
	./scripts/e2e_demo.sh

docker-build:
	docker compose build

docker-up:
	[ -f .env ] || cp .env.example .env
	docker compose up -d

clean:
	rm -rf data/*.sqlite3 .pytest_cache
