PY := .venv/bin/python
export MODEL_DIR ?= $(CURDIR)/models
export DATA_DIR ?= $(CURDIR)/data
export HF_HOME := $(MODEL_DIR)
export FASTEMBED_CACHE_PATH := $(MODEL_DIR)
export PYTHONPATH := $(CURDIR)/src

.PHONY: venv install test test-slow download-model download-models download-models-full run demo eval eval-m4 eval-m5 check-repo eval-llm eval-report datasets sample-pdfs \
        ollama-install ollama-up ollama-down ollama-pull run-llm docker-build docker-up podman-smoke ci-local clean

venv:
	uv venv .venv -p 3.12 || python3 -m venv .venv

install: venv
	uv pip install --python $(PY) -r requirements.lock && uv pip install --python $(PY) -e ".[dev]" \
	  || ($(PY) -m pip install -r requirements.lock && $(PY) -m pip install -e ".[dev]")

test:
	$(PY) -m pytest -q

test-slow: download-models-full
	HF_HUB_OFFLINE=1 $(PY) -m pytest -q -m slow

download-model:
	$(PY) scripts/download_model.py

download-models:  # what the shipped defaults use: en + zh embeddings + zh judge (~0.75 GB); --list shows sizes
	$(PY) scripts/download_model.py --profile default

download-models-full:  # + every judge / embedding compared in the eval (~2.2 GB); add --ollama 0.5b,1.5b for the LLMs
	$(PY) scripts/download_model.py --profile full

run:
	$(PY) -m uvicorn rag.api:app_factory --factory --host 0.0.0.0 --port 8000

run-llm:  # requires `make ollama-up`
	LLM_BACKEND=openai LLM_BASE_URL=http://127.0.0.1:11434/v1 LLM_MODEL=qwen2.5:0.5b $(MAKE) run

demo:  # requires `make run` in another shell
	./scripts/e2e_demo.sh

# ---- evaluation (writes eval/results/*.json + REPORT.md) ----
# Datasets: eval/datasets/{en,zh}.{dev,test}.jsonl (frozen). alpha + refusal threshold are tuned on dev,
# headline numbers come from the held-out test split.
eval: download-models
	HF_HUB_OFFLINE=1 $(PY) -m rag.eval --lang en --model BAAI/bge-small-en-v1.5 --chunk-sizes 800,400
	HF_HUB_OFFLINE=1 $(PY) -m rag.eval --lang zh --model BAAI/bge-small-zh-v1.5 --chunk-sizes 800,200

LLM_MODELS ?= qwen2.5:0.5b
eval-llm: download-models  # requires `make ollama-up`; LLM runs on the test split; ~10 min per model on CPU
	HF_HUB_OFFLINE=1 $(PY) -m rag.eval --lang en --model BAAI/bge-small-en-v1.5 --chunk-sizes 800,400 --llm --llm-models $(LLM_MODELS)
	HF_HUB_OFFLINE=1 $(PY) -m rag.eval --lang zh --model BAAI/bge-small-zh-v1.5 --chunk-sizes 800,200 --llm --llm-models $(LLM_MODELS)

eval-m4: download-models-full  # + answerability judges and query expansion (needs `make ollama-up` and qwen2.5 0.5b/1.5b)
	HF_HUB_OFFLINE=1 $(PY) -m rag.eval --lang en --model BAAI/bge-small-en-v1.5 --chunk-sizes 800,400 --judges nli,nli-t,reranker,reranker-m3 --qe prf,rewrite:qwen2.5:0.5b,rewrite:qwen2.5:1.5b,hyde:qwen2.5:0.5b,hyde:qwen2.5:1.5b
	HF_HUB_OFFLINE=1 $(PY) -m rag.eval --lang zh --model BAAI/bge-small-zh-v1.5 --chunk-sizes 800,200 --judges nli,nli-t,reranker-m3 --qe prf,rewrite:qwen2.5:0.5b,rewrite:qwen2.5:1.5b,hyde:qwen2.5:0.5b,hyde:qwen2.5:1.5b

eval-m5: download-models-full  # M5: dev-only selection; add FINAL=1 for the single test evaluation (needs make ollama-up)
	HF_HUB_OFFLINE=1 $(PY) scripts/build_dev_extra.py en zh
	HF_HUB_OFFLINE=1 $(PY) -m rag.eval_m5 --lang en $(if $(FINAL),--final,)
	HF_HUB_OFFLINE=1 $(PY) -m rag.eval_m5 --lang zh $(if $(FINAL),--final,)

check-repo:  # no large / local-only files in the git tree or history
	./scripts/check_repo.sh

eval-report:
	$(PY) -m rag.eval --report

datasets:  # (re)build dev/test splits from eval/authoring/*.py; refuses to overwrite frozen files
	$(PY) scripts/split_dataset.py en zh

sample-pdfs:  # English PDFs are dependency-free; the Chinese PDF needs Chrome + Noto Sans CJK SC
	$(PY) scripts/make_pdf.py && $(PY) scripts/make_zh_pdf.py

# ---- local LLM (user-level Ollama, everything inside the project dir) ----
ollama-install:
	./scripts/ollama.sh install
ollama-up:
	./scripts/ollama.sh up
ollama-down:
	./scripts/ollama.sh down
ollama-pull:
	./scripts/ollama.sh pull

docker-build:
	docker compose build

podman-smoke:  # no Docker daemon: rootless podman build + run + /health + /ask
	./scripts/docker_smoke.sh

ci-local:  # every step of .github/workflows/ci.yml on a clean copy of the tree
	./scripts/ci_local.sh

docker-up:
	[ -f .env ] || cp .env.example .env
	docker compose up -d

clean:
	rm -rf data/*.sqlite3 .pytest_cache
