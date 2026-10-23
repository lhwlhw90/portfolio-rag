#!/usr/bin/env bash
# Local, scripted equivalent of .github/workflows/ci.yml (`act` needs a Docker daemon, which this
# machine doesn't have). Runs every step of both jobs on a clean copy of the working tree:
#   job "test":   fresh venv -> pip install lock + -e .[dev] -> pytest -q -> eval en/zh ->
#                 download models -> pytest -m slow
#   job "docker": image build (docker if available, else rootless podman via scripts/podman.sh)
# Everything stays inside the project: the copy lives in build/ci/, models in models/.
# Usage: scripts/ci_local.sh [--skip-docker]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WS="$ROOT/build/ci/workspace"
LOG="$ROOT/build/ci/ci.log"
PY="${PYTHON:-python3.12}"
command -v "$PY" >/dev/null || PY=python3
mkdir -p "$ROOT/build/ci"
: > "$LOG"
step() { echo; echo "=== [$1] $2" | tee -a "$LOG"; local t0=$SECONDS; shift 2; ( "$@" ) 2>&1 | tee -a "$LOG"; echo "--- ok ($((SECONDS - t0)) s)" | tee -a "$LOG"; }

# actions/checkout: tracked + untracked-but-not-ignored files (what the next commit will contain)
checkout() {
  rm -rf "$WS" && mkdir -p "$WS"
  (cd "$ROOT" && git ls-files -co --exclude-standard -z | xargs -0 -I{} cp --parents {} "$WS/")
  echo "copied $(cd "$WS" && find . -type f | wc -l) files at $(cd "$ROOT" && git rev-parse --short HEAD)+worktree"
}
setup_python() { "$PY" -m venv "$WS/.venv" && "$WS/.venv/bin/python" --version; }
install() { cd "$WS" && .venv/bin/pip install -q -r requirements.lock && .venv/bin/pip install -q -e ".[dev]"; }
export MODEL_DIR="$ROOT/models" HF_HOME="$ROOT/models"   # = ${{ github.workspace }}/models
run_tests() { cd "$WS" && env -u MODEL_DIR -u HF_HOME .venv/bin/pytest -q; }
run_eval() { cd "$WS" && PYTHONPATH=src .venv/bin/python -m rag.eval --lang en --model BAAI/bge-small-en-v1.5 \
             && PYTHONPATH=src .venv/bin/python -m rag.eval --lang zh --model BAAI/bge-small-zh-v1.5; }
download() { cd "$WS" && PYTHONPATH=src .venv/bin/python scripts/download_model.py BAAI/bge-small-en-v1.5 BAAI/bge-small-zh-v1.5 nli; }
slow_tests() { cd "$WS" && .venv/bin/pytest -q -m slow; }
docker_build() {
  cd "$WS"
  if command -v docker >/dev/null && docker info >/dev/null 2>&1; then docker build -t portfolio-rag .
  else "$ROOT/scripts/podman.sh" build --format docker -t portfolio-rag . ; fi
}

step test "actions/checkout" checkout
step test "actions/setup-python 3.12" setup_python
step test "pip install -r requirements.lock && pip install -e .[dev]" install
step test "pytest -q" run_tests
step test "python -m rag.eval (en, zh)" run_eval
step test "download_model.py" download
step test "pytest -q -m slow" slow_tests
if [[ "${1:-}" != "--skip-docker" ]]; then
  step docker "actions/checkout" checkout
  step docker "docker build -t portfolio-rag ." docker_build
fi
echo; echo "CI (local) PASSED — log: $LOG" | tee -a "$LOG"
