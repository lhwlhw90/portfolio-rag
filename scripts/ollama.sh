#!/usr/bin/env bash
# User-level Ollama (CPU-only) living entirely inside the project:
#   binary  -> tools/ollama/        models -> models/ollama (OLLAMA_MODELS)    HOME -> tools/home-ollama
# Usage: scripts/ollama.sh install | up | down | pull | status
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VER="${OLLAMA_VERSION:-v0.40.9}"
BIN="$ROOT/tools/ollama/bin/ollama"
export OLLAMA_MODELS="$ROOT/models/ollama" OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}" HOME="$ROOT/tools/home-ollama"
GGUF_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
GGUF_SHA="74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"
mkdir -p "$OLLAMA_MODELS" "$HOME"

case "${1:-status}" in
  install)
    mkdir -p "$ROOT/tools/ollama"
    tmp="$(mktemp -d)"
    curl -fL -o "$tmp/ollama.tar.zst" "https://github.com/ollama/ollama/releases/download/$VER/ollama-linux-amd64.tar.zst"
    # CPU only: skip the CUDA/ROCm/Vulkan runtimes (~1.4 GB)
    tar -I zstd -xf "$tmp/ollama.tar.zst" -C "$ROOT/tools/ollama" \
        --exclude='lib/ollama/cuda_*' --exclude='lib/ollama/rocm*' --exclude='lib/ollama/vulkan*'
    rm -rf "$tmp"; "$BIN" --version ;;
  up)
    nohup "$BIN" serve > "$ROOT/tools/ollama.log" 2>&1 &
    for _ in $(seq 30); do curl -sf "http://$OLLAMA_HOST/api/version" >/dev/null && break; sleep 1; done
    curl -sf "http://$OLLAMA_HOST/api/version"; echo ;;
  down)
    pkill -f "$BIN serve" || true ;;
  pull)
    # `ollama pull qwen2.5:0.5b` is the normal path. On networks where the registry's blob CDN
    # is unreachable we import the identical Q4_K_M GGUF from Hugging Face instead.
    if ! "$BIN" pull qwen2.5:0.5b; then
      echo "ollama pull failed -> importing GGUF from Hugging Face" >&2
      mkdir -p "$ROOT/models/gguf"; f="$ROOT/models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf"
      [ -f "$f" ] || curl -fL -o "$f" "$GGUF_URL"
      echo "$GGUF_SHA  $f" | sha256sum -c -
      sed "s#__GGUF__#$f#" "$ROOT/scripts/ollama/Modelfile.qwen2.5.tmpl" > "$ROOT/models/gguf/Modelfile.qwen2.5-0.5b"
      "$BIN" create qwen2.5:0.5b -f "$ROOT/models/gguf/Modelfile.qwen2.5-0.5b"
    fi
    "$BIN" list ;;
  status)
    curl -sf "http://$OLLAMA_HOST/api/version" && echo && "$BIN" list ;;
esac
