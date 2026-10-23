#!/usr/bin/env bash
# User-level Ollama (CPU-only) living entirely inside the project:
#   binary  -> tools/ollama/        models -> models/ollama (OLLAMA_MODELS)    HOME -> tools/home-ollama
# Usage: scripts/ollama.sh install | up | down | pull | status
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VER="${OLLAMA_VERSION:-v0.40.9}"
BIN="$ROOT/tools/ollama/bin/ollama"
export OLLAMA_MODELS="$ROOT/models/ollama" OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}" HOME="$ROOT/tools/home-ollama"
# Hugging Face fallback for `pull`: identical Q4_K_M quantisations of the official Qwen2.5-Instruct releases.
declare -A GGUF_SHA=(
  [0.5b]=74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db
  [1.5b]=6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e
)
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
    # Usage: scripts/ollama.sh pull [0.5b|1.5b]  (default 0.5b)
    # `ollama pull` is the normal path. If the registry's blob CDN is unreachable (seen on networks with a
    # fake-IP DNS proxy: Ollama refuses the redirect as "non-public"), download the identical GGUF from
    # Hugging Face into a temp dir, verify sha256, `ollama create` it, then delete the GGUF again. Ollama
    # keeps its own copy under OLLAMA_MODELS, so no duplicate stays on disk.
    size="${2:-0.5b}"; tag="qwen2.5:$size"
    if "$BIN" list | awk '{print $1}' | grep -qx "$tag"; then echo "$tag already present"; "$BIN" list; exit 0; fi
    if ! "$BIN" pull "$tag"; then
      echo "ollama pull failed -> importing GGUF from Hugging Face" >&2
      sha="${GGUF_SHA[$size]:?unknown size $size}"
      tmp="$(mktemp -d "$ROOT/models/.gguf-XXXX")"; trap 'rm -rf "$tmp"' EXIT
      f="$tmp/qwen2.5-$size-instruct-q4_k_m.gguf"
      curl -fL -o "$f" "https://huggingface.co/Qwen/Qwen2.5-${size^^}-Instruct-GGUF/resolve/main/qwen2.5-$size-instruct-q4_k_m.gguf"
      echo "$sha  $f" | sha256sum -c -
      sed "s#__GGUF__#$f#" "$ROOT/scripts/ollama/Modelfile.qwen2.5.tmpl" > "$tmp/Modelfile"
      "$BIN" create "$tag" -f "$tmp/Modelfile"
    fi
    "$BIN" list ;;
  status)
    curl -sf "http://$OLLAMA_HOST/api/version" && echo && "$BIN" list ;;
esac
