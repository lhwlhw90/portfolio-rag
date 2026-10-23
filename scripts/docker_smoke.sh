#!/usr/bin/env bash
# Build the image and smoke-test it: /health + one upload + one /ask. Uses docker when a daemon is
# reachable, otherwise rootless podman (scripts/podman.sh, no sudo). Models are mounted read-only
# from ./models; data goes to a throw-away dir under build/. The container is removed afterwards.
#   scripts/docker_smoke.sh [image-tag]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${1:-portfolio-rag:smoke}"
PORT="${PORT:-18000}"
NAME="rag-smoke-$$"
if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
  CT=(docker); RUN_USER=()
else
  CT=("$ROOT/scripts/podman.sh")
  # rootless podman without newuidmap = single UID mapping: only container UID 0 exists (it maps
  # to the unprivileged host user, not to real root), so the image's USER 1000 cannot be used
  RUN_USER=(--user 0)
fi
DATA="$ROOT/build/smoke-data"; rm -rf "$DATA"; mkdir -p "$DATA"
cleanup() { "${CT[@]}" rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
echo "== build ($TAG) with ${CT[0]##*/}"
"${CT[@]}" build --format docker -t "$TAG" "$ROOT" 2>&1 | grep -v 'level=warning' | tail -3
echo "== run"
"${CT[@]}" run -d --name "$NAME" "${RUN_USER[@]}" -p "127.0.0.1:$PORT:8000" -v "$ROOT/models:/models:ro" \
  -v "$DATA:/data" -e HF_HUB_OFFLINE=1 "$TAG" 2>&1 | grep -v 'level=warning'
for i in $(seq 1 90); do curl -sf "localhost:$PORT/health" >/dev/null && break; sleep 1; done
echo "== GET /health"; curl -sf "localhost:$PORT/health"; echo
echo "== POST /documents"; curl -sf -F "file=@$ROOT/sample_docs/en/product_faq.txt" "localhost:$PORT/documents"; echo
curl -sf -F "file=@$ROOT/sample_docs/en/release_notes.md" "localhost:$PORT/documents" >/dev/null
echo "== POST /ask"
curl -sf "localhost:$PORT/ask" -H 'Content-Type: application/json' \
  -d '{"question": "How long is the X2 warranty?"}' | tee "$ROOT/build/smoke-answer.json"; echo
grep -q '3-year' "$ROOT/build/smoke-answer.json" && echo "SMOKE OK"
"${CT[@]}" logs "$NAME" 2>&1 | grep -v 'level=warning' | tail -5
