#!/usr/bin/env bash
# Upload the sample docs to a running server and ask a few questions.
set -euo pipefail
BASE=${BASE:-http://127.0.0.1:8000}
for f in sample_docs/${LANG_DIR:-en}/*; do
  curl -sf -F "file=@$f" "$BASE/documents"; echo
done
ask() { curl -sf -H 'Content-Type: application/json' -d "{\"question\": \"$1\"}" "$BASE/ask" | python3 -m json.tool --no-ensure-ascii; }
ask "${1:-How much weight can the robot arm carry?}"
