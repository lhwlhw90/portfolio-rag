#!/usr/bin/env bash
# Pre-release hygiene: what git would publish. Fails if a tracked file (or any blob in history)
# is larger than LIMIT_KB (default 2048), or if local-only dirs are tracked.
set -euo pipefail
cd "$(dirname "$0")/.."
LIMIT_KB="${LIMIT_KB:-2048}"
fail=0
echo "== ignored local dirs (must not be tracked)"
for d in models tools build data .venv; do
  n=$( (git ls-files -- "$d" | grep -v "\.gitkeep$" || true) | wc -l)
  printf '  %-8s on disk %8s, tracked files: %s\n' "$d" "$(du -sh "$d" 2>/dev/null | cut -f1 || echo -)" "$n"
  [ "$n" -eq 0 ] || fail=1
  git check-ignore -q "$d/x" || { echo "  !! $d/ is not ignored"; fail=1; }
done
echo "== tracked tree: $(git ls-files | wc -l) files, $(git ls-files -z | xargs -0 du -cb 2>/dev/null | tail -1 | awk '{printf "%.1f MB", $1/1e6}')"
echo "== largest tracked files"
git ls-files -z | xargs -0 du -b | sort -nr | head -5 | awk '{printf "  %7.0f KB  %s\n", $1/1024, $2}'
echo "== history: $(git rev-list --count --all) commits, .git $(du -sh .git | cut -f1), packed: $(git count-objects -vH | awk -F': ' '/size-pack/{print $2}'), loose: $(git count-objects -vH | awk -F': ' '/^size:/{print $2}')"
echo "== largest blobs in history"
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' \
  | awk '$1=="blob"' | sort -k2 -nr | head -5 | awk '{printf "  %7.0f KB  %s\n", $2/1024, $3}'
big=$(git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' \
  | awk -v l=$((LIMIT_KB*1024)) '$1=="blob" && $2>l {print $3}')
if [ -n "$big" ]; then echo "!! blobs over ${LIMIT_KB} KB:"; echo "$big"; fail=1; fi
bad=$(git ls-files | grep -Ei '\.(onnx|gguf|safetensors|bin|pt|tar|tar\.gz)$|(^|/)\.env$' || true)
if [ -n "$bad" ]; then echo "!! model/secret-like files tracked:"; echo "$bad"; fail=1; fi
[ "$fail" -eq 0 ] && echo "OK: no large or local-only files in the tree or history" || { echo "FAILED"; exit 1; }
