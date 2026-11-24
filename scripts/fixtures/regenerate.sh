#!/usr/bin/env bash
# Regenerates every committed fixture from karpathy/autoresearch at the pinned
# commit, or, with `check`, compares bin/prepare's output in the cache with
# prepare.py's own (see check_prepared.py).
# Usage: scripts/fixtures/regenerate.sh          (or: make fixtures)
#        scripts/fixtures/regenerate.sh check    (or: make check-data)
set -euo pipefail

COMMIT=228791fb499afffb54b46200aca536f79142f117
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CACHE="$ROOT/scripts/fixtures/.cache"
SOURCE="$CACHE/autoresearch-$COMMIT"
VENV="$CACHE/venv"
PYTHON="${PYTHON:-python3}"

# 1. Fetch the original at the pinned commit.
if [ ! -f "$SOURCE/train.py" ]; then
  rm -rf "$SOURCE" && mkdir -p "$SOURCE"
  git -C "$SOURCE" init -q
  git -C "$SOURCE" fetch -q --depth 1 https://github.com/karpathy/autoresearch "$COMMIT"
  git -C "$SOURCE" checkout -q FETCH_HEAD
fi

# 2. Create the Python environment with pinned packages.
if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON" -m venv "$VENV"
fi
"$VENV/bin/pip" install -q -r "$ROOT/scripts/fixtures/requirements.txt"

# 3. Write the fixtures, or check the prepared cache.
if [ "${1:-}" = "check" ]; then
  CACHE_DIR="${AUTORESEARCH_CACHE:-$HOME/.cache/autoresearch.cr}"
  "$ROOT/bin/batch_digests" > "$CACHE/batch_digests.json"
  "$VENV/bin/python" -I "$ROOT/scripts/fixtures/check_prepared.py" "$SOURCE" "$CACHE_DIR" "$CACHE/batch_digests.json"
else
  "$VENV/bin/python" -I "$ROOT/scripts/fixtures/make_fixtures.py" "$SOURCE" "$ROOT/spec/fixtures"
fi
