#!/usr/bin/env bash
# Regenerates every committed fixture from karpathy/autoresearch at the pinned commit.
# Usage: scripts/fixtures/regenerate.sh   (or: make fixtures)
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

# 3. Write the fixtures.
"$VENV/bin/python" -I "$ROOT/scripts/fixtures/make_fixtures.py" "$SOURCE" "$ROOT/spec/fixtures"
