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

# The pinned packages publish wheels for Python 3.12 to 3.14; the fixtures were made with 3.13.
SUPPORTED="3.13 3.12 3.14"

# Prints the Python version of interpreter $1 as major.minor, or nothing if it doesn't run.
python_version() {
  "$1" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || true
}

# Succeeds when version $1 is in SUPPORTED.
supported() {
  case " $SUPPORTED " in *" $1 "*) return 0 ;; *) return 1 ;; esac
}

# 1. Fetch the original at the pinned commit.
if [ ! -f "$SOURCE/train.py" ]; then
  rm -rf "$SOURCE" && mkdir -p "$SOURCE"
  git -C "$SOURCE" init -q
  git -C "$SOURCE" fetch -q --depth 1 https://github.com/karpathy/autoresearch "$COMMIT"
  git -C "$SOURCE" checkout -q FETCH_HEAD
fi

# 2. Choose an installed Python in the supported range: $PYTHON if set, else python3.13,
#    python3.12, python3.14, then python3. Nothing is installed; without one, say what is needed.
if [ -n "${PYTHON:-}" ]; then
  CANDIDATES="$PYTHON"
else
  CANDIDATES="python3.13 python3.12 python3.14 python3"
fi
CHOSEN=""
for candidate in $CANDIDATES; do
  command -v "$candidate" >/dev/null 2>&1 || continue
  version="$(python_version "$candidate")"
  if supported "$version"; then
    CHOSEN="$candidate"
    break
  fi
done
if [ -z "$CHOSEN" ]; then
  {
    if [ -n "${PYTHON:-}" ]; then
      echo "PYTHON=$PYTHON is Python ${version:-unknown or not found}; the pinned packages need 3.12, 3.13 or 3.14."
    else
      echo "No Python 3.12, 3.13 or 3.14 found (tried $CANDIDATES)."
    fi
    echo "Install one (3.13 matches the fixtures), for example with uv, Homebrew or your package manager,"
    echo "then rerun, with PYTHON=/path/to/python3.13 if it isn't on the PATH."
  } >&2
  exit 1
fi
CHOSEN_VERSION="$(python_version "$CHOSEN")"

# 3. Create the Python environment with pinned packages, replacing one made with another version.
if [ -x "$VENV/bin/python" ] && [ "$(python_version "$VENV/bin/python")" != "$CHOSEN_VERSION" ]; then
  echo "Replacing the Python $(python_version "$VENV/bin/python") environment with $CHOSEN_VERSION ($CHOSEN)." >&2
  rm -rf "$VENV"
fi
if [ ! -x "$VENV/bin/python" ]; then
  "$CHOSEN" -m venv "$VENV"
fi
"$VENV/bin/pip" install -q -r "$ROOT/scripts/fixtures/requirements.txt"

# 4. Write the fixtures, or check the prepared cache.
if [ "${1:-}" = "check" ]; then
  CACHE_DIR="${AUTORESEARCH_CACHE:-$HOME/.cache/autoresearch.cr}"
  "$ROOT/bin/batch_digests" > "$CACHE/batch_digests.json"
  "$VENV/bin/python" -I "$ROOT/scripts/fixtures/check_prepared.py" "$SOURCE" "$CACHE_DIR" "$CACHE/batch_digests.json"
else
  "$VENV/bin/python" -I "$ROOT/scripts/fixtures/make_fixtures.py" "$SOURCE" "$ROOT/spec/fixtures"
fi
