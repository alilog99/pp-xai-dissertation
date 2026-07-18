#!/usr/bin/env bash
# Activate the project venv.
# OpenMP for XGBoost/LightGBM is bundled beside their dylibs — do not set DYLD_LIBRARY_PATH.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
unset DYLD_LIBRARY_PATH
# Optional fallback only
if [[ -d "$ROOT/.deps/libomp/lib" ]]; then
  export DYLD_FALLBACK_LIBRARY_PATH="$ROOT/.deps/libomp/lib${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
fi
# shellcheck disable=SC1091
source "$ROOT/venv/bin/activate"
