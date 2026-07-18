#!/usr/bin/env bash
# Activate the project venv under Apple Silicon when possible.
# OpenMP for XGBoost/LightGBM is bundled beside their dylibs — do not set DYLD_LIBRARY_PATH.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
unset DYLD_LIBRARY_PATH

# Optional OpenMP fallback only (never DYLD_LIBRARY_PATH)
if [[ -d "$ROOT/.deps/libomp/lib" ]]; then
  export DYLD_FALLBACK_LIBRARY_PATH="$ROOT/.deps/libomp/lib${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
fi

# shellcheck disable=SC1091
source "$ROOT/venv/bin/activate"

# Detect Rosetta/x86 Python against arm64 wheels
_py_arch="$(python -c 'import platform; print(platform.machine())' 2>/dev/null || echo unknown)"
if [[ "$_py_arch" == "x86_64" ]]; then
  echo "ERROR: Python is running as x86_64 (Rosetta) but packages are arm64."
  echo "Your prompt shows conda (base) and/or Rosetta — that forces the wrong arch."
  echo ""
  echo "Fix — run Streamlit with:"
  echo "  conda deactivate"
  echo "  /bin/bash scripts/run_streamlit.sh"
  echo ""
  echo "Or one-liner:"
  echo "  arch -arm64 /bin/zsh -c 'source scripts/env.sh && streamlit run src/webapp/app.py'"
  return 1 2>/dev/null || exit 1
fi
