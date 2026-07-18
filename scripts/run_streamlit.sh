#!/bin/bash
# Launch Streamlit under native arm64 (avoids Rosetta/x86 vs arm64 NumPy crash).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
unset DYLD_LIBRARY_PATH

# Always use system zsh under arm64 — PATH bash may be Intel Homebrew
exec arch -arm64 /bin/zsh -c "
  source '$ROOT/scripts/env.sh'
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS=1
  echo \"arch=\$(uname -m) python=\$(python -c 'import platform; print(platform.machine())')\"
  python -c 'import numpy; print(\"numpy OK\", numpy.__version__)'
  exec streamlit run src/webapp/app.py --server.port 8501
"
