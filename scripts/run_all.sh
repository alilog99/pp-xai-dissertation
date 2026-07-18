#!/bin/bash
# Run the full PP-XAI modelling pipeline (assumes raw-data already present).
# Prefer: /bin/bash scripts/run_all.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Only re-exec if *this* interpreter is Intel Homebrew bash (not merely PATH order)
_self="${BASH:-}"
if [[ "$_self" == /usr/local/bin/bash ]]; then
  echo "Detected Intel Homebrew bash; re-exec with /bin/bash..."
  exec /bin/bash "$ROOT/scripts/run_all.sh"
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh"
cd "$ROOT"
unset DYLD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "==> arch=$(uname -m) python=$(python -c 'import platform; print(platform.machine())')"
python -c "import numpy; print('numpy OK', numpy.__version__)"

echo "==> 1. Load & filter EPC"
python scripts/load_and_verify_data.py

echo "==> 2. OSM summary / optional join"
python scripts/join_osm.py || true

echo "==> 3. EDA"
python notebooks/01_EDA.py

echo "==> 4. Preprocess"
python scripts/run_preprocess.py

echo "==> 5. Baselines"
python scripts/run_baselines.py

echo "==> 6. Federated"
python scripts/run_federated.py

echo "==> 7. XAI"
python scripts/run_xai.py

echo "==> 8. Evaluation"
python scripts/run_evaluation.py

echo "Pipeline complete. Launch app with:"
echo "  source scripts/env.sh && streamlit run src/webapp/app.py"
