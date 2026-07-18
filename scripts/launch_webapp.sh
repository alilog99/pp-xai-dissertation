#!/usr/bin/env bash
# Launch Streamlit under native arm64 (avoids Rosetta/x86 Homebrew zsh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
unset DYLD_LIBRARY_PATH
exec arch -arm64 /bin/zsh -c "source '$ROOT/scripts/env.sh' && streamlit run src/webapp/app.py --server.port 8501 \"\$@\"" -- "$@"
