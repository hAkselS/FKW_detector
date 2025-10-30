#!/bin/bash

# --- CONFIGURABLE SECTION ---
CONDA_ENV="venv"            # conda/env name (used if conda fallback is needed)
# ----------------------------

# Resolve project directory relative to this script (works when repo is in ~/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Use project-local venv by default, keep logs inside project for portability
LOG_DIR="${PROJECT_DIR}/logs/sys_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/process_control_$(date '+%Y-%m-%d_%H-%M-%S').log"

# Prefer project-local venv activation
VENV_ACTIVATE="$PROJECT_DIR/venv/bin/activate"
if [ -f "$VENV_ACTIVATE" ]; then
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE"
else
    echo "Warning: virtualenv activate not found at $VENV_ACTIVATE" >> "$LOG_FILE" 2>&1
    # Try to fall back to conda (if available on the system)
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base 2>/dev/null)"
        if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1090
            source "$CONDA_BASE/etc/profile.d/conda.sh"
            conda activate "$CONDA_ENV" || echo "Warning: failed to activate conda env $CONDA_ENV" >> "$LOG_FILE" 2>&1
        fi
    else
        echo "No venv or conda found; continuing with system python" >> "$LOG_FILE" 2>&1
    fi
fi

# Run from project root
cd "$PROJECT_DIR" || exit 1

# Run main script and capture output
python3 sys_control/process_control.py >> "$LOG_FILE" 2>&1