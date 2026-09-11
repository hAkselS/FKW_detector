#!/usr/bin/env bash

PROJECT_DIR="$HOME/FKW_detector"
LOG_DIR="$PROJECT_DIR/logs/sys_logs"

mkdir -p "$LOG_DIR"

# Timestamped log file setup
log_file="$LOG_DIR/main_$(date '+%Y-%m-%d_%H-%M-%S').log"

# Virtual environment activation (exit on failure)
VENV_PATH="$PROJECT_DIR/venv/bin/activate"
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
else
    echo "Error: Virtual environment not found at $VENV_PATH" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1

# Change working directory to FKW_detector root
cd "$PROJECT_DIR" || exit 1

python -u "$PROJECT_DIR/sys_control/main.py" >> "$log_file" 2>&1