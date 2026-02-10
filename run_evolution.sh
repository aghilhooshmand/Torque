#!/bin/bash
# Run evolution from config only (no GUI).
# Usage: ./run_evolution.sh   or   bash run_evolution.sh
# Ensure evolution_config.json and dataset are set; then this runs evolution and writes to results/.

cd "$(dirname "$0")"
exec python3 run_evolution_cli.py "$@"
