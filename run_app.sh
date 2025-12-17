#!/bin/bash
# Script to run the Streamlit app with cache clearing

cd "$(dirname "$0")"
source torque-env/bin/activate

echo "Starting Streamlit app..."
echo "If the interface doesn't update, try:"
echo "  1. Stop the app (Ctrl+C)"
echo "  2. Clear cache: rm -rf .streamlit/cache"
echo "  3. Restart: streamlit run app.py --server.runOnSave true"
echo ""

# Run with auto-reload on save
streamlit run app.py --server.runOnSave true

