#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
export YOLO_WALLSTREET_DISABLE_ADJUSTER="${YOLO_WALLSTREET_DISABLE_ADJUSTER:-1}"

"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
"$PYTHON_BIN" src/init_db.py

echo "Starting YOLO-WALLSTREET API on http://127.0.0.1:8000"
echo "Gemma adjuster disabled by default on Mac: YOLO_WALLSTREET_DISABLE_ADJUSTER=${YOLO_WALLSTREET_DISABLE_ADJUSTER}"
echo "Sample request:"
echo "curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{\"ticker\":\"AAPL\",\"horizon\":\"5d\",\"log\":true}'"
uvicorn src.api:app --reload --port 8000
