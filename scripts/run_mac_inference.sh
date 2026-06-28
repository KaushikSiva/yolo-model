#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

echo "Starting YOLO-WALLSTREET API on http://127.0.0.1:8000"
echo "Sample request:"
echo "curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{\"ticker\":\"AAPL\",\"horizon\":\"5d\",\"log\":true}'"
uvicorn src.api:app --reload --port 8000
