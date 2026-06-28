#!/usr/bin/env bash
set -euo pipefail

EXPORT_DIR="exports/yolo-wallstreet-mac"
rm -rf "${EXPORT_DIR}"
mkdir -p "${EXPORT_DIR}/models/production" "${EXPORT_DIR}/config" "${EXPORT_DIR}/src"
cp -R models/production/t1 "${EXPORT_DIR}/models/production/"
cp -R models/production/n1 "${EXPORT_DIR}/models/production/"
cp -R models/production/ensemble "${EXPORT_DIR}/models/production/"
cp -R config "${EXPORT_DIR}/"
cp -R src "${EXPORT_DIR}/"
cp requirements.txt "${EXPORT_DIR}/requirements.txt"
cp README.md "${EXPORT_DIR}/README.md"

printf '%s\n' \
  '# YOLO-WALLSTREET Mac Inference' \
  '' \
  'This export bundle is CPU-safe and does not require CUDA.' \
  '' \
  'Contents:' \
  '- Production t1, n1 metadata, and ensemble artifacts' \
  '- Config files and Python source' \
  '- requirements.txt for Mac/local inference' \
  '' \
  'Notes:' \
  '- Raw datasets are excluded by default.' \
  '- Sync the latest data/processed/features.parquet and data/processed/news_features.parquet when you want up-to-date local inference.' \
  '- MVP n1 inference uses precomputed or stub news features rather than a local Gemma runtime.' \
  '' \
  'Typical steps:' \
  '1. python3 -m venv .venv' \
  '2. source .venv/bin/activate' \
  '3. pip install -r requirements.txt' \
  '4. uvicorn src.api:app --port 8000' \
  > "${EXPORT_DIR}/README_MAC_INFERENCE.md"

echo "Exported Mac bundle to ${EXPORT_DIR}"
