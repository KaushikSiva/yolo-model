#!/usr/bin/env bash
set -euo pipefail

EXPORT_DIR="exports/yolo-wallstreet-mac"
rm -rf "${EXPORT_DIR}"
mkdir -p "${EXPORT_DIR}/models/production" "${EXPORT_DIR}/config" "${EXPORT_DIR}/src" "${EXPORT_DIR}/data/processed"
cp -R models/production/t1 "${EXPORT_DIR}/models/production/"
cp -R models/production/t1_chronos "${EXPORT_DIR}/models/production/"
cp -R models/production/n1 "${EXPORT_DIR}/models/production/"
cp -R models/production/ensemble "${EXPORT_DIR}/models/production/"
cp -R models/production/adjuster "${EXPORT_DIR}/models/production/" 2>/dev/null || true
cp data/processed/features.parquet "${EXPORT_DIR}/data/processed/"
cp data/processed/chronos_features.parquet "${EXPORT_DIR}/data/processed/"
cp data/processed/news_features.parquet "${EXPORT_DIR}/data/processed/"
cp -R config "${EXPORT_DIR}/"
cp -R src "${EXPORT_DIR}/"
cp -R scripts "${EXPORT_DIR}/"
cp requirements.txt "${EXPORT_DIR}/requirements.txt"
cp README.md "${EXPORT_DIR}/README.md"

printf '%s\n' \
  '# YOLO-WALLSTREET Mac Inference' \
  '' \
  'This export bundle is CPU-safe and does not require CUDA.' \
  '' \
  'Contents:' \
  '- Production t1, t1_chronos, n1, ensemble, and optional adjuster artifacts' \
  '- data/processed/features.parquet, chronos_features.parquet, and news_features.parquet' \
  '- Config files and Python source' \
  '- requirements.txt for Mac/local inference' \
  '' \
  'Notes:' \
  '- The Mac path uses precomputed features and disables the Gemma adjuster by default.' \
  '- Set YOLO_WALLSTREET_DISABLE_ADJUSTER=0 only if you also want local adjuster inference and have the required transformer stack.' \
  '' \
  'Typical steps:' \
  '1. bash scripts/run_mac_inference.sh' \
  '2. curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"ticker\":\"AAPL\",\"horizon\":\"5d\",\"log\":true}"' \
  > "${EXPORT_DIR}/README_MAC_INFERENCE.md"

echo "Exported Mac bundle to ${EXPORT_DIR}"
