from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import CANDIDATES_DIR, FEATURES_PATH, T1_FEATURE_COLUMNS, ensure_project_dirs
from src.device import get_device, is_training_gpu_available
from src.modeling import prepare_model_frame, regression_metrics, split_timeframe, version_stamp
from src.utils import save_json, setup_logging, utc_now_iso


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def train_t1_gpu(allow_cpu: bool = False) -> dict | None:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("PyTorch is not installed. Install requirements-gpu.txt to use train_t1_gpu.py.")
        return None

    ensure_project_dirs()
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features file: {FEATURES_PATH}")

    if not is_training_gpu_available() and not allow_cpu:
        print("CUDA is unavailable. Re-run with --allow-cpu true to train on cpu/mps for testing only.")
        return None

    device = get_device()
    df = pd.read_parquet(FEATURES_PATH)
    frame = prepare_model_frame(df, T1_FEATURE_COLUMNS, target_column="future_ret_5d")
    train_df, validation_df, _ = split_timeframe(frame)
    if train_df.empty or validation_df.empty:
        raise RuntimeError("Insufficient train/validation data for t1 GPU baseline.")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[T1_FEATURE_COLUMNS])
    x_val = scaler.transform(validation_df[T1_FEATURE_COLUMNS])
    y_train = train_df["future_ret_5d"].to_numpy(dtype=np.float32)
    y_val = validation_df["future_ret_5d"].to_numpy(dtype=np.float32)

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)

    model = nn.Sequential(
        nn.Linear(len(T1_FEATURE_COLUMNS), 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)

    loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for _ in range(25):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        validation_pred = model(val_tensor).cpu().numpy().reshape(-1)

    metrics = regression_metrics(validation_df["future_ret_5d"], validation_pred)
    model_version = version_stamp("YOLO-WALLSTREET-t1-gpu")
    output_dir = CANDIDATES_DIR / "t1_gpu" / datetime.utcnow().strftime("%Y%m%d%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    joblib.dump(scaler, output_dir / "scaler.joblib")
    save_json(output_dir / "feature_columns.json", {"feature_columns": T1_FEATURE_COLUMNS})

    metadata = {
        "model_name": "YOLO-WALLSTREET-t1",
        "model_version": model_version,
        "trained_at": utc_now_iso(),
        "target": "future_ret_5d",
        "feature_columns": T1_FEATURE_COLUMNS,
        "validation_metrics": metrics,
        "device_used": device,
        "candidate_only": True,
        "mac_inference_supported": device != "cuda",
    }
    save_json(output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-cpu", default="false")
    args = parser.parse_args()
    setup_logging()
    train_t1_gpu(allow_cpu=_parse_bool(args.allow_cpu))


if __name__ == "__main__":
    main()
