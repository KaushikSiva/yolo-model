from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
import shutil
from datetime import datetime

from sqlalchemy import insert

from src.config import ARCHIVED_MODELS_DIR, CANDIDATES_DIR, ENSEMBLE_PRODUCTION_DIR, REPORTS_DIR, T1_PRODUCTION_DIR
from src.db import create_tables, get_engine, model_runs_table
from src.utils import json_dumps, make_id, setup_logging, utc_now_iso


def promote_latest(model_type: str) -> dict:
    evaluation_path = REPORTS_DIR / f"candidate_eval_{model_type}.json"
    if not evaluation_path.exists():
        raise FileNotFoundError(f"Missing evaluation report: {evaluation_path}")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))

    candidate_metrics = evaluation["candidate_metrics"]
    production_metrics = evaluation["production_metrics"]
    candidate_backtest = evaluation["candidate_backtest"]
    production_backtest = evaluation["production_backtest"]

    passed = (
        candidate_metrics["direction_accuracy"] >= production_metrics["direction_accuracy"]
        and candidate_metrics["rmse"] <= production_metrics["rmse"] * 1.02
        and candidate_metrics["top_decile_avg_forward_return"] > production_metrics["top_decile_avg_forward_return"]
        and candidate_backtest["total_return"] > production_backtest["total_return"]
        and candidate_backtest["max_drawdown"] <= production_backtest["max_drawdown"] * 1.10
    )

    result = {"model_type": model_type, "passed": passed, "candidate_version": evaluation["candidate_version"]}
    create_tables(get_engine())

    with get_engine().begin() as connection:
        connection.execute(
            insert(model_runs_table).values(
                run_id=make_id("run"),
                created_at=utc_now_iso(),
                model_name=f"YOLO-WALLSTREET-{model_type}",
                model_version=evaluation["candidate_version"],
                run_type="promotion_check",
                metrics_json=json_dumps(evaluation),
                promoted=int(passed),
            )
        )

    if not passed:
        return result

    production_dir = T1_PRODUCTION_DIR if model_type == "t1" else ENSEMBLE_PRODUCTION_DIR
    candidate_dir = Path(evaluation["candidate_dir"])
    archive_dir = ARCHIVED_MODELS_DIR / f"{model_type}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    if production_dir.exists():
        for item in production_dir.iterdir():
            shutil.move(str(item), archive_dir / item.name)
    for item in candidate_dir.iterdir():
        target = production_dir / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    result["promoted_to"] = str(production_dir)
    result["archived_to"] = str(archive_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["t1", "ensemble"], default="ensemble")
    args = parser.parse_args()
    setup_logging()
    print(json.dumps(promote_latest(args.model), indent=2))


if __name__ == "__main__":
    main()
