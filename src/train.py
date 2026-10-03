from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import pandas as pd

from src.config import (
    FINAL_METRICS_PATH,
    METRICS_PATH,
    MODEL_PATH,
    OUTPUTS_DIR,
    RANDOM_SEED,
    TRAIN_PATH,
)
from src.data_processing import load_csv, split_xy
from src.models import build_experiment_pipelines, build_final_pipeline, regression_metrics
from src.predict import generate_predictions
from src.validation import time_based_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def evaluate_experiments(dev: pd.DataFrame) -> pd.DataFrame:
    train_fold, valid_fold = time_based_split(dev)
    X_train, y_train = split_xy(train_fold)
    X_valid, y_valid = split_xy(valid_fold)
    logger.info(
        "Time split: train=%s rows through %s; valid=%s rows from %s",
        len(train_fold),
        train_fold["date"].max().date(),
        len(valid_fold),
        valid_fold["date"].min().date(),
    )

    records = []
    pipelines = build_experiment_pipelines()
    for name, pipeline in pipelines.items():
        logger.info("Fitting %s", name)
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        pred_valid = pipeline.predict(X_valid)
        pred_train = pipeline.predict(X_train)
        valid_metrics = regression_metrics(y_valid, pred_valid)
        train_metrics = regression_metrics(y_train, pred_train)
        record = {
            "model": name,
            "seed": RANDOM_SEED,
            "train_seconds": round(elapsed, 3),
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "valid_mae": valid_metrics["mae"],
            "valid_rmse": valid_metrics["rmse"],
            "valid_mape": valid_metrics["mape"],
        }
        logger.info(
            "%s valid MAE=%.3f RMSE=%.3f MAPE=%.4f (train MAE=%.3f, %.2fs)",
            name,
            record["valid_mae"],
            record["valid_rmse"],
            record["valid_mape"],
            record["train_mae"],
            elapsed,
        )
        records.append(record)
    metrics = pd.DataFrame(records).sort_values("valid_mae")
    return metrics


def select_model(metrics: pd.DataFrame) -> str:
    return str(metrics.iloc[0]["model"])


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading development data from %s", TRAIN_PATH)
    dev = load_csv(TRAIN_PATH)
    if dev["posted_rate"].isna().any():
        raise ValueError("Training target contains missing values")

    metrics = evaluate_experiments(dev)
    metrics.to_csv(METRICS_PATH, index=False)
    selected = select_model(metrics)
    logger.info("Selected model by time-split MAE: %s", selected)

    X_all, y_all = split_xy(dev)
    logger.info("Refitting %s on all development data (%s rows)", selected, len(dev))
    final_pipeline = build_final_pipeline(selected)
    start = time.perf_counter()
    final_pipeline.fit(X_all, y_all)
    elapsed = time.perf_counter() - start
    joblib.dump({"pipeline": final_pipeline, "model_name": selected}, MODEL_PATH)
    logger.info("Saved pipeline to %s (%.2fs)", MODEL_PATH, elapsed)

    train_pred = final_pipeline.predict(X_all)
    train_metrics = regression_metrics(y_all, train_pred)
    summary = {
        "selected_model": selected,
        "validation_strategy": "time_based_split_cutoff_2025-10-01",
        "primary_metric": "mae",
        "selection_row": metrics.iloc[0].to_dict(),
        "full_train_mae": train_metrics["mae"],
        "full_train_rmse": train_metrics["rmse"],
        "full_train_fit_seconds": elapsed,
        "n_development_rows": int(len(dev)),
    }
    Path(FINAL_METRICS_PATH).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    generate_predictions(MODEL_PATH)
    logger.info("Training complete")


if __name__ == "__main__":
    main()
