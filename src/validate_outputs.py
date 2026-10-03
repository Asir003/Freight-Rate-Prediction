from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np
import pandas as pd

from src.config import (
    DECEMBER_COLUMNS,
    DECEMBER_PREDICTIONS_PATH,
    EXPECTED_VAL_ROWS,
    ID_COL,
    PROJECT_ROOT,
    VALIDATION_PATH,
    VALIDATION_PREDICTIONS_PATH,
)


def fail(message: str) -> None:
    raise SystemExit(f"OUTPUT CHECK FAILED: {message}")


def check_validation_predictions() -> None:
    path = VALIDATION_PREDICTIONS_PATH
    if not path.is_file():
        fail(f"missing {path.name}")
    preds = pd.read_csv(path)
    validation = pd.read_csv(VALIDATION_PATH)
    if list(preds.columns) != [ID_COL, "predicted_rate"]:
        fail("validation_predictions.csv must have columns load_id,predicted_rate")
    if len(preds) != EXPECTED_VAL_ROWS or len(preds) != len(validation):
        fail(f"expected {EXPECTED_VAL_ROWS} prediction rows, found {len(preds)}")
    if preds[ID_COL].duplicated().any():
        fail("duplicate load_id in predictions")
    if not preds[ID_COL].astype(str).equals(validation[ID_COL].astype(str)):
        fail("load_id values or order do not match validation.csv")
    rates = pd.to_numeric(preds["predicted_rate"], errors="coerce")
    if rates.isna().any():
        fail("missing or non-numeric predicted_rate")
    if not np.isfinite(rates).all():
        fail("non-finite predicted_rate")
    if (rates <= 0).any():
        fail("non-positive predicted_rate")


def check_december_predictions() -> None:
    path = DECEMBER_PREDICTIONS_PATH
    if not path.is_file():
        fail(f"missing {path}")
    frame = pd.read_csv(path)
    if list(frame.columns) != DECEMBER_COLUMNS:
        fail("December file columns/order are incorrect")
    if len(frame) != 31:
        fail(f"expected 31 December rows, found {len(frame)}")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any() or dates.duplicated().any():
        fail("invalid or duplicate December dates")
    expected = pd.date_range("2025-12-01", "2025-12-31", freq="D")
    if set(dates) != set(expected):
        fail("December dates must cover 2025-12-01 through 2025-12-31")
    rates = pd.to_numeric(frame["predicted_rate"], errors="coerce")
    if rates.isna().any() or not np.isfinite(rates).all() or (rates <= 0).any():
        fail("December predicted_rate is missing, non-finite, or non-positive")


def run_scorer() -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "score.py"),
        "--predictions",
        str(VALIDATION_PREDICTIONS_PATH),
        "--december-predictions",
        str(DECEMBER_PREDICTIONS_PATH),
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        fail(f"score.py failed:\n{result.stdout}\n{result.stderr}")
    chart = PROJECT_ROOT / "scorer_results" / "candidate_december.png"
    if not chart.is_file():
        fail("scorer did not create scorer_results/candidate_december.png")
    print(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated prediction files")
    parser.parse_args()
    check_validation_predictions()
    check_december_predictions()
    run_scorer()
    print("All output checks passed.")


if __name__ == "__main__":
    main()
