from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    DECEMBER_COLUMNS,
    DECEMBER_INPUT_PATH,
    DECEMBER_PREDICTIONS_PATH,
    ID_COL,
    MODEL_PATH,
    TEMPLATE_PATH,
    VALIDATION_PATH,
    VALIDATION_PREDICTIONS_PATH,
)
from src.data_processing import load_csv
from src.models import clip_rate

logger = logging.getLogger(__name__)


def load_pipeline(model_path: Path = MODEL_PATH):
    payload = joblib.load(model_path)
    if isinstance(payload, dict) and "pipeline" in payload:
        return payload["pipeline"], payload.get("model_name")
    return payload, None


def predict_frame(pipeline, frame: pd.DataFrame) -> np.ndarray:
    preds = clip_rate(pipeline.predict(frame))
    if not np.isfinite(preds).all():
        raise ValueError("Model produced non-finite predictions")
    if (preds <= 0).any():
        raise ValueError("Model produced non-positive predictions")
    return preds


def write_validation_predictions(pipeline) -> pd.DataFrame:
    validation = load_csv(VALIDATION_PATH)
    template = pd.read_csv(TEMPLATE_PATH)
    if list(template.columns) != [ID_COL, "predicted_rate"]:
        raise ValueError("Unexpected validation predictions template columns")
    if len(validation) != len(template):
        raise ValueError("Template length does not match validation.csv")
    if not validation[ID_COL].astype(str).equals(template[ID_COL].astype(str)):
        raise ValueError("Template load_id order does not match validation.csv")

    preds = predict_frame(pipeline, validation)
    output = template.copy()
    output["predicted_rate"] = preds
    output.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)
    logger.info("Wrote %s rows to %s", len(output), VALIDATION_PREDICTIONS_PATH)
    return output


def write_december_predictions(pipeline) -> pd.DataFrame:
    december = load_csv(DECEMBER_INPUT_PATH)
    preds = predict_frame(pipeline, december)
    output = december.copy()
    output["predicted_rate"] = preds
    output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")
    output = output[DECEMBER_COLUMNS]
    DECEMBER_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(DECEMBER_PREDICTIONS_PATH, index=False)
    logger.info("Wrote %s December rows to %s", len(output), DECEMBER_PREDICTIONS_PATH)
    return output


def generate_predictions(model_path: Path = MODEL_PATH) -> None:
    pipeline, name = load_pipeline(model_path)
    logger.info("Generating predictions with %s", name or "saved pipeline")
    write_validation_predictions(pipeline)
    write_december_predictions(pipeline)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    generate_predictions()
