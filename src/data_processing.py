from __future__ import annotations
import logging
import pandas as pd
from src.config import TARGET_COL
logger = logging.getLogger(__name__)


def load_csv(path, parse_dates: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if parse_dates and "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    X = frame.drop(columns=[TARGET_COL], errors="ignore").copy()
    y = frame[TARGET_COL].astype(float) if TARGET_COL in frame.columns else None
    return X, y


def ensure_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in (
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "market_index",
        "quote_signal",
        "weight",
        "distance",
    ):
        if col not in out.columns:
            out[col] = pd.NA
    return out
