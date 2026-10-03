from __future__ import annotations

import pandas as pd

from src.config import TIME_VALID_START


def time_based_split(frame: pd.DataFrame, cutoff: str = TIME_VALID_START):
    """Hold out later dates so validation matches future unseen loads."""
    dates = pd.to_datetime(frame["date"])
    cut = pd.Timestamp(cutoff)
    train_mask = dates < cut
    valid_mask = dates >= cut
    if train_mask.sum() == 0 or valid_mask.sum() == 0:
        raise ValueError(f"Time split at {cutoff} produced an empty fold")
    return frame.loc[train_mask].copy(), frame.loc[valid_mask].copy()
