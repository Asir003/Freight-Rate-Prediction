from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

set_config(transform_output="pandas")

from src.config import RANDOM_SEED, RATE_CEILING, RATE_FLOOR
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, FeatureEngineer


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs(y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None)))
    return {"mae": mae, "rmse": rmse, "mape": mape}


def clip_rate(pred) -> np.ndarray:
    return np.clip(np.asarray(pred, dtype=float), RATE_FLOOR, RATE_CEILING)


class QuoteDistanceRegressor(BaseEstimator, RegressorMixin):
    """Domain baseline: posted_rate ≈ quote_signal * distance after imputation."""

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X)
        return clip_rate(X[:, 0] * X[:, 1])


class DistanceMedianRpmRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        distance = np.asarray(X)[:, 0]
        y = np.asarray(y, dtype=float)
        self.median_rpm_ = float(np.median(y / np.clip(distance, 1e-6, None)))
        return self

    def predict(self, X):
        distance = np.asarray(X)[:, 0]
        return clip_rate(distance * self.median_rpm_)


class RatePerMileModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = np.asarray(y, dtype=float)
        rpm = y / np.clip(X["distance"].to_numpy(), 1e-6, None)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, rpm)
        return self

    def predict(self, X):
        X = pd.DataFrame(X)
        rpm = self.estimator_.predict(X)
        return clip_rate(rpm * X["distance"].to_numpy())


class LogTargetModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, np.log1p(np.asarray(y, dtype=float)))
        return self

    def predict(self, X):
        return clip_rate(np.expm1(self.estimator_.predict(X)))


class ClippedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        return clip_rate(self.estimator_.predict(X))


def _hgb(**kwargs) -> HistGradientBoostingRegressor:
    params = {
        "loss": "absolute_error",
        "learning_rate": 0.06,
        "max_iter": 300,
        "max_depth": 8,
        "min_samples_leaf": 25,
        "l2_regularization": 0.1,
        "random_state": RANDOM_SEED,
        "categorical_features": "from_dtype",
    }
    params.update(kwargs)
    return HistGradientBoostingRegressor(**params)


def _ridge_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
        ]
    )


def build_experiment_pipelines() -> dict[str, Pipeline]:
    hgb = _hgb()
    hgb_shallow = _hgb(max_depth=5, learning_rate=0.08, max_iter=250)
    return {
        "median": Pipeline(
            [
                ("features", FeatureEngineer()),
                ("model", DummyRegressor(strategy="median")),
            ]
        ),
        "distance_median_rpm": Pipeline(
            [
                ("features", FeatureEngineer()),
                (
                    "select",
                    ColumnTransformer([("dist", "passthrough", ["distance"])], remainder="drop"),
                ),
                ("model", DistanceMedianRpmRegressor()),
            ]
        ),
        "quote_x_distance": Pipeline(
            [
                ("features", FeatureEngineer()),
                (
                    "select",
                    ColumnTransformer(
                        [("qd", "passthrough", ["quote_signal", "distance"])],
                        remainder="drop",
                    ),
                ),
                ("model", QuoteDistanceRegressor()),
            ]
        ),
        "ridge": Pipeline(
            [
                ("features", FeatureEngineer()),
                ("prep", _ridge_preprocessor()),
                ("model", ClippedRegressor(Ridge(alpha=3.0))),
            ]
        ),
        "hgb": Pipeline(
            [
                ("features", FeatureEngineer()),
                ("model", ClippedRegressor(hgb)),
            ]
        ),
        "hgb_shallow": Pipeline(
            [
                ("features", FeatureEngineer()),
                ("model", ClippedRegressor(hgb_shallow)),
            ]
        ),
        "hgb_log_target": Pipeline(
            [
                ("features", FeatureEngineer()),
                ("model", LogTargetModel(_hgb(loss="squared_error"))),
            ]
        ),
        "hgb_rpm": Pipeline(
            [
                ("features", FeatureEngineer()),
                ("model", RatePerMileModel(_hgb())),
            ]
        ),
        "linreg_distance": Pipeline(
            [
                ("features", FeatureEngineer()),
                (
                    "select",
                    ColumnTransformer([("dist", "passthrough", ["distance"])], remainder="drop"),
                ),
                ("model", ClippedRegressor(LinearRegression())),
            ]
        ),
    }


def build_final_pipeline(name: str) -> Pipeline:
    pipelines = build_experiment_pipelines()
    if name not in pipelines:
        raise KeyError(f"Unknown model {name}")
    return pipelines[name]
