from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge

from src.data_processing import ensure_feature_columns

EARTH_MILES = 3958.8

CATEGORICAL_FEATURES = ["pickup", "delivery", "equipment"]
NUMERIC_FEATURES = [
    "distance",
    "log_distance",
    "weight",
    "log_weight",
    "weight_missing",
    "weight_was_negative",
    "market_index",
    "quote_signal",
    "market_missing",
    "quote_missing",
    "quote_x_distance",
    "market_x_distance",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "lat_diff",
    "lon_diff",
    "haversine_miles",
    "circuity",
    "month",
    "day",
    "dayofweek",
    "weekofyear",
    "dayofyear",
    "is_weekend",
    "sin_doy",
    "cos_doy",
]


def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_MILES * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Train-only lookup/imputation plus leakage-safe derived features."""

    def __init__(self):
        self.city_coords_ = {}
        self.weight_medians_ = {}
        self.global_weight_median_ = np.nan
        self.quote_medians_ = {}
        self.global_quote_median_ = np.nan
        self.market_imputer_ = None
        self.quote_imputer_ = None
        self.quote_imputer_equipment_ = None

    def fit(self, X: pd.DataFrame, y=None):
        frame = ensure_feature_columns(X)
        self.city_coords_ = self._fit_city_coords(frame)

        weights = pd.to_numeric(frame["weight"], errors="coerce").abs()
        equipment = frame["equipment"].astype(str)
        self.weight_medians_ = weights.groupby(equipment).median().to_dict()
        self.global_weight_median_ = float(weights.median())

        quotes = pd.to_numeric(frame["quote_signal"], errors="coerce")
        self.quote_medians_ = quotes.groupby(equipment).median().to_dict()
        self.global_quote_median_ = float(quotes.median()) if quotes.notna().any() else 2.0

        dates = pd.to_datetime(frame["date"])
        doy = dates.dt.dayofyear.astype(float)
        fourier = np.column_stack(
            [
                np.sin(2 * np.pi * doy / 365.25),
                np.cos(2 * np.pi * doy / 365.25),
            ]
        )
        market = pd.to_numeric(frame["market_index"], errors="coerce")
        market_mask = market.notna()
        self.market_imputer_ = Ridge(alpha=1.0)
        if market_mask.sum() >= 10:
            self.market_imputer_.fit(fourier[market_mask.to_numpy()], market[market_mask])
        else:
            self.market_imputer_.fit(np.array([[0.0, 1.0]]), np.array([float(market.median())]))

        quote_mask = quotes.notna()
        equipment_levels = sorted(equipment.unique())
        self.quote_imputer_equipment_ = equipment_levels
        eq_dummies = pd.get_dummies(equipment).reindex(
            columns=equipment_levels, fill_value=0
        )
        quote_X = np.column_stack([eq_dummies.to_numpy(), fourier])
        self.quote_imputer_ = Ridge(alpha=1.0)
        if quote_mask.sum() >= 10:
            self.quote_imputer_.fit(quote_X[quote_mask.to_numpy()], quotes[quote_mask])
        else:
            self.quote_imputer_.fit(
                np.zeros((1, quote_X.shape[1])), np.array([self.global_quote_median_])
            )
        self.cat_levels_ = {
            "pickup": sorted(frame["pickup"].astype(str).unique()),
            "delivery": sorted(frame["delivery"].astype(str).unique()),
            "equipment": sorted(frame["equipment"].astype(str).unique()),
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = ensure_feature_columns(X)
        n = len(frame)

        pickup = frame["pickup"].astype(str)
        delivery = frame["delivery"].astype(str)
        equipment = frame["equipment"].astype(str)

        pickup_lat, pickup_lon = self._coords_for(pickup, frame["pickup_lat"], frame["pickup_lon"])
        delivery_lat, delivery_lon = self._coords_for(
            delivery, frame["delivery_lat"], frame["delivery_lon"]
        )

        dates = pd.to_datetime(frame["date"])
        if dates.isna().any():
            raise ValueError("FeatureEngineer received invalid dates")

        dayofyear = dates.dt.dayofyear.astype(float)
        sin_doy = np.sin(2 * np.pi * dayofyear / 365.25)
        cos_doy = np.cos(2 * np.pi * dayofyear / 365.25)

        raw_weight = pd.to_numeric(frame["weight"], errors="coerce")
        weight_missing = raw_weight.isna().astype(float)
        weight_was_negative = (raw_weight < 0).fillna(False).astype(float)
        weight = raw_weight.abs()
        weight = weight.fillna(equipment.map(self.weight_medians_))
        weight = weight.fillna(self.global_weight_median_)

        raw_market = pd.to_numeric(frame["market_index"], errors="coerce")
        market_missing = raw_market.isna().astype(float)
        market_hat = self.market_imputer_.predict(np.column_stack([sin_doy, cos_doy]))
        market = raw_market.fillna(pd.Series(market_hat, index=frame.index))

        raw_quote = pd.to_numeric(frame["quote_signal"], errors="coerce")
        quote_missing = raw_quote.isna().astype(float)
        eq_dummies = pd.get_dummies(equipment).reindex(
            columns=self.quote_imputer_equipment_, fill_value=0
        )
        quote_hat = self.quote_imputer_.predict(
            np.column_stack([eq_dummies.to_numpy(), sin_doy, cos_doy])
        )
        quote = raw_quote.fillna(equipment.map(self.quote_medians_))
        quote = quote.fillna(pd.Series(quote_hat, index=frame.index))
        quote = quote.fillna(self.global_quote_median_)

        distance = pd.to_numeric(frame["distance"], errors="coerce").astype(float)
        if distance.isna().any():
            raise ValueError("distance is required at prediction time")

        hav = haversine_miles(pickup_lat, pickup_lon, delivery_lat, delivery_lon)
        circuity = np.divide(
            distance.to_numpy(),
            np.clip(hav, 1e-6, None),
            out=np.ones(n, dtype=float),
            where=np.isfinite(hav),
        )

        out = pd.DataFrame(index=frame.index)
        out["pickup"] = pd.Categorical(pickup, categories=self.cat_levels_["pickup"])
        out["delivery"] = pd.Categorical(delivery, categories=self.cat_levels_["delivery"])
        out["equipment"] = pd.Categorical(equipment, categories=self.cat_levels_["equipment"])
        out["distance"] = distance
        out["log_distance"] = np.log1p(distance.clip(lower=0))
        out["weight"] = weight.astype(float)
        out["log_weight"] = np.log1p(out["weight"].clip(lower=0))
        out["weight_missing"] = weight_missing
        out["weight_was_negative"] = weight_was_negative
        out["market_index"] = market.astype(float)
        out["quote_signal"] = quote.astype(float)
        out["market_missing"] = market_missing
        out["quote_missing"] = quote_missing
        out["quote_x_distance"] = out["quote_signal"] * out["distance"]
        out["market_x_distance"] = out["market_index"] * out["distance"]
        out["pickup_lat"] = pickup_lat
        out["pickup_lon"] = pickup_lon
        out["delivery_lat"] = delivery_lat
        out["delivery_lon"] = delivery_lon
        out["lat_diff"] = out["delivery_lat"] - out["pickup_lat"]
        out["lon_diff"] = out["delivery_lon"] - out["pickup_lon"]
        out["haversine_miles"] = hav
        out["circuity"] = circuity
        out["month"] = dates.dt.month.astype(float)
        out["day"] = dates.dt.day.astype(float)
        out["dayofweek"] = dates.dt.dayofweek.astype(float)
        out["weekofyear"] = dates.dt.isocalendar().week.astype(float)
        out["dayofyear"] = dayofyear
        out["is_weekend"] = (dates.dt.dayofweek >= 5).astype(float)
        out["sin_doy"] = sin_doy
        out["cos_doy"] = cos_doy
        return out[CATEGORICAL_FEATURES + NUMERIC_FEATURES]

    def _fit_city_coords(self, frame: pd.DataFrame) -> dict:
        rows = []
        for city_col, lat_col, lon_col in (
            ("pickup", "pickup_lat", "pickup_lon"),
            ("delivery", "delivery_lat", "delivery_lon"),
        ):
            part = frame[[city_col, lat_col, lon_col]].copy()
            part.columns = ["city", "lat", "lon"]
            rows.append(part)
        coords = pd.concat(rows, ignore_index=True)
        coords["lat"] = pd.to_numeric(coords["lat"], errors="coerce")
        coords["lon"] = pd.to_numeric(coords["lon"], errors="coerce")
        coords = coords.dropna().drop_duplicates("city")
        return {
            str(row.city): (float(row.lat), float(row.lon)) for row in coords.itertuples(index=False)
        }

    def _coords_for(self, cities: pd.Series, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        lat_vals = pd.to_numeric(lat, errors="coerce").to_numpy(dtype=float)
        lon_vals = pd.to_numeric(lon, errors="coerce").to_numpy(dtype=float)
        for i, city in enumerate(cities.tolist()):
            if not np.isfinite(lat_vals[i]) or not np.isfinite(lon_vals[i]):
                lookup = self.city_coords_.get(str(city))
                if lookup is not None:
                    lat_vals[i], lon_vals[i] = lookup
        return lat_vals, lon_vals
