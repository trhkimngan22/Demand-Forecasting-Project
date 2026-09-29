"""Shared calendar and historical features for training and serving."""
import numpy as np
import pandas as pd

KEYS = ["product_id", "store_id"]
CATEGORICAL = ["product_category", "product_subcategory", "location"]
COVARIATES = CATEGORICAL + ["discount", "is_promotion", "holiday", "activity_flag",
                           "temperature", "humidity", "precipitation", "wind"]
CALENDAR = ["day_of_week", "week_of_year", "month", "year"]


def validate_config(config):
    for name in ("forecast_horizon",):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("lags", "rolling_windows"):
        values = config[name]
        if not values or any(type(v) is not int or v < 1 for v in values):
            raise ValueError(f"{name} must contain positive integers")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must not contain duplicates")
    if min(config["rolling_windows"]) < 2:
        raise ValueError("rolling_windows must be at least 2 for sample standard deviation")


def lookback_days(config):
    return max(max(config["lags"]), max(config["rolling_windows"])) + config["forecast_horizon"] - 1


def historical_columns(config):
    return ([f"lag_{lag}" for lag in config["lags"]] +
            [f"rolling_{stat}_{window}" for window in config["rolling_windows"]
             for stat in ("mean", "std")])


def feature_columns(config):
    # Target-date stockout status is unavailable at forecast time.
    return COVARIATES + CALENDAR + historical_columns(config)


def extract_data(raw):
    mapping = {"dt": "date", "sale_amount": "units_ordered",
               "first_category_id": "product_category", "second_category_id": "product_subcategory",
               "city_id": "location", "holiday_flag": "holiday", "avg_temperature": "temperature",
               "avg_humidity": "humidity", "precpt": "precipitation", "avg_wind_level": "wind"}
    frame = raw.rename(columns=mapping).copy()
    frame["is_promotion"] = (frame["discount"] < 1).astype(int)
    return frame[KEYS + ["date", "units_ordered"] + COVARIATES]


def generate_time_features(frame, config):
    """Look up observations by calendar date, never by position across gaps.
    Rolling windows require every daily observation. Missing history produces NaN.
    """
    validate_config(config)
    out = frame.copy().reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    if out[KEYS + ["date"]].isna().any().any():
        raise ValueError("Series identifiers and dates must not be missing")
    if not out["date"].eq(out["date"].dt.normalize()).all():
        raise ValueError("Dates must be daily timestamps at midnight")
    if out.duplicated(KEYS + ["date"]).any():
        raise ValueError("Duplicate product-store dates are not allowed")
    out["day_of_week"] = out.date.dt.dayofweek
    out["week_of_year"] = out.date.dt.isocalendar().week.astype(int)
    out["month"] = out.date.dt.month
    out["year"] = out.date.dt.year
    source = out.set_index(KEYS + ["date"])["units_ordered"]

    def at_offset(days):
        lookup = out[KEYS + ["date"]].copy()
        lookup["date"] -= pd.Timedelta(days=days)
        return source.reindex(pd.MultiIndex.from_frame(lookup)).to_numpy(dtype=float)

    horizon = config["forecast_horizon"]
    for lag in config["lags"]:
        out[f"lag_{lag}"] = at_offset(horizon + lag - 1)
    # Accumulate moments without allocating an N x window matrix.
    for window in config["rolling_windows"]:
        total = np.zeros(len(out))
        squares = np.zeros(len(out))
        for offset in range(horizon, horizon + window):
            values = at_offset(offset)
            total += values
            squares += values * values
        out[f"rolling_mean_{window}"] = total / window
        out[f"rolling_std_{window}"] = np.sqrt(np.maximum(
            (squares - total * total / window) / (window - 1), 0))
    return out


def build_inference_features(payload, config):
    """Require complete history through the forecast origin, excluding later data."""
    target_date = pd.Timestamp(payload["forecast_date"])
    history = pd.DataFrame(payload["history"])
    if history.empty:
        raise ValueError("History must not be empty")
    history["date"] = pd.to_datetime(history["date"])
    if history.date.duplicated().any():
        raise ValueError("Duplicate historical dates are not allowed")
    origin = target_date - pd.Timedelta(days=config["forecast_horizon"])
    if (history.date > origin).any():
        raise ValueError("History must end on or before the forecast origin (forecast_date - horizon)")
    required = pd.date_range(target_date - pd.Timedelta(days=lookback_days(config)), origin)
    if not required.difference(history.date).empty:
        raise ValueError("History must contain all continuous daily observations in the required lookback window")
    for key in KEYS:
        history[key] = payload[key]
    future = {key: payload[key] for key in KEYS + COVARIATES}
    future.update(date=target_date, units_ordered=np.nan)
    combined = pd.concat([history, pd.DataFrame([future])], ignore_index=True)
    return generate_time_features(combined, config).tail(1)[feature_columns(config)]
