"""Train tabular models: python -m src.train --config configs/default.yaml."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import CATEGORICAL, KEYS, extract_data, feature_columns, generate_time_features, historical_columns, lookback_days, validate_config

# Models 
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]


def make_model(name, params, seed, columns):
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    if name in {"linear_regression", "ridge", "lasso"}:
        estimator = {"linear_regression": LinearRegression, "ridge": Ridge, "lasso": Lasso}[name](**params)
    elif name == "lightgbm":
        estimator = LGBMRegressor(**{"random_state": seed, **params})
    elif name == "xgboost":
        estimator = XGBRegressor(**{"random_state": seed, **params})
    elif name == "catboost":
        estimator = CatBoostRegressor(**{"random_seed": seed, "allow_writing_files": False, **params})
    else:
        raise ValueError(f"Unsupported standalone model: {name}")
    numeric = [c for c in columns if c not in CATEGORICAL]
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return Pipeline([("features", preprocessing), ("model", estimator)])


def train(config):
    validate_config(config)
    if not config.get("models"):
        raise ValueError("At least one model is required")
    training = extract_data(pd.read_parquet(ROOT / config["train_path"]))
    evaluation = extract_data(pd.read_parquet(ROOT / config["eval_path"]))
    for frame in (training, evaluation):
        frame["date"] = pd.to_datetime(frame["date"])
        if not np.isfinite(frame.units_ordered).all() or (frame.units_ordered < 0).any():
            raise ValueError("Targets must be finite and nonnegative")
    if training.empty or evaluation.empty:
        raise ValueError("Training and evaluation data must not be empty")
    if training.date.max() >= evaluation.date.min():
        raise ValueError("Evaluation dates must follow all training dates")
    if config.get("max_series") is not None:
        count = config["max_series"]
        if type(count) is not int or count < 1:
            raise ValueError("max_series must be a positive integer or null")
        selected = training[KEYS].drop_duplicates().sort_values(KEYS).head(count)
        training = training.merge(selected, on=KEYS)
        evaluation = evaluation.merge(selected, on=KEYS)
    columns = feature_columns(config)
    historical = historical_columns(config)
    train_ready = generate_time_features(training, config).dropna(subset=historical)
    cutoff = evaluation.date.min() - pd.Timedelta(days=lookback_days(config))
    buffer = training.loc[training.date >= cutoff]
    joined = pd.concat([buffer.assign(is_eval=False), evaluation.assign(is_eval=True)], ignore_index=True)
    eval_ready = generate_time_features(joined, config)
    eval_ready = eval_ready.loc[eval_ready.is_eval].dropna(subset=historical)
    if train_ready.empty or eval_ready.empty:
        raise ValueError("Insufficient complete history for training or evaluation")
    output = ROOT / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for name in config["models"]:
        model = make_model(name, config.get("model_params", {}).get(name, {}), config["seed"], columns)
        start = time.perf_counter()
        model.fit(train_ready[columns], train_ready.units_ordered)
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        prediction = np.maximum(model.predict(eval_ready[columns]), 0)
        inference_seconds = time.perf_counter() - start
        actual = eval_ready.units_ordered.to_numpy()
        result = {"model": name, "RMSE": float(np.sqrt(np.mean((actual - prediction) ** 2))),
                  "WMAPE": float(100 * np.abs(actual - prediction).sum() / actual.sum()) if actual.sum() else None,
                  "train_seconds": train_seconds, "inference_seconds": inference_seconds,
                  "train_rows": len(train_ready), "eval_rows": len(eval_ready)}
        folder = output / name
        folder.mkdir(exist_ok=True)
        artifact = {"format_version": 1, "model_name": name, "model": model,
                    "feature_cols": columns, "config": config, "metrics": result}
        joblib.dump(artifact, folder / "artifact.joblib")
        (folder / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
    leaderboard = pd.DataFrame(results).sort_values(["WMAPE", "RMSE"], na_position="last")
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    (output / "config.yaml").write_text(yaml.safe_dump(config))
    print(leaderboard.to_string(index=False))
    return leaderboard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--max-series", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.max_series is not None:
        config["max_series"] = args.max_series
    train(config)


if __name__ == "__main__":
    main()
