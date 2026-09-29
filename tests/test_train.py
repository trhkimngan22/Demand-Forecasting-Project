import json
import pandas as pd
from src.predict import ForecastRequest, Predictor
from src.train import train
import numpy as np
from src.features import feature_columns, generate_time_features
from src.train import make_model


def test_train_save_reload_from_parquet(tmp_path, frame, config, payload):
    paths = {"train_path": str(tmp_path / "train.parquet"),
             "eval_path": str(tmp_path / "eval.parquet"),
             "output_dir": str(tmp_path / "models")}
    frame.iloc[:50].to_parquet(paths["train_path"])
    frame.iloc[50:57].to_parquet(paths["eval_path"])
    config.update(paths, models=["ridge"], model_params={"ridge": {"alpha": 1.0}})
    leaderboard = train(config)
    assert leaderboard.iloc[0].eval_rows == 7
    assert leaderboard.iloc[0].train_rows == 16
    artifact = tmp_path / "models/ridge/artifact.joblib"
    result = Predictor(artifact).predict(ForecastRequest(**payload))
    assert result["predicted_units"] >= 0
    metrics = json.loads((artifact.parent / "metrics.json").read_text())
    assert metrics["RMSE"] >= 0
    assert len(pd.read_csv(tmp_path / "models/leaderboard.csv")) == 1


def test_all_supported_estimators(frame, config):
    columns = feature_columns(config)
    ready = generate_time_features(frame, config).dropna(subset=columns)
    for name, params in [
        ("linear_regression", {}), ("ridge", {}), ("lasso", {}),
        ("lightgbm", {"n_estimators": 2, "verbosity": -1}),
        ("xgboost", {"n_estimators": 2}),
        ("catboost", {"iterations": 2, "verbose": False}),
    ]:
        model = make_model(name, params, 42, columns)
        model.fit(ready[columns], ready.units_ordered)
        assert np.isfinite(model.predict(ready[columns])).all(), name
