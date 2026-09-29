import copy
import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from src.features import feature_columns, generate_time_features
from src.predict import ForecastRequest, Predictor
from src.train import make_model


@pytest.fixture
def artifact(tmp_path, frame, config):
    columns = feature_columns(config)
    ready = generate_time_features(frame, config).dropna(subset=columns)
    model = make_model("ridge", {}, 42, columns)
    model.fit(ready[columns], ready.units_ordered)
    path = tmp_path / "artifact.joblib"
    joblib.dump({"format_version": 1, "model_name": "ridge", "model": model,
                 "config": config, "feature_cols": columns}, path)
    return path


def test_health_and_prediction_round_trip(artifact, payload):
    with TestClient(create_app(artifact)) as client:
        assert client.get("/health").json()["forecast_horizon"] == 7
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert np.isfinite(response.json()["predicted_units"])
        assert response.json() == Predictor(artifact).predict(ForecastRequest(**payload))


@pytest.mark.parametrize("kind", ["short", "gap", "duplicate", "future", "stale"])
def test_invalid_history(artifact, payload, kind):
    payload = copy.deepcopy(payload)
    if kind == "short":
        payload["history"] = payload["history"][-2:]
    elif kind == "gap":
        del payload["history"][30]
    elif kind == "duplicate":
        payload["history"].append(payload["history"][-1])
    elif kind == "future":
        payload["history"][-1]["date"] = payload["forecast_date"]
    else:
        payload["history"].pop()
    with TestClient(create_app(artifact)) as client:
        assert client.post("/predict", json=payload).status_code == 400


def test_invalid_target_and_missing_fields(artifact, payload):
    with TestClient(create_app(artifact)) as client:
        payload["history"][0]["units_ordered"] = -1
        assert client.post("/predict", json=payload).status_code == 422
        assert client.post("/predict", json={}).status_code == 422


def test_unseen_category(artifact, payload):
    payload["product_category"] = 9999
    with TestClient(create_app(artifact)) as client:
        assert client.post("/predict", json=payload).status_code == 200


def test_missing_artifact_fails_startup(tmp_path):
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(tmp_path / "missing.joblib")):
            pass
