"""Load a standalone artifact and predict from a JSON request."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from datetime import date
from src.features import build_inference_features, validate_config


class HistoricalSale(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    date: date
    units_ordered: float = Field(ge=0)


class ForecastRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    product_id: int
    store_id: int
    forecast_date: date
    product_category: int
    product_subcategory: int
    location: int
    discount: float = Field(ge=0)
    is_promotion: int = Field(ge=0, le=1)
    holiday: int = Field(ge=0, le=1)
    activity_flag: int = Field(ge=0, le=1)
    temperature: float
    humidity: float
    precipitation: float
    wind: float
    history: list[HistoricalSale] = Field(min_length=1)


class Predictor:
    def __init__(self, artifact_path):
        # Load only artifacts produced by a trusted training run.
        self.artifact = joblib.load(artifact_path)
        if self.artifact.get("format_version") != 1:
            raise ValueError("Expected a standalone artifact with format_version=1; retrain with src.train")
        validate_config(self.artifact["config"])

    def predict(self, request):
        config = self.artifact["config"]
        features = build_inference_features(request.model_dump(), config)
        value = float(self.artifact["model"].predict(features[self.artifact["feature_cols"]])[0])
        if not np.isfinite(value):
            raise RuntimeError("Model returned a non-finite prediction")
        return {"product_id": request.product_id, "store_id": request.store_id,
                "forecast_date": request.forecast_date.isoformat(),
                "forecast_horizon": config["forecast_horizon"], "predicted_units": max(value, 0.0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = ForecastRequest.model_validate_json(args.request.read_text())
    print(json.dumps(Predictor(args.model).predict(request), indent=2))


if __name__ == "__main__":
    main()
