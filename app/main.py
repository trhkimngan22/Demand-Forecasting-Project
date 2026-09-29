"""Run from the repository root with: uvicorn app.main:app."""
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from src.predict import ForecastRequest, Predictor

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "saved_models/standalone/lightgbm/artifact.joblib"
logger = logging.getLogger(__name__)


def create_app(model_path=None):
    @asynccontextmanager
    async def lifespan(application):
        # Fail at startup if the artifact is missing or incompatible.
        application.state.predictor = Predictor(model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL))
        yield

    application = FastAPI(title="Demand Forecasting API", version="1.0.0", lifespan=lifespan)

    @application.get("/health")
    def health():
        artifact = application.state.predictor.artifact
        return {"status": "healthy", "model": artifact["model_name"],
                "forecast_horizon": artifact["config"]["forecast_horizon"]}

    @application.post("/predict")
    def predict(request: ForecastRequest):
        try:
            return application.state.predictor.predict(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Forecast failed")
            raise HTTPException(status_code=500, detail="Prediction failed") from exc

    return application


app = create_app()
