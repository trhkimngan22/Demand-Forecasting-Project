# Demand Forecasting Project

A retail demand forecasting project for individual product–store pairs, using FreshRetailNet-50K. The notebook covers data preparation, feature engineering, model comparison, model persistence, and an inference demo with FastAPI.

**LightGBM** achieves the lowest WMAPE in the saved results and is selected for the API demo. The original experiments are preserved in [DemandForecastingProject.ipynb](src/DemandForecastingProject.ipynb). A standalone tabular pipeline and API are also available below.

## Repository structure

```text
Demand-Forecasting-Project/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/default.yaml
├── data/
│   ├── README.md
│   ├── train.parquet          
│   └── eval.parquet            
├── src/
│   ├── __init__.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   └── DemandForecastingProject.ipynb
├── app/
│   ├── __init__.py
│   └── main.py
└── tests/
    ├── conftest.py
    ├── test_features.py
    ├── test_api.py
    └── test_train.py
```

Generated models, virtual environments and caches are excluded from Git. See [data/README.md](data/README.md) for dataset preparation.

## Standalone training and API

The standalone pipeline supports Linear Regression, Ridge, Lasso, XGBoost, LightGBM, and CatBoost. LightGBM is the default. LSTM, TimesFM, EDA, and the original MLflow experiment remain in the research notebook.

### Install and train

Use Python 3.11 or 3.12. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.train --config configs/default.yaml --max-series 20
```

Remove `--max-series 20` to train on all series. Edit `models` in [configs/default.yaml](configs/default.yaml) to compare multiple tabular models. Data and output paths in the configuration are resolved relative to the repository root.

The standalone workflow uses shared calendar-date features, excludes target-date stockout status, and fits preprocessing only on training rows. All selected models are evaluated on the same rows with complete historical features. Missing numeric covariates are imputed from training medians, categorical variables are one-hot encoded with support for unseen categories. This preprocessing differs from the notebook, so its saved leaderboard does not describe standalone results.

Evaluation uses a rolling forecast origin at `target_date - forecast_horizon`. For evaluation periods longer than the horizon, earlier evaluation observations can become available history. This is not a fixed-origin forecast of an arbitrary length. Historical features require complete daily windows, rows without enough history are excluded and retained row counts are recorded in the metrics.

Training writes:

```text
saved_models/standalone/
├── config.yaml
├── leaderboard.csv
└── lightgbm/
    ├── artifact.joblib
    └── metrics.json
```

Each configured model gets its own directory. The artifact contains the fitted preprocessing pipeline, estimator, feature columns, configuration, and metrics. Only load trusted artifacts. Notebook artifacts use a different format and must be retrained with `src.train` before use with this API.

### Run the API

After training LightGBM:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive schema. `GET /health` reports the loaded model. `POST /predict` returns one prediction for the requested date. Set `MODEL_PATH` to a different standalone `artifact.joblib` if needed. The server fails at startup when the artifact is missing or incompatible.

Generate a complete example request from the repository root:

```python
import json
from datetime import date, timedelta
from pathlib import Path

forecast_date = date(2024, 7, 10)
request = {
    "product_id": 38, "store_id": 0,
    "forecast_date": forecast_date.isoformat(),
    "product_category": 5, "product_subcategory": 6, "location": 0,
    "discount": 1.0, "is_promotion": 0, "holiday": 0, "activity_flag": 0,
    "temperature": 29.0, "humidity": 78.0, "precipitation": 0.0, "wind": 2.0,
    "history": [
        {"date": (forecast_date - timedelta(days=offset)).isoformat(),
         "units_ordered": 2.0}
        for offset in range(34, 6, -1)
    ],
}
Path("/tmp/forecast-request.json").write_text(json.dumps(request))
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/forecast-request.json

python -m src.predict \
  --model saved_models/standalone/lightgbm/artifact.joblib \
  --request /tmp/forecast-request.json
```

The sample history is synthetic. Replace it and the covariates with actual inputs. With the default configuration, history must cover every day from **34 days before the forecast date through 7 days before it**, inclusive: 28 observations. Older history is accepted. Observations after the forecast origin are rejected. Unlike the notebook request, each historical item needs only `date` and `units_ordered`. Future promotions and weather must be supplied using information available at the forecast origin. Schema errors return HTTP 422, invalid history returns HTTP 400.

### Run tests

```bash
python -m pytest -q
```

Tests cover calendar offsets, rolling statistics, gaps, series isolation, future-target leakage, training/inference feature equality, model serialization, and API validation. They also exercise all six supported estimators and a Parquet-to-artifact training run. Tests use synthetic data, so no dataset download or pretrained checkpoint is required.

## Dataset and forecasting task

The notebook loads the dataset with `load_dataset("Dingdong-Inc/FreshRetailNet-50K")` and uses its existing `train` and `eval` splits directly. The saved output reports **4,500,000 training rows**, **350,000 evaluation rows**, and approximately **50,000 product–store time series**.

| Component | Usage in the notebook |
| --- | --- |
| Series identifiers | `product_id`, `store_id` |
| Timestamp | `dt`, converted to `date` |
| Target | `sale_amount`, renamed to `units_ordered` |
| Product information | `first_category_id`, `second_category_id` |
| Location | `city_id`, renamed to `location` |
| Promotions | `discount`, `is_promotion`, `activity_flag` |
| Calendar | Day of week, week of year, month, year, and holidays |
| Weather | Temperature, humidity, precipitation, and wind |
| Stockout status | `stockout_flag`, derived from `stock_hour6_22_cnt` |

`units_ordered` is the internal name for `sale_amount`, which contains fractional values. The notebook predicts observed sales.

## Workflow and models

1. Load data, standardize column names, and generate calendar features.
2. Check missing values and compute average autocorrelation across series.
3. Generate lag features and rolling statistics for each product–store pair.
4. Train or initialize eight models and record experiment parameters with MLflow.
5. Evaluate RMSE and WMAPE, then save models and the leaderboard.
6. Analyze feature importance, residuals, and actual versus predicted values.
7. Reload models, measure inference time, and create a FastAPI demo.

| Family | Models compared |
| --- | --- |
| Linear regression | Linear Regression, Ridge, Lasso |
| Gradient boosting | XGBoost, LightGBM, CatBoost |
| Deep learning | Two LSTM layers with 64 and 32 units, dropout of 0.2, trained for 3 epochs |
| Foundation model | Pretrained TimesFM 2.5 using `google/timesfm-2.5-200m-pytorch`, without fine-tuning |

### Temporal configuration

| Parameter | Default value |
| --- | --- |
| `FORECAST_HORIZON` | 7 |
| `SEQUENCE_LENGTH` | 30 |
| `LAGS` | `[1, 7, 14, 21, 28]` |
| `ROLLING_WINDOWS` | `[7, 14]` |
| `LOOKBACK_DAYS` | 34 |
| `SEED` | 42 |

For tabular models, `lag_k` is computed with `shift(k + FORECAST_HORIZON - 1)`. With a horizon of 7, `lag_1` uses the observation 7 rows earlier and `lag_28` uses the observation 34 rows earlier within the same series. Rolling means and standard deviations are computed after shifting the target by 7 rows. These row offsets correspond to days when each series contains continuous daily observations.

## Saved results

The following values come from the notebook's saved **Inference demo** leaderboard. Inference time measures evaluation processing within the corresponding loop, rather than the latency of a single API request.

| Model | WMAPE (%) | RMSE | Inference time (seconds) |
| --- | ---: | ---: | ---: |
| **LightGBM** | **35.5079** | **0.7128** | **52.294** |
| TimesFM | 35.5721 | 0.7286 | 5,083.939 |
| XGBoost | 35.7061 | 0.7518 | 53.327 |
| CatBoost | 36.0978 | 0.7295 | 52.780 |
| Ridge | 36.7617 | 0.7222 | 53.817 |
| Linear Regression | 36.7617 | 0.7222 | 53.075 |
| Lasso | 56.0460 | 1.4706 | 52.223 |
| LSTM | 79.7989 | 1.6917 | 51.195 |

- **RMSE:** the square root of the mean squared prediction error.
- **WMAPE:** `100 × sum(abs(y_true - y_pred)) / sum(y_true)`. The evaluation function returns `NaN` when the target sum is zero.

LightGBM has the lowest errors in this table and substantially lower inference time than TimesFM in the saved run.

**Interpretation limits:** Evaluation is not fully aligned across model families. Tabular models use features shifted by the forecast horizon, TimesFM predicts the entire evaluation segment from training history, LSTM appends only 30 historical days, so a horizon of 7 excludes early evaluation dates with insufficient context. Tabular models also use target-date `stockout_flag` and evaluation-set weather and promotion variables. Their availability at forecast time must be established before treating these results as evidence of deployment performance.

## Running the notebook

### 1. Set up the environment

The notebook metadata records Python **3.12.13**. From the repository root, create a Python 3.12 environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyterlab ipykernel datasets pandas numpy pyarrow \
  scikit-learn xgboost lightgbm catboost statsmodels pmdarima \
  ydata-profiling matplotlib seaborn tensorflow torch timesfm \
  mlflow joblib fastapi pydantic httpx
python -m ipykernel install --user --name demand-forecasting \
  --display-name "Python (Demand Forecasting)"
python -m jupyterlab
```

This dependency list covers the notebook's imports and Parquet loading. Versions are not pinned, and installation has not been verified across operating systems. TimesFM requires a version exposing `TimesFM_2p5_200M_torch` and `ForecastConfig`, as used in the notebook.

The dataset contains millions of rows, and processing creates additional copies in memory. Running the full workflow, particularly LSTM and TimesFM, can require substantial RAM and execution time.

### 2. Choose a data source

**Default:** keep the cells that call `load_dataset(...)` and convert both splits to pandas.

**Local Parquet files:** replace the dataset-loading and split-conversion cells with the following code. It handles a working directory at either the repository root or `src/`:

```python
from pathlib import Path

repo_root = Path.cwd()
if not (repo_root / "data" / "train.parquet").exists():
    repo_root = repo_root.parent

df_train = pd.read_parquet(repo_root / "data" / "train.parquet")
df_eval = pd.read_parquet(repo_root / "data" / "eval.parquet")
```

### 3. Run the notebook

Open `src/DemandForecastingProject.ipynb`, select the **Python (Demand Forecasting)** kernel, and execute the cells in order. You can skip the `!pip install ...` cell if the dependencies are already installed in the environment above.

To try LightGBM first, set `candidate_models = ["lightgbm"]` in **both locations**: before the training loop and in the **Inference demo** section. Changing only the training list leaves the reload loop attempting to load models that have not been saved.

### 4. Inspect the outputs

```text
saved_models/
├── leaderboard.csv
├── best_model.joblib
├── lightgbm/
│   ├── model.joblib
│   └── metadata.joblib
├── lstm/
│   ├── model.keras
│   └── metadata.joblib
└── timesfm/
    └── metadata.joblib
```

This tree illustrates the artifact types. Other tabular models each receive their own directory. `best_model.joblib` stores the top leaderboard row's information, **not the estimator**. TimesFM saves only metadata and reloads its pretrained checkpoint when restored.

MLflow creates the `Demand_Forecasting_Project` experiment and records parameters through `mlflow.log_param` and `mlflow.log_params`.

## Original notebook FastAPI demo

The application is defined inside the notebook and loads LightGBM from `saved_models/lightgbm/`.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Return the service status, model name, and forecast horizon |
| `POST /predict` | Predict demand for a specified product, store, and date |

A prediction request includes product/store information, `forecast_date`, promotion and weather variables, and `history`. History must contain at least **34 continuous daily observations** before the forecast date. The validation also requires the latest historical date to be no earlier than `forecast_date - 7 days`.

The response contains `product_id`, `store_id`, `forecast_date`, `forecast_horizon`, and `predicted_units`. The endpoint returns **one prediction for the requested date**, rather than an array of seven daily forecasts.

The existing test cell uses `TestClient(app)` to call `GET /health`, with a saved HTTP 200 response. The notebook's `/predict` JSON example contains abbreviated history and cannot be submitted as a complete request. This describes the original notebook demo. The standalone `app/main.py` application and `/predict` tests are documented above.
