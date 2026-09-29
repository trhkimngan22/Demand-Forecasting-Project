import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def config():
    return {"forecast_horizon": 7, "lags": [1, 7, 14, 21, 28],
            "rolling_windows": [7, 14], "seed": 42}


@pytest.fixture
def frame():
    n = 70
    return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n),
                         "units_ordered": np.arange(n, dtype=float),
                         "product_id": 38, "store_id": 0, "product_category": 5,
                         "product_subcategory": 6, "location": 0, "discount": 1.0,
                         "is_promotion": 0, "holiday": 0, "activity_flag": 0,
                         "temperature": 20.0, "humidity": 70.0,
                         "precipitation": 0.0, "wind": 2.0})


@pytest.fixture
def payload(frame):
    row = frame.iloc[50].to_dict()
    row.pop("date")
    row.pop("units_ordered")
    row["forecast_date"] = "2024-02-20"
    row["history"] = [{"date": r.date.strftime("%Y-%m-%d"), "units_ordered": r.units_ordered}
                      for r in frame.iloc[:44].itertuples()]
    return row
