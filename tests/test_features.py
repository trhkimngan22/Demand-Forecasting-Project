import numpy as np
import pandas as pd
import pytest
from src.features import build_inference_features, feature_columns, generate_time_features


def test_expected_calendar_lags_and_statistics(frame, config):
    row = generate_time_features(frame, config).iloc[50]
    assert row.lag_1 == 43
    assert row.lag_28 == 16
    assert row.rolling_mean_7 == 40
    assert row.rolling_std_7 == pytest.approx(np.std(np.arange(37, 44), ddof=1))
    assert "stockout_flag" not in feature_columns(config)


def test_api_matches_training_across_forecast_gap(frame, payload, config):
    expected = generate_time_features(frame, config).iloc[[50]][feature_columns(config)]
    actual = build_inference_features(payload, config)
    np.testing.assert_allclose(actual.to_numpy(dtype=float), expected.to_numpy(dtype=float))


def test_missing_day_does_not_shift_other_dates(frame, config):
    frame = frame.drop(index=42)
    result = generate_time_features(frame, config)
    row = result.loc[result.date.eq(pd.Timestamp("2024-02-20"))].iloc[0]
    assert row.lag_1 == 43
    assert np.isnan(row.rolling_mean_7)


def test_series_are_isolated_and_order_independent(frame, config):
    other = frame.assign(store_id=1, units_ordered=frame.units_ordered + 1000)
    mixed = pd.concat([frame, other]).sample(frac=1, random_state=42)
    result = generate_time_features(mixed, config)
    result = result.loc[result.date.eq(pd.Timestamp("2024-02-20"))].set_index("store_id")
    assert result.loc[0, "lag_1"] == 43
    assert result.loc[1, "lag_1"] == 1043


def test_future_targets_cannot_change_features(frame, config):
    original = generate_time_features(frame, config).iloc[50][feature_columns(config)]
    frame.loc[44:, "units_ordered"] = 99999
    modified = generate_time_features(frame, config).iloc[50][feature_columns(config)]
    pd.testing.assert_series_equal(original, modified)


def test_duplicate_dates_rejected(frame, config):
    with pytest.raises(ValueError, match="Duplicate"):
        generate_time_features(pd.concat([frame, frame.iloc[[0]]]), config)
