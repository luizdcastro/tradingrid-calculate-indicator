import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import app
from indicators import FEATURE_COLUMNS, indicators

SAMPLE = json.loads((Path(__file__).parent / 'sample_request.json').read_text())


@pytest.fixture
def candles():
    return pd.DataFrame(SAMPLE['data'])


def test_returns_exactly_the_76_features_in_order(candles):
    out = indicators(candles)
    assert list(out.columns) == FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 76


def test_warmup_rows_are_dropped_and_nothing_is_missing(candles):
    out = indicators(candles)
    assert 0 < len(out) < len(candles)
    assert np.isfinite(out.to_numpy()).all()


def test_features_do_not_depend_on_the_price_level(candles):
    # The model is trained on BTC, ETH and SOL together, so a feature must read the same whether a coin trades at
    # 65,000 or at 65.
    scaled = candles.copy()
    scaled[['open', 'high', 'low', 'close']] /= 1000
    pd.testing.assert_frame_equal(indicators(candles), indicators(scaled), rtol=1e-6, atol=1e-9)


def test_time_of_day_is_neutral_without_timestamps(candles):
    out = indicators(candles)
    assert (out['hour_sin'] == 0).all() and (out['hour_cos'] == 0).all()


def test_time_of_day_is_encoded_on_the_unit_circle(candles):
    timestamps = pd.date_range('2024-01-01', periods=len(candles), freq='5min').strftime('%Y-%m-%dT%H:%M:%S').tolist()
    out = indicators(candles, timestamps=timestamps)
    assert np.allclose(out['hour_sin'] ** 2 + out['hour_cos'] ** 2, 1)
    assert out['hour_sin'].nunique() > 1


def test_the_endpoint_returns_one_row_per_usable_candle():
    response = app.test_client().post('/indicator', json=SAMPLE)
    assert response.status_code == 200
    body = response.get_json()
    assert body['symbol'] == SAMPLE['symbol'] and body['interval'] == SAMPLE['interval']
    assert body['columns'] == FEATURE_COLUMNS
    assert len(body['data']) == len(indicators(pd.DataFrame(SAMPLE['data'])))
