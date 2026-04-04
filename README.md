# tradingrid_calculate_indicator

Feature engineering microservice for a PyTorch LSTM scalping model.  
Receives raw OHLCV candle data and returns 76 normalised, dimensionless technical indicators ready for model input.

---

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Production (gunicorn):
```bash
gunicorn app:app
```

---

## Endpoint

### `POST /indicator`

Computes all technical indicators for a given candle window and returns the clean feature matrix.

**Request body**

```json
{
  "symbol": "BTCUSDT",
  "interval": "5m",
  "timestamps": ["2024-01-01T08:00:00", "2024-01-01T08:05:00", "..."],
  "data": {
    "open":   [30000.0, 30010.0, "..."],
    "high":   [30050.0, 30060.0, "..."],
    "low":    [29990.0, 30000.0, "..."],
    "close":  [30020.0, 30040.0, "..."],
    "volume": [12.5, 14.2, "..."]
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `symbol` | string | yes | Trading pair (e.g. `BTCUSDT`) |
| `interval` | string | yes | Candle timeframe (e.g. `5m`, `15m`) |
| `data.open` | float[] | yes | Open prices |
| `data.high` | float[] | yes | High prices |
| `data.low` | float[] | yes | Low prices |
| `data.close` | float[] | yes | Close prices |
| `data.volume` | float[] | yes | Volumes |
| `timestamps` | string[] | no | ISO-format datetime per candle — enables `hour_sin`/`hour_cos` session features. When omitted both are set to `0`. |

> Minimum recommended input: **200 candles**. The warmup period is ~63 bars (dominated by TEMA-21), so those rows are dropped automatically. Send 500 candles for a full LSTM window.

---

**Response body**

```json
{
  "symbol": "BTCUSDT",
  "interval": "5m",
  "columns": ["return_1", "return_3", "..."],
  "data": [
    { "return_1": 0.0012, "return_3": 0.0031, "rsi_14": 0.58, "..." },
    { "..." }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `symbol` | string | Echoed from request |
| `interval` | string | Echoed from request |
| `columns` | string[] | Ordered list of all 76 feature names |
| `data` | object[] | One object per clean candle, keyed by feature name |

To build the LSTM input tensor: `data` rows × `columns` order = shape `(window, 76)`.

---

## Feature columns (76 total)

### Price action (8)

| Column | Range | Description |
|---|---|---|
| `return_1` | ±5% | 1-candle close return, clipped |
| `return_3` | ±5% | 3-candle close return, clipped |
| `return_5` | ±5% | 5-candle close return, clipped |
| `hl_range` | ≥0 | (high − low) / close |
| `body_ratio` | [−1, 1] | (close − open) / (high − low); 0 on doji |
| `upper_shadow` | [0, 1] | Upper wick / candle range; 0 on doji |
| `lower_shadow` | [0, 1] | Lower wick / candle range; 0 on doji |
| `volume_ratio` | ≥0 | Volume / 20-period rolling mean volume |

### Trend deviations (13)

All values are ratios: `price / MA − 1`. Positive = price above MA, negative = below.

| Column | Description |
|---|---|
| `close_ema9_dev` | close / EMA(9) − 1 |
| `close_ema21_dev` | close / EMA(21) − 1 |
| `close_ema50_dev` | close / EMA(50) − 1 |
| `ema9_ema21_dev` | EMA(9) / EMA(21) − 1 |
| `ema21_ema50_dev` | EMA(21) / EMA(50) − 1 |
| `close_kama_dev` | close / KAMA(30) − 1 |
| `close_dema_dev` | close / DEMA(21) − 1 |
| `close_tema_dev` | close / TEMA(21) − 1 |
| `vwap_dev` | close / rolling VWAP(20) − 1 |
| `close_mama_dev` | close / MAMA − 1 (MESA Adaptive MA) |
| `mama_fama_dev` | MAMA / FAMA − 1 |
| `close_high50_dev` | close / 50-bar high − 1 (≤ 0, proximity to resistance) |
| `close_low50_dev` | close / 50-bar low − 1 (≥ 0, proximity to support) |

### Trend strength (7)

| Column | Range | Description |
|---|---|---|
| `adx` | [0, 1] | ADX(14) / 100 |
| `plus_di` | [0, 1] | +DI(14) / 100 |
| `minus_di` | [0, 1] | −DI(14) / 100 |
| `supertrend` | {0, 1} | SuperTrend binary (1 = uptrend) |
| `aroon_osc` | [−1, 1] | Aroon oscillator(14) / 100 |
| `linearreg_slope` | float | LINEARREG_SLOPE(14) / close |
| `ht_trendmode` | {0, 1} | Hilbert Transform: 1 = trending, 0 = cycling |

### Momentum (13)

| Column | Range | Description |
|---|---|---|
| `rsi_7` | [0, 1] | RSI(7) / 100 |
| `rsi_14` | [0, 1] | RSI(14) / 100 |
| `stoch_k` | [0, 1] | Stochastic %K / 100 |
| `stoch_d` | [0, 1] | Stochastic %D / 100 |
| `stochrsi_k` | [0, 1] | StochRSI %K / 100 — RSI within its own range, more sensitive |
| `stochrsi_d` | [0, 1] | StochRSI %D / 100 |
| `willr` | [0, 1] | Williams %R shifted: (WILLR + 100) / 100 |
| `cci` | [−3, 3] | CCI(14) / 100, clipped at ±3 |
| `macd_hist_norm` | float | MACD histogram / close |
| `ppo` | float | PPO(12,26) / 100 |
| `rocp_10` | ±5% | Rate of change proportion(10), clipped |
| `cmo` | [−1, 1] | Chande Momentum Oscillator(14) / 100 |
| `ultosc` | [0, 1] | Ultimate Oscillator(7,14,28) / 100 |

### Volatility (8)

| Column | Range | Description |
|---|---|---|
| `natr` | [0, 1] | Normalised ATR(14) / 100 |
| `bb_position` | [0, 1] | (close − lower BB) / BB range |
| `bb_width` | ≥0 | BB range / middle band |
| `bb_squeeze` | {0, 1} | 1 when BB is inside Keltner Channel — consolidation before breakout |
| `atr7_close` | ≥0 | ATR(7) / close |
| `stddev5_close` | ≥0 | STDDEV(5) / close |
| `trange_ratio` | ≥0 | Single-candle True Range / close |
| `donchian_position` | [0, 1] | (close − 20-bar low) / (20-bar high − 20-bar low) |

### Volume (5)

| Column | Range | Description |
|---|---|---|
| `mfi` | [0, 1] | Money Flow Index(14) / 100 |
| `obv_roc5` | [−1, 1] | OBV 5-period % change, clipped |
| `ad_roc5` | [−1, 1] | A/D line 5-period % change, clipped |
| `adosc_norm` | [−1, 1] | ADOSC(3,10) 1-period % change, clipped |
| `volume_roc5` | [−1, 1] | Volume 5-period % change, clipped |

### Candlestick patterns (20)

All patterns return `{−1, 0, 1}` (TA-Lib −100/0/100 divided by 100).  
`1` = bullish, `−1` = bearish, `0` = no pattern.

| Column | Pattern |
|---|---|
| `cdlhammer` | Hammer |
| `cdlinvertedhammer` | Inverted Hammer |
| `cdlengulfing` | Engulfing |
| `cdlpiercing` | Piercing Line |
| `cdlmorningstar` | Morning Star |
| `cdl3whitesoldiers` | Three White Soldiers |
| `cdlabandonedbaby` | Abandoned Baby |
| `cdlbreakaway` | Breakaway |
| `cdlhangingman` | Hanging Man |
| `cdlshootingstar` | Shooting Star |
| `cdleveningstar` | Evening Star |
| `cdl3blackcrows` | Three Black Crows |
| `cdldarkcloudcover` | Dark Cloud Cover |
| `cdldoji` | Doji |
| `cdlspinningtop` | Spinning Top |
| `cdlharami` | Harami |
| `cdl3linestrike` | Three-Line Strike |
| `cdldragonflydoji` | Dragonfly Doji |
| `cdlmatchinglow` | Matching Low |
| `cdltasukigap` | Tasuki Gap |

### Time encoding (2)

| Column | Range | Description |
|---|---|---|
| `hour_sin` | [−1, 1] | sin(2π × hour / 24) — requires `timestamps` in request |
| `hour_cos` | [−1, 1] | cos(2π × hour / 24) — requires `timestamps` in request |

> When `timestamps` is not sent both values are `0` (neutral). Provide timestamps to let the model learn Asia/Europe/US session patterns.

---

## LSTM input

```python
import requests
import numpy as np

response = requests.post('http://localhost:5000/indicator', json={
    'symbol': 'BTCUSDT',
    'interval': '5m',
    'timestamps': ['2024-01-01T08:00:00', '2024-01-01T08:05:00', '...'],  # optional
    'data': { 'open': [...], 'high': [...], 'low': [...], 'close': [...], 'volume': [...] }
})

payload = response.json()
columns = payload['columns']   # 76 feature names in correct order
rows    = payload['data']      # list of dicts, one per candle

# Build input tensor shaped (window, 76)
X = np.array([[row[col] for col in columns] for row in rows], dtype=np.float32)
```
