import numpy as np
import pandas as pd
import talib

pd.options.mode.chained_assignment = None

# Every column returned by indicators(). Use len(FEATURE_COLUMNS) as LSTM input size.
FEATURE_COLUMNS = [
    # Price action (8)
    'return_1', 'return_3', 'return_5',
    'hl_range', 'body_ratio', 'upper_shadow', 'lower_shadow',
    'volume_ratio',
    # Trend deviations (13)
    'close_ema9_dev', 'close_ema21_dev', 'close_ema50_dev',
    'ema9_ema21_dev', 'ema21_ema50_dev',
    'close_kama_dev', 'close_dema_dev', 'close_tema_dev',
    'vwap_dev',
    'close_mama_dev', 'mama_fama_dev',
    'close_high50_dev', 'close_low50_dev',
    # Trend strength (7)
    'adx', 'plus_di', 'minus_di', 'supertrend', 'aroon_osc',
    'linearreg_slope', 'ht_trendmode',
    # Momentum (13)
    'rsi_7', 'rsi_14',
    'stoch_k', 'stoch_d',
    'stochrsi_k', 'stochrsi_d',
    'willr', 'cci', 'macd_hist_norm', 'ppo', 'rocp_10', 'cmo', 'ultosc',
    # Volatility (8)
    'natr', 'bb_position', 'bb_width', 'bb_squeeze',
    'atr7_close', 'stddev5_close', 'trange_ratio',
    'donchian_position',
    # Volume (5)
    'mfi', 'obv_roc5', 'ad_roc5', 'adosc_norm', 'volume_roc5',
    # Candlestick patterns (20)
    'cdlhammer', 'cdlinvertedhammer', 'cdlengulfing', 'cdlpiercing',
    'cdlmorningstar', 'cdl3whitesoldiers', 'cdlabandonedbaby', 'cdlbreakaway',
    'cdlhangingman', 'cdlshootingstar', 'cdleveningstar', 'cdl3blackcrows',
    'cdldarkcloudcover', 'cdldoji', 'cdlspinningtop', 'cdlharami',
    'cdl3linestrike', 'cdldragonflydoji', 'cdlmatchinglow', 'cdltasukigap',
    # Time encoding (2) — filled with 0 when timestamps not provided
    'hour_sin', 'hour_cos',
]  # 76 features total


def _supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    high  = df['high'].values.astype(float)
    low   = df['low'].values.astype(float)
    close = df['close'].values.astype(float)

    atr   = talib.ATR(high, low, close, timeperiod=period)
    hl2   = (high + low) / 2.0
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    in_uptrend = np.full(len(df), np.nan)

    for i in range(1, len(df)):
        if np.isnan(atr[i]) or np.isnan(atr[i - 1]):
            continue
        if np.isnan(in_uptrend[i - 1]):
            in_uptrend[i] = 1.0
            continue

        if close[i] > upper[i - 1]:
            in_uptrend[i] = 1.0
        elif close[i] < lower[i - 1]:
            in_uptrend[i] = 0.0
        else:
            in_uptrend[i] = in_uptrend[i - 1]
            if in_uptrend[i] == 1.0 and lower[i] < lower[i - 1]:
                lower[i] = lower[i - 1]
            if in_uptrend[i] == 0.0 and upper[i] > upper[i - 1]:
                upper[i] = upper[i - 1]

    return pd.Series(in_uptrend, index=df.index)


def indicators(df: pd.DataFrame, timestamps=None) -> pd.DataFrame:
    """
    Compute 76 normalised, dimensionless features for an LSTM scalping model.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: open, high, low, close, volume.
    timestamps : list[str] | None
        Optional ISO-format timestamps matching each row (e.g. '2024-01-01T08:00:00').
        When provided, hour_sin and hour_cos encode time-of-day patterns.
        When None, hour_sin and hour_cos are set to 0 (no temporal signal).

    Returns
    -------
    pd.DataFrame
        FEATURE_COLUMNS only, warmup rows dropped, index reset.
    """
    df = df.copy().reset_index(drop=True)

    o = df['open'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    c = df['close'].values.astype(float)
    v = df['volume'].values.astype(float)

    # ── Price action ─────────────────────────────────────────────────────────
    df['return_1'] = df['close'].pct_change(1).clip(-0.05, 0.05)
    df['return_3'] = df['close'].pct_change(3).clip(-0.05, 0.05)
    df['return_5'] = df['close'].pct_change(5).clip(-0.05, 0.05)

    hl = (df['high'] - df['low']).replace(0, np.nan)
    df['hl_range']     = hl / df['close']
    # fillna(0): when high==low (doji), body and shadows are zero by definition
    df['body_ratio']   = ((df['close'] - df['open']) / hl).fillna(0)
    df['upper_shadow'] = ((df['high'] - df[['open', 'close']].max(axis=1)) / hl).fillna(0)
    df['lower_shadow'] = ((df[['open', 'close']].min(axis=1) - df['low']) / hl).fillna(0)

    vol_mean = df['volume'].rolling(20).mean().replace(0, np.nan)
    df['volume_ratio'] = df['volume'] / vol_mean

    # ── Trend deviations ─────────────────────────────────────────────────────
    ema9  = talib.EMA(c, timeperiod=9)
    ema21 = talib.EMA(c, timeperiod=21)
    ema50 = talib.EMA(c, timeperiod=50)
    kama  = talib.KAMA(c, timeperiod=30)
    dema  = talib.DEMA(c, timeperiod=21)
    tema  = talib.TEMA(c, timeperiod=21)
    mama, fama = talib.MAMA(c, fastlimit=0.5, slowlimit=0.05)

    df['close_ema9_dev']  = c / ema9  - 1
    df['close_ema21_dev'] = c / ema21 - 1
    df['close_ema50_dev'] = c / ema50 - 1
    df['ema9_ema21_dev']  = ema9  / ema21 - 1
    df['ema21_ema50_dev'] = ema21 / ema50 - 1
    df['close_kama_dev']  = c / kama  - 1
    df['close_dema_dev']  = c / dema  - 1
    df['close_tema_dev']  = c / tema  - 1

    # Rolling VWAP (20-period): where price sits relative to volume-weighted consensus
    typical = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['vwap_dev'] = df['close'] / vwap - 1

    # MESA Adaptive MAs: non-lagging, cycle-aware — complementary to EMA/KAMA/TEMA
    df['close_mama_dev'] = c / mama - 1
    df['mama_fama_dev']  = mama / fama - 1

    # Distance from 50-period high/low: proximity to key support/resistance levels
    high50 = pd.Series(h).rolling(50).max().values
    low50  = pd.Series(l).rolling(50).min().values
    df['close_high50_dev'] = c / high50 - 1   # ≤ 0: how far below 50-bar high
    df['close_low50_dev']  = c / low50  - 1   # ≥ 0: how far above 50-bar low

    # ── Trend strength ────────────────────────────────────────────────────────
    df['adx']      = talib.ADX(h, l, c, timeperiod=14) / 100
    df['plus_di']  = talib.PLUS_DI(h, l, c, timeperiod=14) / 100
    df['minus_di'] = talib.MINUS_DI(h, l, c, timeperiod=14) / 100
    df['supertrend']   = _supertrend(df)                      # {0.0, 1.0}
    df['aroon_osc']    = talib.AROONOSC(h, l, timeperiod=14) / 100
    df['linearreg_slope'] = talib.LINEARREG_SLOPE(c, timeperiod=14) / c
    df['ht_trendmode']    = talib.HT_TRENDMODE(c).astype(float)  # {0, 1}

    # ── Momentum ──────────────────────────────────────────────────────────────
    df['rsi_7']  = talib.RSI(c, timeperiod=7)  / 100
    df['rsi_14'] = talib.RSI(c, timeperiod=14) / 100

    slowk, slowd = talib.STOCH(h, l, c, fastk_period=5,
                                slowk_period=3, slowk_matype=0,
                                slowd_period=3, slowd_matype=0)
    df['stoch_k'] = slowk / 100
    df['stoch_d'] = slowd / 100

    # StochRSI: RSI measured within its own range — more sensitive than RSI or Stoch alone
    stochrsi_k, stochrsi_d = talib.STOCHRSI(c, timeperiod=14,
                                              fastk_period=5, fastd_period=3,
                                              fastd_matype=0)
    df['stochrsi_k'] = stochrsi_k / 100
    df['stochrsi_d'] = stochrsi_d / 100

    # WILLR: TA-Lib returns [-100, 0] → shift to [0, 1]
    df['willr'] = (talib.WILLR(h, l, c, timeperiod=14) + 100) / 100

    # CCI: divide by 100, clip extreme outliers (raw ±300 → normalised ±3)
    df['cci'] = (talib.CCI(h, l, c, timeperiod=14) / 100).clip(-3.0, 3.0)

    _, _, macdhist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd_hist_norm'] = macdhist / c

    df['ppo']     = talib.PPO(c, fastperiod=12, slowperiod=26, matype=0) / 100
    df['rocp_10'] = talib.ROCP(c, timeperiod=10).clip(-0.05, 0.05)
    df['cmo']     = talib.CMO(c, timeperiod=14) / 100
    df['ultosc']  = talib.ULTOSC(h, l, c, timeperiod1=7, timeperiod2=14, timeperiod3=28) / 100

    # ── Volatility ────────────────────────────────────────────────────────────
    df['natr'] = talib.NATR(h, l, c, timeperiod=14) / 100

    bb_up, bb_mid, bb_lo = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    bb_range = np.where((bb_up - bb_lo) == 0, np.nan, bb_up - bb_lo)
    df['bb_position'] = (c - bb_lo) / bb_range
    df['bb_width']    = (bb_up - bb_lo) / bb_mid

    # BB Squeeze: 1 when BB is inside Keltner Channel → consolidation before breakout
    kc_range = 2 * 1.5 * talib.ATR(h, l, c, timeperiod=20)
    df['bb_squeeze'] = ((bb_up - bb_lo) < kc_range).astype(float)

    df['atr7_close']    = talib.ATR(h, l, c, timeperiod=7) / c
    df['stddev5_close'] = talib.STDDEV(c, timeperiod=5, nbdev=1) / c
    df['trange_ratio']  = talib.TRANGE(h, l, c) / c

    # Donchian channel position: uses actual price extremes (not statistical bands)
    dc_high = pd.Series(h).rolling(20).max().values
    dc_low  = pd.Series(l).rolling(20).min().values
    dc_range = np.where((dc_high - dc_low) == 0, np.nan, dc_high - dc_low)
    df['donchian_position'] = (c - dc_low) / dc_range   # 0=at channel low, 1=at channel high

    # ── Volume ────────────────────────────────────────────────────────────────
    df['mfi'] = talib.MFI(h, l, c, v, timeperiod=14) / 100

    obv = pd.Series(talib.OBV(c, v), index=df.index)
    df['obv_roc5'] = obv.pct_change(5).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0).fillna(0)

    ad = pd.Series(talib.AD(h, l, c, v), index=df.index)
    df['ad_roc5'] = ad.pct_change(5).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0).fillna(0)

    adosc = pd.Series(talib.ADOSC(h, l, c, v, fastperiod=3, slowperiod=10), index=df.index)
    df['adosc_norm'] = adosc.pct_change(1).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0).fillna(0)

    df['volume_roc5'] = df['volume'].pct_change(5).clip(-1.0, 1.0)

    # ── Candlestick patterns (TA-Lib: -100/0/100) → -1/0/1 ──────────────────
    df['cdlhammer']         = talib.CDLHAMMER(o, h, l, c) / 100
    df['cdlinvertedhammer'] = talib.CDLINVERTEDHAMMER(o, h, l, c) / 100
    df['cdlengulfing']      = talib.CDLENGULFING(o, h, l, c) / 100
    df['cdlpiercing']       = talib.CDLPIERCING(o, h, l, c) / 100
    df['cdlmorningstar']    = talib.CDLMORNINGSTAR(o, h, l, c, penetration=0) / 100
    df['cdl3whitesoldiers'] = talib.CDL3WHITESOLDIERS(o, h, l, c) / 100
    df['cdlabandonedbaby']  = talib.CDLABANDONEDBABY(o, h, l, c, penetration=0) / 100
    df['cdlbreakaway']      = talib.CDLBREAKAWAY(o, h, l, c) / 100
    df['cdlhangingman']     = talib.CDLHANGINGMAN(o, h, l, c) / 100
    df['cdlshootingstar']   = talib.CDLSHOOTINGSTAR(o, h, l, c) / 100
    df['cdleveningstar']    = talib.CDLEVENINGSTAR(o, h, l, c, penetration=0) / 100
    df['cdl3blackcrows']    = talib.CDL3BLACKCROWS(o, h, l, c) / 100
    df['cdldarkcloudcover'] = talib.CDLDARKCLOUDCOVER(o, h, l, c, penetration=0) / 100
    df['cdldoji']           = talib.CDLDOJI(o, h, l, c) / 100
    df['cdlspinningtop']    = talib.CDLSPINNINGTOP(o, h, l, c) / 100
    df['cdlharami']         = talib.CDLHARAMI(o, h, l, c) / 100
    df['cdl3linestrike']    = talib.CDL3LINESTRIKE(o, h, l, c) / 100
    df['cdldragonflydoji']  = talib.CDLDRAGONFLYDOJI(o, h, l, c) / 100
    df['cdlmatchinglow']    = talib.CDLMATCHINGLOW(o, h, l, c) / 100
    df['cdltasukigap']      = talib.CDLTASUKIGAP(o, h, l, c) / 100

    # ── Time encoding ─────────────────────────────────────────────────────────
    # sin/cos of hour-of-day captures session patterns (Asia/Europe/US open).
    # When timestamps are not provided both features are 0 (neutral, no signal).
    if timestamps is not None:
        hours = pd.to_datetime(timestamps).hour.values
        df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    else:
        df['hour_sin'] = 0.0
        df['hour_cos'] = 0.0

    # ── Drop warmup rows where any feature is NaN ─────────────────────────────
    return df[FEATURE_COLUMNS].dropna().reset_index(drop=True)
