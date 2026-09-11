from typing import List, Tuple
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("technical_indicators")

DEFAULT_FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "rsi", "macd", "macd_signal", "stoch_k",
    "bb_width", "atr", "obv", "ema_20", "ema_50", "rolling_vol_20"
]

def calculate_technical_indicators(
    df: pd.DataFrame,
    warmup_drop: int = 50
) -> pd.DataFrame:
    """
    Computes pure-pandas technical indicators without lookahead leakage.

    Indicators:
      - Exponential Moving Averages: ema_20, ema_50
      - Relative Strength Index: rsi (14)
      - Moving Average Convergence Divergence: macd (12, 26), macd_signal (9)
      - Stochastic Oscillator: stoch_k (14)
      - Bollinger Bands: bb_mid, bb_upper, bb_lower, bb_width (20, 2 std)
      - Average True Range: atr (14)
      - On-Balance Volume: obv
      - Rolling Daily Volatility: rolling_vol_20 (20-day std of log returns)

    Args:
        df: Clean OHLCV DataFrame sorted chronologically.
        warmup_drop: Number of initial rows to drop for indicator stabilization (default: 50).

    Returns:
        pd.DataFrame: DataFrame enriched with technical indicator columns.
    """
    df_out = df.copy()
    df_out["date"] = pd.to_datetime(df_out["date"])
    df_out.sort_values("date", inplace=True)
    df_out.reset_index(drop=True, inplace=True)

    # 1. Exponential Moving Averages
    df_out["ema_20"] = df_out["Close"].ewm(span=20, adjust=False).mean()
    df_out["ema_50"] = df_out["Close"].ewm(span=50, adjust=False).mean()

    # 2. Relative Strength Index (RSI 14)
    delta = df_out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df_out["rsi"] = 100 - (100 / (1 + rs))

    # 3. MACD (12, 26, 9)
    ema_12 = df_out["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df_out["Close"].ewm(span=26, adjust=False).mean()
    df_out["macd"] = ema_12 - ema_26
    df_out["macd_signal"] = df_out["macd"].ewm(span=9, adjust=False).mean()

    # 4. Stochastic Oscillator %K (14)
    low_14 = df_out["Low"].rolling(14).min()
    high_14 = df_out["High"].rolling(14).max()
    df_out["stoch_k"] = 100 * ((df_out["Close"] - low_14) / (high_14 - low_14 + 1e-9))

    # 5. Bollinger Bands (20, 2 std)
    df_out["bb_mid"] = df_out["Close"].rolling(20).mean()
    bb_std = df_out["Close"].rolling(20).std()
    df_out["bb_upper"] = df_out["bb_mid"] + (2 * bb_std)
    df_out["bb_lower"] = df_out["bb_mid"] - (2 * bb_std)
    df_out["bb_width"] = (df_out["bb_upper"] - df_out["bb_lower"]) / (df_out["bb_mid"] + 1e-9)

    # 6. Average True Range (ATR 14)
    tr1 = df_out["High"] - df_out["Low"]
    tr2 = (df_out["High"] - df_out["Close"].shift(1)).abs()
    tr3 = (df_out["Low"] - df_out["Close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df_out["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()

    # 7. On-Balance Volume (OBV)
    obv_dir = np.sign(df_out["Close"].diff()).fillna(0)
    df_out["obv"] = (obv_dir * df_out["Volume"]).cumsum()

    # 8. 20-Day Rolling Daily Volatility
    df_out["daily_log_ret"] = np.log(df_out["Close"] / df_out["Close"].shift(1))
    df_out["rolling_vol_20"] = df_out["daily_log_ret"].rolling(window=20).std()

    # Drop early warm-up period
    if warmup_drop > 0 and len(df_out) > warmup_drop:
        df_out = df_out.iloc[warmup_drop:].reset_index(drop=True)
        logger.info(f"Dropped {warmup_drop} indicator warm-up rows. Remaining rows: {len(df_out)}")

    return df_out
