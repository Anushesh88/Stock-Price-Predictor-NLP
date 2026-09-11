from typing import Optional
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("labeling")

def create_dynamic_volatility_labels(
    df: pd.DataFrame,
    horizon: int = 3,
    vol_multiplier: float = 0.75
) -> pd.DataFrame:
    """
    Computes dynamic volatility-based multi-horizon labels.

    Formulation:
      Forward log return: R_{t+h} = ln(Close_{t+h} / Close_t)
      Dynamic threshold: Thresh_t = RollingVol20_t * sqrt(h) * vol_multiplier
      Label assignment:
        - Buy  (Class 2): R_{t+h} > Thresh_t
        - Sell (Class 0): R_{t+h} < -Thresh_t
        - Hold (Class 1): -Thresh_t <= R_{t+h} <= Thresh_t

    Trailing rows without full horizon lookahead are dropped.

    Args:
        df: DataFrame containing 'Close' and 'rolling_vol_20'.
        horizon: Prediction horizon in trading days (default: 3).
        vol_multiplier: Scale factor for standard deviation band (default: 0.75).

    Returns:
        pd.DataFrame: DataFrame containing 'fwd_return', 'dynamic_threshold', and integer 'label'.
    """
    df_out = df.copy()

    # Forward log return over the target horizon
    df_out["fwd_return"] = np.log(df_out["Close"].shift(-horizon) / df_out["Close"])

    # Dynamic threshold based on annualized/multi-day volatility
    df_out["dynamic_threshold"] = df_out["rolling_vol_20"] * np.sqrt(horizon) * vol_multiplier

    def assign_label(row):
        ret = row["fwd_return"]
        thresh = row["dynamic_threshold"]
        if pd.isna(ret) or pd.isna(thresh):
            return np.nan
        if ret > thresh:
            return 2  # Buy
        elif ret < -thresh:
            return 0  # Sell
        else:
            return 1  # Hold

    df_out["label"] = df_out.apply(assign_label, axis=1)

    # Drop trailing rows lacking future price data for the horizon
    initial_count = len(df_out)
    df_out.dropna(subset=["label"], inplace=True)
    df_out["label"] = df_out["label"].astype(int)

    logger.info(f"Generated {len(df_out)} dynamic labels (dropped {initial_count - len(df_out)} trailing lookahead rows).")
    counts = df_out["label"].value_counts().sort_index().to_dict()
    logger.info(f"Class Distribution: Sell(0)={counts.get(0, 0)} | Hold(1)={counts.get(1, 0)} | Buy(2)={counts.get(2, 0)}")

    return df_out
