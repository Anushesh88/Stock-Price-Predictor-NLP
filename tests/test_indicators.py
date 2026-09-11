import unittest
import numpy as np
import pandas as pd
from src.features.technical_indicators import calculate_technical_indicators, DEFAULT_FEATURE_COLS

def create_synthetic_ohlcv(n_days=100):
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    rng = np.random.RandomState(42)
    
    close = 1000 + np.cumsum(rng.randn(n_days) * 5)
    high = close + np.abs(rng.randn(n_days) * 3)
    low = close - np.abs(rng.randn(n_days) * 3)
    open_p = close + rng.randn(n_days) * 2
    volume = rng.randint(100000, 500000, size=n_days)

    return pd.DataFrame({
        "date": dates,
        "Open": open_p,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume
    })

class TestTechnicalIndicators(unittest.TestCase):
    def test_calculate_technical_indicators(self):
        df = create_synthetic_ohlcv(n_days=100)
        warmup = 50
        df_feat = calculate_technical_indicators(df, warmup_drop=warmup)

        self.assertEqual(len(df_feat), 100 - warmup)
        for col in DEFAULT_FEATURE_COLS:
            self.assertIn(col, df_feat.columns, f"Missing feature column: {col}")
            self.assertFalse(df_feat[col].isnull().all(), f"Feature {col} is all NaN")
            self.assertFalse(np.isinf(df_feat[col]).any(), f"Feature {col} contains inf")

    def test_rsi_bounds(self):
        df = create_synthetic_ohlcv(n_days=80)
        df_feat = calculate_technical_indicators(df, warmup_drop=20)
        
        self.assertTrue((df_feat["rsi"] >= 0).all() and (df_feat["rsi"] <= 100).all())

if __name__ == "__main__":
    unittest.main()
