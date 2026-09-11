import unittest
import numpy as np
import pandas as pd
from src.features.labeling import create_dynamic_volatility_labels

class TestDynamicLabeling(unittest.TestCase):
    def test_dynamic_volatility_labels(self):
        n = 50
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        close = [100.0] * n
        close[10] = 150.0
        close[20] = 60.0

        df = pd.DataFrame({
            "date": dates,
            "Close": close,
            "rolling_vol_20": [0.02] * n
        })

        horizon = 3
        df_labeled = create_dynamic_volatility_labels(df, horizon=horizon, vol_multiplier=0.75)

        self.assertEqual(len(df_labeled), n - horizon)
        self.assertIn("label", df_labeled.columns)
        self.assertTrue(set(df_labeled["label"].unique()).issubset({0, 1, 2}))

if __name__ == "__main__":
    unittest.main()
