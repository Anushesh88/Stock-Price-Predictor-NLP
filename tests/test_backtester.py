import unittest
import numpy as np
import pandas as pd
from src.evaluation.backtester import run_backtest

class TestBacktester(unittest.TestCase):
    def test_non_overlapping_horizon_sampling(self):
        # 12 trading forward returns
        fwd_returns = [0.03] * 12
        df_test = pd.DataFrame({
            "fwd_return": fwd_returns
        })
        # 12 predictions: all Buy (class 2)
        predictions = [2] * 12
        lookback = 1
        horizon = 3

        # With lookback=1 and horizon=3, 12 samples sampled every 3 days = 4 trades
        results = run_backtest(
            predictions=predictions,
            df_test=df_test,
            lookback=lookback,
            horizon=horizon,
            transaction_cost_bps=0.0005
        )

        self.assertEqual(results["trading_windows"], 4)
        # 4 non-overlapping periods of 3% return = 12% gross
        self.assertAlmostEqual(results["cum_strategy_gross"], 4 * 0.03, places=5)
        # Initial position from 0 to 1 causes 1 trade of 5 bps
        self.assertEqual(results["total_trades"], 1)
        expected_net = (4 * 0.03) - 0.0005
        self.assertAlmostEqual(results["cum_strategy_net"], expected_net, places=5)

if __name__ == "__main__":
    unittest.main()
