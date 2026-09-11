from typing import Dict, List, Union
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("backtester")

def run_backtest(
    predictions: Union[np.ndarray, List[int]],
    df_test: pd.DataFrame,
    lookback: int = 30,
    transaction_cost_bps: float = 0.0005
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Simulates a multi-horizon trading backtest with transaction friction.

    Position Mapping:
      - 2 (Buy)  -> Long (+1)
      - 0 (Sell) -> Short (-1)
      - 1 (Hold) -> Flat (0)

    Trading Dynamics:
      - Strategy return = position * fwd_return
      - Transaction cost charged on position switches: |position_t - position_{t-1}| * cost_bps
      - Net return = Strategy return - Transaction costs

    Args:
        predictions: Model class predictions [0, 1, 2] of length M.
        df_test: Test DataFrame containing 'fwd_return'.
        lookback: Sequence lookback offset (default: 30).
        transaction_cost_bps: Cost per position turnover (default: 0.0005 = 5 bps).

    Returns:
        Dict with cumulative returns, net/gross PnL, and performance summaries.
    """
    preds = np.array(predictions)
    # Map classes to trading positions
    positions = np.array([1 if p == 2 else (-1 if p == 0 else 0) for p in preds])

    # Align with test dataset valid window indices
    test_fwd_returns = df_test["fwd_return"].values[lookback - 1:]

    # Match length
    min_len = min(len(positions), len(test_fwd_returns))
    positions = positions[:min_len]
    test_fwd_returns = test_fwd_returns[:min_len]

    # Gross Strategy Return
    strategy_returns = positions * test_fwd_returns

    # Transaction Friction: Charged whenever position changes
    trades = np.abs(np.diff(positions, prepend=0))
    transaction_costs = trades * transaction_cost_bps
    net_strategy_returns = strategy_returns - transaction_costs

    # Cumulative Series
    cum_market_returns = np.cumsum(test_fwd_returns)
    cum_strategy_gross = np.cumsum(strategy_returns)
    cum_strategy_net = np.cumsum(net_strategy_returns)

    results = {
        "trading_windows": int(min_len),
        "total_trades": int(np.sum(trades > 0)),
        "cum_market_return": float(cum_market_returns[-1]) if len(cum_market_returns) > 0 else 0.0,
        "cum_strategy_gross": float(cum_strategy_gross[-1]) if len(cum_strategy_gross) > 0 else 0.0,
        "cum_strategy_net": float(cum_strategy_net[-1]) if len(cum_strategy_net) > 0 else 0.0,
        "cum_market_series": cum_market_returns,
        "cum_strategy_gross_series": cum_strategy_gross,
        "cum_strategy_net_series": cum_strategy_net,
    }

    logger.info("\n" + "=" * 50)
    logger.info(f"BACKTEST SIMULATION RESULTS ({results['trading_windows']} Trading Windows)")
    logger.info("=" * 50)
    logger.info(f"Total Portfolio Rebalances/Trades: {results['total_trades']}")
    logger.info(f"Cumulative Market (Passive Benchmark): {results['cum_market_return'] * 100:.2f}%")
    logger.info(f"Cumulative Strategy (Gross PnL):       {results['cum_strategy_gross'] * 100:.2f}%")
    logger.info(f"Cumulative Strategy (Net of 5bps Cost): {results['cum_strategy_net'] * 100:.2f}%")
    logger.info("=" * 50)

    return results
