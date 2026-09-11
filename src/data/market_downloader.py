import os
from typing import Optional
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("market_downloader")

def generate_sample_market_prices(
    start_date: str = "2022-01-01",
    periods: int = 500,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generates realistic synthetic OHLCV market data for testing and offline execution.
    """
    logger.info(f"Generating synthetic market prices ({periods} trading days) starting from {start_date}...")
    dates = pd.date_range(start=start_date, periods=periods, freq="B")
    rng = np.random.RandomState(42)

    # Geometric random walk
    log_returns = rng.normal(0.0005, 0.015, size=periods)
    close_prices = 1400.0 * np.exp(np.cumsum(log_returns))
    high_prices = close_prices * (1 + np.abs(rng.normal(0, 0.008, size=periods)))
    low_prices = close_prices * (1 - np.abs(rng.normal(0, 0.008, size=periods)))
    open_prices = low_prices + (high_prices - low_prices) * rng.uniform(0.2, 0.8, size=periods)
    volumes = rng.randint(2_000_000, 15_000_000, size=periods)

    df_prices = pd.DataFrame({
        "date": dates,
        "Open": open_prices,
        "High": high_prices,
        "Low": low_prices,
        "Close": close_prices,
        "Volume": volumes
    })

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_prices.to_parquet(output_path, index=False)
        logger.info(f"Saved sample market prices to: {output_path}")

    return df_prices

def download_market_prices(
    ticker: str = "HDFCBANK.NS",
    start_date: str = "2019-01-01",
    end_date: Optional[str] = None,
    output_path: Optional[str] = None,
    use_sample_on_error: bool = False
) -> pd.DataFrame:
    """
    Downloads historical OHLCV market prices using Yahoo Finance.
    Handles MultiIndex flattening, datetime timezone normalization, and cleaning.

    Args:
        ticker: Symbol to download (default: HDFCBANK.NS).
        start_date: Starting date in YYYY-MM-DD.
        end_date: Ending date in YYYY-MM-DD (optional).
        output_path: Optional path to save Parquet file.

    Returns:
        pd.DataFrame: Cleaned OHLCV DataFrame with columns ['date', 'Open', 'High', 'Low', 'Close', 'Volume'].
    """
    if use_sample_on_error:
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance package not found. Generating realistic synthetic sample market prices...")
            return generate_sample_market_prices(start_date=start_date, periods=500, output_path=output_path)
    else:
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is required for downloading live market data. "
                "Install it with 'pip install yfinance' or pass use_sample_on_error=True."
            )

    logger.info(f"Downloading historical market prices for '{ticker}' from {start_date}...")
    try:
        df_prices = yf.download(ticker, start=start_date, end=end_date, progress=False)
    except Exception as e:
        if use_sample_on_error:
            logger.warning(f"Error fetching from Yahoo Finance ({e}). Falling back to sample prices.")
            return generate_sample_market_prices(start_date=start_date, periods=500, output_path=output_path)
        raise

    if df_prices is None or df_prices.empty:
        if use_sample_on_error:
            logger.warning(f"Empty data returned for '{ticker}'. Falling back to sample prices.")
            return generate_sample_market_prices(start_date=start_date, periods=500, output_path=output_path)
        raise ValueError(f"No price data returned for ticker '{ticker}'. Please check the symbol and dates.")

    # Flatten MultiIndex columns if present
    if isinstance(df_prices.columns, pd.MultiIndex):
        df_prices.columns = [col[0] for col in df_prices.columns]

    df_prices = df_prices.reset_index()
    
    # Standardize date column
    date_col = next((c for c in df_prices.columns if 'date' in c.lower()), 'Date')
    df_prices.rename(columns={date_col: 'date'}, inplace=True)
    df_prices['date'] = pd.to_datetime(df_prices['date']).dt.tz_localize(None)

    required_cols = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df_prices = df_prices[required_cols].dropna().sort_values('date').reset_index(drop=True)

    logger.info(f"Successfully downloaded {len(df_prices)} trading days for {ticker} ({df_prices['date'].min().strftime('%Y-%m-%d')} to {df_prices['date'].max().strftime('%Y-%m-%d')}).")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_prices.to_parquet(output_path, index=False)
        logger.info(f"Saved market prices to: {output_path}")

    return df_prices
