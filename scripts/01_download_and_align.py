import argparse
import glob
import os
import sys
import yaml

# Ensure repository root is in python module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.market_downloader import download_market_prices
from src.data.text_ingestion import (
    get_trading_day_mapper,
    ingest_bse_filings,
    ingest_news_csv,
    align_multimodal_data,
)
from src.utils.logger import get_logger

logger = get_logger("01_download_and_align")

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Download OHLCV Market Prices and Align Multimodal Text")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--ticker", default=None, help="Stock ticker (overrides config)")
    parser.add_argument("--sample", action="store_true", help="Generate synthetic market data instead of live download")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ticker = args.ticker or cfg["project"]["ticker"]
    start_date = cfg["project"]["start_date"]
    end_date = cfg["project"]["end_date"]
    processed_dir = cfg["paths"]["processed_data_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Download Market Prices
    prices_path = os.path.join(processed_dir, "market_prices.parquet")
    df_prices = download_market_prices(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        output_path=prices_path,
        use_sample_on_error=args.sample
    )

    # 2. Text Ingestion
    map_fn = get_trading_day_mapper(df_prices["date"])
    text_records = []

    # Ingest BSE filings
    bse_pattern = cfg["paths"].get("bse_pattern", "")
    if bse_pattern:
        text_records.extend(ingest_bse_filings(bse_pattern, map_fn))

    # Ingest News CSV files from data/raw/
    raw_dir = cfg["paths"].get("raw_data_dir", "data/raw")
    news_csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    logger.info(f"Found {len(news_csv_files)} news CSV file(s) in {raw_dir}: {[os.path.basename(f) for f in news_csv_files]}")
    for n_csv in news_csv_files:
        text_records.extend(ingest_news_csv(n_csv, map_fn))

    # 3. Align Multimodal Stream
    df_aligned = align_multimodal_data(df_prices, text_records)
    aligned_path = os.path.join(processed_dir, "phase1_aligned.parquet")
    df_aligned.to_parquet(aligned_path, index=False)
    logger.info(f"Phase 1 Complete. Gold aligned dataset saved to: {aligned_path}")

if __name__ == "__main__":
    main()
