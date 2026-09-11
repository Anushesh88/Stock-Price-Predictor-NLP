import os
import glob
import json
from typing import List, Optional
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("text_ingestion")

def get_trading_day_mapper(trading_days: pd.Series):
    """
    Creates a mapping function that maps any timestamp to the nearest eligible trading day.
    Applies the Indian market closing rule (15:30 IST):
    Any news arriving after 15:30 IST is assigned to the subsequent trading day.
    """
    sorted_days = np.sort(pd.to_datetime(trading_days.unique()))

    def map_to_trading_day(ts) -> Optional[pd.Timestamp]:
        if pd.isna(ts):
            return None
        ts = pd.to_datetime(ts)
        # 15:30 IST cutoff: if past 15:30, news affects next day's market open
        effective_date = ts.date() if (ts.hour < 15 or (ts.hour == 15 and ts.minute <= 30)) else ts.date() + pd.Timedelta(days=1)
        effective_ts = pd.Timestamp(effective_date)
        future_days = sorted_days[sorted_days >= effective_ts]
        return future_days[0] if len(future_days) > 0 else None

    return map_to_trading_day


def ingest_bse_filings(bse_pattern: str, map_fn) -> List[dict]:
    """
    Ingests official BSE regulatory filings from JSON files.
    """
    bse_files = glob.glob(bse_pattern, recursive=True)
    if not bse_files:
        logger.warning(f"No BSE JSON files found matching pattern: {bse_pattern}")
        return []

    logger.info(f"Ingesting BSE filings from {len(bse_files)} file(s)...")
    records = []
    
    for fpath in bse_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                items = data.get("Table") or data.get("Table1") or data.get("data") or [data]
            elif isinstance(data, list):
                items = data
            else:
                items = []

            for item in items:
                if not isinstance(item, dict):
                    continue

                ts_str = (
                    item.get("NEWS_DT") or item.get("DT_TM") or item.get("DisseminationTime") or
                    item.get("timestamp") or item.get("News_dt") or item.get("dt_tm")
                )
                headline = item.get("NEWSSUB") or item.get("HEADLINE") or item.get("headline") or item.get("Subject") or ""
                desc = item.get("NEWS_BODY") or item.get("MORE") or item.get("description") or item.get("Body") or ""

                if ts_str and (headline or desc):
                    ts = pd.to_datetime(ts_str, errors="coerce")
                    if pd.notna(ts):
                        ts = ts.tz_localize(None) if ts.tzinfo is None else ts.tz_convert(None)
                        tdate = map_fn(ts)
                        if tdate:
                            clean_txt = f"{headline.strip()}. {desc.strip()}".strip()
                            records.append({"trading_date": tdate, "clean_text": clean_txt, "source": "BSE"})
        except Exception as e:
            logger.debug(f"Error parsing BSE file {fpath}: {e}")
            continue

    logger.info(f"Extracted {len(records)} BSE filing records.")
    return records


def ingest_news_csv(csv_path: str, map_fn) -> List[dict]:
    """
    Ingests news articles or sentiment headlines from CSV files.
    Robustly parses dates across Business Standard, Kaggle, and custom formats.
    """
    if not os.path.exists(csv_path):
        logger.warning(f"News CSV not found at: {csv_path}")
        return []

    logger.info(f"Ingesting news from CSV: {csv_path}")
    df_news = pd.read_csv(csv_path)

    # Prioritize standard timestamp columns
    if "timestamp" in df_news.columns:
        date_col = "timestamp"
    elif "date" in df_news.columns:
        date_col = "date"
    else:
        date_col = next((c for c in df_news.columns if any(k in c.lower() for k in ["date", "time", "timestamp"])), None)

    headline_col = next((c for c in df_news.columns if any(k in c.lower() for k in ["headline", "title", "subject"])), None)
    desc_col = next((c for c in df_news.columns if any(k in c.lower() for k in ["desc", "body", "text"]) and c != headline_col), None)

    if not date_col:
        logger.warning(f"Could not identify a date/timestamp column in {csv_path}.")
        return []

    # Parse dates with support for Indian standard timestamps (e.g. '08:07:25 29/04/2025 pm IST')
    series_str = df_news[date_col].astype(str).str.replace(r"\s*(am|pm)\s*ist", "", case=False, regex=True).str.strip()
    parsed_dates = pd.to_datetime(series_str, format="%H:%M:%S %d/%m/%Y", errors="coerce")
    fallback_dates = pd.to_datetime(df_news[date_col], errors="coerce")
    df_news["_clean_dt"] = parsed_dates.fillna(fallback_dates).dt.tz_localize(None)

    records = []
    for _, row in df_news.dropna(subset=["_clean_dt"]).iterrows():
        tdate = map_fn(row["_clean_dt"])
        if not tdate:
            continue

        parts = []
        if headline_col and pd.notna(row.get(headline_col)):
            parts.append(str(row[headline_col]).strip())
        if desc_col and pd.notna(row.get(desc_col)):
            parts.append(str(row[desc_col]).strip())

        text = ". ".join(parts).strip()
        if text:
            records.append({"trading_date": tdate, "clean_text": text, "source": os.path.basename(csv_path)})

    logger.info(f"Extracted {len(records)} news records from {csv_path}.")
    return records


def align_multimodal_data(
    df_prices: pd.DataFrame,
    text_records: List[dict],
    default_empty_text: str = "No major news or filings reported for HDFC Bank on this trading day."
) -> pd.DataFrame:
    """
    Aggregates multi-source text records per trading session and merges with OHLCV prices.
    Deduplicates text entries and fills quiet days with default neutral text.
    """
    if not text_records:
        logger.warning("No text records provided. Populating all trading days with neutral text.")
        df_aligned = df_prices.copy()
        df_aligned["aggregated_text"] = default_empty_text
        return df_aligned

    df_text = pd.DataFrame(text_records)
    # Deduplicate text
    df_text = df_text.drop_duplicates(subset=["trading_date", "clean_text"]).reset_index(drop=True)

    # Group daily text into single aggregated string separated by pipe delimiter
    df_daily = df_text.groupby("trading_date")["clean_text"].apply(lambda texts: " | ".join(texts)).reset_index()
    df_daily.rename(columns={"trading_date": "date", "clean_text": "aggregated_text"}, inplace=True)

    # Merge left with market prices
    df_aligned = pd.merge(df_prices, df_daily, on="date", how="left")
    df_aligned["aggregated_text"] = df_aligned["aggregated_text"].fillna(default_empty_text)

    coverage_days = (df_aligned["aggregated_text"] != default_empty_text).sum()
    coverage_pct = (coverage_days / len(df_aligned)) * 100
    logger.info(f"Multimodal Alignment Complete: {len(df_aligned)} total trading days | {coverage_days} days with active news ({coverage_pct:.2f}% coverage).")

    return df_aligned
