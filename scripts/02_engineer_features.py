import argparse
import os
import sys
import yaml
import pandas as pd

# Ensure repository root is in python module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.technical_indicators import calculate_technical_indicators
from src.features.labeling import create_dynamic_volatility_labels
from src.features.scaler import SimpleStandardScaler
from src.utils.logger import get_logger

logger = get_logger("02_engineer_features")

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Technical Feature Engineering, Labeling & Partitioning")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    processed_dir = cfg["paths"]["processed_data_dir"]
    aligned_path = os.path.join(processed_dir, "phase1_aligned.parquet")

    if not os.path.exists(aligned_path):
        raise FileNotFoundError(f"Missing {aligned_path}. Run scripts/01_download_and_align.py first.")

    df = pd.read_parquet(aligned_path)
    logger.info(f"Loaded aligned dataset with {len(df)} rows.")

    # 1. Technical Indicators
    warmup = cfg["features"]["warmup_drop"]
    df_feat = calculate_technical_indicators(df, warmup_drop=warmup)

    # 2. Dynamic Volatility Multi-Horizon Labeling
    horizon = cfg["features"]["horizon"]
    vol_mult = cfg["features"]["vol_multiplier"]
    df_labeled = create_dynamic_volatility_labels(df_feat, horizon=horizon, vol_multiplier=vol_mult)

    # 3. Chronological Train / Val / Test Partitioning
    train_ratio = cfg["split"]["train_ratio"]
    val_ratio = cfg["split"]["val_ratio"]

    n = len(df_labeled)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))

    df_train = df_labeled.iloc[:train_idx].copy()
    df_val = df_labeled.iloc[train_idx:val_idx].copy()
    df_test = df_labeled.iloc[val_idx:].copy()

    logger.info(f"Partition Splits -> Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # 4. Strict Leakage-Free Scaling (Fitted exclusively on Train partition)
    feature_cols = cfg["features"]["feature_cols"]
    scaler = SimpleStandardScaler()
    scaler.fit(df_train, feature_cols)

    df_train_scaled = scaler.transform(df_train)
    df_val_scaled = scaler.transform(df_val)
    df_test_scaled = scaler.transform(df_test)

    # 5. Save Artifacts
    scaler_path = os.path.join(processed_dir, "scaler.json")
    scaler.save(scaler_path)

    df_train_scaled.to_parquet(os.path.join(processed_dir, "train_partition.parquet"), index=False)
    df_val_scaled.to_parquet(os.path.join(processed_dir, "val_partition.parquet"), index=False)
    df_test_scaled.to_parquet(os.path.join(processed_dir, "test_partition.parquet"), index=False)

    logger.info("Phase 2 Complete: Scaled partitions and scaler parameters successfully saved.")

if __name__ == "__main__":
    main()
