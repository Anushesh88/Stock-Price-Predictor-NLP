import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch

# Ensure repository root is in python module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import DataLoader
from src.data.dataset import StockMultimodalDataset
from src.models.gated_multimodal import GatedMultimodalStockPredictor
from src.evaluation.evaluate import evaluate_model
from src.evaluation.backtester import run_backtest
from src.utils.logger import get_logger

logger = get_logger("05_backtest")

def main():
    parser = argparse.ArgumentParser(description="Phase 5: Out-of-Sample Evaluation and Backtesting Simulation")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", default=None, help="Path to trained model checkpoint")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    processed_dir = cfg["paths"]["processed_data_dir"]
    artifacts_dir = cfg["paths"]["artifacts_dir"]
    feature_cols = cfg["features"]["feature_cols"]
    lookback = cfg["features"]["lookback"]
    cost_bps = cfg["backtest"]["transaction_cost_bps"]

    checkpoint_path = args.checkpoint or os.path.join(artifacts_dir, "best_gated_focal_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}. Run scripts/04_train.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Test Partition & Embeddings
    df_test = pd.read_parquet(os.path.join(processed_dir, "test_partition.parquet"))
    test_emb = np.load(os.path.join(processed_dir, "test_embeddings.npy"))

    test_dataset = StockMultimodalDataset(df_test, test_emb, feature_cols, lookback=lookback)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 2. Load Model
    model = GatedMultimodalStockPredictor(
        num_features=len(feature_cols),
        text_dim=cfg["nlp"]["embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        lstm_layers=cfg["model"]["lstm_layers"],
        dropout=cfg["model"]["dropout"],
        num_classes=cfg["model"]["num_classes"]
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    logger.info(f"Loaded best checkpoint from: {checkpoint_path}")

    # 3. Evaluation
    targets, preds, report = evaluate_model(model, test_loader, device=device)

    # 4. Multi-Horizon Backtesting Simulation
    results = run_backtest(
        predictions=preds,
        df_test=df_test,
        lookback=lookback,
        transaction_cost_bps=cost_bps
    )

    logger.info("Phase 5 Complete: Evaluation and Financial Backtest finished.")

if __name__ == "__main__":
    main()
