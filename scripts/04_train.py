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
from src.models.focal_loss import FocalLoss
from src.training.trainer import MultimodalTrainer
from src.utils.logger import get_logger

logger = get_logger("04_train")

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Train Gated Multimodal Predictor with Focal Loss")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    processed_dir = cfg["paths"]["processed_data_dir"]
    artifacts_dir = cfg["paths"]["artifacts_dir"]
    feature_cols = cfg["features"]["feature_cols"]
    lookback = cfg["features"]["lookback"]

    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr = cfg["training"]["learning_rate"]
    weight_decay = cfg["training"]["weight_decay"]
    gamma = cfg["training"]["focal_gamma"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target Training Device: {device}")

    # 1. Load Data Partitions
    df_train = pd.read_parquet(os.path.join(processed_dir, "train_partition.parquet"))
    df_val = pd.read_parquet(os.path.join(processed_dir, "val_partition.parquet"))

    train_emb = np.load(os.path.join(processed_dir, "train_embeddings.npy"))
    val_emb = np.load(os.path.join(processed_dir, "val_embeddings.npy"))

    train_dataset = StockMultimodalDataset(df_train, train_emb, feature_cols, lookback=lookback)
    val_dataset = StockMultimodalDataset(df_val, val_emb, feature_cols, lookback=lookback)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Dataset Windows -> Train: {len(train_dataset)} | Validation: {len(val_dataset)}")

    # 2. Compute Class Balance & Weights for Focal Loss
    train_labels = [train_dataset[i][2].item() for i in range(len(train_dataset))]
    counts = np.bincount(train_labels, minlength=cfg["model"]["num_classes"])
    # Inverse frequency weights
    alpha_weights = len(train_labels) / (len(counts) * np.maximum(counts, 1))
    alpha_tensor = torch.tensor(alpha_weights, dtype=torch.float32).to(device)

    # 3. Model & Loss Setup
    model = GatedMultimodalStockPredictor(
        num_features=len(feature_cols),
        text_dim=cfg["nlp"]["embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        lstm_layers=cfg["model"]["lstm_layers"],
        dropout=cfg["model"]["dropout"],
        num_classes=cfg["model"]["num_classes"]
    )

    criterion = FocalLoss(alpha=alpha_tensor, gamma=gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 4. Run Training
    trainer = MultimodalTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        artifacts_dir=artifacts_dir
    )

    trainer.train(train_loader, val_loader, epochs=epochs)

if __name__ == "__main__":
    main()
