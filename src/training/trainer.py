import os
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.training.metrics import calculate_metrics
from src.utils.logger import get_logger

logger = get_logger("trainer")

class MultimodalTrainer:
    """
    Handles model training, validation, learning rate scheduling, and best checkpoint saving.
    """
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        max_grad_norm: float = 1.0,
        artifacts_dir: str = "artifacts"
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_grad_norm = max_grad_norm
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)

        self.model.to(self.device)
        self.best_val_f1 = -1.0
        self.best_model_path = os.path.join(self.artifacts_dir, "best_gated_focal_model.pth")

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for x_num, x_text, y in train_loader:
            x_num = x_num.to(self.device)
            x_text = x_text.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            logits, _ = self.model(x_num, x_text)
            loss = self.criterion(logits, y)
            loss.backward()

            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(len(train_loader), 1)

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, Dict]:
        self.model.eval()
        total_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for x_num, x_text, y in data_loader:
                x_num = x_num.to(self.device)
                x_text = x_text.to(self.device)
                y = y.to(self.device)

                logits, _ = self.model(x_num, x_text)
                loss = self.criterion(logits, y)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy().tolist())
                val_targets.extend(y.cpu().numpy().tolist())

        avg_loss = total_loss / max(len(data_loader), 1)
        metrics = calculate_metrics(val_targets, val_preds)
        return avg_loss, metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 25
    ) -> Dict[str, float]:
        logger.info(f"Starting model training for {epochs} epochs on device: {self.device}...")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            val_loss, val_metrics = self.evaluate(val_loader)
            val_f1 = val_metrics["macro_f1"]
            val_acc = val_metrics["accuracy"]

            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Macro-F1: {val_f1:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                torch.save(self.model.state_dict(), self.best_model_path)
                logger.info(f"  -> Checkpoint Saved! New Best Val Macro-F1: {self.best_val_f1:.4f} to {self.best_model_path}")

        logger.info(f"Training Complete. Best Model Checkpoint: {self.best_model_path} (Macro-F1: {self.best_val_f1:.4f})")
        return {"best_val_f1": self.best_val_f1, "checkpoint_path": self.best_model_path}
