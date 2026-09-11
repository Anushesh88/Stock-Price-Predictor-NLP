from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from src.utils.logger import get_logger

logger = get_logger("evaluate")

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    target_names: List[str] = None
) -> Tuple[List[int], List[int], str]:
    """
    Evaluates the trained model on test data and prints classification report.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for test partition.
        device: Evaluation device.
        target_names: Label names (default: ['Sell (0)', 'Hold (1)', 'Buy (2)']).

    Returns:
        Tuple of (all_targets, all_preds, report_text)
    """
    if target_names is None:
        target_names = ["Sell (0)", "Hold (1)", "Buy (2)"]

    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_num, x_text, y in test_loader:
            x_num = x_num.to(device)
            x_text = x_text.to(device)
            logits, _ = model(x_num, x_text)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

    report = classification_report(all_targets, all_preds, target_names=target_names, zero_division=0)
    logger.info("\n" + "=" * 50)
    logger.info("TEST SET CLASSIFICATION REPORT")
    logger.info("=" * 50 + "\n" + report)

    return all_targets, all_preds, report
