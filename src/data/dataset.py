from typing import List, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class StockMultimodalDataset(Dataset):
    """
    PyTorch Dataset providing aligned numerical time series sequences, FinBERT embeddings, and multi-class labels.

    Args:
        df_scaled (pd.DataFrame): Preprocessed and scaled DataFrame containing features and 'label'.
        embeddings (np.ndarray): 2D array of dense text embeddings of shape [N, embedding_dim].
        feature_cols (List[str]): List of column names used as numerical inputs.
        lookback (int): Sequence length of historical days (default: 30).
    """
    def __init__(
        self,
        df_scaled: pd.DataFrame,
        embeddings: np.ndarray,
        feature_cols: List[str],
        lookback: int = 30
    ):
        self.features = df_scaled[feature_cols].values.astype(np.float32)
        if "label" in df_scaled.columns:
            self.labels = df_scaled["label"].values.astype(np.int64)
        else:
            self.labels = np.zeros(len(df_scaled), dtype=np.int64)

        self.embeddings = embeddings.astype(np.float32)
        self.lookback = lookback

        if len(self.features) < lookback:
            raise ValueError(f"Dataset length ({len(self.features)}) is less than lookback window ({lookback}).")

        # Valid indices start after lookback - 1 days of history
        self.valid_indices = list(range(lookback - 1, len(self.features)))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.lookback + 1

        x_num = self.features[start_idx:end_idx + 1]  # Shape: [lookback, num_features]
        x_text = self.embeddings[end_idx]             # Shape: [embedding_dim]
        y = self.labels[end_idx]                      # Scalar int class label

        return (
            torch.tensor(x_num, dtype=torch.float32),
            torch.tensor(x_text, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )

# Backward-compatible alias for notebook naming
HDFCDataset = StockMultimodalDataset
