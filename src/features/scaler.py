import json
from typing import Dict, List, Optional
import pandas as pd

class SimpleStandardScaler:
    """
    Leakage-free, self-contained standard scaler with JSON serialization.
    Replaces heavy/incompatible sklearn versions with pure pandas math.
    """
    def __init__(self):
        self.mean_: Optional[Dict[str, float]] = None
        self.scale_: Optional[Dict[str, float]] = None
        self.feature_cols: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame, feature_cols: List[str]):
        """
        Calculates sample mean and sample standard deviation strictly on the training partition.
        """
        self.feature_cols = list(feature_cols)
        self.mean_ = df[self.feature_cols].mean(axis=0).to_dict()
        std = df[self.feature_cols].std(axis=0)
        # Epsilon floor prevents zero-division for constant features
        self.scale_ = std.replace(0, 1e-8).to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies z-score scaling: (x - mean) / std.
        """
        if self.mean_ is None or self.scale_ is None or self.feature_cols is None:
            raise RuntimeError("SimpleStandardScaler must be fitted before transforming.")

        df_out = df.copy()
        for col in self.feature_cols:
            mean = self.mean_[col]
            std = self.scale_[col]
            df_out[col] = (df_out[col] - mean) / std
        return df_out

    def fit_transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        return self.fit(df, feature_cols).transform(df)

    def save(self, filepath: str):
        """
        Serializes scaler parameters to JSON.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "mean": self.mean_,
                "scale": self.scale_,
                "feature_cols": self.feature_cols
            }, f, indent=4)

    @classmethod
    def load(cls, filepath: str) -> "SimpleStandardScaler":
        """
        Loads scaler parameters from JSON.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        scaler = cls()
        scaler.mean_ = data["mean"]
        scaler.scale_ = data["scale"]
        scaler.feature_cols = data["feature_cols"]
        return scaler
