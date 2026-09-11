from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedMultimodalStockPredictor(nn.Module):
    """
    Multimodal Stock Predictor fusing numerical technical time series with financial news embeddings.

    Architecture:
      1. Numerical Encoder: 2-layer Bidirectional LSTM processing 30-day lookback sequences.
      2. Text Projection: MLP projecting 768-d FinBERT [CLS] dense embedding into LSTM latent dimension.
      3. Cross-Attention: Softmax attention over the 30-day temporal sequence conditioned on news context.
      4. Dynamic Sigmoid Gating: Computes adaptive mixing ratio between numerical and textual context.
      5. Classification Head: Fully connected layers with LayerNorm and Dropout mapping to 3 classes (Sell, Hold, Buy).

    Args:
        num_features: Number of numerical input features (default: 15).
        text_dim: Dimension of text embeddings from FinBERT (default: 768).
        hidden_dim: Hidden dimension for LSTM and projection layers (default: 96).
        lstm_layers: Number of stacked LSTM layers (default: 2).
        dropout: Dropout rate (default: 0.4).
        num_classes: Output class count (default: 3).
    """
    def __init__(
        self,
        num_features: int = 15,
        text_dim: int = 768,
        hidden_dim: int = 96,
        lstm_layers: int = 2,
        dropout: float = 0.4,
        num_classes: int = 3
    ):
        super(GatedMultimodalStockPredictor, self).__init__()

        # 1. Numerical Branch: Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.lstm_out_dim = hidden_dim * 2

        # 2. Text Branch: FinBERT Projection Layer
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.lstm_out_dim)
        )

        # 3. Gating Layer: Dynamically balances technical representation and news context
        self.gate_layer = nn.Sequential(
            nn.Linear(self.lstm_out_dim * 2, self.lstm_out_dim),
            nn.Sigmoid()
        )

        # 4. Final Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_out_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        x_num: torch.Tensor,
        x_text: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x_num: Tensor of shape [Batch, Lookback, NumFeatures]
            x_text: Tensor of shape [Batch, TextDim]

        Returns:
            Tuple of:
              - logits: Tensor of shape [Batch, NumClasses]
              - attn_weights: Attention weights over lookback days [Batch, Lookback]
        """
        # Numerical LSTM Pass -> [Batch, Lookback, 2 * hidden_dim]
        lstm_out, (hn, _) = self.lstm(x_num)

        # Text Projection Pass -> [Batch, 2 * hidden_dim]
        proj_text = self.text_proj(x_text)
        text_key = proj_text.unsqueeze(1)  # [Batch, 1, 2 * hidden_dim]

        # Scaled dot-product cross-attention over the 30-day temporal sequence
        scale = lstm_out.size(-1) ** 0.5
        scores = torch.sum(lstm_out * text_key, dim=-1, keepdim=True) / scale  # [Batch, Lookback, 1]
        attn_weights = F.softmax(scores, dim=1)                                # [Batch, Lookback, 1]
        context_vec = torch.sum(attn_weights * lstm_out, dim=1)                 # [Batch, 2 * hidden_dim]

        # Pooled numerical representation from final forward and backward LSTM states
        lstm_pooled = torch.cat([hn[-2], hn[-1]], dim=1)                       # [Batch, 2 * hidden_dim]

        # Adaptive Gating
        gate = self.gate_layer(torch.cat([lstm_pooled, context_vec], dim=1))    # [Batch, 2 * hidden_dim]
        gated_context = gate * context_vec

        # Multimodal Concatenation & Logits
        fused = torch.cat([lstm_pooled, gated_context], dim=1)                  # [Batch, 4 * hidden_dim]
        logits = self.classifier(fused)                                         # [Batch, num_classes]

        return logits, attn_weights.squeeze(-1)
