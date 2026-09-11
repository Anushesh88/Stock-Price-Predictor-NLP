from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    r"""
    Multi-Class Focal Loss for imbalanced financial classification.

    Loss Formulation:
      FL(p_t) = - \alpha_t * (1 - p_t)^\gamma * \log(p_t)

    Args:
        alpha: Optional 1D Tensor of per-class weights of shape [NumClasses].
        gamma: Focusing parameter for modulating easy/hard samples (default: 2.0).
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted raw scores of shape [Batch, NumClasses]
            targets: Ground truth class indices of shape [Batch]
        """
        # Unweighted cross entropy without reduction to compute genuine p_t
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_term = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss

        return focal_loss.mean()
