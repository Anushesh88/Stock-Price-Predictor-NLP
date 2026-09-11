import unittest
import torch
from src.models.gated_multimodal import GatedMultimodalStockPredictor
from src.models.focal_loss import FocalLoss

class TestModelForward(unittest.TestCase):
    def test_model_forward_pass_and_attention_shapes(self):
        batch_size = 4
        lookback = 30
        num_features = 15
        text_dim = 768
        num_classes = 3

        model = GatedMultimodalStockPredictor(
            num_features=num_features,
            text_dim=text_dim,
            hidden_dim=32,
            lstm_layers=2,
            dropout=0.2,
            num_classes=num_classes
        )

        x_num = torch.randn(batch_size, lookback, num_features)
        x_text = torch.randn(batch_size, text_dim)

        logits, attn_weights = model(x_num, x_text)

        self.assertEqual(logits.shape, (batch_size, num_classes))
        self.assertEqual(attn_weights.shape, (batch_size, lookback))

        # Attention weights along timeline should sum to 1
        attn_sums = attn_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(attn_sums, torch.ones(batch_size), atol=1e-5))

    def test_focal_loss_backward(self):
        batch_size = 4
        num_classes = 3

        logits = torch.randn(batch_size, num_classes, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 1], dtype=torch.long)
        weights = torch.tensor([1.0, 0.5, 1.2], dtype=torch.float32)

        criterion = FocalLoss(alpha=weights, gamma=2.0)
        loss = criterion(logits, targets)

        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any())

if __name__ == "__main__":
    unittest.main()
