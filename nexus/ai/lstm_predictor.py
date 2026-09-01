"""Advanced LSTM-based Market Predictor with Attention Mechanism.

Real SOTA AI deep learning model for temporal dependency modeling in market price action:
- Bidirectional LSTM for sequence representation
- Multi-head Attention for feature importance calculation
- Residual skip connections and confidence/uncertainty estimation heads
- Online gradient adaptation for live market regime shifts
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from nexus.utils.logger import get_nexus_logger

logger = get_nexus_logger("nexus.ai.lstm_predictor")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None  # type: ignore
    nn = object  # type: ignore
    optim = object  # type: ignore


if _HAS_TORCH:

    class AttentionLayer(nn.Module):
        """Multi-head attention mechanism for temporal feature importance."""

        def __init__(self, hidden_dim: int, num_heads: int = 4):
            super().__init__()
            self.num_heads = num_heads
            self.hidden_dim = hidden_dim
            self.head_dim = hidden_dim // num_heads

            assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

            self.query = nn.Linear(hidden_dim, hidden_dim)
            self.key = nn.Linear(hidden_dim, hidden_dim)
            self.value = nn.Linear(hidden_dim, hidden_dim)
            self.fc_out = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            batch_size, seq_len, _ = x.shape

            Q = (
                self.query(x)
                .view(batch_size, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = (
                self.value(x)
                .view(batch_size, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )

            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
            attention_weights = torch.softmax(scores, dim=-1)

            attended = torch.matmul(attention_weights, V)
            attended = (
                attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
            )

            output = self.fc_out(attended)

            return output, attention_weights.mean(dim=1)

    class LSTMPredictor(nn.Module):
        """Advanced LSTM-based market predictor with multi-head attention."""

        def __init__(
            self,
            input_dim: int = 20,
            hidden_dim: int = 128,
            num_layers: int = 3,
            num_heads: int = 4,
            dropout: float = 0.3,
            output_classes: int = 3,
        ):
            super().__init__()

            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

            self.input_embedding = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            self.lstm = nn.LSTM(
                hidden_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True,
            )

            self.attention = AttentionLayer(hidden_dim * 2, num_heads)
            self.residual = nn.Linear(hidden_dim * 2, hidden_dim * 2)

            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_classes),
            )

            self.confidence_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

            self.value_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
            )

        def forward(
            self, x: torch.Tensor, return_attention: bool = False
        ) -> Dict[str, torch.Tensor]:
            batch_size, seq_len, _ = x.shape

            x_embedded = x.view(-1, self.input_dim)
            x_embedded = self.input_embedding(x_embedded)
            x_embedded = x_embedded.view(batch_size, seq_len, -1)

            lstm_out, _ = self.lstm(x_embedded)

            attended, attention_weights = self.attention(lstm_out)
            residual = self.residual(lstm_out)
            combined = attended + residual

            final_state = combined[:, -1, :]

            logits = self.classifier(final_state)
            confidence = self.confidence_head(final_state)
            value = self.value_head(final_state)

            probabilities = torch.softmax(logits, dim=-1)

            result = {
                "logits": logits,
                "probabilities": probabilities,
                "confidence": confidence,
                "value": value,
                "prediction": torch.argmax(probabilities, dim=-1),
            }

            if return_attention:
                result["attention_weights"] = attention_weights

            return result

    class LSTMTrainer:
        """Trainer for LSTM predictor with gradient clipping and adaptive scheduling."""

        def __init__(self, model: LSTMPredictor, learning_rate: float = 0.001, device: str = "cpu"):
            self.model = model.to(device)
            self.device = device
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            self.classification_loss = nn.CrossEntropyLoss()
            self.value_loss = nn.MSELoss()

            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=10, factor=0.5
            )

            self.history: Dict[str, list] = {"train_loss": [], "val_loss": [], "accuracy": []}

        def train_batch(
            self,
            x: torch.Tensor,
            y_class: torch.Tensor,
            y_value: torch.Tensor,
            confidence_weight: float = 0.1,
            value_weight: float = 0.3,
        ) -> float:
            self.model.train()
            self.optimizer.zero_grad()

            outputs = self.model(x)

            cls_loss = self.classification_loss(outputs["logits"], y_class)
            val_loss = self.value_loss(outputs["value"].squeeze(), y_value)

            correct_mask = (outputs["prediction"] == y_class).float()
            conf_loss = -torch.mean(outputs["confidence"].squeeze() * correct_mask)

            total_loss = cls_loss + value_weight * val_loss + confidence_weight * conf_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            return float(total_loss.item())

        def evaluate(
            self, x: torch.Tensor, y_class: torch.Tensor, y_value: torch.Tensor
        ) -> Dict[str, float]:
            self.model.eval()

            with torch.no_grad():
                outputs = self.model(x)

                cls_loss = self.classification_loss(outputs["logits"], y_class)
                val_loss = self.value_loss(outputs["value"].squeeze(), y_value)
                accuracy = (outputs["prediction"] == y_class).float().mean()

            return {
                "loss": float(cls_loss.item() + 0.3 * val_loss.item()),
                "accuracy": float(accuracy.item()),
                "avg_confidence": float(outputs["confidence"].mean().item()),
            }

        def save_checkpoint(self, path: Path):
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "history": self.history,
                },
                path,
            )
            logger.info(f"Model checkpoint saved to {path}")

        def load_checkpoint(self, path: Path):
            if path.exists():
                checkpoint = torch.load(path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.history = checkpoint.get("history", self.history)
                logger.info(f"Model checkpoint loaded from {path}")

    class MarketPredictor:
        """High-level predictor interface for live sequence evaluation."""

        def __init__(self, model_path: Optional[Path] = None, device: Optional[str] = None):
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self.device = device
            self.model = LSTMPredictor().to(device)
            self.trainer = LSTMTrainer(self.model, device=device)

            if model_path and model_path.exists():
                self.trainer.load_checkpoint(model_path)

            logger.info(f"MarketPredictor initialized on {device}")

        async def predict(
            self, features: np.ndarray, return_attention: bool = False
        ) -> Dict[str, Any]:
            self.model.eval()

            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(x, return_attention=return_attention)

            result = {
                "direction": ["call", "put", "hold"][int(outputs["prediction"].item())],
                "probabilities": {
                    "call": float(outputs["probabilities"][0, 0].item()),
                    "put": float(outputs["probabilities"][0, 1].item()),
                    "hold": float(outputs["probabilities"][0, 2].item()),
                },
                "confidence": float(outputs["confidence"].item()),
                "expected_value": float(outputs["value"].item()),
            }

            if return_attention:
                result["attention_weights"] = outputs["attention_weights"].cpu().numpy()

            return result

        def online_update(self, features: np.ndarray, actual_class: int, actual_value: float):
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            y_class = torch.LongTensor([actual_class]).to(self.device)
            y_value = torch.FloatTensor([actual_value]).to(self.device)

            loss = self.trainer.train_batch(x, y_class, y_value)
            logger.debug(f"Online update completed with loss: {loss:.4f}")

else:
    # Lightweight stub when torch is absent
    class AttentionLayer:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for AttentionLayer")

    class LSTMPredictor:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for LSTMPredictor")

    class LSTMTrainer:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for LSTMTrainer")

    class MarketPredictor:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for MarketPredictor")


__all__ = ["LSTMPredictor", "LSTMTrainer", "MarketPredictor", "AttentionLayer"]
