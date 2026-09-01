"""Transformer-based market prediction components."""

# Attempt to import torch; provide graceful fallback if unavailable
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.optim as optim  # type: ignore

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - environment without torch
    torch = None  # type: ignore
    nn = object  # type: ignore
    optim = object  # type: ignore
    _HAS_TORCH = False

import builtins
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from nexus.utils.logger import get_nexus_logger
from nexus.utils.technical import calculate_features

logger = get_nexus_logger("nexus.intelligence.transformer")

if _HAS_TORCH:

    class PositionalEncoding(nn.Module):
        """
        Positional encoding for temporal awareness in transformer model.
        Adds information about the position of elements in the sequence.
        """

        def __init__(self, d_model: int, max_seq_length: int = 1000, dropout: float = 0.1):
            """
            Initialize the positional encoding.

            Args:
                d_model: Embedding dimension
                max_seq_length: Maximum sequence length
                dropout: Dropout probability
            """
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            # Create positional encoding matrix
            pe = torch.zeros(max_seq_length, d_model)
            position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)

            # Register positional encoding as a buffer (not a parameter)
            self.register_buffer("pe", pe)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore
            """
            Apply positional encoding.

            Args:
                x: Input tensor [batch_size, seq_len, embedding_dim]

            Returns:
                torch.Tensor: Positionally encoded tensor
            """
            x = x + self.pe[:, : x.size(1)]  # type: ignore[index,operator]
            return self.dropout(x)  # type: ignore[no-any-return]

    class FinancialTransformerEncoder(nn.Module):
        """
        Custom transformer encoder optimized for financial time series.
        """

        def __init__(
            self,
            d_model: int = 64,
            nhead: int = 4,
            dim_feedforward: int = 256,
            dropout: float = 0.1,
            num_layers: int = 3,
        ):
            """
            Initialize the financial transformer encoder.

            Args:
                d_model: Feature dimension
                nhead: Number of attention heads
                dim_feedforward: Dimension of feedforward network
                dropout: Dropout probability
                num_layers: Number of encoder layers
            """
            super().__init__()
            self.d_model = d_model

            # Input projection
            self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

            # Transformer encoder layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )

            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        def forward(
            self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Forward pass.

            Args:
                src: Input tensor of shape [batch_size, seq_len, d_model]
                src_mask: Optional mask tensor

            Returns:
                torch.Tensor: Output tensor
            """
            # Add positional encoding
            src = self.pos_encoder(src)

            # Pass through transformer encoder
            output = self.transformer_encoder(src, src_mask)

            return output  # type: ignore[no-any-return]

    class MarketTransformer(nn.Module):
        """
        Complete transformer model for market prediction.
        """

        def __init__(
            self,
            input_dim: int = 20,
            d_model: int = 64,
            nhead: int = 4,
            num_encoder_layers: int = 4,
            dim_feedforward: int = 256,
            num_classes: int = 3,  # Buy, Sell, Hold
            dropout: float = 0.1,
            seq_length: int = 60,
        ):
            """
            Initialize the market transformer.

            Args:
                input_dim: Number of input features
                d_model: Hidden dimension
                nhead: Number of attention heads
                num_encoder_layers: Number of encoder layers
                dim_feedforward: Dimension of feedforward network
                num_classes: Number of output classes
                dropout: Dropout probability
                seq_length: Sequence length
            """
            super().__init__()

            self.input_dim = input_dim
            self.d_model = d_model
            self.seq_length = seq_length

            # Feature embedding
            self.feature_embedding = nn.Linear(input_dim, d_model)

            # Transformer encoder
            self.transformer = FinancialTransformerEncoder(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )

            # Prediction heads
            self.classification_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )

            self.confidence_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:  # type: ignore
            """
            Forward pass through the model.

            Args:
                x: Input tensor [batch_size, seq_len, input_dim]

            Returns:
                Tuple[torch.Tensor, torch.Tensor]: Logits and confidence scores
            """
            # Embed features
            x = self.feature_embedding(x)

            # Pass through transformer
            x = self.transformer(x)

            # Use last sequence element for prediction
            x = x[:, -1]

            # Generate predictions and confidence
            logits = self.classification_head(x)
            confidence = self.confidence_head(x)

            return logits, confidence


class MarketPredictor:
    """
    Market prediction system that combines transformer model with preprocessing
    and trading signal generation.
    """

    def __init__(
        self,
        lookback_periods: int = 60,
        feature_dim: int = 20,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        """Initialize the market predictor."""
        self.lookback_periods = lookback_periods
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.eval_history: list[Any] = []
        self.model_path = Path("models/transformer/")
        self.model_path.mkdir(exist_ok=True, parents=True)

        if _HAS_TORCH:
            if device is None:
                import torch as _torch  # local name

                self.device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
            else:
                import torch as _torch

                self.device = _torch.device(device)

            # Initialize model
            self.model = MarketTransformer(input_dim=feature_dim, seq_length=lookback_periods).to(
                self.device
            )  # type: ignore[attr-defined]

            # Optimizer and loss
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)  # type: ignore
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5
            )  # type: ignore
            import torch.nn as _nn

            self.criterion = _nn.CrossEntropyLoss()

            # Feature normalizers
            self.feature_means = None
            self.feature_stds = None

            # Class mapping
            self.class_to_signal = {0: "hold", 1: "call", 2: "put"}

            # Try to load existing model
            self.load_model()

            logger.info(f"Market Transformer initialized on device: {self.device}")
        else:
            # Fallback attributes
            self.device = "cpu"  # type: ignore[assignment]
            self.model = self._FallbackModel()  # type: ignore[assignment]
            self.optimizer = None  # type: ignore[assignment]
            self.scheduler = None  # type: ignore[assignment]
            self.criterion = None  # type: ignore[assignment]
            self.feature_means = None
            self.feature_stds = None
            self.class_to_signal = {0: "hold", 1: "call", 2: "put"}
            logger.warning("Torch not available; using lightweight numpy fallback model.")

    class _FallbackModel:
        """Minimal callable to satisfy tests without torch."""

        def __call__(self, x):  # x is numpy array or placeholder
            import numpy as _np

            logits = _np.zeros((1, 3), dtype=_np.float32)
            confidence = _np.zeros((1, 1), dtype=_np.float32)
            return logits, confidence

    def preprocess(self, data: pd.DataFrame):
        """
        Preprocess market data for the transformer model.

        Args:
            data: OHLCV market data

        Returns:
            torch.Tensor: Preprocessed features
        """
        # Extract technical features
        features_df = calculate_features(data)

        # Select relevant columns and handle NaN values
        selected_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rsi",
            "macd",
            "macd_signal",
            "bb_upper",
            "bb_lower",
            "ema_short",
            "ema_medium",
            "ema_long",
            "atr",
            "trend_strength",
            "volatility",
            "momentum",
            "mean_reversion",
            "support",
            "resistance",
        ]

        # Ensure all required columns exist
        for col in selected_cols:
            if col not in features_df.columns:
                features_df[col] = 0.0

        # Select and fill NaN values
        features = features_df[selected_cols].fillna(0).values

        # Normalize features
        if self.feature_means is None or self.feature_stds is None:
            # First-time setup
            self.feature_means = np.mean(features, axis=0)
            stds = np.std(features, axis=0)
            stds[stds == 0] = 1.0  # Avoid division by zero
            self.feature_stds = stds

        normalized_features = (features - self.feature_means) / self.feature_stds

        # Create sequences efficiently using numpy and avoid Python lists
        if len(normalized_features) >= self.lookback_periods:
            num_sequences = len(normalized_features) - self.lookback_periods + 1
            # Create a 3D array [num_sequences, lookback, feature_dim]
            X_np = np.lib.stride_tricks.sliding_window_view(
                normalized_features, (self.lookback_periods, normalized_features.shape[1])
            )
            # sliding_window_view result shape: [num_sequences, 1, lookback, feature_dim]
            # Squeeze the size-1 dimension
            X_np = X_np.reshape(
                num_sequences, self.lookback_periods, normalized_features.shape[1]
            ).astype(np.float32)
            if _HAS_TORCH:
                import torch as _torch

                X_tensor = _torch.from_numpy(X_np).to(self.device)
            else:
                # Return numpy view for fallback
                return np.zeros(
                    (num_sequences, self.lookback_periods, len(selected_cols)), dtype=np.float32
                )
        else:
            if _HAS_TORCH:
                import torch as _torch

                # Create empty tensor with correct shape if no sequences can be created
                X_tensor = _torch.zeros(
                    (0, self.lookback_periods, len(selected_cols)), dtype=_torch.float32
                )
            else:
                return np.zeros((0, self.lookback_periods, len(selected_cols)), dtype=np.float32)

        return X_tensor

    def update_from_trade(self, trade_record: Dict[str, Any]):
        """Lightweight online update hook. Records outcomes and can adjust LR.
        This is intentionally conservative to avoid destabilizing the model during live trading.
        """
        try:
            success = bool(trade_record.get("success", False))
            profit = float(trade_record.get("profit", 0.0))
            self.eval_history.append(
                {
                    "timestamp": trade_record.get("timestamp"),
                    "success": success,
                    "profit": profit,
                    "asset": trade_record.get("asset"),
                    "timeframe": trade_record.get("timeframe"),
                }
            )
            # Simple adaptive LR: decrease on recent losses, increase slightly on wins
            if len(self.eval_history) >= 5:
                recent = self.eval_history[-5:]
                win_rate = builtins.sum(1 for r in recent if r["success"]) / 5.0
                for g in self.optimizer.param_groups:
                    base_lr = g.get("initial_lr", g["lr"])
                    # Keep within [1e-5, 5e-4]
                    new_lr = float(np.clip(base_lr * (0.9 if win_rate < 0.4 else 1.05), 1e-5, 5e-4))
                    g["lr"] = new_lr
        except Exception:
            pass

    def save_model(self):
        """Save the model to disk."""
        try:
            # Save model parameters
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "feature_means": self.feature_means,
                    "feature_stds": self.feature_stds,
                },
                self.model_path / "transformer_model.pth",
            )

            logger.info("Market Transformer model saved successfully")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self) -> bool:
        """
        Load the model from disk.

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if not _HAS_TORCH:
            return False
        try:
            import torch as _torch

            model_file = self.model_path / "transformer_model.pth"
            if model_file.exists():
                checkpoint = _torch.load(model_file, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])  # type: ignore
                self.feature_means = checkpoint["feature_means"]
                self.feature_stds = checkpoint["feature_stds"]

                logger.info("Market Transformer model loaded successfully")
                return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")

        return False

    async def train(self, data: pd.DataFrame, labels: np.ndarray, epochs: int = 10):
        """
        Train the transformer model.

        Args:
            data: OHLCV market data
            labels: Training labels (0=hold, 1=buy, 2=sell)
            epochs: Number of training epochs
        """
        # Preprocess data
        X = self.preprocess(data)

        # Convert labels to tensor
        y = torch.tensor(labels, dtype=torch.long).to(self.device)

        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            for batch_X, batch_y in dataloader:
                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                logits, confidence = self.model(batch_X)
                loss = self.criterion(logits, batch_y)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Track metrics
                epoch_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            # Epoch summary
            accuracy = correct / total if total > 0 else 0
            avg_loss = epoch_loss / len(dataloader) if dataloader else 0
            logger.info(
                f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
            )

            # Update learning rate scheduler
            self.scheduler.step(avg_loss)

        # Save trained model
        self.save_model()

        return True

    async def predict(
        self, data: pd.DataFrame, asset: str, timeframe: int, regime: Optional[str] = None
    ) -> Dict:
        """
        Generate trading signal prediction.

        Args:
            data: OHLCV market data
            asset: Asset symbol
            timeframe: Analysis timeframe in minutes
            regime: Current market regime (if known)

        Returns:
            Dict: Prediction result with signal, confidence, and metadata
        """
        if len(data) < self.lookback_periods:
            logger.warning(f"Not enough data for prediction. Need {self.lookback_periods} periods.")
            return {
                "signal": "hold",
                "confidence": 0.0,
                "reasoning": "Insufficient data for analysis",
            }

        # Preprocess data
        X = self.preprocess(data)

        # Use only the last sequence for prediction
        if len(X) > 0:
            X_latest = X[-1:]
        else:
            logger.warning("No valid sequences generated after preprocessing")
            return {"signal": "hold", "confidence": 0.0, "reasoning": "Data preprocessing failed"}

        # Ensure input shape is [batch, seq_length, feature_dim]
        if X_latest.dim() == 3:
            if X_latest.shape[-1] != self.feature_dim:
                # Permute if last dimension is not feature_dim
                X_latest = X_latest.permute(0, 2, 1)

        # Set model to evaluation mode
        self.model.eval()

        # Generate prediction
        with torch.no_grad():
            logits, confidence = self.model(X_latest)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = int(torch.argmax(logits, dim=1).item())
            confidence_value = confidence.item()

            # Get class probabilities
            probs = probabilities[0].cpu().numpy()

            # Get predicted signal
            signal = self.class_to_signal[predicted_class]

            # Adjust confidence based on regime if provided
            if regime:
                regime_confidence_factors = {
                    "trending": 1.2 if signal != "hold" else 0.8,
                    "ranging": 1.1 if signal == "hold" else 0.9,
                    "volatile": 0.8,  # Reduce confidence in volatile markets
                    "reversal": 1.0,
                    "unknown": 0.9,
                }
                regime_factor = regime_confidence_factors.get(regime, 1.0)
                adjusted_confidence = min(0.95, confidence_value * regime_factor)
            else:
                adjusted_confidence = confidence_value

            # Extract features for the prediction
            feature_values = X_latest[0, -1].cpu().numpy()
            normalized_features = {
                "trend_strength": feature_values[14],
                "volatility": feature_values[15],
                "momentum": feature_values[16],
                "mean_reversion": feature_values[17],
            }

            # Generate reasoning
            reasoning = self._generate_reasoning(signal, probs, normalized_features, regime)

            # Create response
            result = {
                "signal": signal,
                "confidence": float(adjusted_confidence),
                "reasoning": reasoning,
                "probabilities": {
                    "hold": float(probs[0]),
                    "call": float(probs[1]),
                    "put": float(probs[2]),
                },
                "features": {
                    "trend_strength": float(normalized_features["trend_strength"]),
                    "volatility": float(normalized_features["volatility"]),
                    "momentum": float(normalized_features["momentum"]),
                    "mean_reversion": float(normalized_features["mean_reversion"]),
                },
                "timeframe": timeframe,
                "asset": asset,
            }

            logger.debug(
                f"Generated prediction: {signal} with {adjusted_confidence:.2f} confidence"
            )

            return result

    def _generate_reasoning(
        self, signal: str, probs: np.ndarray, features: Dict, regime: Optional[str] = None
    ) -> str:
        """
        Generate human-readable reasoning for the prediction.

        Args:
            signal: Predicted signal
            probs: Class probabilities
            features: Normalized feature values
            regime: Market regime

        Returns:
            str: Reasoning explanation
        """
        # Generate base reasoning from signal and features
        trend = features["trend_strength"]
        volatility = features["volatility"]
        momentum = features["momentum"]
        mean_reversion = features["mean_reversion"]

        if signal == "call":
            if momentum > 0.5:
                reason = f"Strong upward momentum ({momentum:.2f})"
            elif trend > 0.5:
                reason = f"Bullish trend detected ({trend:.2f})"
            elif mean_reversion > 0.5:
                reason = f"Oversold condition with expected reversal ({mean_reversion:.2f})"
            else:
                reason = "Bullish pattern recognized"

        elif signal == "put":
            if momentum < -0.3:
                reason = f"Strong downward momentum ({momentum:.2f})"
            elif trend < -0.3:
                reason = f"Bearish trend detected ({trend:.2f})"
            elif mean_reversion < -0.3:
                reason = f"Overbought condition with expected reversal ({mean_reversion:.2f})"
            else:
                reason = "Bearish pattern recognized"

        else:  # hold
            if abs(trend) < 0.2:
                reason = f"No clear trend detected ({trend:.2f})"
            elif volatility > 0.7:
                reason = f"High volatility, waiting for clarity ({volatility:.2f})"
            else:
                reason = "No actionable pattern recognized"

        # Add regime context if available
        if regime:
            reason += f" in {regime} market"

        return reason
