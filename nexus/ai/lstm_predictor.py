"""
Advanced LSTM-based Market Predictor with Attention Mechanism.

This is a REAL AI model, not just pattern matching. It uses:
- Bidirectional LSTM for temporal dependencies
- Multi-head attention for feature importance
- Dropout and batch normalization for regularization
- Online learning capability for market adaptation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

from nexus.utils.logger import get_nexus_logger

logger = get_nexus_logger("nexus.ai.lstm_predictor")


class AttentionLayer(nn.Module):
    """Multi-head attention mechanism for feature importance."""

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

        # Linear transformations
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        output = self.fc_out(attended)

        return output, attention_weights.mean(dim=1)  # Average over heads


class LSTMPredictor(nn.Module):
    """
    Advanced LSTM-based market predictor with attention and residual connections.

    This is a real deep learning model that learns market patterns from data,
    not just following predefined rules.
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
        output_classes: int = 3,  # Buy, Sell, Hold
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = AttentionLayer(hidden_dim * 2, num_heads)

        # Residual connection
        self.residual = nn.Linear(hidden_dim * 2, hidden_dim * 2)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_classes)
        )

        # Confidence head (for uncertainty estimation)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Value prediction head (for regression)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, sequence_length, features)
            return_attention: Whether to return attention weights

        Returns:
            Dictionary containing predictions, confidence, and optionally attention
        """
        batch_size, seq_len, _ = x.shape

        # Embed input features
        x_embedded = x.view(-1, self.input_dim)
        x_embedded = self.input_embedding(x_embedded)
        x_embedded = x_embedded.view(batch_size, seq_len, -1)

        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x_embedded)

        # Apply attention
        attended, attention_weights = self.attention(lstm_out)

        # Residual connection
        residual = self.residual(lstm_out)
        combined = attended + residual

        # Use last timestep for prediction
        final_state = combined[:, -1, :]

        # Get predictions
        logits = self.classifier(final_state)
        confidence = self.confidence_head(final_state)
        value = self.value_head(final_state)

        # Softmax for probabilities
        probabilities = torch.softmax(logits, dim=-1)

        result = {
            'logits': logits,
            'probabilities': probabilities,
            'confidence': confidence,
            'value': value,
            'prediction': torch.argmax(probabilities, dim=-1)
        }

        if return_attention:
            result['attention_weights'] = attention_weights

        return result


class LSTMTrainer:
    """
    Trainer for LSTM predictor with online learning capabilities.
    """

    def __init__(
        self,
        model: LSTMPredictor,
        learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.value_loss = nn.MSELoss()

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'accuracy': []
        }

    def train_batch(
        self,
        x: torch.Tensor,
        y_class: torch.Tensor,
        y_value: torch.Tensor,
        confidence_weight: float = 0.1,
        value_weight: float = 0.3
    ) -> float:
        """
        Train on a single batch.

        Args:
            x: Input features (batch, seq_len, features)
            y_class: Target classes (batch,)
            y_value: Target values (batch,)
            confidence_weight: Weight for confidence loss
            value_weight: Weight for value prediction loss

        Returns:
            Total loss value
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(x)

        # Calculate losses
        cls_loss = self.classification_loss(outputs['logits'], y_class)
        val_loss = self.value_loss(outputs['value'].squeeze(), y_value)

        # Confidence regularization (encourage high confidence for correct predictions)
        correct_mask = (outputs['prediction'] == y_class).float()
        conf_loss = -torch.mean(outputs['confidence'].squeeze() * correct_mask)

        # Combined loss
        total_loss = cls_loss + value_weight * val_loss + confidence_weight * conf_loss

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return total_loss.item()

    def evaluate(
        self,
        x: torch.Tensor,
        y_class: torch.Tensor,
        y_value: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate model on validation data."""
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(x)

            cls_loss = self.classification_loss(outputs['logits'], y_class)
            val_loss = self.value_loss(outputs['value'].squeeze(), y_value)

            accuracy = (outputs['prediction'] == y_class).float().mean()

        return {
            'loss': cls_loss.item() + 0.3 * val_loss.item(),
            'accuracy': accuracy.item(),
            'avg_confidence': outputs['confidence'].mean().item()
        }

    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        logger.info(f"Model checkpoint saved to {path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.history = checkpoint.get('history', self.history)
            logger.info(f"Model checkpoint loaded from {path}")
        else:
            logger.warning(f"Checkpoint not found: {path}")


class MarketPredictor:
    """
    High-level interface for market prediction using LSTM.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[str] = None
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        self.model = LSTMPredictor().to(device)
        self.trainer = LSTMTrainer(self.model, device=device)

        if model_path and model_path.exists():
            self.trainer.load_checkpoint(model_path)

        logger.info(f"MarketPredictor initialized on {device}")

    async def predict(
        self,
        features: np.ndarray,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """
        Make prediction on market features.

        Args:
            features: Array of shape (sequence_length, num_features)
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with prediction, confidence, and value
        """
        self.model.eval()

        # Convert to tensor and add batch dimension
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(x, return_attention=return_attention)

        # Convert to numpy for easier use
        result = {
            'direction': ['call', 'put', 'hold'][outputs['prediction'].item()],
            'probabilities': {
                'call': outputs['probabilities'][0, 0].item(),
                'put': outputs['probabilities'][0, 1].item(),
                'hold': outputs['probabilities'][0, 2].item()
            },
            'confidence': outputs['confidence'].item(),
            'expected_value': outputs['value'].item()
        }

        if return_attention:
            result['attention_weights'] = outputs['attention_weights'].cpu().numpy()

        return result

    def online_update(
        self,
        features: np.ndarray,
        actual_class: int,
        actual_value: float
    ):
        """
        Perform online learning update with new data.

        Args:
            features: Market features
            actual_class: Actual outcome (0=call won, 1=put won, 2=hold)
            actual_value: Actual profit/loss
        """
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        y_class = torch.LongTensor([actual_class]).to(self.device)
        y_value = torch.FloatTensor([actual_value]).to(self.device)

        loss = self.trainer.train_batch(x, y_class, y_value)
        logger.debug(f"Online update completed with loss: {loss:.4f}")


    __all__ = ['LSTMPredictor', 'LSTMTrainer', 'MarketPredictor', 'AttentionLayer']
"""
Advanced LSTM-based Market Predictor with Attention Mechanism.

This is a REAL AI model, not just pattern matching. It uses:
- Bidirectional LSTM for temporal dependencies
- Multi-head attention for feature importance
- Dropout and batch normalization for regularization
- Online learning capability for market adaptation

Note: Requires PyTorch to be installed.
"""

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    # Provide dummy classes when PyTorch is not available
    class nn:
        class Module:
            pass
    torch = None

import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

from nexus.utils.logger import get_nexus_logger

logger = get_nexus_logger("nexus.ai.lstm_predictor")


if not _HAS_TORCH:
    # Dummy implementations when PyTorch is not installed
    class AttentionLayer:
        """Dummy attention layer - PyTorch not installed."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for LSTM predictor. Install with: uv pip install torch")

    class LSTMPredictor:
        """Dummy LSTM predictor - PyTorch not installed."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for LSTM predictor. Install with: uv pip install torch")

    class LSTMTrainer:
        """Dummy LSTM trainer - PyTorch not installed."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for LSTM predictor. Install with: uv pip install torch")

    class MarketPredictor:
        """Dummy market predictor - PyTorch not installed."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for LSTM predictor. Install with: uv pip install torch")

    __all__ = ['LSTMPredictor', 'LSTMTrainer', 'MarketPredictor', 'AttentionLayer']

else:
    # Real implementations when PyTorch is available


class AttentionLayer(nn.Module):
    """Multi-head attention mechanism for feature importance."""

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

        # Linear transformations
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        output = self.fc_out(attended)

        return output, attention_weights.mean(dim=1)  # Average over heads


class LSTMPredictor(nn.Module):
    """
    Advanced LSTM-based market predictor with attention and residual connections.

    This is a real deep learning model that learns market patterns from data,
    not just following predefined rules.
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
        output_classes: int = 3,  # Buy, Sell, Hold
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = AttentionLayer(hidden_dim * 2, num_heads)

        # Residual connection
        self.residual = nn.Linear(hidden_dim * 2, hidden_dim * 2)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_classes)
        )

        # Confidence head (for uncertainty estimation)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Value prediction head (for regression)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, sequence_length, features)
            return_attention: Whether to return attention weights

        Returns:
            Dictionary containing predictions, confidence, and optionally attention
        """
        batch_size, seq_len, _ = x.shape

        # Embed input features
        x_embedded = x.view(-1, self.input_dim)
        x_embedded = self.input_embedding(x_embedded)
        x_embedded = x_embedded.view(batch_size, seq_len, -1)

        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x_embedded)

        # Apply attention
        attended, attention_weights = self.attention(lstm_out)

        # Residual connection
        residual = self.residual(lstm_out)
        combined = attended + residual

        # Use last timestep for prediction
        final_state = combined[:, -1, :]

        # Get predictions
        logits = self.classifier(final_state)
        confidence = self.confidence_head(final_state)
        value = self.value_head(final_state)

        # Softmax for probabilities
        probabilities = torch.softmax(logits, dim=-1)

        result = {
            'logits': logits,
            'probabilities': probabilities,
            'confidence': confidence,
            'value': value,
            'prediction': torch.argmax(probabilities, dim=-1)
        }

        if return_attention:
            result['attention_weights'] = attention_weights

        return result


class LSTMTrainer:
    """
    Trainer for LSTM predictor with online learning capabilities.
    """

    def __init__(
        self,
        model: LSTMPredictor,
        learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.value_loss = nn.MSELoss()

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'accuracy': []
        }

    def train_batch(
        self,
        x: torch.Tensor,
        y_class: torch.Tensor,
        y_value: torch.Tensor,
        confidence_weight: float = 0.1,
        value_weight: float = 0.3
    ) -> float:
        """
        Train on a single batch.

        Args:
            x: Input features (batch, seq_len, features)
            y_class: Target classes (batch,)
            y_value: Target values (batch,)
            confidence_weight: Weight for confidence loss
            value_weight: Weight for value prediction loss

        Returns:
            Total loss value
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(x)

        # Calculate losses
        cls_loss = self.classification_loss(outputs['logits'], y_class)
        val_loss = self.value_loss(outputs['value'].squeeze(), y_value)

        # Confidence regularization (encourage high confidence for correct predictions)
        correct_mask = (outputs['prediction'] == y_class).float()
        conf_loss = -torch.mean(outputs['confidence'].squeeze() * correct_mask)

        # Combined loss
        total_loss = cls_loss + value_weight * val_loss + confidence_weight * conf_loss

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return total_loss.item()

    def evaluate(
        self,
        x: torch.Tensor,
        y_class: torch.Tensor,
        y_value: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate model on validation data."""
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(x)

            cls_loss = self.classification_loss(outputs['logits'], y_class)
            val_loss = self.value_loss(outputs['value'].squeeze(), y_value)

            accuracy = (outputs['prediction'] == y_class).float().mean()

        return {
            'loss': cls_loss.item() + 0.3 * val_loss.item(),
            'accuracy': accuracy.item(),
            'avg_confidence': outputs['confidence'].mean().item()
        }

    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        logger.info(f"Model checkpoint saved to {path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.history = checkpoint.get('history', self.history)
            logger.info(f"Model checkpoint loaded from {path}")
        else:
            logger.warning(f"Checkpoint not found: {path}")


class MarketPredictor:
    """
    High-level interface for market prediction using LSTM.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[str] = None
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        self.model = LSTMPredictor().to(device)
        self.trainer = LSTMTrainer(self.model, device=device)

        if model_path and model_path.exists():
            self.trainer.load_checkpoint(model_path)

        logger.info(f"MarketPredictor initialized on {device}")

    async def predict(
        self,
        features: np.ndarray,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """
        Make prediction on market features.

        Args:
            features: Array of shape (sequence_length, num_features)
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with prediction, confidence, and value
        """
        self.model.eval()

        # Convert to tensor and add batch dimension
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(x, return_attention=return_attention)

        # Convert to numpy for easier use
        result = {
            'direction': ['call', 'put', 'hold'][outputs['prediction'].item()],
            'probabilities': {
                'call': outputs['probabilities'][0, 0].item(),
                'put': outputs['probabilities'][0, 1].item(),
                'hold': outputs['probabilities'][0, 2].item()
            },
            'confidence': outputs['confidence'].item(),
            'expected_value': outputs['value'].item()
        }

        if return_attention:
            result['attention_weights'] = outputs['attention_weights'].cpu().numpy()

        return result

    def online_update(
        self,
        features: np.ndarray,
        actual_class: int,
        actual_value: float
    ):
        """
        Perform online learning update with new data.

        Args:
            features: Market features
            actual_class: Actual outcome (0=call won, 1=put won, 2=hold)
            actual_value: Actual profit/loss
        """
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        y_class = torch.LongTensor([actual_class]).to(self.device)
        y_value = torch.FloatTensor([actual_value]).to(self.device)

        loss = self.trainer.train_batch(x, y_class, y_value)
        logger.debug(f"Online update completed with loss: {loss:.4f}")


__all__ = ['LSTMPredictor', 'LSTMTrainer', 'MarketPredictor', 'AttentionLayer']

