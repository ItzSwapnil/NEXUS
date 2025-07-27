"""
Model management module for NEXUS.

This module handles model training, evaluation, and prediction for the NEXUS trading system.
"""

import asyncio
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from nexus.config import ModelConfig

logger = logging.getLogger("nexus.models")

class LSTMModel(nn.Module):
    """LSTM model for time series prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, dropout: float = 0.2):
        """
        Initialize the LSTM model.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units
            num_layers: Number of LSTM layers
            output_dim: Number of output dimensions
            dropout: Dropout rate
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_dim)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        
        return out

class BaseModel:
    """Base class for all models."""
    
    def __init__(self, name: str, model_type: str, parameters: Dict[str, Any]):
        """
        Initialize the base model.
        
        Args:
            name: Model name
            model_type: Model type
            parameters: Model parameters
        """
        self.name = name
        self.model_type = model_type
        self.parameters = parameters
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.trained = False
        self.last_trained = None
        self.metrics: Dict[str, float] = {}
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Dict[str, float]: Training metrics
        """
        raise NotImplementedError("Subclasses must implement train method")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Predictions
        """
        raise NotImplementedError("Subclasses must implement predict method")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Probability predictions
        """
        raise NotImplementedError("Subclasses must implement predict_proba method")
    
    def save(self, path: Path) -> None:
        """
        Save the model.
        
        Args:
            path: Path to save the model
        """
        raise NotImplementedError("Subclasses must implement save method")
    
    def load(self, path: Path) -> bool:
        """
        Load the model.
        
        Args:
            path: Path to load the model from
            
        Returns:
            bool: True if model was loaded successfully, False otherwise
        """
        raise NotImplementedError("Subclasses must implement load method")

class SklearnModel(BaseModel):
    """Wrapper for scikit-learn models."""
    
    def __init__(self, name: str, model_type: str, parameters: Dict[str, Any]):
        """
        Initialize the scikit-learn model.
        
        Args:
            name: Model name
            model_type: Model type
            parameters: Model parameters
        """
        super().__init__(name, model_type, parameters)
        
        # Create the model based on type
        if model_type == "random_forest":
            self.model = RandomForestClassifier(**parameters)
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(**parameters)
        elif model_type == "logistic_regression":
            self.model = LogisticRegression(**parameters)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Dict[str, float]: Training metrics
        """
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="binary"),
            "recall": recall_score(y_test, y_pred, average="binary"),
            "f1": f1_score(y_test, y_pred, average="binary")
        }
        
        self.metrics = metrics
        self.trained = True
        self.last_trained = datetime.now()
        
        logger.info(f"Model {self.name} trained successfully with metrics: {metrics}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.trained:
            raise RuntimeError("Model not trained")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Probability predictions
        """
        if not self.trained:
            raise RuntimeError("Model not trained")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make probability predictions
        return self.model.predict_proba(X_scaled)
    
    def save(self, path: Path) -> None:
        """
        Save the model.
        
        Args:
            path: Path to save the model
        """
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_columns": self.feature_columns,
                "trained": self.trained,
                "last_trained": self.last_trained,
                "metrics": self.metrics
            }, f)
        
        logger.info(f"Model {self.name} saved to {path}")
    
    def load(self, path: Path) -> bool:
        """
        Load the model.
        
        Args:
            path: Path to load the model from
            
        Returns:
            bool: True if model was loaded successfully, False otherwise
        """
        try:
            # Load the model
            with open(path, "rb") as f:
                data = pickle.load(f)
            
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.feature_columns = data["feature_columns"]
            self.trained = data["trained"]
            self.last_trained = data["last_trained"]
            self.metrics = data["metrics"]
            
            logger.info(f"Model {self.name} loaded from {path}")
            return True
            
        except Exception as e:
            logger.exception(f"Error loading model {self.name} from {path}: {e}")
            return False

class PyTorchModel(BaseModel):
    """Wrapper for PyTorch models."""
    
    def __init__(self, name: str, model_type: str, parameters: Dict[str, Any]):
        """
        Initialize the PyTorch model.
        
        Args:
            name: Model name
            model_type: Model type
            parameters: Model parameters
        """
        super().__init__(name, model_type, parameters)
        
        self.sequence_length = parameters.get("sequence_length", 10)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model parameters
        self.input_dim = parameters.get("input_dim", 10)
        self.hidden_dim = parameters.get("hidden_dim", 64)
        self.num_layers = parameters.get("num_layers", 2)
        self.output_dim = parameters.get("output_dim", 1)
        self.dropout = parameters.get("dropout", 0.2)
        
        # Training parameters
        self.batch_size = parameters.get("batch_size", 32)
        self.learning_rate = parameters.get("learning_rate", 0.001)
        self.num_epochs = parameters.get("num_epochs", 50)
    
    def _prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare data for LSTM.
        
        Args:
            X: Features
            y: Labels (optional)
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Prepared data
        """
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Reshape for LSTM: (batch_size, sequence_length, input_dim)
        X_reshaped = []
        y_reshaped = []
        
        for i in range(len(X_scaled) - self.sequence_length):
            X_reshaped.append(X_scaled[i:i+self.sequence_length])
            if y is not None:
                y_reshaped.append(y[i+self.sequence_length])
        
        X_tensor = torch.FloatTensor(np.array(X_reshaped)).to(self.device)
        
        if y is not None:
            y_tensor = torch.FloatTensor(np.array(y_reshaped)).to(self.device)
            return X_tensor, y_tensor
        
        return X_tensor, None
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Dict[str, float]: Training metrics
        """
        # Update input dimension
        self.input_dim = X.shape[1]
        
        # Create model
        self.model = LSTMModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=self.output_dim,
            dropout=self.dropout
        ).to(self.device)
        
        # Prepare data
        X_tensor, y_tensor = self._prepare_data(X, y)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Evaluate on test set
            self.model.eval()
            test_loss = 0.0
            y_true = []
            y_pred = []
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y.unsqueeze(1))
                    test_loss += loss.item()
                    
                    # Convert to binary predictions
                    preds = (outputs > 0.5).float()
                    
                    y_true.extend(batch_y.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy().flatten())
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
            recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
            f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
            
            logger.debug(
                f"Epoch {epoch+1}/{self.num_epochs}, "
                f"Train Loss: {train_loss/len(train_loader):.4f}, "
                f"Test Loss: {test_loss/len(test_loader):.4f}, "
                f"Accuracy: {accuracy:.4f}, "
                f"Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, "
                f"F1: {f1:.4f}"
            )
        
        # Final evaluation
        self.model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = self.model(batch_X)
                preds = (outputs > 0.5).float()
                
                y_true.extend(batch_y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy().flatten())
        
        # Calculate final metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="binary", zero_division=0)
        }
        
        self.metrics = metrics
        self.trained = True
        self.last_trained = datetime.now()
        
        logger.info(f"Model {self.name} trained successfully with metrics: {metrics}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.trained or self.model is None:
            raise RuntimeError("Model not trained")
        
        # Prepare data
        X_tensor, _ = self._prepare_data(X)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            preds = (outputs > 0.5).float()
        
        return preds.cpu().numpy().flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Probability predictions
        """
        if not self.trained or self.model is None:
            raise RuntimeError("Model not trained")
        
        # Prepare data
        X_tensor, _ = self._prepare_data(X)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
        
        return outputs.cpu().numpy().flatten()
    
    def save(self, path: Path) -> None:
        """
        Save the model.
        
        Args:
            path: Path to save the model
        """
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        model_path = path.with_suffix(".pt")
        torch.save(self.model.state_dict(), model_path)
        
        # Save metadata
        metadata_path = path.with_suffix(".pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "feature_columns": self.feature_columns,
                "trained": self.trained,
                "last_trained": self.last_trained,
                "metrics": self.metrics,
                "parameters": self.parameters
            }, f)
        
        logger.info(f"Model {self.name} saved to {path}")
    
    def load(self, path: Path) -> bool:
        """
        Load the model.
        
        Args:
            path: Path to load the model from
            
        Returns:
            bool: True if model was loaded successfully, False otherwise
        """
        try:
            # Load metadata
            metadata_path = path.with_suffix(".pkl")
            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
            
            self.scaler = data["scaler"]
            self.feature_columns = data["feature_columns"]
            self.trained = data["trained"]
            self.last_trained = data["last_trained"]
            self.metrics = data["metrics"]
            self.parameters.update(data["parameters"])
            
            # Update model parameters
            self.input_dim = self.parameters.get("input_dim", 10)
            self.hidden_dim = self.parameters.get("hidden_dim", 64)
            self.num_layers = self.parameters.get("num_layers", 2)
            self.output_dim = self.parameters.get("output_dim", 1)
            self.dropout = self.parameters.get("dropout", 0.2)
            
            # Create model
            self.model = LSTMModel(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                output_dim=self.output_dim,
                dropout=self.dropout
            ).to(self.device)
            
            # Load model weights
            model_path = path.with_suffix(".pt")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            logger.info(f"Model {self.name} loaded from {path}")
            return True
            
        except Exception as e:
            logger.exception(f"Error loading model {self.name} from {path}: {e}")
            return False

class ModelManager:
    """
    Model manager for the NEXUS trading system.
    
    This class handles model training, evaluation, and prediction for the NEXUS trading system.
    """
    
    def __init__(self, model_configs: List[ModelConfig]):
        """
        Initialize the model manager.
        
        Args:
            model_configs: List of model configurations
        """
        self.model_configs = model_configs
        self.models: Dict[str, BaseModel] = {}
        self.models_dir = Path("./models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.initialized = False
    
    def initialize(self) -> None:
        """Initialize the model manager."""
        logger.info("Initializing model manager")
        
        # Create models
        for config in self.model_configs:
            if not config.enabled:
                continue
            
            try:
                # Create model based on type
                if config.type in ["random_forest", "gradient_boosting", "logistic_regression"]:
                    model = SklearnModel(config.name, config.type, config.parameters)
                elif config.type in ["lstm", "gru", "rnn"]:
                    model = PyTorchModel(config.name, config.type, config.parameters)
                else:
                    logger.warning(f"Unknown model type: {config.type}")
                    continue
                
                # Try to load model
                model_path = self.models_dir / f"{config.name}"
                if model_path.with_suffix(".pkl").exists() or model_path.with_suffix(".pt").exists():
                    if model.load(model_path):
                        logger.info(f"Model {config.name} loaded successfully")
                    else:
                        logger.warning(f"Failed to load model {config.name}, will train from scratch")
                
                # Add model to dictionary
                self.models[config.name] = model
                
            except Exception as e:
                logger.exception(f"Error initializing model {config.name}: {e}")
        
        self.initialized = True
        logger.info(f"Model manager initialized with {len(self.models)} models")
    
    def update_models(self, data: Dict[str, Dict[int, pd.DataFrame]]) -> None:
        """
        Update models with new data.
        
        Args:
            data: Dictionary of feature DataFrames
        """
        if not self.initialized:
            raise RuntimeError("Model manager not initialized")
        
        logger.info("Updating models")
        
        # Prepare training data
        X_train, y_train = self._prepare_training_data(data)
        
        if len(X_train) == 0 or len(y_train) == 0:
            logger.warning("No training data available")
            return
        
        # Train each model
        for name, model in self.models.items():
            try:
                logger.info(f"Training model {name}")
                
                # Set feature columns
                model.feature_columns = list(X_train.columns)
                
                # Train model
                metrics = model.train(X_train.values, y_train.values)
                
                # Save model
                model_path = self.models_dir / f"{name}"
                model.save(model_path)
                
                logger.info(f"Model {name} trained and saved with metrics: {metrics}")
                
            except Exception as e:
                logger.exception(f"Error training model {name}: {e}")
    
    def _prepare_training_data(self, data: Dict[str, Dict[int, pd.DataFrame]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data from feature DataFrames.
        
        Args:
            data: Dictionary of feature DataFrames
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Features and labels
        """
        all_features = []
        all_labels = []
        
        # Process each asset and timeframe
        for asset, timeframes in data.items():
            for timeframe, df in timeframes.items():
                if len(df) < 100:  # Skip if not enough data
                    continue
                
                # Create features
                features = df.copy()
                
                # Create labels (1 if price goes up in next candle, 0 otherwise)
                features["target"] = (features["close"].shift(-1) > features["close"]).astype(int)
                
                # Drop rows with NaN values
                features = features.dropna()
                
                if len(features) == 0:
                    continue
                
                # Extract features and labels
                X = features.drop(["timestamp", "open", "high", "low", "close", "volume", "target"], axis=1, errors="ignore")
                y = features["target"]
                
                all_features.append(X)
                all_labels.append(y)
        
        # Combine all data
        if not all_features or not all_labels:
            return pd.DataFrame(), pd.DataFrame()
        
        X_combined = pd.concat(all_features, axis=0)
        y_combined = pd.concat(all_labels, axis=0)
        
        return X_combined, y_combined
    
    def predict(self, model_name: str, features: pd.DataFrame) -> Tuple[int, float]:
        """
        Make a prediction using a specific model.
        
        Args:
            model_name: Name of the model to use
            features: Feature DataFrame
            
        Returns:
            Tuple[int, float]: Prediction (1 for up, 0 for down) and confidence
        """
        if not self.initialized:
            raise RuntimeError("Model manager not initialized")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if not model.trained:
            raise RuntimeError(f"Model {model_name} not trained")
        
        # Prepare features
        X = features[model.feature_columns].values
        
        # Make prediction
        try:
            # Get probability
            proba = model.predict_proba(X)
            
            # Get prediction
            if isinstance(proba, np.ndarray) and proba.ndim > 0:
                if proba.shape[1] > 1:  # Multi-class
                    pred = np.argmax(proba, axis=1)[-1]
                    confidence = proba[-1, pred]
                else:  # Binary
                    confidence = proba[-1]
                    pred = 1 if confidence > 0.5 else 0
            else:  # Single value
                confidence = float(proba)
                pred = 1 if confidence > 0.5 else 0
            
            return pred, confidence
            
        except Exception as e:
            logger.exception(f"Error making prediction with model {model_name}: {e}")
            return 0, 0.5  # Default to neutral prediction
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict[str, Any]: Model information
        """
        if not self.initialized:
            raise RuntimeError("Model manager not initialized")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        return {
            "name": model.name,
            "type": model.model_type,
            "trained": model.trained,
            "last_trained": model.last_trained,
            "metrics": model.metrics,
            "parameters": model.parameters
        }
    
    def get_all_models_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all models.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of model information
        """
        if not self.initialized:
            raise RuntimeError("Model manager not initialized")
        
        return {name: self.get_model_info(name) for name in self.models}