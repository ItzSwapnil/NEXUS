"""
AI Model Training Infrastructure.

This module provides the complete training pipeline for all AI models in NEXUS.
"""

import asyncio
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from nexus.ai.deep_rl_agent import DeepRLAgent
from nexus.ai.lstm_predictor import MarketPredictor
from nexus.data.provider import DataProvider, FeatureEngineer
from nexus.utils.device import get_best_device
from nexus.utils.logger import get_nexus_logger

logger = get_nexus_logger("nexus.ai.train_models")


class ModelTrainingPipeline:
    """
    Complete training pipeline for NEXUS AI models with GPU acceleration support.
    """

    def __init__(
        self,
        data_dir: Path = Path("data"),
        model_dir: Path = Path("models"),
        device: Optional[str] = None,
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model_dir.mkdir(exist_ok=True, parents=True)

        if device is None:
            device = get_best_device(enable_gpu=True)
        self.device = device

        logger.info(f"Training pipeline initialized on GPU/Device: {device}")

    async def prepare_training_data(
        self, asset: str = "EURUSD", timeframe: int = 5, samples: int = 10000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data with features and labels.

        Returns:
            Tuple of (features, labels_class, labels_value)
        """
        logger.info(f"Preparing training data for {asset} ({samples} samples)...")

        # Initialize data provider
        provider = DataProvider()
        engineer = FeatureEngineer()

        # Get historical data
        df = await provider.get_ohlcv(
            symbol=asset,
            timeframe=timeframe,
            limit=samples + 100,  # Extra for feature calculation
            source="synthetic",  # Use synthetic for training demo
        )

        # Add technical features
        df = engineer.add_features(df)

        # Remove NaN rows
        df = df.dropna()

        # Extract features (last 20 columns as example)
        feature_cols = [col for col in df.columns if col not in ["timestamp", "volume"]][-20:]

        # Create sequences for LSTM
        sequence_length = 60
        features = []
        labels_class = []
        labels_value = []

        for i in range(len(df) - sequence_length - 1):
            # Feature sequence
            seq = df[feature_cols].iloc[i : i + sequence_length].values
            features.append(seq)

            # Labels (next timestep)
            next_close = df["close"].iloc[i + sequence_length]
            current_close = df["close"].iloc[i + sequence_length - 1]

            # Classification: 0=call, 1=put, 2=hold
            price_change = (next_close - current_close) / current_close
            if price_change > 0.001:  # 0.1% threshold
                label_class = 0  # Call
            elif price_change < -0.001:
                label_class = 1  # Put
            else:
                label_class = 2  # Hold

            labels_class.append(label_class)
            labels_value.append(price_change * 100)  # As percentage

        X = np.array(features)
        y_class = np.array(labels_class)
        y_value = np.array(labels_value)

        logger.info(f"Prepared {len(X)} training samples")
        logger.info(f"Feature shape: {X.shape}")
        logger.info(
            f"Label distribution: Call={np.sum(y_class == 0)}, Put={np.sum(y_class == 1)}, Hold={np.sum(y_class == 2)}"
        )

        return X, y_class, y_value

    async def train_lstm(
        self, epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001
    ):
        """Train LSTM predictor."""
        logger.info("Starting LSTM training...")

        # Prepare data
        X, y_class, y_value = await self.prepare_training_data()

        # Split train/val
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_class_train, y_class_val = y_class[:split_idx], y_class[split_idx:]
        y_value_train, y_value_val = y_value[:split_idx], y_value[split_idx:]

        # Initialize model
        predictor = MarketPredictor(device=self.device)
        trainer = predictor.trainer

        # Training loop
        best_val_acc = 0.0
        for epoch in range(epochs):
            # Training
            train_losses = []
            for i in range(0, len(X_train), batch_size):
                batch_x = torch.FloatTensor(X_train[i : i + batch_size]).to(self.device)
                batch_y_class = torch.LongTensor(y_class_train[i : i + batch_size]).to(self.device)
                batch_y_value = torch.FloatTensor(y_value_train[i : i + batch_size]).to(self.device)

                loss = trainer.train_batch(batch_x, batch_y_class, batch_y_value)
                train_losses.append(loss)

            # Validation
            val_metrics = trainer.evaluate(
                torch.FloatTensor(X_val).to(self.device),
                torch.LongTensor(y_class_val).to(self.device),
                torch.FloatTensor(y_value_val).to(self.device),
            )

            avg_train_loss = np.mean(train_losses)

            logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f"Train Loss={avg_train_loss:.4f}, "
                f"Val Loss={val_metrics['loss']:.4f}, "
                f"Val Acc={val_metrics['accuracy']:.3f}, "
                f"Confidence={val_metrics['avg_confidence']:.3f}"
            )

            # Save best model
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                save_path = self.model_dir / "lstm_predictor_best.pth"
                trainer.save_checkpoint(save_path)
                logger.info(f"✓ New best model saved (acc={best_val_acc:.3f})")

        logger.info(f"LSTM training complete! Best accuracy: {best_val_acc:.3f}")

    async def train_dqn(self, episodes: int = 1000, max_steps: int = 100):
        """Train Deep RL agent."""
        logger.info("Starting DQN training...")

        # Initialize agent
        agent = DeepRLAgent(state_dim=20, action_dim=3, device=self.device)

        # Prepare environment (simplified trading sim)
        X, _, y_value = await self.prepare_training_data()

        episode_rewards = []

        for episode in range(episodes):
            # Random starting point
            start_idx = np.random.randint(0, len(X) - max_steps)
            total_reward = 0.0
            epsilon = max(0.01, 0.5 * (0.995**episode))

            for step in range(max_steps):
                idx = start_idx + step
                state = X[idx].flatten()

                # Select action
                action, _ = agent.select_action(state, epsilon)

                # Simulate environment
                actual_return = y_value[idx]

                # Reward: based on whether action was correct
                if (
                    (action == 0 and actual_return > 0)
                    or (action == 1 and actual_return < 0)
                    or (action == 2 and abs(actual_return) < 0.1)
                ):
                    reward = abs(actual_return)
                else:
                    reward = -abs(actual_return)

                total_reward += reward

                # Next state
                next_idx = min(idx + 1, len(X) - 1)
                next_state = X[next_idx].flatten()
                done = step == max_steps - 1

                # Store transition
                agent.store_transition(state, action, reward, next_state, done)

                # Learn
                agent.learn()

            episode_rewards.append(total_reward)

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                logger.info(
                    f"Episode {episode + 1}/{episodes}: "
                    f"Avg Reward={avg_reward:.3f}, "
                    f"Epsilon={epsilon:.3f}"
                )

            # Save checkpoint
            if (episode + 1) % 100 == 0:
                save_path = self.model_dir / f"dqn_agent_ep{episode + 1}.pth"
                agent.save(save_path)

        # Save final model
        final_path = self.model_dir / "dqn_agent_final.pth"
        agent.save(final_path)
        logger.info("DQN training complete!")


async def main():
    """Main training entry point."""
    print("=" * 80)
    print("NEXUS AI Model Training Pipeline")
    print("=" * 80)
    print()

    pipeline = ModelTrainingPipeline()

    print("Select training option:")
    print("  1. Train LSTM Predictor")
    print("  2. Train DQN Agent")
    print("  3. Train Both")
    print()

    choice = input("Enter choice (1-3): ")

    if choice == "1":
        await pipeline.train_lstm(epochs=50)
    elif choice == "2":
        await pipeline.train_dqn(episodes=1000)
    elif choice == "3":
        await pipeline.train_lstm(epochs=50)
        await pipeline.train_dqn(episodes=1000)
    else:
        print("Invalid choice!")

    print()
    print("Training complete!")


if __name__ == "__main__":
    asyncio.run(main())
