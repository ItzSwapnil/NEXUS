"""
Advanced Reinforcement Learning Agent for NEXUS Trading System

This module implements a sophisticated RL agent using Deep Q-Networks (DQN) with:
- Double DQN for stable learning
- Prioritized Experience Replay
- Dueling network architecture
- Multi-step returns
- Adaptive exploration strategies
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import math
import asyncio
import pandas as pd
from pathlib import Path

from nexus.utils.logger import get_nexus_logger
from nexus.utils.technical import calculate_features

logger = get_nexus_logger("nexus.intelligence.rl_agent")

# Define experience tuple structure
Experience = namedtuple('Experience',
                        field_names=['state', 'action', 'reward', 'next_state', 'done'])

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for more efficient learning.
    Stores experiences with priorities based on TD error.
    """

    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        """
        Initialize the buffer.

        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform sampling)
            beta: Importance sampling exponent (1 = no correction)
            beta_increment: Beta increment per sampling
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        """
        Add an experience to memory with max priority.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Initialize with max priority for new experiences
        max_priority = self.priorities.max() if self.size > 0 else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append(Experience(state, action, reward, next_state, done))
        else:
            self.memory[self.position] = Experience(state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[List[Experience], torch.Tensor, torch.Tensor]:
        """
        Sample a batch of experiences based on priorities.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (experiences, indices, weights)
        """
        if self.size < batch_size:
            # Not enough samples yet
            return [], torch.tensor([]), torch.tensor([])

        # Get sampling probabilities from priorities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices based on probabilities
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)

        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        weights = torch.tensor(weights, dtype=torch.float)

        # Get experiences
        experiences = [self.memory[idx] for idx in indices]

        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return experiences, torch.tensor(indices), weights

    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor):
        """
        Update priorities for sampled experiences.

        Args:
            indices: Indices of sampled experiences
            priorities: New TD error priorities
        """
        for idx, priority in zip(indices.tolist(), priorities.tolist()):
            self.priorities[idx] = max(1e-6, priority)  # Small epsilon to avoid zero priority

    def __len__(self) -> int:
        """Get current buffer size."""
        return self.size


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture with separate value and advantage streams.
    This improves learning by separating state value from action advantages.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize the network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        # Feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Value stream (estimates state value)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Advantage stream (estimates action advantages)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            x: Input state tensor

        Returns:
            torch.Tensor: Q-values for each action
        """
        features = self.feature_layer(x)

        # Get value and advantages
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine value and advantages using dueling architecture
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values


class RLAgent:
    """
    Advanced RL agent for trading using Double DQN with dueling architecture.
    """

    # Define actions
    ACTION_BUY = 0   # "call" in Quotex
    ACTION_SELL = 1  # "put" in Quotex
    ACTION_HOLD = 2  # Do nothing

    def __init__(
        self,
        state_dim: int = 20,
        hidden_dim: int = 128,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        gamma: float = 0.99,
        learning_rate: float = 0.0005,
        target_update: int = 10,
        device: str = None
    ):
        """
        Initialize the RL agent.

        Args:
            state_dim: Dimension of state space
            hidden_dim: Hidden layer dimension
            buffer_capacity: Replay buffer capacity
            batch_size: Training batch size
            gamma: Discount factor
            learning_rate: Learning rate
            target_update: Target network update frequency
            device: Computing device ('cuda', 'cpu')
        """
        self.state_dim = state_dim
        self.action_dim = 3  # Buy, Sell, Hold
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update

        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Create policy and target networks
        self.policy_net = DuelingDQN(state_dim, self.action_dim, hidden_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, self.action_dim, hidden_dim).to(self.device)

        # Initialize target network with policy network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network always in eval mode

        # Create optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Create replay buffer
        self.memory = PrioritizedReplayBuffer(buffer_capacity)

        # Training parameters
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.1  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Exploration rate decay
        self.steps_done = 0
        self.update_count = 0

        # Feature normalizers
        self.feature_means = None
        self.feature_stds = None

        # Model persistence
        self.model_path = Path("models/rl_agent/")
        self.model_path.mkdir(exist_ok=True, parents=True)

        # Action mapping
        self.action_to_signal = {
            self.ACTION_BUY: "call",
            self.ACTION_SELL: "put",
            self.ACTION_HOLD: "hold"
        }

        # Load existing model if available
        self.load_model()

        logger.info(f"RL Agent initialized on device: {self.device}")

    def preprocess_state(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess market data into state representation.

        Args:
            data: OHLCV market data

        Returns:
            np.ndarray: State vector
        """
        # Extract technical features
        features_df = calculate_features(data)

        # Select relevant features for state representation
        selected_cols = [
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
            'ema_short', 'ema_medium', 'ema_long', 'atr',
            'trend_strength', 'volatility', 'momentum',
            'mean_reversion', 'volume_trend',
            'close_norm', 'high_low_diff', 'daily_return',
            'price_pattern', 'support', 'resistance'
        ]

        # Ensure all required columns exist
        for col in selected_cols:
            if col not in features_df.columns:
                features_df[col] = 0.0

        # Get most recent state (last row)
        state = features_df[selected_cols].fillna(0).iloc[-1].values

        # Normalize state
        if self.feature_means is None or self.feature_stds is None:
            # Use all data for normalization statistics first time
            all_states = features_df[selected_cols].fillna(0).values
            self.feature_means = np.mean(all_states, axis=0)
            self.feature_stds = np.std(all_states, axis=0)
            self.feature_stds[self.feature_stds == 0] = 1.0  # Avoid division by zero

        normalized_state = (state - self.feature_means) / self.feature_stds

        return normalized_state

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            explore: Whether to use exploration

        Returns:
            int: Selected action
        """
        if explore and random.random() < self.epsilon:
            # Random exploration
            return random.randint(0, self.action_dim - 1)
        else:
            # Greedy exploitation
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()

    def optimize_model(self):
        """Train the model using a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return

        # Sample experiences from memory
        experiences, indices, weights = self.memory.sample(self.batch_size)

        # Convert experiences to tensors
        states = torch.tensor(np.array([e.state for e in experiences]),
                             dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array([e.action for e in experiences]),
                              dtype=torch.long).to(self.device)
        rewards = torch.tensor(np.array([e.reward for e in experiences]),
                              dtype=torch.float).to(self.device)
        next_states = torch.tensor(np.array([e.next_state for e in experiences]),
                                  dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array([e.done for e in experiences]),
                            dtype=torch.bool).to(self.device)
        weights = weights.to(self.device)

        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute next Q values using Double DQN approach
        # 1. Get actions from policy net
        next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
        # 2. Get Q-values from target net for these actions
        next_q_values = self.target_net(next_states).gather(1, next_actions)
        # 3. Compute target Q values
        target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))

        # Compute TD errors for prioritization
        td_errors = torch.abs(current_q_values - target_q_values).detach() + 1e-6

        # Compute loss with importance sampling weights
        loss = (weights.unsqueeze(1) * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

        self.optimizer.step()

        # Update priorities in replay buffer
        self.memory.update_priorities(indices, td_errors.squeeze(1).cpu())

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.memory.add(state, action, reward, next_state, done)

    def store_experience(self, state, action, reward, next_state, done):
        """Store an experience in the replay buffer."""
        if hasattr(self, 'replay_buffer'):
            self.replay_buffer.add(state, action, reward, next_state, done)

    def save_model(self):
        """Save model weights and parameters."""
        try:
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'feature_means': self.feature_means,
                'feature_stds': self.feature_stds,
                'epsilon': self.epsilon,
                'steps_done': self.steps_done,
                'update_count': self.update_count
            }, self.model_path / "rl_agent.pth")

            logger.info("RL Agent model saved successfully")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self) -> bool:
        """
        Load model weights and parameters.

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            model_file = self.model_path / "rl_agent.pth"
            if model_file.exists():
                checkpoint = torch.load(model_file, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint['policy_net'])
                self.target_net.load_state_dict(checkpoint['target_net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.feature_means = checkpoint['feature_means']
                self.feature_stds = checkpoint['feature_stds']
                self.epsilon = checkpoint['epsilon']
                self.steps_done = checkpoint['steps_done']
                self.update_count = checkpoint['update_count']

                logger.info("RL Agent model loaded successfully")
                return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")

        return False

    def confidence_from_q_values(self, q_values: torch.Tensor, action: int) -> float:
        """
        Calculate confidence from Q-values.

        Args:
            q_values: Q-values for all actions
            action: Selected action

        Returns:
            float: Confidence score (0.0-1.0)
        """
        # Convert to numpy for easier manipulation
        q_vals = q_values.cpu().numpy()

        # Get Q-value for selected action
        q_value = q_vals[action]

        # Calculate advantage over other actions
        other_actions = [a for a in range(self.action_dim) if a != action]
        other_q_values = [q_vals[a] for a in other_actions]

        # If no other actions, return high confidence
        if not other_q_values:
            return 0.9

        # Calculate advantage
        max_other_q = max(other_q_values)
        advantage = q_value - max_other_q

        # Convert advantage to confidence (0.0-1.0)
        confidence = sigmoid(advantage)

        return confidence

    async def predict(self, data: pd.DataFrame, asset: str = None, timeframe: int = None, regime: str = None) -> Dict:
        """
        Generate a trading signal prediction.

        Args:
            data: OHLCV market data
            asset: Asset symbol (optional)
            timeframe: Analysis timeframe (optional)
            regime: Market regime (optional)

        Returns:
            Dict: Prediction result with signal, confidence, and metadata
        """
        # Preprocess state
        state = self.preprocess_state(data)

        # Select action (without exploration for prediction)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action = q_values.max(1)[1].item()

        # Get signal from action
        signal = self.action_to_signal[action]

        # Calculate confidence
        confidence = self.confidence_from_q_values(q_values[0], action)

        # Adjust confidence based on regime
        if regime:
            regime_confidence_factors = {
                "trending": 1.1 if action != self.ACTION_HOLD else 0.9,
                "ranging": 1.1 if action == self.ACTION_HOLD else 0.9,
                "volatile": 0.8,  # Reduce confidence in volatile markets
                "reversal": 1.0,
                "unknown": 0.9
            }
            regime_factor = regime_confidence_factors.get(regime, 1.0)
            adjusted_confidence = min(0.95, confidence * regime_factor)
        else:
            adjusted_confidence = confidence

        # Get Q-values for reasoning
        q_values_list = q_values[0].cpu().tolist()

        # Generate reasoning
        reasoning = self._generate_reasoning(action, q_values_list, regime)

        # Create result
        result = {
            "signal": signal,
            "confidence": float(adjusted_confidence),
            "reasoning": reasoning,
            "q_values": {
                "call": float(q_values_list[self.ACTION_BUY]),
                "put": float(q_values_list[self.ACTION_SELL]),
                "hold": float(q_values_list[self.ACTION_HOLD])
            },
            "action": int(action),
            "timeframe": timeframe,
            "asset": asset
        }

        return result

    def _generate_reasoning(self, action: int, q_values: List[float], regime: Optional[str] = None) -> str:
        """
        Generate reasoning for prediction based on Q-values.

        Args:
            action: Selected action
            q_values: Q-values for all actions
            regime: Current market regime

        Returns:
            str: Reasoning explanation
        """
        action_names = {
            self.ACTION_BUY: "Call (Buy)",
            self.ACTION_SELL: "Put (Sell)",
            self.ACTION_HOLD: "Hold"
        }

        # Calculate advantages
        advantages = {}
        for a in range(self.action_dim):
            if a != action:
                advantages[a] = q_values[action] - q_values[a]

        # Create reasoning based on advantages
        reasoning = f"{action_names[action]} action selected with Q-value {q_values[action]:.3f}"

        if advantages:
            max_adv_action = max(advantages.items(), key=lambda x: x[1])
            if max_adv_action[1] > 1.0:
                reasoning += f", strong advantage ({max_adv_action[1]:.2f}) over {action_names[max_adv_action[0]]}"
            elif max_adv_action[1] > 0.5:
                reasoning += f", moderate advantage over alternatives"
            else:
                reasoning += f", slight advantage over alternatives"

        # Add regime context
        if regime:
            reasoning += f" in {regime} market condition"

        return reasoning


def sigmoid(x: float) -> float:
    """Sigmoid function that converts values to range (0, 1)."""
    return 1.0 / (1.0 + math.exp(-x))
