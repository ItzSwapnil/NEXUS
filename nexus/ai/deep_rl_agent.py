"""
Deep Q-Network (DQN) Agent for Trading
Author: Swapnil De Sarkar
Created: 2025

Advanced Deep Q-Network (DQN) with Prioritized Experience Replay.

This is a reinforcement learning agent I developed that learns optimal trading strategies
through interaction with the market, using dueling architecture and noisy layers.
"""

import random
from collections import namedtuple
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from nexus.utils.logger import get_nexus_logger

logger = get_nexus_logger("nexus.ai.deep_rl_agent")

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration (NoisyNet)."""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w_eps: torch.Tensor = getattr(self, "weight_epsilon")  # noqa: B009
            b_eps: torch.Tensor = getattr(self, "bias_epsilon")  # noqa: B009
            weight = self.weight_mu + self.weight_sigma * w_eps
            bias = self.bias_mu + self.bias_sigma * b_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture for better value estimation.

    Separates state value and action advantages for more stable learning.
    """

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, use_noisy: bool = True
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_noisy = use_noisy

        # Shared feature extraction
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Value stream (estimates state value)
        if use_noisy:
            self.value_stream = nn.Sequential(
                NoisyLinear(hidden_dim, hidden_dim // 2), nn.ReLU(), NoisyLinear(hidden_dim // 2, 1)
            )
        else:
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1)
            )

        # Advantage stream (estimates action advantages)
        if use_noisy:
            self.advantage_stream = nn.Sequential(
                NoisyLinear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                NoisyLinear(hidden_dim // 2, action_dim),
            )
        else:
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim),
            )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.features(state)

        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine value and advantages
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))

        return q_values  # type: ignore[no-any-return]

    def reset_noise(self):
        """Reset noise in noisy layers."""
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay for more efficient learning.
    """

    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: List[Transition] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def push(self, transition: Transition, priority: Optional[float] = None):
        """Add transition with priority."""
        if priority is None:
            priority = max(self.priorities) if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """Sample batch with importance sampling weights."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        # Calculate sampling probabilities
        priorities = self.priorities[: len(self.buffer)]
        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        samples = [self.buffer[idx] for idx in indices]

        return samples, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities, strict=False):
            self.priorities[idx] = priority + 1e-6  # Small epsilon to avoid zero priority

    def __len__(self):
        return len(self.buffer)


class DeepRLAgent:
    """
    Advanced Deep Reinforcement Learning agent for trading.

    Features:
    - Double DQN for reduced overestimation
    - Dueling architecture for better value estimation
    - Prioritized experience replay
    - Noisy networks for exploration
    - Multi-step returns
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,  # 0: Call, 1: Put, 2: Hold
        hidden_dim: int = 256,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100000,
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Q-networks
        self.q_network = DuelingDQN(state_dim, action_dim, hidden_dim).to(device)
        self.target_network = DuelingDQN(state_dim, action_dim, hidden_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Experience replay
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)

        # Learning step counter
        self.learn_step_counter = 0

        logger.info(
            f"DeepRLAgent initialized on {device} with {state_dim} states and {action_dim} actions"
        )

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> Tuple[int, float]:
        """
        Select action using epsilon-greedy policy.

        Returns:
            Tuple of (action, q_value)
        """
        if random.random() < epsilon:
            action = random.randrange(self.action_dim)
            q_value = 0.0
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = q_values.argmax(dim=1).item()
                q_value = q_values[0, action].item()

        return action, q_value

    def store_transition(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ):
        """Store transition in replay buffer."""
        transition = Transition(state, action, reward, next_state, done)
        self.replay_buffer.push(transition)

    def learn(self) -> Optional[float]:
        """
        Perform one learning step.

        Returns:
            Loss value if learning occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        beta = min(1.0, 0.4 + self.learn_step_counter * 0.0001)
        batch, indices, weights = self.replay_buffer.sample(self.batch_size, beta)

        # Unpack batch
        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
        actions = torch.LongTensor([t.action for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(self.device)
        dones = torch.FloatTensor([t.done for t in batch]).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: use Q-network to select action, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q_values = (
                self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            )
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Calculate TD errors for priority update
        td_errors = torch.abs(current_q_values - target_q_values)

        # Weighted loss
        loss = (
            weights_tensor
            * nn.functional.smooth_l1_loss(current_q_values, target_q_values, reduction="none")
        ).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        # Soft update target network
        self._soft_update_target_network()

        # Reset noise
        self.q_network.reset_noise()
        self.target_network.reset_noise()

        self.learn_step_counter += 1

        return float(loss.item())

    def _soft_update_target_network(self):
        """Soft update of target network parameters."""
        for target_param, param in zip(
            self.target_network.parameters(), self.q_network.parameters(), strict=False
        ):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path: Path):
        """Save agent state."""
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "learn_step_counter": self.learn_step_counter,
            },
            path,
        )
        logger.info(f"Agent saved to {path}")

    def load(self, path: Path):
        """Load agent state."""
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.q_network.load_state_dict(checkpoint["q_network"])
            self.target_network.load_state_dict(checkpoint["target_network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.learn_step_counter = checkpoint["learn_step_counter"]
            logger.info(f"Agent loaded from {path}")
        else:
            logger.warning(f"Checkpoint not found: {path}")


__all__ = ["DeepRLAgent", "DuelingDQN", "PrioritizedReplayBuffer"]
