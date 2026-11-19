"""Reinforcement learning agent implementation.

Provides an in-memory experience replay buffer and Q-learning updates.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class RLAgent:
    """Linear Q-learning agent with experience replay."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        buffer_capacity: int = 10000,
        lr: float = 0.01,
        gamma: float = 0.95,
    ):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.buffer_capacity = buffer_capacity
        self.lr = lr
        self.gamma = gamma
        self.memory: Deque[Transition] = deque(maxlen=buffer_capacity)
        # Linear Q-value approximator: Q(s, a) = weights[:, a] . s + bias[a]
        self.weights: np.ndarray = np.zeros((state_dim, 3), dtype=np.float32)
        self.bias: np.ndarray = np.zeros(3, dtype=np.float32)

    def store_transition(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ) -> None:
        self.memory.append(
            Transition(state=state, action=action, reward=reward, next_state=next_state, done=done)
        )

    def learn_from_trade(self, trade_result: Dict[str, Any]) -> None:
        """Record trade result and perform temporal-difference Q-update."""
        state = trade_result.get("state")
        action = int(trade_result.get("action", 0))
        reward = float(trade_result.get("reward", 0.0))
        next_state = trade_result.get("next_state", state)
        done = bool(trade_result.get("done", False))
        if state is not None and next_state is not None:
            self.store_transition(state, action, reward, next_state, done)
            s = np.asarray(state, dtype=np.float32).flatten()
            ns = np.asarray(next_state, dtype=np.float32).flatten()
            if len(s) == self.state_dim and len(ns) == self.state_dim and 0 <= action < 3:
                q_current = float(np.dot(s, self.weights[:, action]) + self.bias[action])
                q_next_max = 0.0 if done else float(np.max(np.dot(ns, self.weights) + self.bias))
                td_error = (reward + self.gamma * q_next_max) - q_current
                self.weights[:, action] += self.lr * td_error * s
                self.bias[action] += self.lr * td_error

    async def predict(
        self,
        data: Any,
        asset: str = "EURUSD",
        timeframe: int = 1,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate trading decision using estimated Q-values."""
        if hasattr(data, "select_dtypes"):
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                state_vec = numeric_data.iloc[-1].values
                if len(state_vec) >= self.state_dim:
                    state_vec = state_vec[-self.state_dim :]
                else:
                    state_vec = np.pad(state_vec, (0, self.state_dim - len(state_vec)))
            else:
                state_vec = np.zeros(self.state_dim, dtype=np.float32)
        else:
            state_vec = np.zeros(self.state_dim, dtype=np.float32)

        q_values = np.dot(state_vec, self.weights) + self.bias
        action_idx = int(np.argmax(q_values))
        action_map = {0: "hold", 1: "buy", 2: "sell"}
        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / np.sum(exp_q)
        confidence = float(probs[action_idx])

        return {
            "signal": action_map.get(action_idx, "hold"),
            "confidence": confidence,
            "q_values": q_values.tolist(),
            "reasoning": f"RL Q-decision({action_map.get(action_idx, 'hold')})",
        }

    def sample_batch(self, batch_size: int):  # pragma: no cover
        if len(self.memory) < batch_size:
            return list(self.memory)
        idx = np.random.choice(len(self.memory), batch_size, replace=False)
        mem_list = list(self.memory)
        return [mem_list[i] for i in idx]


__all__ = ["RLAgent"]
