"""Lightweight RLAgent implementation for tests.

Provides a simple in-memory replay buffer and placeholder learning hook.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Deque, Any, Dict
from collections import deque
import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class RLAgent:
    def __init__(self, state_dim: int, hidden_dim: int, buffer_capacity: int = 10000):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.buffer_capacity = buffer_capacity
        self.memory: Deque[Transition] = deque(maxlen=buffer_capacity)

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.append(Transition(state=state, action=action, reward=reward, next_state=next_state, done=done))

    def learn_from_trade(self, trade_result: Dict[str, Any]) -> None:
        """Record trade result as a transition. Placeholder for learning logic."""
        state = trade_result.get("state")
        action = trade_result.get("action", 0)
        reward = float(trade_result.get("reward", 0.0))
        next_state = trade_result.get("next_state", state)
        done = bool(trade_result.get("done", False))
        if state is not None and next_state is not None:
            self.store_transition(state, int(action), reward, next_state, done)

    def sample_batch(self, batch_size: int):  # pragma: no cover
        if len(self.memory) < batch_size:
            return list(self.memory)
        idx = np.random.choice(len(self.memory), batch_size, replace=False)
        mem_list = list(self.memory)
        return [mem_list[i] for i in idx]


__all__ = ["RLAgent"]

