"""
Model Checkpointing and Persistence Manager for NEXUS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from nexus.utils.device import get_best_device
from nexus.utils.logger import get_nexus_logger

logger = get_nexus_logger("nexus.intelligence.checkpointing")

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


class ModelCheckpointManager:
    """
    Manages saving and loading model checkpoints and ensemble weights.
    """

    def __init__(self, checkpoint_dir: str = "models"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = get_best_device()

    def save_checkpoint(
        self,
        transformer_model: Optional[Any] = None,
        bilstm_model: Optional[Any] = None,
        rl_agent: Optional[Any] = None,
        ensemble_weights: Optional[Dict[str, float]] = None,
    ) -> bool:
        """
        Save all component models and ensemble weights.
        """
        success = True

        if _HAS_TORCH and torch is not None:
            # Save Transformer
            if transformer_model is not None and hasattr(transformer_model, "state_dict"):
                try:
                    torch.save(
                        transformer_model.state_dict(),
                        self.checkpoint_dir / "transformer.pt",
                    )
                    logger.info("Saved Transformer model checkpoint")
                except Exception as e:
                    logger.error("Failed to save Transformer checkpoint: %s", e)
                    success = False

            # Save Bi-LSTM Attention
            if bilstm_model is not None and hasattr(bilstm_model, "state_dict"):
                try:
                    torch.save(
                        bilstm_model.state_dict(),
                        self.checkpoint_dir / "bilstm_attention.pt",
                    )
                    logger.info("Saved Bi-LSTM Attention model checkpoint")
                except Exception as e:
                    logger.error("Failed to save Bi-LSTM Attention checkpoint: %s", e)
                    success = False

            # Save Deep RL Agent
            if rl_agent is not None and hasattr(rl_agent, "q_network"):
                try:
                    rl_state = {
                        "q_network": rl_agent.q_network.state_dict(),
                        "target_network": rl_agent.target_network.state_dict(),
                    }
                    torch.save(rl_state, self.checkpoint_dir / "deep_rl_agent.pt")
                    logger.info("Saved Deep RL Agent model checkpoint")
                except Exception as e:
                    logger.error("Failed to save Deep RL Agent checkpoint: %s", e)
                    success = False

        # Save Ensemble Weights JSON
        if ensemble_weights:
            try:
                weights_path = self.checkpoint_dir / "ensemble_weights.json"
                with open(weights_path, "w", encoding="utf-8") as f:
                    json.dump(ensemble_weights, f, indent=2)
                logger.info("Saved Ensemble weights JSON: %s", ensemble_weights)
            except Exception as e:
                logger.error("Failed to save Ensemble weights JSON: %s", e)
                success = False

        return success

    def load_checkpoint(
        self,
        transformer_model: Optional[Any] = None,
        bilstm_model: Optional[Any] = None,
        rl_agent: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Load component model weights and saved ensemble weights.

        Returns:
            Dict[str, float]: Loaded ensemble weights if available.
        """
        loaded_weights: Dict[str, float] = {}

        if _HAS_TORCH and torch is not None:
            # Load Transformer
            t_path = self.checkpoint_dir / "transformer.pt"
            if (
                t_path.exists()
                and transformer_model is not None
                and hasattr(transformer_model, "load_state_dict")
            ):
                try:
                    state = torch.load(t_path, map_location=self.device)
                    transformer_model.load_state_dict(state)
                    logger.info("Loaded Transformer model checkpoint")
                except Exception as e:
                    logger.warning("Could not load Transformer checkpoint: %s", e)

            # Load Bi-LSTM
            b_path = self.checkpoint_dir / "bilstm_attention.pt"
            if (
                b_path.exists()
                and bilstm_model is not None
                and hasattr(bilstm_model, "load_state_dict")
            ):
                try:
                    state = torch.load(b_path, map_location=self.device)
                    bilstm_model.load_state_dict(state)
                    logger.info("Loaded Bi-LSTM Attention model checkpoint")
                except Exception as e:
                    logger.warning("Could not load Bi-LSTM Attention checkpoint: %s", e)

            # Load Deep RL Agent
            rl_path = self.checkpoint_dir / "deep_rl_agent.pt"
            if rl_path.exists() and rl_agent is not None and hasattr(rl_agent, "q_network"):
                try:
                    state = torch.load(rl_path, map_location=self.device)
                    rl_agent.q_network.load_state_dict(state["q_network"])
                    rl_agent.target_network.load_state_dict(state["target_network"])
                    logger.info("Loaded Deep RL Agent model checkpoint")
                except Exception as e:
                    logger.warning("Could not load Deep RL Agent checkpoint: %s", e)

        # Load Ensemble Weights JSON
        weights_path = self.checkpoint_dir / "ensemble_weights.json"
        if weights_path.exists():
            try:
                with open(weights_path, "r", encoding="utf-8") as f:
                    loaded_weights = json.load(f)
                logger.info("Loaded Ensemble weights JSON: %s", loaded_weights)
            except Exception as e:
                logger.warning("Could not load Ensemble weights JSON: %s", e)

        return loaded_weights
