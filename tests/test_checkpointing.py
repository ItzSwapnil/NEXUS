"""Unit tests for Model Checkpointing and Persistence."""

from nexus.intelligence.checkpointing import ModelCheckpointManager


def test_checkpoint_manager_saves_and_loads_ensemble_weights(tmp_path):
    manager = ModelCheckpointManager(checkpoint_dir=str(tmp_path))

    weights = {"transformer": 0.35, "lstm": 0.40, "rl_agent": 0.25}
    saved = manager.save_checkpoint(ensemble_weights=weights)
    assert saved is True

    loaded = manager.load_checkpoint()
    assert loaded == weights


def test_checkpoint_manager_handles_missing_directory(tmp_path):
    nested_dir = tmp_path / "deep" / "models"
    manager = ModelCheckpointManager(checkpoint_dir=str(nested_dir))

    weights = {"transformer": 0.50, "lstm": 0.50}
    manager.save_checkpoint(ensemble_weights=weights)

    assert (nested_dir / "ensemble_weights.json").exists()
