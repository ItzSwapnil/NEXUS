from nexus.intelligence.adaptive import Experience, ExperienceStore


def test_experience_store_settles_only_after_feedback(tmp_path) -> None:
    store = ExperienceStore(tmp_path / "experiences.sqlite3")
    experience = Experience(
        experience_id="e1",
        timestamp=1.0,
        asset="EURUSD",
        state={"novelty_score": 0.2},
        action="long",
        forecast={"expected_return": 0.001},
        source="test",
    )
    store.append(experience)
    assert store.resolved_replay() == []
    assert store.update_outcome("e1", {"status": "settled", "reward": 0.01, "win": True})
    resolved = store.resolved_replay()
    assert len(resolved) == 1
    assert resolved[0].reward == 0.01
    assert resolved[0].outcome["status"] == "settled"
