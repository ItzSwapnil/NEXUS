import pytest
from nexus.core.engine import NexusEngine
from nexus.utils.config import NexusSettings

@pytest.fixture
def engine():
    settings = NexusSettings()
    return NexusEngine(settings=settings, demo_mode=True, auto_login=False)

def test_engine_initialization(engine):
    assert engine is not None
    assert hasattr(engine, 'strategy_registry')
    assert hasattr(engine, 'model_registry')
    assert hasattr(engine, 'risk_registry')
    assert hasattr(engine, 'meta_strategy') or engine.meta_strategy is None

def test_register_strategy(engine):
    class DummyStrategy:
        pass
    engine.register_strategy('dummy', DummyStrategy())
    assert 'dummy' in engine.strategy_registry
    engine.unregister_strategy('dummy')
    assert 'dummy' not in engine.strategy_registry

def test_emotional_state_update(engine):
    trade_result = {'success': True}
    engine.update_emotional_state(trade_result)
    assert 0.0 <= engine.emotion_state['greed'] <= 1.0
    trade_result = {'success': False}
    engine.update_emotional_state(trade_result)
    assert 0.0 <= engine.emotion_state['fear'] <= 1.0

def test_advanced_risk_management(engine):
    context = {}
    base_amount = 1000.0
    pos_size = engine.advanced_risk_management(context, base_amount)
    assert pos_size >= 1.0

