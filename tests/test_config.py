import pytest
from nexus.utils.config import NexusSettings, QuotexSettings, TradingSettings, AISettings, MemorySettings, RegimeDetectorSettings, TransformerSettings, RLAgentSettings, EvolutionSettings


def creates_valid_settings_with_minimum_fields():
    settings = NexusSettings(
        quotex=QuotexSettings(email='a@b.com', password='pw'),
        trading=TradingSettings()
    )
    assert isinstance(settings, NexusSettings)
    assert settings.quotex.email == 'a@b.com'
    assert settings.trading.base_trade_amount == 5.0


def raises_error_if_quotex_missing():
    with pytest.raises(Exception):
        NexusSettings(trading=TradingSettings())


def raises_error_if_trading_missing():
    with pytest.raises(Exception):
        NexusSettings(quotex=QuotexSettings(email='a@b.com', password='pw'))


def accepts_all_fields_and_defaults():
    settings = NexusSettings(
        quotex=QuotexSettings(email='a@b.com', password='pw'),
        trading=TradingSettings(),
        ai=AISettings(enable_gpu=False, num_workers=2),
        memory=MemorySettings(capacity=5000),
        regime_detector=RegimeDetectorSettings(n_regimes=2),
        transformer=TransformerSettings(feature_dim=16),
        rl_agent=RLAgentSettings(state_dim=8),
        evolution=EvolutionSettings(population_size=5),
        environment='production',
        enable_gpu=False,
        num_workers=2,
        log_level='DEBUG',
        data_dir='custom_data',
        models_dir='custom_models',
        logs_dir='custom_logs',
        version='3.0.0'
    )
    assert settings.environment == 'production'
    assert settings.ai.enable_gpu is False
    assert settings.memory.capacity == 5000
    assert settings.transformer.feature_dim == 16
    assert settings.version == '3.0.0'


def string_fields_are_trimmed_and_valid():
    settings = NexusSettings(
        quotex=QuotexSettings(email='  a@b.com  ', password=' pw '),
        trading=TradingSettings()
    )
    assert settings.quotex.email.strip() == 'a@b.com'
    assert settings.quotex.password.strip() == 'pw'


def default_values_are_set_for_optional_fields():
    settings = NexusSettings(
        quotex=QuotexSettings(email='a@b.com', password='pw'),
        trading=TradingSettings()
    )
    assert settings.ai.enable_gpu is True
    assert settings.memory.dimension == 128
    assert settings.regime_detector.n_regimes == 4
    assert settings.transformer.batch_size == 128
    assert settings.rl_agent.hidden_dim == 64
    assert settings.evolution.mutation_rate == 0.1

