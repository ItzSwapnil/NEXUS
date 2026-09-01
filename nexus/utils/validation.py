"""
Input validation utilities for NEXUS trading system.

Provides comprehensive validation for trading parameters, asset symbols,
amounts, timeframes, and other critical inputs to ensure system integrity.
"""

import re
from decimal import Decimal, InvalidOperation
from typing import Any, List, Optional, Tuple

from nexus.utils.logger import get_nexus_logger

logger = get_nexus_logger("nexus.utils.validation")


class ValidationError(ValueError):
    """Custom exception for validation errors."""

    pass


def validate_asset_symbol(symbol: str) -> str:
    """
    Validate and normalize asset symbol.

    Args:
        symbol: Asset symbol to validate

    Returns:
        str: Normalized symbol (uppercase, stripped)

    Raises:
        ValidationError: If symbol is invalid
    """
    if not symbol or not isinstance(symbol, str):
        raise ValidationError("Asset symbol must be a non-empty string")

    normalized = symbol.strip().upper()

    # Check length (typically 3-10 characters for forex/crypto/stocks)
    if not (2 <= len(normalized) <= 20):
        raise ValidationError(f"Asset symbol length invalid: {normalized}")

    # Allow alphanumeric, spaces, and common separators
    if not re.match(r"^[A-Z0-9\s&\-/]+$", normalized):
        raise ValidationError(f"Asset symbol contains invalid characters: {normalized}")

    return normalized


def validate_trade_amount(
    amount: float, min_amount: float = 1.0, max_amount: float = 10000.0
) -> float:
    """
    Validate trade amount is within acceptable range.

    Args:
        amount: Trade amount to validate
        min_amount: Minimum allowed amount
        max_amount: Maximum allowed amount

    Returns:
        float: Validated amount

    Raises:
        ValidationError: If amount is invalid
    """
    try:
        amount_decimal = Decimal(str(amount))
    except (InvalidOperation, ValueError, TypeError) as err:
        raise ValidationError(f"Invalid amount format: {amount}") from err

    if amount_decimal <= 0:
        raise ValidationError(f"Amount must be positive: {amount}")

    if amount_decimal < Decimal(str(min_amount)):
        raise ValidationError(f"Amount {amount} below minimum {min_amount}")

    if amount_decimal > Decimal(str(max_amount)):
        raise ValidationError(f"Amount {amount} exceeds maximum {max_amount}")

    return float(amount_decimal)


def validate_direction(direction: str) -> str:
    """
    Validate trade direction.

    Args:
        direction: Trade direction (call/put, buy/sell, up/down)

    Returns:
        str: Normalized direction ('call' or 'put')

    Raises:
        ValidationError: If direction is invalid
    """
    if not direction or not isinstance(direction, str):
        raise ValidationError("Direction must be a non-empty string")

    normalized = direction.strip().lower()

    # Map various common terms to call/put
    call_synonyms = {"call", "buy", "up", "long", "higher", "green"}
    put_synonyms = {"put", "sell", "down", "short", "lower", "red"}

    if normalized in call_synonyms:
        return "call"
    elif normalized in put_synonyms:
        return "put"
    else:
        raise ValidationError(f"Invalid direction: {direction}. Use 'call' or 'put'")


def validate_expiration(expiration: Any, min_seconds: int = 60, max_seconds: int = 3600) -> int:
    """
    Validate expiration time in seconds.

    Args:
        expiration: Expiration time in seconds
        min_seconds: Minimum allowed expiration
        max_seconds: Maximum allowed expiration

    Returns:
        int: Validated expiration in seconds

    Raises:
        ValidationError: If expiration is invalid
    """
    try:
        exp_int = int(expiration)
    except (ValueError, TypeError) as err:
        raise ValidationError(f"Expiration must be an integer: {expiration}") from err

    if exp_int < min_seconds:
        raise ValidationError(f"Expiration {exp_int}s below minimum {min_seconds}s")

    if exp_int > max_seconds:
        raise ValidationError(f"Expiration {exp_int}s exceeds maximum {max_seconds}s")

    return exp_int


def validate_timeframe(timeframe: Any, allowed_timeframes: Optional[List[int]] = None) -> int:
    """
    Validate timeframe in minutes.

    Args:
        timeframe: Timeframe in minutes
        allowed_timeframes: List of allowed timeframes (optional)

    Returns:
        int: Validated timeframe

    Raises:
        ValidationError: If timeframe is invalid
    """
    if allowed_timeframes is None:
        allowed_timeframes = [1, 5, 15, 30, 60, 240, 1440]  # Standard timeframes

    try:
        tf_int = int(timeframe)
    except (ValueError, TypeError) as err:
        raise ValidationError(f"Timeframe must be an integer: {timeframe}") from err

    if tf_int <= 0:
        raise ValidationError(f"Timeframe must be positive: {timeframe}")

    if allowed_timeframes and tf_int not in allowed_timeframes:
        logger.warning(f"Timeframe {tf_int} not in standard list: {allowed_timeframes}")

    return tf_int


def validate_payout(payout: float, min_payout: float = 0.0, max_payout: float = 100.0) -> float:
    """
    Validate payout percentage.

    Args:
        payout: Payout percentage
        min_payout: Minimum allowed payout
        max_payout: Maximum allowed payout

    Returns:
        float: Validated payout

    Raises:
        ValidationError: If payout is invalid
    """
    try:
        payout_float = float(payout)
    except (ValueError, TypeError) as err:
        raise ValidationError(f"Payout must be a number: {payout}") from err

    if payout_float < min_payout:
        raise ValidationError(f"Payout {payout_float} below minimum {min_payout}")

    if payout_float > max_payout:
        raise ValidationError(f"Payout {payout_float} exceeds maximum {max_payout}")

    return payout_float


def validate_win_rate(win_rate: float) -> float:
    """
    Validate win rate is between 0 and 1.

    Args:
        win_rate: Win rate to validate

    Returns:
        float: Validated win rate

    Raises:
        ValidationError: If win rate is invalid
    """
    try:
        wr_float = float(win_rate)
    except (ValueError, TypeError) as err:
        raise ValidationError(f"Win rate must be a number: {win_rate}") from err

    if not (0.0 <= wr_float <= 1.0):
        raise ValidationError(f"Win rate must be between 0 and 1: {win_rate}")

    return wr_float


def validate_confidence(confidence: float) -> float:
    """
    Validate confidence score is between 0 and 1.

    Args:
        confidence: Confidence score to validate

    Returns:
        float: Validated confidence

    Raises:
        ValidationError: If confidence is invalid
    """
    try:
        conf_float = float(confidence)
    except (ValueError, TypeError) as err:
        raise ValidationError(f"Confidence must be a number: {confidence}") from err

    if not (0.0 <= conf_float <= 1.0):
        raise ValidationError(f"Confidence must be between 0 and 1: {confidence}")

    return conf_float


def validate_weights(weights: dict) -> dict:
    """
    Validate and normalize ensemble weights.

    Args:
        weights: Dictionary of weights

    Returns:
        dict: Normalized weights (sum to 1.0)

    Raises:
        ValidationError: If weights are invalid
    """
    if not weights or not isinstance(weights, dict):
        raise ValidationError("Weights must be a non-empty dictionary")

    # Ensure all values are numeric and positive
    normalized = {}
    total = 0.0

    for key, value in weights.items():
        try:
            weight = float(value)
        except (ValueError, TypeError) as err:
            raise ValidationError(f"Weight for {key} must be numeric: {value}") from err

        if weight < 0:
            raise ValidationError(f"Weight for {key} cannot be negative: {weight}")

        normalized[key] = weight
        total += weight

    # Normalize to sum to 1.0
    if total <= 0:
        raise ValidationError("Sum of weights must be positive")

    return {k: v / total for k, v in normalized.items()}


def validate_trade_params(
    asset: str,
    direction: str,
    amount: float,
    expiration: Any,
    min_amount: float = 1.0,
    max_amount: float = 10000.0,
) -> Tuple[str, str, float, int]:
    """
    Validate all trade parameters at once.

    Args:
        asset: Asset symbol
        direction: Trade direction
        amount: Trade amount
        expiration: Expiration time
        min_amount: Minimum allowed amount
        max_amount: Maximum allowed amount

    Returns:
        Tuple of validated (asset, direction, amount, expiration)

    Raises:
        ValidationError: If any parameter is invalid
    """
    validated_asset = validate_asset_symbol(asset)
    validated_direction = validate_direction(direction)
    validated_amount = validate_trade_amount(amount, min_amount, max_amount)
    validated_expiration = validate_expiration(expiration)

    return validated_asset, validated_direction, validated_amount, validated_expiration


__all__ = [
    "ValidationError",
    "validate_asset_symbol",
    "validate_trade_amount",
    "validate_direction",
    "validate_expiration",
    "validate_timeframe",
    "validate_payout",
    "validate_win_rate",
    "validate_confidence",
    "validate_weights",
    "validate_trade_params",
]
