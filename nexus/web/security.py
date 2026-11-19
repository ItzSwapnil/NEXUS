"""Authentication helpers for NEXUS service-to-service API access."""

from __future__ import annotations

from hmac import compare_digest
from typing import Mapping


def extract_service_token(headers: Mapping[str, str]) -> str | None:
    """Read a service token from a dedicated or standard bearer header."""
    normalized_headers = {key.lower(): value for key, value in headers.items()}
    direct_token = normalized_headers.get("x-nexus-api-token", "").strip()
    if direct_token:
        return direct_token

    authorization = normalized_headers.get("authorization", "").strip()
    scheme, _, token = authorization.partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


def has_valid_service_token(headers: Mapping[str, str], expected_token: str) -> bool:
    """Compare a supplied service token without leaking match timing."""
    supplied_token = extract_service_token(headers)
    return bool(supplied_token and expected_token and compare_digest(supplied_token, expected_token))
