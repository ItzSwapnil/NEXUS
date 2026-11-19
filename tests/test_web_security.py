from nexus.web.security import extract_service_token, has_valid_service_token


def test_extracts_dedicated_service_token_case_insensitively():
    assert extract_service_token({"X-Nexus-Api-Token": "token-value"}) == "token-value"


def test_extracts_bearer_service_token():
    assert extract_service_token({"Authorization": "Bearer token-value"}) == "token-value"


def test_rejects_missing_or_invalid_service_token():
    assert not has_valid_service_token({}, "expected")
    assert not has_valid_service_token({"Authorization": "Bearer wrong"}, "expected")


def test_accepts_matching_service_token():
    assert has_valid_service_token({"Authorization": "Bearer expected"}, "expected")
