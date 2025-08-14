import pytest

try:
    from playwright.sync_api import sync_playwright  # type: ignore
    PLAYWRIGHT_AVAILABLE = True
except Exception:  # pragma: no cover
    PLAYWRIGHT_AVAILABLE = False


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
def test_playwright_smoke_and_internal_health_metrics():
    # Playwright browser smoke test without any web server
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception:
            pytest.skip("Playwright Chromium browser not installed")
        page = browser.new_page()
        page.goto("data:text/html,<html><body><h1>NEXUS OK</h1></body></html>")
        assert page.text_content("h1") == "NEXUS OK"
        browser.close()

    # Validate internal health/metrics without FastAPI/uvicorn
    from nexus.api import health, metrics

    h = health()
    assert h.get("status") == "ok"
    assert h.get("lang") == "en"

    m = metrics()
    assert isinstance(m, dict)
    for k in ["total_trades", "winning_trades", "losing_trades", "total_profit"]:
        assert k in m
