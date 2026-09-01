"""Interactive Browser Sign-In Helper for NEXUS.

Launches a visible Chromium browser window so the user can interactively complete
sign-in and Cloudflare verification. Captures the resulting session cookies and SSID
token directly into .env.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _update_env_file(env_path: Path, updates: dict[str, str]) -> None:
    load_dotenv(env_path)
    lines = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    updated_keys = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in line:
            key = line.split("=", 1)[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                updated_keys.add(key)
                continue
        new_lines.append(line)

    for k, v in updates.items():
        if k not in updated_keys:
            new_lines.append(f"{k}={v}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


async def run_interactive_login() -> None:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("Playwright is not installed. Install via: uv pip install playwright")
        return

    print("=" * 60)
    print("NEXUS - Interactive Browser Sign-In Helper")
    print("=" * 60)

    try:
        async with async_playwright() as p:
            print("Launching visible browser window...")
            browser = await p.chromium.launch(
                headless=False,
                args=["--disable-blink-features=AutomationControlled"],
            )
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            )
            page = await context.new_page()

            sign_in_url = "https://qxbroker.com/en/sign-in"
            print(f"Navigating to {sign_in_url}...")
            await page.goto(sign_in_url)

            # Auto-fill credentials from .env if present
            import os
            import time

            load_dotenv()
            email = os.getenv("QUOTEX_EMAIL", "")
            password = os.getenv("QUOTEX_PASSWORD", "")

            autofill_submitted = False
            last_submit_attempt = 0.0

            async def try_autofill_and_submit() -> bool:
                nonlocal last_submit_attempt
                now = time.time()
                if now - last_submit_attempt < 3.0:
                    return False
                last_submit_attempt = now
                try:
                    # Target the Sign-In form specifically
                    email_input = page.locator(
                        'form[action*="sign-in"] input[name="email"], input[autocomplete="username"]:visible, form:visible input[name="email"]:visible'
                    ).first
                    pass_input = page.locator(
                        'form[action*="sign-in"] input[name="password"], #password-input:visible, form:visible input[name="password"]:visible'
                    ).first
                    submit_btn = page.locator(
                        'form[action*="sign-in"] button.modal-sign__block-button, form[action*="sign-in"] button:has-text("Sign in"), button:has-text("Sign in"):visible, button.modal-sign__block-button:visible'
                    ).first

                    if await email_input.count() > 0 and await email_input.is_visible():
                        curr_email = await email_input.input_value()
                        if curr_email != email:
                            print(f"Auto-filling credentials for {email}...")
                            await email_input.fill(email, timeout=3000)
                            await pass_input.fill(password, timeout=3000)
                            print("✓ Auto-filled email and password!")
                            await asyncio.sleep(0.5)

                        if await submit_btn.count() > 0 and await submit_btn.is_visible():
                            btn_text = (await submit_btn.inner_text()).strip()
                            print(
                                f"✓ Automatically clicking '{btn_text}' button and submitting form..."
                            )
                            try:
                                await submit_btn.click(timeout=3000)
                            except Exception:
                                pass
                            try:
                                await pass_input.press("Enter")
                            except Exception:
                                pass
                            try:
                                await page.evaluate("""() => {
                                    const f = document.querySelector('form[action*="sign-in"]') || document.querySelector('form:not([action*="sign-up"])');
                                    if (f) {
                                        const btn = f.querySelector('button.modal-sign__block-button') ||
                                                    Array.from(f.querySelectorAll('button')).find(b => b.innerText.toLowerCase().includes('sign in'));
                                        if (btn) btn.click();
                                        else if (f.requestSubmit) f.requestSubmit();
                                        else f.submit();
                                    }
                                }""")
                            except Exception:
                                pass
                            print("✓ Sign-In submitted successfully!")
                            return True
                except Exception as autofill_err:
                    print(f"Auto-fill notice: {autofill_err}")
                return False

            if email and password:
                # Wait briefly for page/Cloudflare to settle, then attempt initial auto-fill
                for _ in range(10):
                    await asyncio.sleep(1)
                    if await try_autofill_and_submit():
                        autofill_submitted = True
                        break

            captured_ws_token = ""

            def on_websocket(ws: Any) -> None:
                def on_framesent(payload: Any) -> None:
                    nonlocal captured_ws_token
                    try:
                        txt = (
                            payload
                            if isinstance(payload, str)
                            else payload.decode("utf-8", errors="ignore")
                        )
                        if "authorization" in txt:
                            idx = txt.find("{")
                            end_idx = txt.rfind("}")
                            if idx != -1 and end_idx != -1:
                                import json

                                data = json.loads(txt[idx : end_idx + 1])
                                tok = data.get("session") or data.get("token")
                                if tok and isinstance(tok, str) and len(tok) > 5:
                                    captured_ws_token = tok
                    except Exception:
                        pass

                ws.on("framesent", on_framesent)

            async def on_response(response: Any) -> None:
                nonlocal captured_ws_token
                try:
                    if "cabinets/digest" in response.url or "/digest" in response.url:
                        data = await response.json()
                        tok = data.get("data", {}).get("token")
                        if tok and isinstance(tok, str) and len(tok) > 5:
                            captured_ws_token = tok
                except Exception:
                    pass

            page.on("websocket", on_websocket)
            page.on("response", on_response)

            print(
                "\n👉 The helper will automatically capture your session "
                "once the browser lands on the Trading Platform (https://qxbroker.com/en/trade).\n"
            )

            token_found = False
            cookies_str = ""
            ssid_token = ""

            for i in range(300):
                await asyncio.sleep(1)
                current_url = page.url

                # If still on sign-in page and not submitted yet, retry submission
                if not autofill_submitted and email and password and "/sign-in" in current_url:
                    if await try_autofill_and_submit():
                        autofill_submitted = True

                # Check for 2FA prompt
                try:
                    two_fa_input = page.locator(
                        'input[name="code"]:visible, input[name="pin"]:visible, input[name="two_factor"]:visible, input[autocomplete="one-time-code"]:visible'
                    ).first
                    if await two_fa_input.count() > 0 and await two_fa_input.is_visible():
                        if i % 10 == 0:
                            print(
                                "👉 2FA verification prompt detected! Please enter your PIN/code in the browser window."
                            )
                except Exception:
                    pass

                # If on trade page, try active token extraction
                if "/trade" in current_url or "/demo-trade" in current_url:
                    if not captured_ws_token:
                        try:
                            eval_tok = await page.evaluate("""async () => {
                                if (window.settings && window.settings.token) return window.settings.token;
                                if (localStorage.getItem('token')) return localStorage.getItem('token');
                                try {
                                    const r = await fetch('/api/v1/cabinets/digest');
                                    if (r.ok) {
                                        const d = await r.json();
                                        if (d && d.data && d.data.token) return d.data.token;
                                    }
                                } catch (e) {}
                                return '';
                            }""")
                            if eval_tok and isinstance(eval_tok, str) and len(eval_tok) > 5:
                                captured_ws_token = eval_tok
                        except Exception:
                            pass

                cookies = await context.cookies()
                cookie_dict = {
                    c["name"]: c["value"] for c in cookies if "name" in c and "value" in c
                }

                # Only proceed if we captured the genuine WebSocket auth token
                if captured_ws_token:
                    parts = [f"{k}={v}" for k, v in cookie_dict.items()]
                    cookies_str = "; ".join(parts)
                    ssid_token = captured_ws_token
                    token_found = True
                    print(f"\n✓ Logged in! Captured Verified WebSocket Token: {ssid_token[:10]}...")
                    print("✓ Captured all session cookies!")
                    break

                if i % 10 == 0:
                    if "/trade" in current_url or "/demo-trade" in current_url:
                        print("Waiting for trading session initialization... (Trade page active)")
                    else:
                        print(f"Waiting for login... (Current page: {current_url})")

            await browser.close()

            if token_found and cookies_str:
                env_file = Path(".env")
                updates = {
                    "QUOTEX_COOKIES": cookies_str,
                    "QUOTEX_USER_AGENT": (
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                }
                if ssid_token:
                    updates["QUOTEX_SSID"] = ssid_token
                _update_env_file(env_file, updates)
                print("✓ Updated .env file with fresh session tokens and User-Agent!")
                print("You can now run: uv run python -m nexus.main --gui --demo")

            else:
                print("⚠ Sign-in timed out or session tokens were not captured.")
    except Exception as err:
        err_msg = str(err)
        print(f"\n⚠ Browser launch notice: {err_msg}\n")
        if "shared libraries" in err_msg or "libasound" in err_msg:
            print("To enable full Playwright browser window on Ubuntu 24.04+, run:")
            print("  sudo apt-get install -y libasound2t64 libgdk-pixbuf-2.0-0 libgtk-3-0t64")

        print(
            "\nAlternatively, copy your laravel_session cookie from your regular browser into .env:"
        )
        print("  QUOTEX_SSID=your_laravel_session_cookie_value")


if __name__ == "__main__":
    asyncio.run(run_interactive_login())
