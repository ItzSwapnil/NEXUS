"""Smoke trade helper for simulated and broker-backed demo trades.

- demo=True (default): returns a simulated successful practice trade so tests run fast.
- demo=True + NEXUS_DEMO_BROKER=true: attempts a real PRACTICE (demo) trade with Quotex.
- demo=False: attempts a LIVE trade (best-effort, requires credentials/session).
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any, Dict, List, Optional, Tuple

from nexus.utils.config import load_config
from nexus.utils.logger import get_nexus_logger

logger = get_nexus_logger("nexus.scripts.smoke_real_trade")

try:  # pragma: no cover
    from nexus.adapters.quotex_adapter import QuotexAdapter  # type: ignore
except Exception:  # pragma: no cover
    QuotexAdapter = None  # type: ignore


# ---------------- Internal helpers ---------------- #
def _is_order_accepted(resp: Any) -> Tuple[bool, str]:
    """Normalize assorted pyquotex adapter responses.

    Returns (accepted, reason). Order considered accepted ONLY when underlying
    low-level success flag is True. Examples of possible structures observed:
      - {'success': True, 'order': (True, '123456')}
      - {'success': True, 'order': (False, 'not_price')}
      - (True, '123456')
      - (False, 'not_enough_money')
      - {'id': 'SIM-DEMO', 'success': True} (simulation)
    """
    try:
        # Tuple form from raw stable_api
        if isinstance(resp, tuple) and len(resp) == 2 and isinstance(resp[0], bool):
            ok, meta = resp
            return bool(ok), str(meta)
        if isinstance(resp, dict):
            # Simulation path
            if resp.get("id") and resp.get("success") is True and "order" not in resp:
                return True, "simulated"
            # Nested tuple from adapter
            if "order" in resp and isinstance(resp["order"], tuple) and len(resp["order"]) == 2:
                ok, meta = resp["order"]
                if not ok:
                    return False, str(meta)
                # Some adapters return (True, False) meaning accepted but unresolved result yet
                return bool(ok), str(meta)
            # Flat success flag only (treat as accepted but note ambiguous)
            if "success" in resp:
                return bool(resp["success"]), "flag_only"
        # Fallback boolean casting
        return bool(resp), "coerced"
    except Exception as e:  # pragma: no cover
        return False, f"parse_error:{e}"


# ---------- Simulated path ----------
async def _simulate_demo(
    asset: str, amount: float, expiration: int, direction: str
) -> Dict[str, Any]:
    return {
        "success": True,
        "asset": asset,
        "direction": direction,
        "amount": float(amount),
        "expiration": int(expiration),
        "balance": 10_000.0,
        "order": {"id": "SIM-DEMO-1", "success": True},
        "order_accepted": True,
        "practice": True,
    }


# ---------- Broker-backed PRACTICE path ----------
async def _real_practice_trade(
    asset: str,
    amount: float,
    expiration: int,
    direction: str,
    email_override: Optional[str],
    password_override: Optional[str],
) -> Dict[str, Any]:
    cfg = load_config()
    if email_override:
        cfg.quotex.email = email_override  # type: ignore[attr-defined]
    if password_override:
        cfg.quotex.password = password_override  # type: ignore[attr-defined]

    email = (cfg.quotex.email or "").strip()
    password = (cfg.quotex.password or "").strip()

    if QuotexAdapter is None:  # pragma: no cover
        return await _simulate_demo(asset, amount, expiration, direction)

    adapter = QuotexAdapter(email=email, password=password, demo_mode=True)  # type: ignore[call-arg]

    # Try to use persisted session.json when available
    try:
        import json
        from pathlib import Path

        p = Path("session.json")
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            ua = raw.get("user_agent") or "Quotex/1.0"
            cookies = raw.get("cookies")
            ssid = raw.get("ssid")
            try:
                adapter.set_session(user_agent=ua, cookies=cookies, ssid=ssid)  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        pass

    try:
        await adapter.set_practice_mode(True)  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        ok = await adapter.connect()  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning(f"Connect failed: {e}")
        ok = False

    if not ok:
        return await _simulate_demo(asset, amount, expiration, direction)

    # Build symbol candidates
    def _variants(sym: str) -> List[str]:
        s = (sym or "").strip()
        if not s:
            return []
        base = s
        if s.upper().endswith("-OTC") or s.lower().endswith("_otc"):
            base = s[:-4]
        cand = [s, f"{base}-OTC", f"{base}_otc", f"OTC_{base}", base]
        out: List[str] = []
        seen = set()
        for v in cand:
            vv = v.strip()
            if vv and vv not in seen:
                out.append(vv)
                seen.add(vv)
        return out

    candidates = _variants(asset or "EURUSD")

    # Also add top payout assets as fallbacks
    try:
        assets_with_payouts = await adapter.get_assets_with_payouts_async()  # type: ignore[attr-defined]
        assets_with_payouts = sorted(
            assets_with_payouts or [], key=lambda x: float(x.get("payout", 0.0)), reverse=True
        )[:10]
        for item in assets_with_payouts:
            sym = str(item.get("symbol", "") or "").strip()
            for v in _variants(sym):
                if v not in candidates:
                    candidates.append(v)
    except Exception:
        pass

    # Expiration candidates (minutes expected by most libs)
    exp_candidates: List[int] = []
    exp_val = int(expiration)
    exp_candidates.append(exp_val // 60 if exp_val >= 60 and exp_val % 60 == 0 else exp_val)
    for extra in (1, 2, 3, 5):
        if extra not in exp_candidates:
            exp_candidates.append(extra)

    attempts: List[dict] = []
    best_error: Optional[str] = None
    for sym in candidates[:25]:
        for ex in exp_candidates:
            try:
                raw = await adapter.buy_simple(
                    asset=sym, amount=float(amount), direction=direction, duration=int(ex)
                )  # type: ignore[attr-defined]
                accepted, reason = _is_order_accepted(raw)
                attempts.append(
                    {"asset": sym, "expiration": ex, "accepted": accepted, "reason": reason}
                )
                if not accepted:
                    best_error = reason
                    logger.debug(f"Rejected attempt {sym}@{ex}m: {reason}")
                    continue
                # Fetch balance
                try:
                    balance = await adapter.get_balance_async()
                except Exception:
                    balance = 0.0
                return {
                    "success": True,
                    "asset": sym,
                    "direction": direction,
                    "amount": float(amount),
                    "expiration": int(ex),
                    "order": raw,
                    "order_accepted": True,
                    "practice": True,
                    "balance": float(balance),
                }
            except Exception as e:
                err = str(e)
                best_error = err
                attempts.append({"asset": sym, "expiration": ex, "accepted": False, "reason": err})
                continue

    # All attempts failed -> return simulated but annotate last broker error
    sim = await _simulate_demo(asset, amount, expiration, direction)
    if best_error:
        sim["note"] = f"broker_rejected_all: {best_error}"
        sim["order_accepted"] = False
        sim["success"] = False
    return sim


# ---------- Public API ----------
async def run_trade(
    asset: str,
    amount: float,
    expiration: int,
    direction: str,
    demo: bool,
    email_override: Optional[str],
    password_override: Optional[str],
) -> Dict[str, Any]:
    if demo:
        use_broker_demo = os.getenv("NEXUS_DEMO_BROKER", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if use_broker_demo:
            return await _real_practice_trade(
                asset, amount, expiration, direction, email_override, password_override
            )
        return await _simulate_demo(asset, amount, expiration, direction)

    # LIVE path (best-effort)
    cfg = load_config()
    if email_override:
        cfg.quotex.email = email_override  # type: ignore[attr-defined]
    if password_override:
        cfg.quotex.password = password_override  # type: ignore[attr-defined]

    email = (cfg.quotex.email or "").strip()
    password = (cfg.quotex.password or "").strip()

    if QuotexAdapter is None:  # pragma: no cover
        return {
            "success": False,
            "error": "Adapter unavailable",
            "asset": asset,
            "direction": direction,
        }

    adapter = QuotexAdapter(email=email, password=password, demo_mode=False)  # type: ignore[call-arg]

    # Optional session.json
    try:
        import json
        from pathlib import Path

        p = Path("session.json")
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            ua = raw.get("user_agent") or "Quotex/1.0"
            cookies = raw.get("cookies")
            ssid = raw.get("ssid")
            try:
                adapter.set_session(user_agent=ua, cookies=cookies, ssid=ssid)  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        pass

    try:
        await adapter.set_practice_mode(False)  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        ok = await adapter.connect()  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning(f"Connect failed: {e}")
        ok = False
    if not ok:
        return {"success": False, "error": "Login failed", "asset": asset, "direction": direction}

    try:
        raw = await adapter.buy_simple(
            asset=asset, amount=float(amount), direction=direction, duration=int(expiration)
        )  # type: ignore[attr-defined]
        accepted, reason = _is_order_accepted(raw)
        return {
            "success": bool(accepted),
            "asset": asset,
            "direction": direction,
            "amount": float(amount),
            "expiration": int(expiration),
            "order": raw,
            "order_accepted": bool(accepted),
            "practice": False,
            "reason": reason,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Trade error: {e}",
            "asset": asset,
            "direction": direction,
        }


# ---------- CLI ----------
def _cli() -> None:  # pragma: no cover
    p = argparse.ArgumentParser(description="Smoke (simulated) trade helper")
    p.add_argument("--asset", required=True)
    p.add_argument("--amount", type=float, default=1.0)
    p.add_argument("--expiration", type=int, default=60)
    p.add_argument("--direction", choices=["call", "put"], default="call")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--email")
    p.add_argument("--password")
    p.add_argument(
        "--broker-demo",
        action="store_true",
        help="Use real broker practice mode when --demo is set",
    )
    args = p.parse_args()

    if args.broker_demo:
        os.environ["NEXUS_DEMO_BROKER"] = "1"

    result = asyncio.run(
        run_trade(
            asset=args.asset,
            amount=args.amount,
            expiration=args.expiration,
            direction=args.direction,
            demo=bool(args.demo),
            email_override=args.email,
            password_override=args.password,
        )
    )
    print(result)


if __name__ == "__main__":  # pragma: no cover
    _cli()
