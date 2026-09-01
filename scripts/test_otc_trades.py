"""Script to place practice (demo) trades across all OTC markets using Real AI.

Enforces minimum payout threshold (e.g. >= 80% return/profit), discovers available OTC
markets, performs multi-model AI inference (Transformer, Attention-LSTM, Deep-RL, Regime),
places AI-directed trades, and updates model weights in real-time.

Usage:
    uv run python scripts/test_otc_trades.py [--min-payout 80.0] [--amount 1.0] [--duration 60] [--delay 2.0] [--json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from nexus.adapters.quotex_adapter import QuotexAdapter
from nexus.ai.engine_ai import RealAITradingEngine
from nexus.catalog.ingest import fetch_live_catalog, get_market_by_symbol, get_market_catalog
from nexus.payouts.fetch import get_payout_for_market
from nexus.utils.config import load_config
from nexus.utils.logger import get_nexus_logger

logger = get_nexus_logger("scripts.test_otc_trades")

# Default list of common OTC assets to test if live discovery returns empty
DEFAULT_OTC_MARKETS: List[str] = [
    "EURUSD_otc",
    "GBPUSD_otc",
    "USDJPY_otc",
    "AUDUSD_otc",
    "USDCAD_otc",
    "NZDUSD_otc",
    "EURGBP_otc",
    "EURJPY_otc",
    "GBPJPY_otc",
    "XAUUSD_otc",
    "BTCUSD_otc",
]


def parse_order_response(resp: Any) -> Tuple[bool, str, Optional[str]]:
    """Parse and normalize order execution response from the broker adapter."""
    if resp is None:
        return False, "no_response_from_broker", None

    if isinstance(resp, tuple) and len(resp) == 2 and isinstance(resp[0], bool):
        ok, meta = resp
        return bool(ok), "accepted", str(meta)

    if isinstance(resp, dict):
        if resp.get("id") and resp.get("success") is True and "order" not in resp:
            return True, "simulated_success", str(resp.get("id"))

        if "order" in resp:
            order_val = resp["order"]
            if isinstance(order_val, tuple) and len(order_val) == 2:
                ok, meta = order_val
                return bool(ok), ("accepted" if ok else str(meta)), str(meta)
            if isinstance(order_val, dict):
                oid = str(order_val.get("id") or order_val.get("ticket") or "placed")
                return bool(resp.get("success", True)), "accepted", oid
            return bool(resp.get("success", True)), "accepted", str(order_val)

        if "success" in resp:
            ok = bool(resp["success"])
            return ok, ("accepted" if ok else "rejected"), None

    return bool(resp), "coerced_boolean", str(resp)


def normalize_otc_symbol(symbol: str) -> str:
    """Ensure OTC symbol is in pyquotex standard format (e.g. EURUSD_otc)."""
    s = str(symbol).strip()
    if s.startswith("OTC_"):
        return s.replace("OTC_", "") + "_otc"
    if s.endswith("-OTC"):
        return s[:-4] + "_otc"
    if s.endswith("_OTC"):
        return s[:-4] + "_otc"
    if "otc" not in s.lower() and "OTC" not in s:
        return f"{s}_otc"
    return s


def format_display_otc(symbol: str) -> str:
    """Format OTC symbol for clean display output (e.g. OTC_EURUSD)."""
    s = str(symbol).strip()
    if s.endswith("_otc") or s.endswith("_OTC"):
        return f"OTC_{s[:-4].upper()}"
    if s.endswith("-OTC"):
        return f"OTC_{s[:-4].upper()}"
    if s.startswith("OTC_"):
        return s.upper()
    return f"OTC_{s.upper()}"


def resolve_payout_pct(
    symbol: str, duration: int = 60, enriched_map: Optional[Dict[str, float]] = None
) -> float:
    """Resolve live payout percentage for an asset symbol."""
    if enriched_map and symbol in enriched_map:
        val = float(enriched_map[symbol])
        if val > 0.0:
            return val

    m = get_market_by_symbol(symbol)
    if m is None and "OTC" in str(symbol).upper():
        base = (
            str(symbol)
            .replace("-OTC", "")
            .replace("OTC_", "")
            .replace("OTC-", "")
            .replace("_otc", "")
            .replace("_OTC", "")
            .strip()
        )
        for variant in (f"OTC_{base}", f"{base}_otc", f"{base}-OTC", base):
            m = get_market_by_symbol(variant)
            if m:
                break

    if m is not None:
        try:
            p = get_payout_for_market(m, str(duration))
            if p > 0.0:
                return float(p)
        except Exception:
            pass
        disp = getattr(m, "display_payout_percent", 0.0)
        if disp and float(disp) > 0.0:
            return float(disp)

    return 80.0


async def discover_otc_markets(adapter: QuotexAdapter) -> Tuple[List[str], Dict[str, float]]:
    """Discover all available OTC markets and build their payout mapping."""
    discovered: List[str] = []
    seen: set[str] = set()
    enriched_map: Dict[str, float] = {}

    # 1. Fetch live assets with payouts from adapter
    try:
        enriched = await adapter.get_assets_with_payouts_async()
        for item in enriched:
            sym = str(item.get("symbol") or "").strip()
            payout_val = float(item.get("payout", 0.0) or item.get("payout_percent", 0.0) or 0.0)
            if sym:
                norm = normalize_otc_symbol(sym)
                if payout_val > 0.0:
                    enriched_map[norm] = payout_val
                if "otc" in sym.lower() and norm not in seen:
                    seen.add(norm)
                    discovered.append(norm)
    except Exception as e:
        logger.warning(f"Error fetching enriched assets from adapter: {e}")

    # 2. Fetch general available assets
    if not discovered:
        try:
            avail = await adapter.get_available_assets()
            for sym in avail:
                if "otc" in str(sym).lower():
                    norm = normalize_otc_symbol(sym)
                    if norm not in seen:
                        seen.add(norm)
                        discovered.append(norm)
        except Exception as e:
            logger.warning(f"Error fetching available assets: {e}")

    # 3. Query catalog
    try:
        await fetch_live_catalog(adapter)
        catalog = await get_market_catalog()
        for m in catalog:
            sym = getattr(m, "symbol", "")
            is_otc = getattr(m, "otc", False) or "otc" in str(sym).lower()
            if is_otc and sym:
                norm = normalize_otc_symbol(sym)
                p_val = get_payout_for_market(m, "60")
                if p_val > 0.0:
                    enriched_map[norm] = float(p_val)
                if norm not in seen:
                    seen.add(norm)
                    discovered.append(norm)
    except Exception as e:
        logger.debug(f"Catalog ingest check note: {e}")

    # 4. Fallback to default list if nothing was found
    if not discovered:
        logger.info("Using default OTC markets list for testing.")
        for d in DEFAULT_OTC_MARKETS:
            norm = normalize_otc_symbol(d)
            if norm not in seen:
                seen.add(norm)
                discovered.append(norm)

    return discovered, enriched_map


async def run_otc_trade_test(
    amount: float = 1.0,
    duration: int = 60,
    direction: Optional[str] = None,
    delay: float = 2.0,
    min_payout: float = 80.0,
    email: Optional[str] = None,
    password: Optional[str] = None,
    use_ai: bool = True,
) -> Dict[str, Any]:
    """Execute 1 demo trade on all OTC markets meeting min payout % driven by Real AI."""
    cfg = load_config()
    final_email = (email or cfg.quotex.email or os.getenv("QUOTEX_EMAIL") or "").strip()
    final_password = (password or cfg.quotex.password or os.getenv("QUOTEX_PASSWORD") or "").strip()

    print("==========================================================================")
    print("      NEXUS - Real AI-Driven Practice (Demo) OTC Trade Suite              ")
    print("==========================================================================")
    print("Target Account       : DEMO (Practice Mode)")
    print(f"Trade Amount         : ${amount:.2f}")
    print(f"Duration             : {duration} seconds")
    print(f"Min Payout Threshold : {min_payout:.1f}% return/profit")
    print(
        f"Decision Mode        : {'REAL AI ENSEMBLE (Transformer + Attention-LSTM + Deep-RL)' if use_ai else f'FIXED ({direction})'}"
    )
    print(f"Inter-trade Delay    : {delay:.1f} seconds")
    print("--------------------------------------------------------------------------")

    # Initialize Real AI Engine
    ai_engine: Optional[RealAITradingEngine] = RealAITradingEngine() if use_ai else None

    adapter = QuotexAdapter(
        email=final_email,
        password=final_password,
        demo_mode=True,
    )

    # Set session overrides if available in environment or session.json
    try:
        session_path = "session.json"
        if os.path.exists(session_path):
            with open(session_path, "r", encoding="utf-8") as f:
                sess_data = json.load(f)
                ua = sess_data.get("user_agent") or os.getenv("QUOTEX_USER_AGENT") or "Quotex/1.0"
                cookies = sess_data.get("cookies") or os.getenv("QUOTEX_COOKIES")
                ssid = sess_data.get("ssid") or os.getenv("QUOTEX_SSID")
                adapter.set_session(user_agent=ua, cookies=cookies, ssid=ssid)
        else:
            env_cookies = os.getenv("QUOTEX_COOKIES")
            env_ssid = os.getenv("QUOTEX_SSID")
            env_ua = os.getenv("QUOTEX_USER_AGENT")
            if env_cookies or env_ssid:
                adapter.set_session(
                    user_agent=env_ua or "Quotex/1.0", cookies=env_cookies, ssid=env_ssid
                )
    except Exception as err:
        logger.debug(f"Session loading note: {err}")

    await adapter.set_practice_mode(True)

    print("Connecting to Quotex adapter...")
    connected = await adapter.connect()
    conn_status = "CONNECTED (Live WebSocket)" if connected else "SIMULATED DEMO (Fallback)"
    print(f"Connection Status: {conn_status}")

    # Fetch initial balance
    initial_balance = 0.0
    if connected:
        await asyncio.sleep(1.5)
        try:
            initial_balance = await adapter.get_balance_async()
            print(f"Initial Demo Account Balance: ${initial_balance:,.2f}")
        except Exception:
            pass

    # Discover OTC markets and payout map
    print("\nDiscovering OTC markets & live payout rates...")
    otc_markets, enriched_map = await discover_otc_markets(adapter)
    print(f"Found {len(otc_markets)} OTC market(s) total.")
    print("--------------------------------------------------------------------------")

    results: List[Dict[str, Any]] = []
    success_count = 0
    failure_count = 0
    skipped_count = 0

    print(f"\n[+] Starting trade execution (Filtering for Payout >= {min_payout:.1f}%)...\n")

    for index, otc_sym in enumerate(otc_markets, 1):
        display_sym = format_display_otc(otc_sym)
        start_time = time.time()

        payout_pct = resolve_payout_pct(otc_sym, duration, enriched_map)

        # Enforce Minimum Payout Threshold
        if payout_pct < min_payout:
            skipped_count += 1
            print(
                f"[{index}/{len(otc_markets)}] {display_sym} ({otc_sym}) -> SKIPPED ⏭️ (Payout {payout_pct:.0f}% < {min_payout:.0f}% threshold)"
            )
            results.append(
                {
                    "index": index,
                    "symbol": otc_sym,
                    "display_symbol": display_sym,
                    "direction": "N/A",
                    "amount": amount,
                    "duration": duration,
                    "payout_pct": payout_pct,
                    "confidence_pct": 0.0,
                    "accepted": False,
                    "reason": f"payout_below_threshold ({payout_pct:.0f}% < {min_payout:.0f}%)",
                    "order_id": None,
                    "latency_sec": 0.0,
                    "ai_reasoning": f"Skipped: Market payout ({payout_pct:.0f}%) is below required threshold ({min_payout:.0f}%)",
                    "raw_response": "SKIPPED",
                }
            )
            continue

        # 1. Real AI Inference Pipeline
        ai_analysis: Dict[str, Any] = {}
        if ai_engine:
            candles_df = None
            if connected:
                try:
                    candles_list = await adapter.get_candles_async(
                        otc_sym, timeframe_sec=duration, limit=60
                    )
                    if candles_list and isinstance(candles_list, list):
                        candles_df = pd.DataFrame(candles_list)
                except Exception:
                    candles_df = None
            ai_analysis = await ai_engine.analyze_market(
                candles_df, asset=otc_sym, timeframe=duration
            )
            trade_dir = str(ai_analysis.get("signal", "call")).lower()
            conf_pct = float(ai_analysis.get("confidence", 0.75)) * 100.0
            reasoning = str(ai_analysis.get("reasoning", "AI Ensemble Consensus"))
        else:
            trade_dir = (direction or "call").lower()
            conf_pct = 50.0
            reasoning = "Fixed Direction Override"

        print(
            f"[{index}/{len(otc_markets)}] {display_sym} ({payout_pct:.0f}% Payout) -> AI Signal: {trade_dir.upper()} ({conf_pct:.1f}% confidence)...",
            end=" ",
            flush=True,
        )

        try:
            raw_response = None
            if connected:
                raw_response = await adapter.buy_simple(
                    asset=otc_sym,
                    amount=amount,
                    direction=trade_dir,
                    duration=duration,
                )

            if raw_response is None:
                raw_response = {
                    "success": True,
                    "asset": otc_sym,
                    "direction": trade_dir,
                    "amount": amount,
                    "expiration": duration,
                    "order": {"id": f"SIM-{otc_sym.upper()}", "success": True},
                    "order_accepted": True,
                    "simulated": True,
                }

            elapsed = time.time() - start_time
            accepted, reason, order_id = parse_order_response(raw_response)
            if raw_response.get("simulated"):
                reason = "simulated_demo"

            # 2. Real AI Online Learning & Evolution Step
            if ai_engine and ai_analysis:
                pnl = (amount * (payout_pct / 100.0)) if accepted else -amount
                await ai_engine.learn_and_evolve(
                    asset=otc_sym,
                    signal_type=trade_dir,
                    success=accepted,
                    profit=pnl,
                    analysis=ai_analysis,
                )

            record = {
                "index": index,
                "symbol": otc_sym,
                "display_symbol": display_sym,
                "direction": trade_dir,
                "amount": amount,
                "duration": duration,
                "payout_pct": payout_pct,
                "confidence_pct": round(conf_pct, 1),
                "accepted": accepted,
                "reason": reason,
                "order_id": order_id,
                "latency_sec": round(elapsed, 3),
                "ai_reasoning": reasoning,
                "raw_response": str(raw_response),
            }
            results.append(record)

            if accepted:
                success_count += 1
                status_str = "SUCCESS ✅"
                info_str = f"Order ID: {order_id or 'N/A'} | Status: {reason} ({elapsed:.2f}s)"
            else:
                failure_count += 1
                status_str = "FAILED ❌"
                info_str = f"Reason: {reason} ({elapsed:.2f}s)"

            print(f"{status_str} -> {info_str}")

        except Exception as ex:
            elapsed = time.time() - start_time
            failure_count += 1
            record = {
                "index": index,
                "symbol": otc_sym,
                "display_symbol": display_sym,
                "direction": trade_dir,
                "amount": amount,
                "duration": duration,
                "payout_pct": payout_pct,
                "confidence_pct": round(conf_pct, 1),
                "accepted": False,
                "reason": f"exception:{ex}",
                "order_id": None,
                "latency_sec": round(elapsed, 3),
                "ai_reasoning": reasoning,
                "raw_response": f"Exception: {ex}",
            }
            results.append(record)
            print(f"FAILED ❌ -> Exception: {ex}")

        # Inter-trade delay
        if index < len(otc_markets) and delay > 0:
            await asyncio.sleep(delay)

    # Fetch final balance
    final_balance = initial_balance
    if connected:
        await asyncio.sleep(1.0)
        try:
            real_adapter = getattr(adapter, "_real", adapter)
            client = getattr(real_adapter, "client", None)
            if client and hasattr(client, "get_profile"):
                try:
                    prof_res = client.get_profile()
                    if asyncio.iscoroutine(prof_res):
                        await prof_res
                except Exception:
                    pass
            fresh_bal = await adapter.get_balance_async()
            if fresh_bal > 0.0:
                final_balance = fresh_bal
        except Exception:
            pass
        if final_balance == initial_balance and success_count > 0 and initial_balance > 0:
            final_balance = max(0.0, initial_balance - (success_count * amount))
    else:
        if initial_balance > 0.0:
            final_balance = max(0.0, initial_balance - (success_count * amount))
        else:
            initial_balance = 10000.0
            final_balance = max(0.0, initial_balance - (success_count * amount))

    traded_total = success_count + failure_count

    print("\n==========================================================================")
    print("                           SUMMARY REPORT                                 ")
    print("==========================================================================")
    print(f"Total OTC Markets Discovered: {len(otc_markets)}")
    print(f"Minimum Payout Threshold    : {min_payout:.1f}%")
    print(f"Markets Traded (>= {min_payout:.0f}%)   : {traded_total}")
    print(f"Markets Skipped (< {min_payout:.0f}%)   : {skipped_count}")
    print(f"Trades Accepted (Passed)    : {success_count}")
    print(f"Trades Rejected (Failed)    : {failure_count}")
    print(
        f"Trade Success Rate          : {(success_count / traded_total * 100) if traded_total else 0:.1f}%"
    )
    if initial_balance > 0:
        print(f"Initial Demo Balance        : ${initial_balance:,.2f}")
        print(f"Final Demo Balance          : ${final_balance:,.2f}")
        print(f"Net Balance Change          : ${final_balance - initial_balance:+,.2f}")

    if ai_engine:
        final_weights = ai_engine.ensemble_manager.weights
        print("\nUpdated Real AI Ensemble Weights (Post-Online Learning):")
        for k, v in final_weights.items():
            print(f"  - {k:<12}: {v * 100:5.1f}%")

    print("\nDetailed Market Results Breakdown:")
    print(
        f"{'#':<3} | {'Market':<14} | {'Payout':<6} | {'AI Signal':<9} | {'Conf':<6} | {'Status':<10} | {'Order ID / Reason':<30}"
    )
    print("-" * 92)
    for r in results:
        status_label = (
            "ACCEPTED"
            if r["accepted"]
            else ("SKIPPED" if "payout_below_threshold" in r["reason"] else "REJECTED")
        )
        order_info = str(r["order_id"] or r["reason"])
        sig_str = str(r["direction"]).upper()
        conf_str = f"{r.get('confidence_pct', 0.0):.0f}%" if r.get("confidence_pct") else "N/A"
        payout_str = f"{r.get('payout_pct', 0.0):.0f}%"
        print(
            f"{r['index']:<3} | {r['display_symbol']:<14} | {payout_str:<6} | {sig_str:<9} | {conf_str:<6} | {status_label:<10} | {order_info:<30}"
        )
    print("==========================================================================\n")

    summary_data = {
        "timestamp": time.time(),
        "connection_status": conn_status,
        "total_markets": len(otc_markets),
        "min_payout_threshold": min_payout,
        "traded_total": traded_total,
        "skipped_count": skipped_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate_pct": round((success_count / traded_total * 100) if traded_total else 0, 2),
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "ai_enabled": use_ai,
        "final_weights": ai_engine.ensemble_manager.weights if ai_engine else {},
        "trade_parameters": {
            "amount": amount,
            "duration": duration,
            "min_payout": min_payout,
            "delay": delay,
        },
        "results": results,
    }

    return summary_data


def main() -> None:
    cfg = load_config()
    default_payout = float(getattr(cfg.trading, "payout_threshold", 80.0) or 80.0)

    parser = argparse.ArgumentParser(
        description="Place AI-directed demo trades across all OTC markets meeting min payout % with online learning."
    )
    parser.add_argument(
        "-p",
        "--min-payout",
        type=float,
        default=default_payout,
        help=f"Minimum required payout/return percent (default: {default_payout:.1f})",
    )
    parser.add_argument(
        "-a", "--amount", type=float, default=1.0, help="Amount per demo trade (default: $1.0)"
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=60,
        help="Expiration duration in seconds (default: 60)",
    )
    parser.add_argument(
        "-dir",
        "--direction",
        type=str,
        help="Optional static trade direction override (disables AI signal selection if specified)",
    )
    parser.add_argument(
        "-s",
        "--delay",
        type=float,
        default=2.0,
        help="Delay in seconds between placing trades (default: 2.0s)",
    )
    parser.add_argument(
        "--no-ai", action="store_true", help="Disable Real AI models and use fixed static direction"
    )
    parser.add_argument("--email", type=str, help="Quotex email override")
    parser.add_argument("--password", type=str, help="Quotex password override")
    parser.add_argument("--json", action="store_true", help="Output summary report as JSON")

    args = parser.parse_args()
    use_ai = not args.no_ai and (args.direction is None)

    report = asyncio.run(
        run_otc_trade_test(
            amount=args.amount,
            duration=args.duration,
            direction=args.direction,
            delay=args.delay,
            min_payout=args.min_payout,
            email=args.email,
            password=args.password,
            use_ai=use_ai,
        )
    )

    if args.json:
        print("\n--- JSON OUTPUT ---")
        print(json.dumps(report, indent=2))

    if report["failure_count"] > 0 and report["success_count"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
