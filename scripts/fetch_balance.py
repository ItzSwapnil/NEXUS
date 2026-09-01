import asyncio

from nexus.adapters.quotex_adapter import QuotexAdapter
from nexus.utils.config import load_runtime_settings
from nexus.utils.logger import get_nexus_logger

logger = get_nexus_logger("scripts.fetch_balance")


async def main() -> None:
    settings = load_runtime_settings()
    print("============================================================")
    print("           NEXUS - Fetching Live Quotex Balance            ")
    print("============================================================")
    adapter = QuotexAdapter(
        email=settings.quotex.email,
        password=settings.quotex.password,
        demo_mode=True,
    )
    print("Connecting & authenticating to Quotex WebSocket...")
    connected = await adapter.connect()
    print(f"Connection Status: {'SUCCESS' if connected else 'FAILED'}")

    if connected:
        # Give WebSocket a brief moment to receive the balance payload
        await asyncio.sleep(2.0)
        balance = await adapter.get_balance_async()
        print("------------------------------------------------------------")
        print(f"💰 LIVE QUOTEX DEMO BALANCE: ${balance:,.2f}")
        print("------------------------------------------------------------")
    else:
        print("Failed to authenticate to Quotex.")


if __name__ == "__main__":
    asyncio.run(main())
