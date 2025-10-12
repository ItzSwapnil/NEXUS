
import pytest
from nexus.scripts.smoke_real_trade import run_trade

@pytest.mark.asyncio
async def test_place_demo_trade():
    """
    Tests placing a trade on the demo account.
    """
    # Define the parameters for the demo trade
    asset = "EURUSD-OTC"
    amount = 1.0
    expiration = 60
    direction = "call"
    demo = True
    email_override = "test@example.com"  # Use a dummy email for the test
    password_override = "testpassword"   # Use a dummy password for the test

    # Run the trade function
    result = await run_trade(
        asset=asset,
        amount=amount,
        expiration=expiration,
        direction=direction,
        demo=demo,
        email_override=email_override,
        password_override=password_override,
    )

    # Assert that the trade was successful
    assert result["success"], f"Demo trade failed: {result.get('error', 'No error message')}"
    assert result["practice"] is True
    assert result["order_accepted"] is True
