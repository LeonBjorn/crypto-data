"""Prove the Hyperliquid broker against the real testnet, in stages.

The unit tests drive a fake venue, which is the right way to test this project's
own guards and cannot tell you whether ccxt can actually reach Hyperliquid. This
script answers that, and it is deliberately not part of the test suite: it needs
the network, and the suite forbids the network so that a forgotten mock fails
loudly instead of passing slowly.

Run it by hand:

    uv run python scripts/testnet_check.py

It works in stages and stops at the first one that cannot proceed. Stages 1-3
need no credentials at all and check the things most likely to be wrong. Stage 4
needs an API wallet and still sends nothing. Stage 5 is the only one that places
an order, is refused unless `--place-order` is passed, and is refused again
unless the network is testnet.

    HYPERLIQUID_WALLET_ADDRESS   the account
    HYPERLIQUID_PRIVATE_KEY      an API/agent wallet key -- never the master key

Nothing here can touch mainnet: the network is hard-coded to testnet below, and
the broker refuses mainnet without a second environment variable regardless.
"""

import argparse
import os
import sys

from paper.broker import BUY
from paper.hyperliquid import ADDRESS_ENV, KEY_ENV, HyperliquidBroker, LiveTradingRefused

TESTNET_SYMBOL = "BTC/USDC:USDC"


def head(n, title):
    print(f"\n{n}. {title}")


def ok(message):
    print(f"   ok    {message}")


def bad(message):
    print(f"   FAIL  {message}")


def note(message):
    print(f"         {message}")


def stage_reachable():
    """Can ccxt reach the testnet at all, and is it really the testnet."""
    head(1, "reaching the testnet")
    import ccxt

    client = ccxt.hyperliquid({"enableRateLimit": True})
    client.set_sandbox_mode(True)

    urls = client.urls.get("api")
    target = urls if isinstance(urls, str) else str(urls)
    if "testnet" not in target.lower():
        bad(f"sandbox mode did not switch the endpoint: {target}")
        return None
    ok(f"endpoint is testnet ({target[:60]})")

    markets = client.load_markets()
    ok(f"loaded {len(markets)} markets")
    if TESTNET_SYMBOL not in markets:
        bad(f"{TESTNET_SYMBOL} not listed; available example: {sorted(markets)[:3]}")
        return None
    ok(f"{TESTNET_SYMBOL} is listed")
    return client


def stage_prices(client):
    """Does the testnet quote a price we could size an order against."""
    head(2, "reading a price")
    ticker = client.fetch_ticker(TESTNET_SYMBOL)
    price = ticker.get("last") or ticker.get("close")
    if not price:
        bad(f"no usable price in the ticker: {ticker}")
        return None
    ok(f"{TESTNET_SYMBOL} last {price:,.2f}")
    return float(price)


def stage_guards(price):
    """The refusals, against the live client rather than a fake one.

    These are the same assertions the unit tests make. Repeating them here is
    not redundant: the tests prove the logic, this proves the logic is still
    wired to the thing that will actually send orders.
    """
    head(3, "the guards, with a real client underneath")
    broker = HyperliquidBroker(
        network="testnet",
        dry_run=True,
        allowed_symbols={TESTNET_SYMBOL},
        max_order_notional=25.0,
        max_total_exposure=100.0,
    )
    note(broker.describe())

    qty = 20.0 / price
    fill = broker.market(TESTNET_SYMBOL, BUY, qty, price, 0)
    ok(f"dry run returned a fill of {fill.qty:.6f} without sending anything")

    checks = [
        ("symbol outside the allow-list", lambda: broker.market("ETH/USDC:USDC", BUY, qty, price, 0)),
        ("order above the per-order cap", lambda: broker.market(TESTNET_SYMBOL, BUY, 1.0, price, 0)),
    ]
    for label, call in checks:
        try:
            call()
            bad(f"{label} was NOT refused")
            return None
        except LiveTradingRefused:
            ok(f"{label} refused")

    import pathlib
    switch = pathlib.Path("state/TESTNET_STOP")
    switch.parent.mkdir(parents=True, exist_ok=True)
    switch.write_text("", encoding="utf-8")
    stopped = HyperliquidBroker(
        network="testnet", dry_run=True, kill_switch=str(switch),
        allowed_symbols={TESTNET_SYMBOL}, max_order_notional=25.0,
    )
    try:
        stopped.market(TESTNET_SYMBOL, BUY, qty, price, 0)
        bad("the kill switch did NOT stop the order")
        switch.unlink(missing_ok=True)
        return None
    except LiveTradingRefused:
        ok("kill switch refused the order")
    finally:
        switch.unlink(missing_ok=True)

    return broker


def stage_authenticated(price):
    """With credentials: can we read our own account. Still sends nothing."""
    head(4, "authenticated reads")
    if not os.environ.get(ADDRESS_ENV) or not os.environ.get(KEY_ENV):
        note(f"skipped -- set {ADDRESS_ENV} and {KEY_ENV} to run this stage")
        note("use an API/agent wallet, which cannot withdraw. Never the master key.")
        return None

    broker = HyperliquidBroker(
        network="testnet", dry_run=True,
        allowed_symbols={TESTNET_SYMBOL},
        max_order_notional=25.0, max_total_exposure=100.0,
    )
    balances = broker.balances()
    ok(f"balances read: {balances.get('total') or balances}")
    ok(f"positions read: {broker.positions() or 'flat'}")
    ok(f"exposure: {broker.exposure():,.2f}")
    return broker


def stage_order(price, confirm):
    """The only stage that sends anything. Testnet, tiny, and opt-in twice."""
    head(5, "placing one real testnet order")
    if not confirm:
        note("skipped -- pass --place-order to send a single small testnet order")
        return
    if not os.environ.get(ADDRESS_ENV):
        note("skipped -- no credentials")
        return

    broker = HyperliquidBroker(
        network="testnet", dry_run=False,
        allowed_symbols={TESTNET_SYMBOL},
        max_order_notional=25.0, max_total_exposure=100.0,
    )
    qty = 15.0 / price
    note(f"sending market buy of {qty:.6f} {TESTNET_SYMBOL} (~$15 of testnet funds)")
    fill = broker.market(TESTNET_SYMBOL, BUY, qty, price, 0)
    ok(f"filled {fill.qty:.6f} at {fill.price:,.2f}, fee {fill.fee}")
    note(f"effective {fill.effective_price:,.2f} -- read back from the venue, not assumed")
    ok(f"positions now: {broker.positions()}")
    note("close it in the Hyperliquid testnet interface when you are done.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--place-order", action="store_true",
                        help="send one small order on TESTNET (stage 5)")
    args = parser.parse_args()

    print("Hyperliquid testnet check -- mainnet is unreachable from this script")

    client = stage_reachable()
    if client is None:
        return 1
    price = stage_prices(client)
    if price is None:
        return 1
    if stage_guards(price) is None:
        return 1
    stage_authenticated(price)
    stage_order(price, args.place_order)

    print("\ndone. Stages that were skipped need credentials; see docs/hyperliquid-setup.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
