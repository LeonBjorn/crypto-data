"""A Hyperliquid broker, defaulting to testnet and to placing nothing at all.

This is the first code in the project capable of moving real money, and the
whole repository has until now defended the property that no such code existed.
That property is being given up deliberately and in one place, so the rest of
this file is mostly the conditions under which it is given up.

Read the order of defences before the code:

1. *It does not trade unless told twice.* `dry_run` is the default and logs the
   order it would have placed. Turning it off is a constructor argument.
2. *It is on testnet unless told otherwise.* Mainnet requires both an explicit
   argument and the environment variable named below. One of the two on its own
   is not enough, because one of the two is what a mistake looks like.
3. *A file can stop it.* If the kill switch exists, every order is refused
   before anything is signed. `touch` is faster than finding the process.
4. *Caps are checked before signing, not after.* Per-order notional, total
   exposure, and an allow-list of symbols. Exceeding one raises rather than
   truncating the order: silently trading less than asked is its own bug.
5. *There is no withdrawal path.* Not disabled -- absent. The class has no
   method that could move funds off the venue, so no bug in it can.

CREDENTIALS
-----------
Read from the environment, never from a config file and never from an argument
that might be logged. On Hyperliquid the signing material is a *private key*,
which is a different and much sharper object than an exchange API key: it is the
wallet. Use an **API wallet** (an agent wallet), which can place orders and
cannot withdraw, and keep the master key nowhere near this process.

    HYPERLIQUID_WALLET_ADDRESS   the account being traded
    HYPERLIQUID_PRIVATE_KEY      an API/agent wallet key, never the master key
    HYPERLIQUID_ALLOW_MAINNET    must equal "yes" before mainnet is reachable

WHAT IT DOES NOT ASSUME
-----------------------
The engine issues an intent and reads a Fill back. A live venue partially fills,
slips, and rejects, so the Fill returned here is built from what the venue
reported and never from what was requested. That is the entire reason the seam
was written this way months before there was anything behind it.
"""

import logging
import os
import time
from pathlib import Path

from paper.broker import BUY, SELL, BrokerError, Fill

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_KILL_SWITCH",
    "HyperliquidBroker",
    "LiveTradingRefused",
]

# Its presence stops everything. A path rather than a flag because a file can be
# created by anyone with a shell, from anywhere, without this process
# cooperating -- which is the property you want at the moment you want it.
DEFAULT_KILL_SWITCH = "state/STOP"

MAINNET_ENV = "HYPERLIQUID_ALLOW_MAINNET"
ADDRESS_ENV = "HYPERLIQUID_WALLET_ADDRESS"
KEY_ENV = "HYPERLIQUID_PRIVATE_KEY"


class LiveTradingRefused(BrokerError):
    """Raised when a live order is refused by one of this module's own guards.

    Deliberately distinct from a venue error. "Hyperliquid rejected this" and
    "this project refused to send it" are different events and the second should
    never be mistaken for a network problem and retried.
    """


class HyperliquidBroker:
    """Places orders on Hyperliquid, or says exactly what it would have placed.

    Implements the same three methods as `PaperBroker`, so the engine cannot
    tell them apart. The differences are all in what can go wrong.
    """

    def __init__(
        self,
        *,
        network="testnet",
        dry_run=True,
        allowed_symbols=(),
        max_order_notional=100.0,
        max_total_exposure=500.0,
        kill_switch=DEFAULT_KILL_SWITCH,
        client=None,
    ):
        if network not in ("testnet", "mainnet"):
            raise LiveTradingRefused(f"network must be testnet or mainnet, got {network!r}")

        # Two independent switches for mainnet, on purpose. A single one is a
        # single thing to get wrong, and this is the one place in the project
        # where getting it wrong costs money rather than time.
        if network == "mainnet" and os.environ.get(MAINNET_ENV) != "yes":
            raise LiveTradingRefused(
                f"mainnet was requested but {MAINNET_ENV} is not set to 'yes'. "
                f"Both are required. If you did not deliberately set out to trade "
                f"real money in this run, this is the guard working."
            )

        self.network = network
        self.dry_run = bool(dry_run)
        self.allowed_symbols = set(allowed_symbols)
        self.max_order_notional = float(max_order_notional)
        self.max_total_exposure = float(max_total_exposure)
        self.kill_switch = Path(kill_switch)
        self.orders = []

        self._client = client if client is not None else self._build_client()

    # -- construction -------------------------------------------------------

    def _build_client(self):
        """A ccxt client, in sandbox mode unless mainnet was demanded twice."""
        import ccxt

        address = os.environ.get(ADDRESS_ENV)
        key = os.environ.get(KEY_ENV)
        if not address or not key:
            if self.dry_run:
                # A dry run with no credentials is a legitimate thing to want:
                # it exercises the wiring and the guards without holding a key
                # at all. It cannot read positions, so the exposure cap has
                # nothing to check -- which is sound only because nothing is
                # sent. It says so rather than appearing to be armed.
                log.warning(
                    "no credentials in the environment; running unauthenticated. "
                    "Orders are logged, positions cannot be read, and the exposure "
                    "cap is not enforceable in this mode."
                )
                return None
            raise LiveTradingRefused(
                f"set {ADDRESS_ENV} and {KEY_ENV} in the environment. They are "
                f"deliberately not read from any config file, so that a key "
                f"cannot arrive in the repository by being pasted into one. Use "
                f"an API/agent wallet that cannot withdraw."
            )

        client = ccxt.hyperliquid({
            "walletAddress": address,
            "privateKey": key,
            "enableRateLimit": True,
        })
        if self.network == "testnet":
            client.set_sandbox_mode(True)
        return client

    # -- the guards ---------------------------------------------------------

    def _refuse_if_stopped(self):
        if self.kill_switch.exists():
            raise LiveTradingRefused(
                f"the kill switch {self.kill_switch} exists, so no order was sent. "
                f"Delete it to resume."
            )

    def _check(self, symbol, qty, reference_price):
        """Every limit, before anything is signed."""
        self._refuse_if_stopped()

        if self.allowed_symbols and symbol not in self.allowed_symbols:
            raise LiveTradingRefused(
                f"{symbol} is not in the allowed list {sorted(self.allowed_symbols)}. "
                f"An allow-list is the cheapest protection against a config error "
                f"pointing the engine at a market nobody meant to trade."
            )

        notional = abs(qty * reference_price)
        if notional > self.max_order_notional:
            raise LiveTradingRefused(
                f"order of {notional:,.2f} exceeds the per-order cap of "
                f"{self.max_order_notional:,.2f}. Raise the cap deliberately or "
                f"send less -- this is not truncated automatically, because "
                f"quietly trading less than asked is its own bug."
            )

        exposure = self.exposure() + notional
        if exposure > self.max_total_exposure:
            raise LiveTradingRefused(
                f"this order would take total exposure to {exposure:,.2f}, over "
                f"the cap of {self.max_total_exposure:,.2f}."
            )

    # -- the interface the engine uses --------------------------------------

    def market(self, symbol, side, qty, reference_price, timestamp) -> Fill:
        """Send a market order and report what came back.

        The Fill is built from the venue's answer -- filled quantity, average
        price, fee actually charged -- and never from the request. A live venue
        fills partially, and code that assumes otherwise keeps a position it does
        not have.
        """
        if side not in (BUY, SELL):
            raise BrokerError(f"side must be {BUY!r} or {SELL!r}, got {side!r}")
        if qty <= 0:
            raise BrokerError(f"quantity must be positive, got {qty!r}")

        self._check(symbol, qty, reference_price)

        if self.dry_run:
            log.info(
                "DRY RUN: would %s %.8f %s at ~%.6f (%s)",
                side, qty, symbol, reference_price, self.network,
            )
            self.orders.append(
                {"symbol": symbol, "side": side, "qty": qty,
                 "reference_price": reference_price, "dry_run": True}
            )
            # Priced as the paper broker would, and flagged, so a dry run is
            # legible as a simulation rather than mistaken for an execution.
            return Fill(
                symbol=symbol, side=side, qty=float(qty),
                price=float(reference_price), effective_price=float(reference_price),
                fee=0.0, timestamp=int(timestamp),
            )

        order = self._client.create_order(
            symbol=symbol, type="market", side=side, amount=qty,
            price=reference_price,  # hyperliquid needs a reference for slippage bounds
        )
        self.orders.append(order)
        return self._fill_from(order, symbol, side, timestamp)

    def _fill_from(self, order, symbol, side, timestamp) -> Fill:
        """Turn a venue order into a Fill, refusing to guess at anything."""
        filled = float(order.get("filled") or 0.0)
        average = order.get("average") or order.get("price")

        if not filled or not average:
            raise BrokerError(
                f"{symbol}: the order returned no fill "
                f"(filled={order.get('filled')!r}, average={order.get('average')!r}, "
                f"status={order.get('status')!r}). Nothing is assumed about the "
                f"position from an order that did not report one."
            )

        average = float(average)
        fee_cost = float((order.get("fee") or {}).get("cost") or 0.0)

        # Effective price is the average worsened by the fee actually charged, in
        # the direction that hurts. Here it is measured rather than assumed,
        # which is the whole difference from the paper broker.
        per_unit_fee = fee_cost / filled if filled else 0.0
        effective = average + per_unit_fee if side == BUY else average - per_unit_fee

        return Fill(
            symbol=symbol, side=side, qty=filled, price=average,
            effective_price=effective, fee=fee_cost,
            timestamp=int(order.get("timestamp") or timestamp or time.time() * 1000),
        )

    def positions(self) -> dict:
        """Open quantity per symbol, as the venue understands it.

        The venue is the authority, not this process's memory. Live fills drift
        from what was intended -- partials, rejections, liquidations that
        happened while nothing was running -- so state is read back rather than
        assumed on every run.
        """
        if self._client is None:
            return {}
        held = {}
        for position in self._client.fetch_positions() or []:
            amount = float(position.get("contracts") or 0.0)
            if amount:
                side = position.get("side")
                held[position["symbol"]] = -amount if side == "short" else amount
        return held

    def balances(self) -> dict:
        balance = self._client.fetch_balance() or {}
        return {"free": balance.get("free", {}), "total": balance.get("total", {})}

    def exposure(self) -> float:
        """Absolute notional currently open, for the cap check."""
        if self._client is None:
            # Only reachable in an unauthenticated dry run, where no order is
            # sent and there is therefore no exposure to cap.
            return 0.0
        try:
            total = 0.0
            for position in self._client.fetch_positions() or []:
                total += abs(float(position.get("notional") or 0.0))
            return total
        except Exception:
            # A cap that cannot be evaluated must not be treated as satisfied.
            raise LiveTradingRefused(
                "could not read current positions from the venue, so the "
                "exposure cap cannot be checked. Refusing to trade blind."
            )

    def describe(self) -> str:
        return (
            f"hyperliquid {self.network}"
            f"{' (dry run)' if self.dry_run else ' LIVE'}, "
            f"per-order cap {self.max_order_notional:,.0f}, "
            f"exposure cap {self.max_total_exposure:,.0f}"
        )
