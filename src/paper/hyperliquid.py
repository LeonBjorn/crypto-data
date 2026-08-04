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

THREE THINGS THAT ARE NOT THE SAME ON A VENUE AS IN THE STORE
-------------------------------------------------------------
*The symbol.* The store is keyed by Binance spot pairs -- `BTC/USDT` -- and
Hyperliquid trades perpetuals called `BTC/USDC:USDC`. Nothing translated between
them until now, so an engine handed this broker would have asked for a market
that does not exist. Mapping is resolved against the venue's own listings rather
than by string surgery, and a symbol with no perpetual is refused rather than
quietly skipped. Note that testnet lists fewer markets than mainnet: XRP has a
perpetual on mainnet and none on testnet, so a testnet rehearsal of this
project's five symbols can only ever cover four.

*The quantity.* Venues round to a lot size, and `amount_to_precision` will
happily return zero. ADA's step is one whole unit, so a fifth of a unit becomes
nothing at all -- and an order for nothing is not a small order, it is a bug that
looks like a filled position of size zero. Rounding is applied here and a result
of zero is refused.

*Closing versus opening.* A sell that is meant to close a long will open a short
instead if the position has already gone. Exits are sent reduce-only, so the
worst case is that nothing happens rather than that the book silently flips.

WHAT IT DOES NOT ASSUME
-----------------------
The engine issues an intent and reads a Fill back. A live venue partially fills,
slips, and rejects, so the Fill returned here is built from what the venue
reported and never from what was requested. That is the entire reason the seam
was written this way months before there was anything behind it.
"""

import logging
import os
import re
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

# An Ethereum-style address: 0x and forty hex digits, lower or mixed case. The
# venue rejects anything else with a 422 that names neither the field nor the
# reason, so it is worth checking here where the message can say which.
ADDRESS_PATTERN = re.compile(r"^0x[0-9a-fA-F]{40}$")

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
        symbol_map=None,
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
        # Explicit mapping wins; otherwise it is resolved from the venue's own
        # listings the first time one is needed. Explicit exists so a test, or
        # an operator who disagrees with the resolution, can pin it.
        self.symbol_map = dict(symbol_map or {})
        self._markets = None
        self.orders = []

        # Whether the client can act as an account, as opposed to merely read
        # public data. A dry run with no key still gets a real client -- symbol
        # resolution and lot sizes are public information and are most of what a
        # rehearsal needs to check.
        self.authenticated = True
        self._client = client if client is not None else self._build_client()

    # -- construction -------------------------------------------------------

    def _build_client(self):
        """A ccxt client, in sandbox mode unless mainnet was demanded twice."""
        import ccxt

        # Stripped before anything else. A variable set with a trailing newline
        # or a copied-in space is the single easiest mistake to make here, and
        # the venue's answer to it is a 422 reading "failed to deserialize the
        # JSON body" from deep inside ccxt -- which names neither the field nor
        # the whitespace. Diagnosed the hard way; refused clearly from now on.
        address = (os.environ.get(ADDRESS_ENV) or "").strip()
        key = (os.environ.get(KEY_ENV) or "").strip()
        if address and not ADDRESS_PATTERN.match(address):
            raise LiveTradingRefused(
                f"{ADDRESS_ENV} is not a well-formed address: expected 0x "
                f"followed by 40 hex characters, got {len(address)} characters "
                f"starting {address[:6]!r}. Check for a stray space or newline "
                f"-- the venue rejects those with an unhelpful 422."
            )
        if not address or not key:
            if self.dry_run:
                # A dry run with no credentials is a legitimate thing to want:
                # it exercises the wiring and the guards without holding a key at
                # all. It still gets a real client, because market listings and
                # lot sizes are public and resolving them is most of the value of
                # a rehearsal. What it cannot do is read the account, so the
                # exposure cap has nothing to check -- sound only because nothing
                # is sent, and said out loud rather than left to be assumed.
                log.warning(
                    "no credentials in the environment; running unauthenticated. "
                    "Symbols and lot sizes resolve normally, orders are logged "
                    "rather than sent, and the exposure cap is not enforceable."
                )
                self.authenticated = False
                public = ccxt.hyperliquid({"enableRateLimit": True})
                if self.network == "testnet":
                    public.set_sandbox_mode(True)
                return public
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

    # -- translating between the store and the venue -------------------------

    def _load_markets(self):
        if self._markets is None:
            if self._client is None:
                raise LiveTradingRefused(
                    "cannot resolve venue symbols without a client. Pass "
                    "symbol_map explicitly for an unauthenticated dry run."
                )
            self._markets = self._client.load_markets()
        return self._markets

    def resolve(self, symbol):
        """The venue's name for a symbol the store knows by another.

        Matched on the base asset and required to be a perpetual, because
        `BTC/USDC` and `BTC/USDC:USDC` both exist on this venue and are
        different instruments. A base with no perpetual raises rather than
        falling back to spot -- silently trading a different instrument from the
        one the strategy chose is the failure this method exists to prevent.
        """
        if symbol in self.symbol_map:
            return self.symbol_map[symbol]

        base = symbol.split("/")[0].upper()
        markets = self._load_markets()
        matches = [
            name for name, market in markets.items()
            if market.get("swap") and (market.get("base") or "").upper() == base
        ]
        if not matches:
            listed = sorted(
                (m.get("base") or "") for m in markets.values() if m.get("swap")
            )
            raise LiveTradingRefused(
                f"{symbol} has no perpetual market on this venue "
                f"({self.network}). {base} is not among the {len(listed)} listed. "
                f"Note testnet lists fewer markets than mainnet -- XRP is a "
                f"perpetual on mainnet and not on testnet."
            )
        resolved = sorted(matches)[0]
        self.symbol_map[symbol] = resolved
        return resolved

    def _round_quantity(self, venue_symbol, qty):
        """Quantity at the venue's lot size, refusing a result of zero.

        `amount_to_precision` rounds down, and for an asset whose step is a whole
        unit a fractional order becomes zero. Sending that is not a small trade;
        it is an order for nothing that will read afterwards as a position of
        size zero.
        """
        if self._client is None:
            return float(qty)
        rounded = float(self._client.amount_to_precision(venue_symbol, qty))
        if rounded <= 0:
            step = (self._load_markets().get(venue_symbol, {})
                    .get("precision", {}).get("amount"))
            raise LiveTradingRefused(
                f"{venue_symbol}: a quantity of {qty:.10f} rounds to zero at this "
                f"venue's lot size of {step}. The position is too small to express "
                f"here -- raise the capital or drop the symbol, but do not send an "
                f"order for nothing."
            )
        return rounded

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

    def market(self, symbol, side, qty, reference_price, timestamp, *, reduce_only=False) -> Fill:
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

        # The allow-list is checked against the symbol as the caller knows it,
        # so an operator writes the names they configured rather than the
        # venue's spelling of them.
        self._check(symbol, qty, reference_price)
        venue_symbol = self.resolve(symbol)
        qty = self._round_quantity(venue_symbol, qty)

        if self.dry_run:
            log.info(
                "DRY RUN: would %s %.8f %s (%s) at ~%.6f (%s)%s",
                side, qty, venue_symbol, symbol, reference_price, self.network,
                " reduce-only" if reduce_only else "",
            )
            self.orders.append(
                {"symbol": symbol, "venue_symbol": venue_symbol, "side": side,
                 "qty": qty, "reference_price": reference_price,
                 "reduce_only": reduce_only, "dry_run": True}
            )
            # Priced as the paper broker would, and flagged, so a dry run is
            # legible as a simulation rather than mistaken for an execution.
            return Fill(
                symbol=symbol, side=side, qty=float(qty),
                price=float(reference_price), effective_price=float(reference_price),
                fee=0.0, timestamp=int(timestamp),
            )

        params = {"reduceOnly": True} if reduce_only else {}
        order = self._client.create_order(
            symbol=venue_symbol, type="market", side=side, amount=qty,
            price=reference_price,  # hyperliquid needs a reference for slippage bounds
            params=params,
        )
        self.orders.append(order)
        # Reported under the name the caller used, not the venue's, so the
        # ledger stays keyed the same way the store and every earlier trade are.
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

    def close(self, symbol, side, qty, reference_price, timestamp) -> Fill:
        """Exit a position, reduce-only.

        A plain sell meant to close a long will open a short if the position has
        already gone -- liquidated, closed by hand, or never opened because an
        earlier order was rejected. Reduce-only makes the worst case "nothing
        happened" instead of "the book is now short".
        """
        return self.market(symbol, side, qty, reference_price, timestamp, reduce_only=True)

    def positions(self) -> dict:
        """Open quantity per symbol, as the venue understands it.

        The venue is the authority, not this process's memory. Live fills drift
        from what was intended -- partials, rejections, liquidations that
        happened while nothing was running -- so state is read back rather than
        assumed on every run.
        """
        if self._client is None or not self.authenticated:
            return {}
        held = {}
        for position in self._client.fetch_positions() or []:
            amount = float(position.get("contracts") or 0.0)
            if amount:
                side = position.get("side")
                held[position["symbol"]] = -amount if side == "short" else amount
        return held

    def balances(self) -> dict:
        if not self.authenticated:
            raise LiveTradingRefused(
                "balances need credentials; this client is unauthenticated."
            )
        balance = self._client.fetch_balance() or {}
        return {"free": balance.get("free", {}), "total": balance.get("total", {})}

    def exposure(self) -> float:
        """Absolute notional currently open, for the cap check."""
        if self._client is None or not self.authenticated:
            # Only reachable in an unauthenticated dry run, where no order is
            # sent and there is therefore no exposure to cap.
            return 0.0
        try:
            total = 0.0
            for position in self._client.fetch_positions() or []:
                total += abs(float(position.get("notional") or 0.0))
            return total
        except Exception as failure:
            # A cap that cannot be evaluated must not be treated as satisfied.
            # The cause is carried in the message as well as chained: the
            # original version said only that the read failed, and the reason
            # sat six frames down a traceback that ended in this refusal --
            # which reads like the guard being the problem rather than reporting
            # one.
            raise LiveTradingRefused(
                f"could not read positions from the venue, so the exposure cap "
                f"cannot be checked and no order will be sent. The venue said: "
                f"{type(failure).__name__}: {str(failure)[-160:]}"
            ) from failure

    def describe(self) -> str:
        return (
            f"hyperliquid {self.network}"
            f"{' (dry run)' if self.dry_run else ' LIVE'}, "
            f"per-order cap {self.max_order_notional:,.0f}, "
            f"exposure cap {self.max_total_exposure:,.0f}"
        )
