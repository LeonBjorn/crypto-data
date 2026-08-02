"""The execution seam: the one place an intention becomes a fill.

The engine never assumes it got what it asked for. It says "buy this much of
that" and then reads the Fill that comes back, taking the price, the quantity
and the fee from the answer rather than from the request. That sounds like
ceremony while the only broker fills perfectly out of a candle it was handed,
and it is the entire reason a live broker can be added later without touching
the engine: a real venue partially fills, slips, and rejects, and code that
reads its fills back already handles all three.

This is the same shape as `collector.exchange`, and for the same reason. That
module is the one place the project talks to a venue for *data*; this is the one
place it will talk to a venue for *orders*. Everything above both deals in plain
numbers.

WHAT A LIVE BROKER WILL HAVE TO ADD
-----------------------------------
Nothing in the interface, which is the point -- but a great deal behind it, and
it is worth writing down now while it is still free to be honest about it. A
live implementation needs credentials kept outside this repository, a hard cap
on order size and total exposure, an allowed-symbol list, a kill switch, and a
reconciliation step that treats the venue's own `positions()` as the truth
rather than this process's memory of it. `positions()` and `balances()` are on
the interface for that last reason: paper knows its own state perfectly and does
not need to ask, and live must ask, every run.

LONG AND SHORT
--------------
`side` is spelled out rather than inferred from the sign of a quantity, and
short is a legal value even though no rule in this project produces a sell
signal yet. The venue this is heading for is a perpetuals exchange, where
shorting is the ordinary case rather than an exotic one -- and the long-only
constraint is exactly what made every measured result bleed in a falling market.
Leaving the seam long-only would mean rebuilding it at the moment that stops
being acceptable.
"""

from dataclasses import dataclass
from typing import Optional, Protocol

from signals.trades import DEFAULT_COSTS, Costs

__all__ = [
    "BUY",
    "Broker",
    "BrokerError",
    "Fill",
    "PaperBroker",
    "SELL",
]

BUY = "buy"
SELL = "sell"


class BrokerError(Exception):
    """Raised when an order cannot be placed or filled as asked."""


@dataclass(frozen=True)
class Fill:
    """What actually happened, as the venue reports it.

    `price` is the raw fill price -- what the tape says. `effective_price` is
    that price after the costs of trading it, worsened in whichever direction
    hurts: a buy pays more, a sell receives less. Both are carried because they
    answer different questions and collapsing them loses one. The ledger records
    `price`, so a row can be checked against a chart; the cash arithmetic uses
    `effective_price`, so what the account believes it has is what it would
    really have.

    Keeping them apart is also what lets a live broker be honest. There, `price`
    is the venue's reported average fill and `fee` is the fee it actually
    charged, so `effective_price` stops being an assumption and becomes a
    measurement, with no change anywhere above.
    """

    symbol: str
    side: str
    qty: float
    price: float
    effective_price: float
    fee: float
    timestamp: int

    @property
    def notional(self) -> float:
        """What the position is worth at the raw fill price."""
        return self.qty * self.price

    @property
    def cash(self) -> float:
        """The signed effect on cash: negative for a buy, positive for a sell."""
        value = self.qty * self.effective_price
        return -value if self.side == BUY else value


class Broker(Protocol):
    """What the engine needs from anything that can execute.

    Deliberately four methods. Anything richer -- limit orders, reduce-only
    flags, leverage -- belongs to a specific venue and would leak that venue's
    vocabulary into an engine that should not know it.
    """

    def market(self, symbol: str, side: str, qty: float, reference_price: float,
               timestamp: int) -> Fill:
        """Buy or sell `qty` at the market, and report what was filled."""

    def positions(self) -> dict:
        """Open quantity per symbol, as the venue understands it."""

    def balances(self) -> dict:
        """Free and total balance, as the venue understands it."""


class PaperBroker:
    """Fills every order at the price it was handed, minus costs.

    The reference price is not a suggestion here, it is the answer: the engine
    passes the candle's open, or the level a stop was triggered at, and that is
    what the trade gets. There is no queue, no depth and no rejection, which is
    the one respect in which paper trading is easier than the real thing and the
    thing to remember when comparing the two later.

    Costs default to the same numbers the backtest uses, so a paper run and a
    backtest of the same period are priced identically and any difference
    between them is about timing rather than about fees.
    """

    def __init__(self, costs: Costs = DEFAULT_COSTS):
        self.costs = costs
        self._positions: dict = {}
        self._fills: list = []

    def market(self, symbol, side, qty, reference_price, timestamp) -> Fill:
        if side not in (BUY, SELL):
            raise BrokerError(f"side must be {BUY!r} or {SELL!r}, got {side!r}")
        if qty <= 0:
            raise BrokerError(f"quantity must be positive, got {qty!r}")
        if not reference_price > 0:
            raise BrokerError(f"reference price must be positive, got {reference_price!r}")

        per_side = self.costs.per_side
        effective = (
            reference_price * (1 + per_side)
            if side == BUY
            else reference_price * (1 - per_side)
        )

        fill = Fill(
            symbol=symbol,
            side=side,
            qty=float(qty),
            price=float(reference_price),
            effective_price=float(effective),
            # Reported separately from the price for the live case, where the
            # two really are separate numbers on the statement.
            fee=float(qty * reference_price * self.costs.fee),
            timestamp=int(timestamp),
        )

        held = self._positions.get(symbol, 0.0)
        self._positions[symbol] = held + fill.qty if side == BUY else held - fill.qty
        if abs(self._positions[symbol]) < 1e-12:
            self._positions.pop(symbol)

        self._fills.append(fill)
        return fill

    def positions(self) -> dict:
        return dict(self._positions)

    def balances(self) -> dict:
        """Paper holds no cash of its own; the Account is the authority.

        Returned empty rather than omitted so that code written against the
        interface does not have to ask which broker it is talking to.
        """
        return {}

    @property
    def fills(self) -> list:
        """Every fill this broker produced, oldest first."""
        return list(self._fills)
