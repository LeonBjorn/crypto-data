"""Driving the engine against a real venue, with the brakes on.

The paper engine and this share every line that decides *what* to trade. What
differs is that a mistake here costs money, so this module is mostly refusals.

THE ONE THAT MATTERS MOST
-------------------------
*A live runner must never replay history.* The paper account is built to replay:
hand it a fresh state and it walks two years of candles, which is exactly right
when the orders are imaginary and catastrophic when they are not -- roughly two
thousand market orders, at today's prices, for signals that fired months ago.

So the first run of a live account **arms** and acts on nothing. It records where
"now" is and stops. Only bars that arrive afterwards are traded. There is no flag
to skip this, because the situation in which someone wants to skip it is exactly
the situation in which they have misunderstood it.

THE OTHERS
----------
*The venue is the authority.* Positions are read back before anything is
decided. This process's memory is a cache, and it is wrong whenever a fill was
partial, an order was rejected, something was liquidated, or a human touched the
account. A disagreement stops the run rather than being reconciled by guesswork.

*Catching up is not the same as trading.* If the runner has been down, the bars
it missed are history, and history is not tradeable -- the prices in it are gone.
Past a small number of missed bars it refuses and asks for a decision instead of
firing a queue of orders into a market that has moved.

*Its ledger is its own.* Nothing here writes to the paper account's state. The
forward record is the only out-of-sample evidence this project has and it is not
going to be polluted by rehearsal fills.
"""

import logging

from paper.account import Account
from paper.broker import BUY, SELL
from paper.engine import Bar, PaperBook
from signals.trades import DEFAULT_COSTS

log = logging.getLogger(__name__)

__all__ = ["LiveError", "LiveRunner", "MAX_CATCHUP_BARS"]

# Bars the runner will act on in a single pass. A handful covers an ordinary
# restart or a machine that slept through an hour. Beyond it, the missed bars are
# history: their prices are gone, and filling them now means trading a queue of
# stale signals into a market that has already moved.
MAX_CATCHUP_BARS = 3


class LiveError(Exception):
    """Raised when a live run cannot proceed safely."""


class LiveRunner:
    """One account, one venue, and a great deal of reluctance.

    Wraps the same `PaperBook` the paper account uses -- the trade logic is
    identical by construction, which is the only reason results from one can be
    compared to the other -- and adds the checks that only matter when the fills
    are real.
    """

    def __init__(
        self,
        symbols,
        *,
        hold,
        broker,
        account=None,
        costs=DEFAULT_COSTS,
        stop=None,
        target=None,
        trail=None,
        max_catchup=MAX_CATCHUP_BARS,
    ):
        if not symbols:
            raise LiveError("a live runner needs at least one symbol")

        self.symbols = list(symbols)
        self.broker = broker
        self.account = account if account is not None else Account.unlimited()
        self.max_catchup = int(max_catchup)
        self.armed_at = None
        self.cursor = None

        self.books = {
            symbol: PaperBook(
                symbol, hold=hold, stop=stop, target=target, trail=trail,
                costs=costs, account=self.account, broker=broker,
            )
            for symbol in self.symbols
        }

    # -- arming --------------------------------------------------------------

    @property
    def armed(self) -> bool:
        return self.armed_at is not None

    def arm(self, frames):
        """Record where "now" is, and trade nothing.

        The whole point of the first run. Every bar in the store already
        happened; acting on any of it would be placing orders for signals whose
        prices are gone.
        """
        latest = max(int(frame["timestamp"].iloc[-1]) for frame in frames.values())
        self.armed_at = latest
        self.cursor = latest
        log.info("armed at %s; no orders placed. Bars after this are tradeable.", latest)
        return latest

    # -- reconciliation ------------------------------------------------------

    def reconcile(self):
        """Compare what the venue holds against what this process believes.

        Returns a dict of disagreements, empty when they match. Read *before*
        anything is decided, because every downstream decision assumes the
        current position is known, and this process's copy of it is a cache that
        goes stale whenever reality intervenes.
        """
        venue = self.broker.positions()
        local = {}
        for symbol, book in self.books.items():
            held = sum(position.qty for position in book.positions)
            if held:
                local[symbol] = held

        disagreements = {}
        for symbol in set(venue) | set(local):
            there = float(venue.get(symbol, 0.0))
            here = float(local.get(symbol, 0.0))
            # A tolerance, because a venue rounds to its lot size and this does
            # not. Anything larger than that is a real disagreement.
            if abs(there - here) > max(abs(here), abs(there)) * 1e-6 + 1e-12:
                disagreements[symbol] = {"venue": there, "local": here}
        return disagreements

    def require_agreement(self):
        """Stop the run unless the venue and this process agree.

        Deliberately not self-healing. Adopting the venue's position silently
        would paper over the interesting question -- *why* they diverged -- and
        the answers include a rejected order, a partial fill nobody noticed, and
        a liquidation. None of those should be discovered by a runner quietly
        adjusting its books and carrying on.
        """
        disagreements = self.reconcile()
        if disagreements:
            lines = ", ".join(
                f"{symbol}: venue {values['venue']:.8f} vs local {values['local']:.8f}"
                for symbol, values in sorted(disagreements.items())
            )
            raise LiveError(
                f"the venue and this process disagree about {len(disagreements)} "
                f"position(s): {lines}. Refusing to trade on a position this "
                f"process may be wrong about. Find out why they diverged -- a "
                f"rejected order, a partial fill, a liquidation, or a human -- "
                f"before running again."
            )
        return True

    # -- advancing -----------------------------------------------------------

    def advance(self, frames, signals):
        """Act on bars newer than the cursor. Returns how many were acted on."""
        if not self.armed:
            raise LiveError(
                "this runner has never been armed. Call arm() first: it records "
                "where the present is and trades nothing, so that history is not "
                "replayed into a live venue."
            )

        self.require_agreement()

        timeline = sorted(
            {int(value) for frame in frames.values() for value in frame["timestamp"]}
        )
        pending = [stamp for stamp in timeline if stamp > (self.cursor or 0)]

        if len(pending) > self.max_catchup:
            raise LiveError(
                f"{len(pending)} bars have passed since the last run, and the "
                f"limit is {self.max_catchup}. Those bars are history: their "
                f"prices are gone, and filling them now would fire a queue of "
                f"stale signals into a market that has moved. Either accept the "
                f"gap by re-arming, or investigate why the runner was down."
            )

        positions = {
            symbol: {int(stamp): index for index, stamp in enumerate(frame["timestamp"])}
            for symbol, frame in frames.items()
        }

        acted = 0
        for stamp in pending:
            for symbol in self.symbols:
                index = positions.get(symbol, {}).get(stamp)
                if index is None:
                    continue
                row = frames[symbol].iloc[index]
                self.books[symbol].advance(
                    Bar(
                        index=index, timestamp=stamp,
                        open=float(row["open"]), high=float(row["high"]),
                        low=float(row["low"]), close=float(row["close"]),
                    ),
                    signal=bool(signals[symbol].iloc[index]),
                )
            self.cursor = stamp
            acted += 1
        return acted

    # -- state ---------------------------------------------------------------

    def to_state(self) -> dict:
        return {
            "armed_at": self.armed_at,
            "cursor": self.cursor,
            "positions": [
                {
                    "symbol": position.symbol,
                    "entry_time": int(position.entry_time),
                    "entry_price": position.entry_price,
                    "effective_entry": position.effective_entry,
                    "qty": position.qty,
                    "peak": position.peak,
                }
                for book in self.books.values() for position in book.positions
            ],
            "ledger": [row for book in self.books.values() for row in book.closed],
            "venue": self.broker.describe() if hasattr(self.broker, "describe") else None,
        }

    def describe(self) -> str:
        state = "armed" if self.armed else "NOT ARMED (will trade nothing)"
        return f"live runner, {len(self.symbols)} symbol(s), {state}"
