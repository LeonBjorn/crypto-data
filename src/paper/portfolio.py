"""Several symbols, one wallet, and the bookkeeping that survives a restart.

`PaperBook` handles one symbol's positions and knows nothing about the others.
That is the right split for the arithmetic -- a bar of ETH says nothing about
SOL -- but it is the wrong split for money, because all five books spend from
the same account. This module is the join: one book per symbol, one Account
between them, advanced together so that the cash one of them spends is cash the
next one cannot.

ADVANCING IN TIME ORDER, NOT SYMBOL ORDER
-----------------------------------------
The bars are walked outermost-by-time and innermost-by-symbol, rather than
finishing one symbol before starting the next. It would be simpler the other
way and it would be wrong: BTC would spend the whole wallet across two years
before ADA had seen its first candle, and the cap would be decided by
alphabetical order rather than by what happened first. Walking by time means the
account is in the state it would really have been in when each bar arrived.

Within one timestamp the symbols are still in a fixed order, and that order does
decide who gets the last slot when two fire on the same bar. There is no honest
way around that -- a real account has the same problem and resolves it by
whichever request arrived first -- so it is at least made deterministic, and
noted here rather than discovered later.

RESUMING
--------
The cursor is a timestamp, not a bar number. Bar numbers are positions in
whatever frame happened to be loaded, and the frame grows every time the
collector runs, so a number that meant "yesterday at noon" one day means
something else the next. A timestamp means the same thing forever, and mapping
it back to a position is arithmetic the store's continuity guarantees.
"""

import numbers

import pandas as pd

from paper.account import Account
from paper.engine import LEDGER_COLUMNS, Bar, PaperBook, Position
from paper.state import MAX_REJECTIONS_KEPT
from signals.trades import Costs

__all__ = ["Portfolio", "PortfolioError"]


class PortfolioError(Exception):
    """Raised when a portfolio cannot be built or resumed as asked."""


class Portfolio:
    """One account, one book per symbol, advanced together through time."""

    def __init__(self, symbols, *, hold, stop=None, target=None, trail=None,
                 costs: Costs, account: Account):
        if not symbols:
            raise PortfolioError("a portfolio needs at least one symbol")

        self.symbols = list(symbols)
        self.hold = int(hold)
        self.stop = stop
        self.target = target
        self.trail = trail
        self.costs = costs
        self.account = account

        self.books = {
            symbol: PaperBook(
                symbol, hold=hold, stop=stop, target=target, trail=trail,
                costs=costs, account=account,
            )
            for symbol in self.symbols
        }
        # Timestamp of the last bar acted on. None means nothing has been seen,
        # which is what a first run looks like.
        self.cursor = None
        self.rejections_total = 0
        # Rejections carried in from earlier runs. Only a tail of them is saved,
        # so the count has to be remembered separately from the list -- and a
        # resumed run must add to it rather than replace it, or every restart
        # would report that the wallet had never refused anything.
        self._rejections_before = 0

    # -- moving forward -----------------------------------------------------

    def advance(self, frames, signals) -> int:
        """Walk every bar newer than the cursor. Returns how many were acted on.

        `frames` and `signals` cover the whole stored history, not just the new
        part. They have to: a rule needs its warm-up, and a two-hundred-bar
        average of the last three bars is not a two-hundred-bar average. Only
        the bars past the cursor are *acted* on, which is what makes running
        this hourly and running it once a week produce the same ledger.
        """
        timeline = sorted(
            {int(value) for frame in frames.values() for value in frame["timestamp"]}
        )
        if self.cursor is not None:
            timeline = [stamp for stamp in timeline if stamp > self.cursor]

        positions = {
            symbol: {int(stamp): index for index, stamp in enumerate(frame["timestamp"])}
            for symbol, frame in frames.items()
        }

        acted = 0
        for stamp in timeline:
            for symbol in self.symbols:
                index = positions.get(symbol, {}).get(stamp)
                if index is None:
                    # This symbol has no candle at this moment. A shorter
                    # history, or a hole the store refused to paper over.
                    continue
                row = frames[symbol].iloc[index]
                self.books[symbol].advance(
                    Bar(
                        index=index,
                        timestamp=stamp,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                    ),
                    signal=bool(signals[symbol].iloc[index]),
                )
            self.cursor = stamp
            acted += 1

        self.rejections_total = self._rejections_before + len(self.account.rejections)
        return acted

    # -- what it has to say -------------------------------------------------

    def ledger(self) -> pd.DataFrame:
        """Every completed round trip across every symbol, oldest exit first."""
        tables = [book.ledger() for book in self.books.values()]
        combined = pd.concat(tables, ignore_index=True) if tables else None
        if combined is None or combined.empty:
            return pd.DataFrame(
                {name: pd.Series(dtype=dtype) for name, dtype in LEDGER_COLUMNS.items()}
            )
        return combined.sort_values("exit_time").reset_index(drop=True)

    def open_positions(self) -> list:
        return [position for book in self.books.values() for position in book.positions]

    def marks(self, frames) -> dict:
        """What each symbol's open positions are worth at its latest close."""
        values = {}
        for symbol, book in self.books.items():
            if not book.positions or symbol not in frames or frames[symbol].empty:
                continue
            last_close = float(frames[symbol]["close"].iloc[-1])
            values[symbol] = book.marks(last_close)
        return values

    def equity(self, frames) -> float:
        return self.account.equity(self.marks(frames))

    # -- surviving a restart ------------------------------------------------

    def to_state(self) -> dict:
        """Everything needed to carry on exactly where this left off."""
        return {
            "cursor": self.cursor,
            "cash": self.account.cash,
            "positions": [
                {
                    "symbol": position.symbol,
                    "entry_time": int(position.entry_time),
                    "entry_price": position.entry_price,
                    "effective_entry": position.effective_entry,
                    "qty": position.qty,
                    "peak": position.peak,
                }
                for position in self.open_positions()
            ],
            "ledger": self.ledger().to_dict(orient="records"),
            "rejections_total": self.rejections_total,
            "rejections_recent": [
                {"bar": r.bar, "symbol": r.symbol, "reason": r.reason}
                for r in self.account.rejections[-MAX_REJECTIONS_KEPT:]
            ],
        }

    def restore(self, state, frames):
        """Put back what `to_state` saved.

        Positions are restored by timestamp and converted to the bar number they
        occupy in *this* frame, because the frame has grown since they were
        written and a stored bar number would now point at the wrong candle.
        """
        if state is None:
            return self

        cursor = state.get("cursor")
        if cursor is not None and not isinstance(cursor, numbers.Integral):
            raise PortfolioError(f"saved cursor should be epoch milliseconds, got {cursor!r}")
        self.cursor = None if cursor is None else int(cursor)

        self.account.cash = float(state.get("cash", self.account.starting_capital))
        self._rejections_before = int(state.get("rejections_total", 0))
        self.rejections_total = self._rejections_before

        indices = {
            symbol: {int(stamp): index for index, stamp in enumerate(frame["timestamp"])}
            for symbol, frame in frames.items()
        }

        for saved in state.get("positions", []):
            symbol = saved["symbol"]
            book = self.books.get(symbol)
            if book is None:
                raise PortfolioError(
                    f"saved state holds a position in {symbol}, which is not in the "
                    f"current symbol list. Add it back, or start over with --reset."
                )

            entry_time = int(saved["entry_time"])
            entry_bar = indices.get(symbol, {}).get(entry_time)
            if entry_bar is None:
                raise PortfolioError(
                    f"the candle a {symbol} position was opened on ({entry_time}) is "
                    f"no longer in the store, so its holding period cannot be placed. "
                    f"Backfill further, or start over with --reset."
                )

            book.positions.append(
                Position(
                    symbol=symbol,
                    entry_bar=entry_bar,
                    entry_time=entry_time,
                    entry_price=float(saved["entry_price"]),
                    effective_entry=float(saved["effective_entry"]),
                    qty=float(saved["qty"]),
                    final_bar=entry_bar + self.hold,
                    peak=float(saved["peak"]),
                )
            )
            # The account counts positions, and a restored one is still open.
            # Zero cash, because the money left when the position was opened and
            # the cash restored above already reflects that -- charging it again
            # would spend it twice.
            self.account.opened(symbol, 0.0)

        for row in state.get("ledger", []):
            symbol = row.get("symbol")
            if symbol in self.books:
                self.books[symbol].closed.append(row)

        return self
