"""The online trade engine: one candle in, whatever that candle changed out.

`signals.trades.round_trips` is handed the whole file and computes every trade
at once. This does the same arithmetic with the opposite information: it is told
about one closed candle, and must already have remembered everything that
matters about the ones before it. That is the only shape a live system can have,
because the future genuinely has not happened yet.

The two must agree, exactly, and the test that says so is the reason to believe
anything this package prints. Replay the stored candles through `advance` one at
a time with an unlimited account, and the ledger that falls out has to match
`round_trips` trade for trade and price for price. Anything else means one of
the two is wrong about what a trade is, and a paper run that cannot be compared
to a backtest cannot be compared to anything.

THE ORDER INSIDE ONE BAR
------------------------
Everything hangs on this, and it is not arbitrary -- each step is the online
spelling of a decision `round_trips` already made.

1. *Positions whose time is up leave at this bar's open.* Before anything else,
   because `round_trips` sells at the open of `entry_bar + hold` and what that
   bar does afterwards is somebody else's problem. A trade sold at the open
   cannot then be stopped out by the low of the same bar.

2. *An entry armed by the previous bar fills at this bar's open.* Never at the
   close that produced the signal -- that price had already gone past by the
   time the rule could be computed. This is the single most important line in
   the package, and it is the same one `trades.py` calls the most common way a
   backtest lies.

3. *Every open position is checked against this bar's high and low.* Including
   one opened in step 2, because you have held it since that bar's open and its
   low is yours.

4. *A signal on this bar arms an entry for the next one.* Last, so that a
   signal cannot fill on the bar that produced it however the steps are read.

WHAT IT REFUSES TO GUESS
------------------------
The engine does not decide what it can afford; the Account does. It does not
decide what price it got; the Broker reports that. It asks both and records the
answers, which is what will let the same code drive a real venue -- where the
answer to "what did I get" is frequently not what was asked for.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from paper.account import Account
from paper.broker import BUY, SELL, PaperBroker
from signals.trades import DEFAULT_COSTS, Costs

__all__ = [
    "Bar",
    "LEDGER_COLUMNS",
    "PaperBook",
    "Position",
]

# The ledger mirrors signals.trades.TRADE_COLUMNS so a paper run and a backtest
# can be put side by side without translating one into the other, plus the three
# facts a backtest has no use for and a wallet cannot do without.
LEDGER_COLUMNS = {
    "symbol": "object",
    "entry_bar": "int64",
    "entry_time": "int64",
    "entry_price": "float64",
    "exit_bar": "int64",
    "exit_time": "int64",
    "exit_price": "float64",
    "exit_reason": "object",
    "bars_held": "int64",
    "gross_return": "float64",
    "net_return": "float64",
    "qty": "float64",
    "cash_in": "float64",
    "cash_out": "float64",
}


@dataclass
class Bar:
    """One closed candle, named so the fields cannot be passed in the wrong order."""

    index: int
    timestamp: int
    open: float
    high: float
    low: float
    close: float


@dataclass
class Position:
    """One open trade, and everything needed to decide when it ends.

    `peak` is the highest high seen since entry and is what a trailing stop
    rides. It is carried on the position rather than recomputed because
    recomputing it would need the bars since entry, which is exactly the history
    a live process does not keep.
    """

    symbol: str
    entry_bar: int
    entry_time: int
    entry_price: float
    effective_entry: float
    qty: float
    final_bar: int
    peak: float

    @property
    def cash_in(self) -> float:
        return self.qty * self.effective_entry


class PaperBook:
    """One symbol's positions, advanced one candle at a time.

    A book per symbol, sharing one Account, so that the account's cash and cap
    are felt across the portfolio while the bar arithmetic stays per-symbol and
    checkable on paper. `hold`, `stop`, `target` and `trail` mean exactly what
    they mean in `signals.trades`, because they are handed to the same
    comparisons.
    """

    def __init__(
        self,
        symbol,
        *,
        hold,
        stop=None,
        target=None,
        trail=None,
        costs: Costs = DEFAULT_COSTS,
        account: Optional[Account] = None,
        broker=None,
    ):
        self.symbol = symbol
        self.hold = int(hold)
        self.stop = stop
        self.target = target
        self.trail = trail
        self.costs = costs
        self.account = account if account is not None else Account.unlimited()
        self.broker = broker if broker is not None else PaperBroker(costs)

        self.positions: list = []
        self.closed: list = []
        self._armed = False
        self._armed_from = None

    # -- the one method that matters ---------------------------------------

    def advance(self, bar: Bar, signal: bool = False) -> list:
        """Take one closed candle. Returns the trades that ended on it.

        Safe to call for every bar of a history or once an hour forever; there
        is no difference between the two as far as this method is concerned,
        which is the property that makes a paper run resumable.
        """
        finished = []

        # 1. Time exits, at this bar's open, before anything can happen to them.
        for position in [p for p in self.positions if bar.index >= p.final_bar]:
            finished.append(self._close(position, bar, bar.open, "hold"))

        # 2. An entry armed by the previous bar fills at this open.
        if self._armed:
            self._armed = False
            self._try_open(bar)

        # 3. Stops, targets and trails, against this bar's range.
        for position in list(self.positions):
            exit_at = self._bar_exit(position, bar)
            if exit_at is not None:
                price, reason = exit_at
                finished.append(self._close(position, bar, price, reason))
            elif bar.high > position.peak:
                # Only after the low has been checked. A candle says nothing
                # about which of its extremes came first, so letting this bar's
                # high lift the stop clear of this bar's low would be choosing
                # the flattering reading of an unanswerable question.
                position.peak = bar.high

        # 4. A signal here arms the next bar, never this one.
        if signal:
            self._armed = True
            self._armed_from = bar.index

        return finished

    # -- the parts it is made of -------------------------------------------

    def _bar_exit(self, position, bar):
        """`(price, reason)` if the position ends inside this bar, else None.

        The fixed levels and the trailing level are resolved separately and then
        compared, which is how `round_trips` does it and gives the same answer
        for the same reason: within one bar the pessimistic exit wins, and
        across bars the earlier one does.
        """
        fixed = None
        if self.stop is not None and bar.low <= position.entry_price * (1 - self.stop):
            level = position.entry_price * (1 - self.stop)
            # A bar that opened below the stop never offered the stop. Filling
            # there would invent a counterparty, on precisely the worst trades.
            fixed = (min(bar.open, level), "stop")
        elif self.target is not None and bar.high >= position.entry_price * (1 + self.target):
            level = position.entry_price * (1 + self.target)
            fixed = (max(bar.open, level), "target")

        trailed = None
        if self.trail is not None:
            level = position.peak * (1 - self.trail)
            if bar.low <= level:
                trailed = (min(bar.open, level), "trail")

        if fixed is not None and trailed is not None:
            return trailed if trailed[0] < fixed[0] else fixed
        return fixed if fixed is not None else trailed

    def _try_open(self, bar):
        """Open a position at this bar's open, if the account allows it."""
        refusal = self.account.refusal(self.symbol)
        if refusal is not None:
            self.account.reject(bar.index, self.symbol, refusal, at=bar.timestamp)
            return None

        notional = self.account.notional_for(self.symbol)
        # Sized off the effective price so that the cash actually leaving the
        # account is the notional asked for, fees included rather than added on
        # top of a full-size position the wallet could not quite afford.
        effective = bar.open * (1 + self.costs.per_side)
        qty = notional / effective

        fill = self.broker.market(self.symbol, BUY, qty, bar.open, bar.timestamp)

        position = Position(
            symbol=self.symbol,
            entry_bar=bar.index,
            entry_time=bar.timestamp,
            entry_price=fill.price,
            effective_entry=fill.effective_price,
            qty=fill.qty,
            final_bar=bar.index + self.hold,
            # The peak starts at the entry price, not at this bar's high. Step 3
            # has not run yet for this position, so its low is still to be
            # tested -- and testing it against a level this same bar's high had
            # already lifted is the flattering reading of a candle that does not
            # say which extreme came first. The high is folded in afterwards, by
            # the same step that folds in every later bar's.
            peak=fill.price,
        )
        self.positions.append(position)
        self.account.opened(self.symbol, fill.qty * fill.effective_price)
        return position

    def _close(self, position, bar, price, reason):
        """Sell a position at `price` and write the ledger row."""
        fill = self.broker.market(self.symbol, SELL, position.qty, price, bar.timestamp)
        self.positions.remove(position)
        self.account.closed(self.symbol, fill.qty * fill.effective_price)

        row = {
            "symbol": self.symbol,
            "entry_bar": position.entry_bar,
            "entry_time": position.entry_time,
            "entry_price": position.entry_price,
            "exit_bar": bar.index,
            "exit_time": bar.timestamp,
            "exit_price": fill.price,
            "exit_reason": reason,
            "bars_held": bar.index - position.entry_bar,
            # Computed from the raw and effective prices exactly as round_trips
            # does, rather than from the cash moved. The two agree, and deriving
            # them the same way means they cannot stop agreeing.
            "gross_return": fill.price / position.entry_price - 1,
            "net_return": fill.effective_price / position.effective_entry - 1,
            "qty": position.qty,
            "cash_in": position.cash_in,
            "cash_out": fill.qty * fill.effective_price,
        }
        self.closed.append(row)
        return row

    # -- what it has to say afterwards -------------------------------------

    def ledger(self) -> pd.DataFrame:
        """Every completed round trip, oldest first."""
        if not self.closed:
            return pd.DataFrame(
                {name: pd.Series(dtype=dtype) for name, dtype in LEDGER_COLUMNS.items()}
            )
        return pd.DataFrame(self.closed).astype(LEDGER_COLUMNS)

    def marks(self, last_close) -> float:
        """What the open positions are worth at `last_close`. Display only."""
        return sum(position.qty * last_close for position in self.positions)

    def run(self, candles, signals) -> pd.DataFrame:
        """Replay a whole frame one bar at a time. Returns the ledger.

        The convenience the invariant test is written against, and the thing a
        catch-up run does when a scheduled process has been off for a day: there
        is no separate batch path, only this loop.
        """
        opens = candles["open"].to_numpy(dtype="float64")
        highs = candles["high"].to_numpy(dtype="float64")
        lows = candles["low"].to_numpy(dtype="float64")
        closes = candles["close"].to_numpy(dtype="float64")
        times = candles["timestamp"].to_numpy()
        fired = signals.to_numpy()

        for index in range(len(candles)):
            self.advance(
                Bar(
                    index=index,
                    timestamp=int(times[index]),
                    open=float(opens[index]),
                    high=float(highs[index]),
                    low=float(lows[index]),
                    close=float(closes[index]),
                ),
                signal=bool(fired[index]),
            )

        return self.ledger()
