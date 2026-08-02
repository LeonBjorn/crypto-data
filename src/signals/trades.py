"""Trades: turning an opinion about a bar into a price someone could have paid.

A rule says "bar 4,201 was a reason to buy". This module answers the only
question that makes that statement worth anything: if you had believed it, what
would you actually have got?

THE ENTRY, WHICH IS THE WHOLE POINT
-----------------------------------
A signal on bar *t* buys at the open of bar *t+1*. Never at bar *t*'s close.

The close of bar *t* is the number the rule was computed from, and it is not
known until bar *t* has finished -- at which moment it stops being a price you
can trade and becomes a price you watched go past. The next open is the first
number in the file you could have acted on. The difference between those two
choices is small on any single trade and enormous in aggregate, because it is
exactly the size of the move the rule was reacting to. A crossover rule scored
at the signal bar's close is being paid for noticing the thing that already
happened.

This is the most common way a backtest lies, and it does not look like lying
while you are writing it. It looks like using the number you already have.

THE EXIT
--------
Symmetrically, an exit by time leaves at an open too: hold N bars and you are
out at the open of bar *t+1+N*, having lived through bars *t+1* to *t+N*.

Stops and targets are checked bar by bar across those bars, using each bar's low
and high. Both default to off, which matters more than it looks: with them off,
the result measures the *signal* and nothing else, so a rule that turns out to
be worthless cannot be rescued by a lucky stop, and a rule that works cannot be
credited to one. Turn them on afterwards and the difference between the two runs
is what the exit logic was worth.

A trailing stop is the third exit, and off by default for the same reason. It
begins as a plain stop `trail` below the entry and then ratchets up to sit
`trail` below the highest high seen since entry, never loosening. Where a fixed
stop asks "has this fallen too far from where I bought it", the trailing one
asks "has this given back too much of what it was ahead" -- which is the exit a
breakout wants, since the whole thesis of a breakout is that a move already
under way tends to run until it visibly stops rather than for a fixed number of
hours. It obeys the same two rules as the fixed stop below: a bar that gaps
through the level fills at its open, and the peak it trails is the one locked in
by earlier bars, not one set by the same candle whose low is being tested.

Two decisions inside that scan are worth stating out loud, because both are
places where a reasonable-looking choice quietly improves the results:

*When one bar touches both levels, the stop is taken.* An hourly candle records
a high and a low and says nothing about which came first. The data cannot answer
the question, so the answer is chosen, and it is chosen pessimistically. The
optimistic reading would inflate results specifically on the most volatile bars,
which is where the money is.

*When a bar opens beyond a level, the fill is the open.* A market that gaps four
percent through your stop did not fill you at your stop; there was nobody there.
Filling at the level would invent a counterparty, and it would do it precisely
on the worst trades, which is where the damage compounds. The same rule applies
upward: a gap through the target fills at the open too. The rule is "you got the
open", not "you got the worse of the two" -- gaps are not a punishment, they are
just what happened.

COSTS
-----
Fees and slippage default to real numbers rather than zero, and switching them
off takes an explicit `costs=FREE`.

That asymmetry is deliberate. A round trip at 0.1% taker fee plus 0.05% assumed
slippage per side costs about 0.3%, which is in the same range as the entire
edge of most hourly rules -- so a zero-cost backtest is not a slightly optimistic
backtest, it is often the difference between a winner and a loser. Defaults are
what you get when you are in a hurry, and being in a hurry should not be the
thing that makes the numbers look good.

Every trade carries both `gross_return` and `net_return` so the size of the toll
is always visible rather than baked in.

WHAT IS NOT HERE
----------------
No position sizing and no account. Each trade is scored as a percentage move on
its own, and trades are allowed to overlap: a signal is not suppressed because
an earlier one is still open. That measures the *rule* -- every signal gets
scored on its merits and the sample stays large. It does not model one wallet
with finite cash, where results would depend on which signal happened to arrive
first. That is a different question and it belongs to a different module.

No statistics either. Counting, averaging and comparing against random entries
is increment 5, and keeping it out means this file can be checked against frames
worked out on paper.

Bar numbers here are *positions*, not index labels. `prices.load` returns a
plain 0..n-1 index, and positions are what makes "the next bar" unambiguous.
"""

import numbers
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "Book",
    "Costs",
    "DEFAULT_COSTS",
    "DEFAULT_HOLDS",
    "FREE",
    "TradeError",
    "across_holds",
    "round_trips",
]


# The columns a trade table always has, even when it has no rows. Downstream
# code should never have to ask whether anything traded before it can look at a
# column name.
TRADE_COLUMNS = {
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
}

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close")

# Columns the fill model actually reads. Deliberately not all of them: the close
# is never used to fill anything, and listing it here would be checking data
# that no decision depends on.
USED_COLUMNS = ("open", "high", "low")

# Quarter of a day, a day, three days, a week. Four rather than one because a
# rule that works at exactly one holding period and nowhere near it has told you
# something, and what it has told you is usually "this was chance".
DEFAULT_HOLDS = (6, 24, 72, 168)


class TradeError(ValueError):
    """Raised when the request cannot be answered honestly.

    A ValueError subclass, matching PriceError and RuleError, so the CLI can
    catch one broad type and print a message rather than a traceback.
    """


@dataclass(frozen=True)
class Costs:
    """What a single side of a round trip costs, as fractions of the fill.

    The fee is a known number -- Binance spot taker is 0.1%. The slippage is an
    assumption, and a deliberately unkind one: hourly signals are traded at
    market, and market orders move. Both are charged on entry and on exit.
    """

    fee: float = 0.001
    slippage: float = 0.0005

    @property
    def per_side(self) -> float:
        return self.fee + self.slippage

    def describe(self) -> str:
        if self.per_side == 0:
            return "no costs"
        return f"{self.fee:.2%} fee + {self.slippage:.3%} slippage per side"


DEFAULT_COSTS = Costs()
FREE = Costs(fee=0.0, slippage=0.0)


@dataclass(frozen=True)
class Book:
    """Every completed round trip, plus what it took to get them.

    `dropped` is not a footnote. Signals that fired too near the end of the file
    to finish their hold are not short trades, they are absences, and a book
    that reported only its trades would let a long holding period quietly delete
    the most recent months without saying so.
    """

    trades: pd.DataFrame
    signals: int
    dropped: int
    hold: int
    costs: Costs
    stop: Optional[float] = None
    target: Optional[float] = None
    trail: Optional[float] = None

    def _exits(self) -> str:
        if not len(self.trades):
            return ""
        counts = self.trades["exit_reason"].value_counts()
        return " Exits: " + ", ".join(f"{count} by {reason}" for reason, count in counts.items())

    def summary(self) -> str:
        return (
            f"{self.hold}-bar hold: {self.signals} signal(s), "
            f"{len(self.trades)} trade(s), {self.dropped} dropped for want of "
            f"history. {self.costs.describe()}.{self._exits()}"
        )


def _whole_number(value, name):
    # bool is an Integral in Python, and `hold=True` is a typo, not a request
    # for a one-bar hold.
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TradeError(f"{name} must be a whole number of bars, got {value!r}.")
    if int(value) < 1:
        raise TradeError(f"{name} must be at least 1 bar, got {value!r}.")
    return int(value)


def _fraction(value, name, upper):
    """A positive fraction, optionally capped.

    Stops are capped below 1.0 because a stop at or beyond 100% is a stop at or
    below zero, which no price reaches. Targets have no cap, because prices have
    no ceiling and doubling is a strange target rather than an impossible one.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TradeError(f"{name} must be a fraction such as 0.02, got {value!r}.")
    value = float(value)
    if value <= 0:
        raise TradeError(f"{name} must be greater than zero, got {value!r}.")
    if upper is not None and value >= upper:
        raise TradeError(f"{name} must be less than {upper}, got {value!r}.")
    return value


def _checked_costs(costs):
    if not isinstance(costs, Costs):
        raise TradeError(f"costs must be a Costs, got {type(costs).__name__}.")
    for name in ("fee", "slippage"):
        value = getattr(costs, name)
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TradeError(f"costs.{name} must be a number, got {value!r}.")
        if float(value) < 0:
            raise TradeError(f"costs.{name} cannot be negative, got {value!r}.")
    if costs.per_side >= 1:
        raise TradeError(
            f"costs of {costs.per_side:.2%} per side would consume the whole position."
        )
    return costs


def _checked_candles(candles):
    if not isinstance(candles, pd.DataFrame):
        raise TradeError(f"candles must be a DataFrame, got {type(candles).__name__}.")
    missing = [name for name in REQUIRED_COLUMNS if name not in candles.columns]
    if missing:
        raise TradeError(f"candles are missing column(s): {', '.join(missing)}.")
    return candles


def _checked_signals(signals, candles):
    if not isinstance(signals, pd.Series):
        raise TradeError(f"signals must be a Series, got {type(signals).__name__}.")
    # Length before index, because "2 signals for 400 candles" tells you what to
    # go and fix and "the index does not match" does not.
    if len(signals) != len(candles):
        raise TradeError(f"got {len(signals)} signal(s) for {len(candles)} candle(s).")
    if not signals.index.equals(candles.index):
        raise TradeError("the signals are indexed differently from the candles.")
    if signals.dtype != bool:
        raise TradeError(f"signals must be boolean, got dtype {signals.dtype}.")
    return signals


def _empty_trades():
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in TRADE_COLUMNS.items()})


def _exit_of(entry_bar, final_bar, entry_price, opens, highs, lows, stop, target):
    """Where the trade ends: (bar, price, reason).

    The bars lived through are `entry_bar` up to but not including `final_bar`.
    `final_bar`'s open is the time-based exit, so what happens *within*
    `final_bar` is no longer the trade's problem -- it was sold at that open.
    """
    lived = slice(entry_bar, final_bar)
    stop_level = None if stop is None else entry_price * (1 - stop)
    target_level = None if target is None else entry_price * (1 + target)
    first_stop = np.inf
    first_target = np.inf

    if stop_level is not None:
        touched = np.flatnonzero(lows[lived] <= stop_level)
        if len(touched):
            first_stop = int(touched[0])
    if target_level is not None:
        touched = np.flatnonzero(highs[lived] >= target_level)
        if len(touched):
            first_target = int(touched[0])

    if first_stop == np.inf and first_target == np.inf:
        return final_bar, opens[final_bar], "hold"

    # `<=` rather than `<`: an equal index means both were touched in the same
    # bar, and the bar does not say which came first, so the stop is assumed.
    if first_stop <= first_target:
        bar = entry_bar + first_stop
        return bar, min(opens[bar], stop_level), "stop"

    bar = entry_bar + first_target
    return bar, max(opens[bar], target_level), "target"


def _trailing_exit(entry_bar, final_bar, entry_price, opens, highs, lows, trail):
    """The first bar a trailing stop is hit, as `(bar, price)`, or None.

    The stop rides `trail` below the highest high seen so far and never loosens.
    It starts at `entry_price * (1 - trail)` -- a plain stop -- and ratchets up
    as new highs are made, so what it protects is the peak of the trade rather
    than the entry.

    The peak used at each bar is the one set by *earlier* bars, not by the bar
    whose low is being tested. A single candle records a high and a low with no
    order between them, so the honest reading is the one that does not let this
    bar's own high lift the stop out of the way of this bar's own low first --
    the same pessimism the fixed stop uses when a bar touches two levels.

    A bar that gaps below the level fills at its open, not the level: a market
    that opened straight through the stop never offered the stop price, and
    inventing it would flatter exactly the worst exits. This mirrors _exit_of.
    """
    peak = entry_price
    for bar in range(entry_bar, final_bar):
        level = peak * (1 - trail)
        if lows[bar] <= level:
            return bar, min(opens[bar], level)
        if highs[bar] > peak:
            peak = highs[bar]
    return None


def round_trips(candles, signals, *, hold, stop=None, target=None, trail=None, costs=DEFAULT_COSTS):
    """Score every signal as a round trip. Returns a Book.

    `hold` is in bars. `stop` and `target` are fractions of the entry price --
    0.02 is a stop two percent below it -- and both are off by default.
    """
    hold = _whole_number(hold, "hold")
    stop = _fraction(stop, "stop", upper=1.0)
    target = _fraction(target, "target", upper=None)
    trail = _fraction(trail, "trail", upper=1.0)
    costs = _checked_costs(costs)
    candles = _checked_candles(candles)
    signals = _checked_signals(signals, candles)

    opens = candles["open"].to_numpy(dtype="float64")
    highs = candles["high"].to_numpy(dtype="float64")
    lows = candles["low"].to_numpy(dtype="float64")
    times = candles["timestamp"].to_numpy()
    bars = len(candles)

    fired = np.flatnonzero(signals.to_numpy())
    rows = []
    dropped = 0

    for signal_bar in fired:
        entry_bar = int(signal_bar) + 1
        final_bar = entry_bar + hold
        # The time-based exit needs `final_bar`'s open to exist. A signal that
        # cannot reach it has not made a shorter trade, it has made no trade.
        if final_bar >= bars:
            dropped += 1
            continue

        span = slice(entry_bar, final_bar + 1)
        for name, values in zip(USED_COLUMNS, (opens, highs, lows)):
            if not np.isfinite(values[span]).all():
                raise TradeError(
                    f"the {name} price is missing somewhere in bars "
                    f"{entry_bar}-{final_bar}, which a trade would have to hold through."
                )

        entry_price = opens[entry_bar]
        exit_bar, exit_price, reason = _exit_of(
            entry_bar, final_bar, entry_price, opens, highs, lows, stop, target
        )

        # The trailing stop is layered on top rather than folded into _exit_of,
        # so the fixed stop/target logic that the tests pin down stays untouched.
        # It wins only if it would have fired first -- an earlier bar, or the
        # same bar at a worse price, keeping the tie-break pessimistic.
        if trail is not None:
            trailed = _trailing_exit(
                entry_bar, final_bar, entry_price, opens, highs, lows, trail
            )
            if trailed is not None:
                trail_bar, trail_price = trailed
                if trail_bar < exit_bar or (
                    trail_bar == exit_bar and trail_price < exit_price
                ):
                    exit_bar, exit_price, reason = trail_bar, trail_price, "trail"

        paid = entry_price * (1 + costs.per_side)
        received = exit_price * (1 - costs.per_side)
        rows.append(
            {
                "entry_bar": entry_bar,
                "entry_time": times[entry_bar],
                "entry_price": entry_price,
                "exit_bar": exit_bar,
                "exit_time": times[exit_bar],
                "exit_price": exit_price,
                "exit_reason": reason,
                "bars_held": exit_bar - entry_bar,
                "gross_return": exit_price / entry_price - 1,
                "net_return": received / paid - 1,
            }
        )

    table = _empty_trades() if not rows else pd.DataFrame(rows).astype(TRADE_COLUMNS)
    return Book(
        trades=table,
        signals=int(len(fired)),
        dropped=dropped,
        hold=hold,
        costs=costs,
        stop=stop,
        target=target,
        trail=trail,
    )


def across_holds(candles, signals, holds=DEFAULT_HOLDS, **kwargs):
    """The same signals scored at several holding periods. Returns {hold: Book}."""
    return {
        _whole_number(hold, "hold"): round_trips(candles, signals, hold=hold, **kwargs)
        for hold in holds
    }
