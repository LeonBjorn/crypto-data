"""Rules: the point where numbers become an opinion about buying.

A rule takes a frame of candles and returns one boolean per candle. True means
"this bar is a reason to buy". Everything a rule needs to know it reads from
bars at or before the one it is answering about, and everything it decides is
recorded as a plain boolean so that the next layer -- turning signals into round
trips with fees -- has nothing left to interpret.

Three of them to begin with, all conventional, all with their textbook
parameters. That is deliberate. The temptation in this part of a project is to
invent a rule, tune it until the backtest looks good, and then discover that the
tuning *was* the result. Starting from settings someone else chose, decades ago,
for reasons unrelated to this data, makes the first measurement an honest one.
None of them is expected to work.

THE EDGE
--------
Every rule reports only the bar where its condition turns on, never the run of
bars where it happens to hold. "RSI below 30" can be true for a week; reporting
all of it would turn one dip into forty opportunities, and the trade count, hit
rate and average return computed downstream would all describe the same event
forty times while looking like a healthy sample.

Turning on requires an earlier bar where the condition was *known* and false.
Warm-up does not count as false, because it is not false -- it is unknown. That
distinction bites on exactly one bar per series and is worth the care: if a
20/50 crossover fired on bar 49 merely because the fast average happened to sit
above the slow one the first time both existed, that is not a crossing, it is an
artefact of where the file starts, and it would produce one fabricated trade per
symbol that no amount of extra history could remove.

The same suppression is applied to breakout, where it is arguably too strict --
a breakout is a statement about one bar, not a change of state, so a genuine
break on the first fully-warmed bar gets discarded. One signal per symbol, out
of seventeen thousand bars, in exchange for "fired" meaning one thing across the
whole project. Cheap.

WHAT IS NOT HERE
---------------
No position sizing, no stop, no exit, no fees, and no opinion about how many
signals may be open at once. A rule answers one question about one bar. Trades
are increment 4, and keeping them out means a rule can be checked against a
series worked out on paper.
"""

import inspect
import numbers
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import indicators as ind

__all__ = [
    "RULES",
    "Rule",
    "RuleError",
    "apply",
    "breakout",
    "get",
    "ma_cross",
    "names",
    "rsi_oversold",
]


# Which candle columns each rule reads. Named constants rather than tuples
# written out twice, because they appear both in the function that checks them
# and in the registry entry that advertises them, and two copies of a fact are
# one copy plus a chance to disagree.
CLOSE_ONLY = ("close",)
HIGH_AND_CLOSE = ("high", "close")


class RuleError(ValueError):
    """Raised for a rule that cannot be run as asked.

    Covers the things the indicators cannot know about: a frame that is not a
    frame or is missing a column, a rule name that does not exist, a parameter
    that does not exist, and parameter combinations that are individually legal
    but jointly meaningless. Period and window shapes are left to
    `IndicatorError`, which already checks them and says so precisely; both are
    ValueError, so a caller that does not care can catch one thing.
    """


def _frame(candles, columns, where):
    """Check `candles` is a frame carrying `columns`, and return it.

    The column check is up front rather than left to a KeyError because the
    KeyError says only the column name, and a caller who handed over the wrong
    frame needs to be told which of several frames in play was wrong.
    """
    if not isinstance(candles, pd.DataFrame):
        raise RuleError(
            f"{where} expects a candle DataFrame, got {type(candles).__name__}. "
            f"Rules take the frame that prices.load returns; the indicators are "
            f"the ones that take a bare Series."
        )
    missing = [name for name in columns if name not in candles.columns]
    if missing:
        raise RuleError(
            f"{where} needs the column(s) {', '.join(missing)}, which this frame "
            f"does not have. It has: {', '.join(map(str, candles.columns))}"
        )
    return candles


def _condition(truth, defined):
    """Fold a boolean test and a "was this even computable" mask into one Series.

    Three states, held as 1.0, 0.0 and NaN, because two booleans would have to
    be carried around together and the pair would eventually get separated. The
    NaN is what stops warm-up from being mistaken for a settled False, which is
    the whole reason this is not just `fast > slow`.
    """
    return truth.astype("float64").where(defined)


def _turns_true(condition, name):
    """The bar where `condition` changes from a known False to a known True.

    NaN compares equal to nothing, so a bar in warm-up neither fires nor lets
    the bar after it fire -- which is exactly the intent, stated in two lines
    rather than in a special case.
    """
    now = condition.to_numpy(dtype="float64")
    before = np.full(len(now), np.nan)
    before[1:] = now[:-1]
    fired = (now == 1.0) & (before == 0.0)
    return pd.Series(fired, index=condition.index, name=name)


def ma_cross(candles, *, fast=20, slow=50):
    """Buy when the fast moving average rises through the slow one.

    The oldest trend-following rule there is. Its weakness is well known and
    worth stating before any result arrives: it is late by construction, since
    the averages have to move before they can cross, and in a market that
    chops sideways it buys every false start. Whether that costs more than the
    trends it catches is what increment 5 is for.

    Simple averages rather than exponential, because the crossing bar is then a
    fact about the last `slow` closes and nothing else, and an exponential
    average never fully forgets anything.
    """
    _frame(candles, CLOSE_ONLY, "ma-cross")
    close = candles["close"]
    fast_ma = ind.sma(close, fast)
    slow_ma = ind.sma(close, slow)
    # After the indicators, so that a nonsense window is reported as a nonsense
    # window rather than as a comparison failure.
    if fast >= slow:
        raise RuleError(
            f"ma-cross: fast must be shorter than slow, got fast={fast} and "
            f"slow={slow}. Equal windows compare a series with itself, which is "
            f"never greater, so the rule would run happily and never fire once."
        )
    return _turns_true(
        _condition(fast_ma > slow_ma, fast_ma.notna() & slow_ma.notna()), "ma-cross"
    )


def rsi_oversold(candles, *, period=ind.DEFAULT_RSI_PERIOD, level=30):
    """Buy when the RSI drops through `level`.

    The contrarian one: a price that has fallen far enough, fast enough, is
    assumed to be due a bounce. Note what this rule is *not* -- it fires on the
    way down, at the moment the market becomes oversold, not when it recovers
    back above the level. The recovery variant is at least as common and is a
    different rule with a different lateness; if the numbers in increment 5
    suggest it, it belongs beside this one rather than hidden behind a flag.
    """
    _frame(candles, CLOSE_ONLY, "rsi-oversold")
    if isinstance(level, bool) or not isinstance(level, numbers.Real):
        raise RuleError(
            f"rsi-oversold: level must be a number, got {level!r} "
            f"({type(level).__name__})"
        )
    if not 0 < level < 100:
        raise RuleError(
            f"rsi-oversold: level must be between 0 and 100 exclusive, got {level}. "
            f"RSI cannot leave that range, so a level at either end is a rule that "
            f"always fires or never does."
        )
    strength = ind.rsi(candles["close"], period=period)
    return _turns_true(
        _condition(strength < level, strength.notna()), "rsi-oversold"
    )


def breakout(candles, *, window=20):
    """Buy when the close clears the highest high of the previous `window` bars.

    The `.shift(1)` below is the entire rule. Without it the window includes the
    bar being tested, and the bar that breaks out is by definition the highest
    bar in the neighbourhood, so the level rises to meet the price and the rule
    never fires -- silently, forever, while looking perfectly sensible.

    Highs set the level and the close breaks it, which is the conservative
    pairing. Using the high on both sides would fire on a price that was touched
    at an unknown moment inside the hour; a close is something that had actually
    happened by the time the bar ended, which is when a decision could be made.
    """
    _frame(candles, HIGH_AND_CLOSE, "breakout")
    prior_high = ind.rolling_high(candles["high"], window).shift(1)
    close = candles["close"]
    return _turns_true(
        _condition(close > prior_high, prior_high.notna()), "breakout"
    )


def _defaults_of(function):
    """The keyword defaults a rule declares in its own signature.

    Read out of the function rather than written down again in the registry.
    The command line needs to print the defaults without running anything, and
    an earlier version of this file kept a second copy for that purpose --
    which is two places to change a number and one place to forget. Mutation
    testing found it: changing `window=20` to `window=10` in the signature
    broke nothing, because every call went through the registry's copy.
    """
    return {
        name: parameter.default
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


@dataclass(frozen=True)
class Rule:
    """A rule, its command-line name, and what it needs to run."""

    name: str
    function: Callable
    description: str
    columns: tuple = CLOSE_ONLY

    @property
    def defaults(self):
        """The rule's parameters and their defaults, straight from the source."""
        return _defaults_of(self.function)


RULES = {
    rule.name: rule
    for rule in (
        Rule(
            name="ma-cross",
            function=ma_cross,
            description="20-bar average rises through the 50-bar average",
            columns=CLOSE_ONLY,
        ),
        Rule(
            name="rsi-oversold",
            function=rsi_oversold,
            description="14-bar RSI drops through 30",
            columns=CLOSE_ONLY,
        ),
        Rule(
            name="breakout",
            function=breakout,
            description="close clears the highest high of the previous 20 bars",
            columns=HIGH_AND_CLOSE,
        ),
    )
}


def names():
    """Every rule name, sorted, for help text and for iterating in tests."""
    return sorted(RULES)


def get(name):
    """The Rule registered under `name`."""
    try:
        return RULES[name]
    except (KeyError, TypeError):
        raise RuleError(
            f"no rule called {name!r}. Available: {', '.join(names())}"
        ) from None


def apply(name, candles, **overrides):
    """Run a rule by name, with `overrides` on top of its defaults.

    Unknown parameters are refused rather than ignored. A silently dropped
    `--param windwo=10` would run the default window and report the result as
    though it had been tuned, which is worse than a crash by exactly the amount
    of time it takes to notice.
    """
    rule = get(name)
    unknown = sorted(set(overrides) - set(rule.defaults))
    if unknown:
        raise RuleError(
            f"{name} has no parameter(s) {', '.join(unknown)}. "
            f"It takes: {', '.join(sorted(rule.defaults))}"
        )
    return rule.function(candles, **{**rule.defaults, **overrides})
