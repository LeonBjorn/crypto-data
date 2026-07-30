"""Moving averages, RSI and rolling extremes.

The arithmetic layer, and nothing else. Every function here takes one pandas
Series and returns one pandas Series of the same length with the same index. No
file access, no configuration, no notion of a symbol or a timeframe, no opinion
about whether a number is a reason to buy. That belongs in `rules.py`, and
keeping it out of here is what makes these functions checkable against values
computed by hand.

A Series rather than the whole candle frame, deliberately. It means `sma` works
on closes, on volume, or on another indicator's output without knowing the
difference, and it means `rolling_high` can be handed the `high` column, which
is the one a breakout rule actually wants.

Two rules run through everything below.

The first is that no function may look forward. The value at bar t is computed
from bars up to and including t and from nothing else. This is not a matter of
taste: an indicator that reads one bar ahead produces a backtest that makes
money and a live system that does not, and the gap between them is invisible
because both are computed by the same code. The tests enforce it mechanically by
recomputing on truncated data.

The second is that a warm-up period must be NaN rather than a partial answer. A
twenty-bar average of three bars is not a twenty-bar average, but it is a
perfectly plausible-looking float, and nothing downstream can tell them apart.
"""

import numbers

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

# Wilder's original, and the default on every charting package. Pinned here
# rather than left to each caller so that "the RSI" means one thing across the
# project; a later change would silently invalidate every earlier result.
DEFAULT_RSI_PERIOD = 14

# What RSI reports when a window contains neither a gain nor a loss -- a price
# that has not moved at all. The formula is 0/0 there, so this is a convention
# rather than a result. Neutral is the only defensible choice: the alternatives
# say "maximally oversold" or "maximally overbought" about a flat line.
FLAT_RSI = 50.0

__all__ = [
    "DEFAULT_RSI_PERIOD",
    "FLAT_RSI",
    "IndicatorError",
    "ema",
    "rolling_high",
    "rolling_low",
    "rsi",
    "sma",
]


class IndicatorError(ValueError):
    """Raised for input these functions will not compute on.

    A ValueError subclass to match `ConfigError` in the collector, so a caller
    that only cares that the input was wrong can catch the broad type while the
    CLI catches this one and prints the message rather than a traceback.
    """


def _as_float_series(values, where):
    """
    Validate an input Series and return an independent float64 copy.

    `where` names the calling function so the message says which indicator
    objected rather than leaving that to be worked out from a traceback.

    The copy is not incidental. The intended usage is

        candles["sma20"] = sma(candles["close"], 20)

    which hands this function a live column of the caller's DataFrame. Anything
    that wrote through that reference would alter the price history itself, and
    every indicator computed afterwards would be computed on different data --
    a bug with no error message and no obvious first symptom.
    """
    if not isinstance(values, pd.Series):
        raise IndicatorError(
            f"{where} expects a pandas Series, got {type(values).__name__}. "
            f"If you have a DataFrame of candles, pass one column of it, "
            f"e.g. sma(candles['close'], 20)."
        )

    if not is_numeric_dtype(values):
        raise IndicatorError(
            f"{where} expects a numeric Series, got dtype {values.dtype}. "
            f"A dtype of 'object' usually means the values are strings."
        )

    present = values.notna().to_numpy()
    if present.any():
        first = int(present.argmax())
        if not present[first:].all():
            holes = np.flatnonzero(~present[first:]) + first
            shown = ", ".join(str(int(pos)) for pos in holes[:5])
            more = "" if len(holes) <= 5 else f" and {len(holes) - 5} more"
            raise IndicatorError(
                f"{where} was given a Series with {len(holes)} gap(s) in the "
                f"middle of it, at position(s) {shown}{more}. Leading NaN is "
                f"fine -- that is another indicator warming up -- but a NaN "
                f"between two numbers means a missing bar, and averaging across "
                f"it produces a value spanning a discontinuity that looks "
                f"exactly like a normal one. Load prices with "
                f"require_continuous=True, or fill the hole with `collect`."
            )

    return values.astype("float64")


def _check_period(value, name, where):
    """
    Validate a window, span or period: a whole number of bars, at least one.

    `numbers.Integral` rather than `int` so that a numpy integer -- which is
    what you get from anything that has been through a DataFrame -- is accepted.
    bool is excluded explicitly because it satisfies Integral and True == 1, so
    the obvious check would silently treat `window=True` as `window=1`.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise IndicatorError(
            f"{where}: {name} must be a whole number of bars, "
            f"got {value!r} ({type(value).__name__})"
        )
    if value < 1:
        raise IndicatorError(
            f"{where}: {name} must be at least 1, got {value}"
        )
    return int(value)


def sma(values, window):
    """
    Simple moving average: the mean of the last `window` bars, inclusive of now.

    NaN until `window` bars are available. `min_periods` is passed explicitly
    even though it already defaults to `window`, because that default is the
    single most important thing about this function and a default is easy to
    change by accident.
    """
    values = _as_float_series(values, "sma")
    window = _check_period(window, "window", "sma")
    return values.rolling(window=window, min_periods=window).mean()


def ema(values, span):
    """
    Exponential moving average, weighting recent bars more heavily.

    `span` is pandas' parameter and the one charting packages mean by "period":
    a span of 12 gives the smoothing factor alpha = 2/(12+1) that a 12-period
    EMA is defined by. It is not a count of bars averaged -- an EMA never fully
    forgets anything -- which is why the argument is not called `window`.

    `adjust=False` gives the recursive form, ema[t] = alpha*price[t] +
    (1-alpha)*ema[t-1], which is what every platform displays and what a live
    system would compute bar by bar. `adjust=True`, the pandas default, computes
    a mathematically tidier version whose early values differ, and matching a
    chart matters more here than matching a textbook.

    `min_periods=span` is the part worth reading twice. Left alone, `ewm` returns
    a value from the very first bar, and that value is the first price. A 26-span
    EMA at bar 3 is a barely-smoothed copy of the price, and a crossover rule
    comparing it against a 12-span EMA at bar 3 is comparing two barely-smoothed
    copies of the same series, which cross constantly and mean nothing. Masking
    the warm-up costs `span` bars of history and removes a whole class of
    imaginary signals.
    """
    values = _as_float_series(values, "ema")
    span = _check_period(span, "span", "ema")
    return values.ewm(span=span, adjust=False, min_periods=span).mean()


def _wilder_average(values, period):
    """
    Wilder's smoothed average: seed with a simple mean, then a running update.

    avg[period] = mean(values[1..period])
    avg[t]      = (avg[t-1] * (period - 1) + values[t]) / period

    Written out as a loop rather than as `ewm(alpha=1/period, adjust=False)`,
    which is the tempting one-liner and is not the same thing. The shortcut seeds
    with the *first* value instead of the mean of the first `period`, so its
    output is wrong for the whole warm-up and slightly wrong forever afterwards,
    and it reports a value from the first bar where Wilder is still warming up.
    Checked rather than assumed: on the six-price example in the tests the
    shortcut gives 83.33 where Wilder gives 75.0.

    Leading NaN is skipped, so the seed starts at the first real observation.
    `_as_float_series` has already refused any NaN after that point, so the
    window being averaged is guaranteed to be free of holes.

    500 bars per symbol per run is nothing, so a readable loop is worth more here
    than a vectorised expression nobody can check against the definition.
    """
    array = values.to_numpy(dtype="float64")
    out = np.full(len(array), np.nan)
    present = ~np.isnan(array)

    if int(present.sum()) < period:
        return pd.Series(out, index=values.index)

    start = int(present.argmax())
    seed_end = start + period

    average = array[start:seed_end].mean()
    out[seed_end - 1] = average

    for position in range(seed_end, len(array)):
        average = (average * (period - 1) + array[position]) / period
        out[position] = average

    return pd.Series(out, index=values.index)


def rsi(values, period=DEFAULT_RSI_PERIOD):
    """
    Relative strength index: recent average gain against recent average loss,
    on a 0-100 scale.

    The first value lands at index `period`, not `period - 1`. N differences need
    N+1 bars, so a 14-period RSI has nothing to say until the fifteenth. This is
    the easiest off-by-one in the file to make and the hardest to notice, because
    being one bar out shifts every value by one bar for the entire history and
    the shape of the line barely changes.

    The two divide-by-zero cases are handled explicitly rather than left to
    floating-point arithmetic. With no losses at all, the ratio is infinite and
    `100 - 100/(1 + inf)` evaluates to NaN, so the indicator would go blank at
    exactly the moment it is most extreme -- fourteen straight up bars. With
    neither gains nor losses it is 0/0, which is a question the formula cannot
    answer and `FLAT_RSI` answers by convention.
    """
    values = _as_float_series(values, "rsi")
    period = _check_period(period, "period", "rsi")

    delta = values.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    average_gain = _wilder_average(gain, period).to_numpy()
    average_loss = _wilder_average(loss, period).to_numpy()

    out = np.full(len(values), np.nan)
    computed = ~np.isnan(average_gain)

    # Falling or mixed: the ordinary case, and the only one where the ratio is
    # safe to take. A run of pure losses lands here too, with a gain of zero,
    # giving RS = 0 and RSI = 0 without needing a branch of its own.
    ordinary = computed & (average_loss > 0)
    strength = average_gain[ordinary] / average_loss[ordinary]
    out[ordinary] = 100.0 - 100.0 / (1.0 + strength)

    out[computed & (average_loss == 0) & (average_gain > 0)] = 100.0
    out[computed & (average_loss == 0) & (average_gain == 0)] = FLAT_RSI

    return pd.Series(out, index=values.index)


def rolling_high(values, window):
    """
    The highest value over the last `window` bars, including the current one.

    Including the current bar is the ordinary convention and it is the wrong
    thing for a breakout rule, which is worth stating here because the mistake is
    silent. "Price broke above the 20-bar high" cannot mean a high that the
    breaking bar helped set -- the current bar's high is part of that maximum, so
    price can never exceed it and the rule simply never fires. The prior high is
    `rolling_high(candles["high"], 20).shift(1)`, and increment 2 does that
    shifting inside the rule where the intent is visible.
    """
    values = _as_float_series(values, "rolling_high")
    window = _check_period(window, "window", "rolling_high")
    return values.rolling(window=window, min_periods=window).max()


def rolling_low(values, window):
    """
    The lowest value over the last `window` bars, including the current one.

    The same caveat as `rolling_high` applies in mirror image: a stop or a
    breakdown rule wants the prior low, which needs `.shift(1)`.
    """
    values = _as_float_series(values, "rolling_low")
    window = _check_period(window, "window", "rolling_low")
    return values.rolling(window=window, min_periods=window).min()
