"""Getting candles out of the store and into a state fit to compute on.

Thin on purpose. The Parquet reading, the path layout and the gap arithmetic all
already exist in `collector` and are already tested there, so this module calls
them rather than growing a second copy that can drift. What it adds is a change
of attitude.

The collector is tolerant because it has to be. A missing file is a normal first
run. A short range is a reason to go and fetch more. Half a symbol is progress.
None of that survives contact with analysis: once candles reach an indicator, a
missing bar stops looking like a missing bar and starts looking like a price that
did not move, a short range stops looking short and starts looking like a
complete answer to a question nobody asked.

So this module refuses things the collector shrugs at, and the refusals are the
point:

    a symbol that was never collected      -- zero rows and zero trades looks
                                              exactly like a rule that never fired
    a range wider than the store holds     -- a two-year backtest quietly
                                              becoming a two-month one
    a hole inside the requested range      -- a 20-bar average spanning a
                                              discontinuity, looking ordinary
    a duplicated timestamp                 -- one bar counted twice, invisible
                                              to gap detection

The one case where refusing would be wrong is a hole *outside* the range being
asked about. Exchanges have permanent holes in old history, and if any hole
anywhere disqualified a file then one bad hour in 2024 would block every backtest
over 2026 -- which would not make anyone careful, it would just make
`require_continuous=False` a habit.

This module reads the store and never writes to it.

Times are epoch milliseconds throughout, matching the store. Parsing '2024-08-01'
is the CLI's job, deliberately in one place: a project that converts dates in two
modules is a project whose two modules will eventually disagree about what a date
means.
"""

import datetime as dt
import numbers

from collector import gaps, store
from collector.timeframes import timeframe_to_ms

# How many gaps to name before summarising. A message taller than a terminal
# window is a message that gets scrolled past rather than read, and the first
# few tell you as much about the shape of the problem as all of them.
MAX_GAPS_SHOWN = 5

__all__ = ["PriceError", "load"]


class PriceError(ValueError):
    """Raised when the store cannot answer the question that was asked.

    A ValueError subclass, matching ConfigError and TimeframeError, so the CLI
    can catch one broad type and print a message instead of a traceback.
    """


def _utc(ms):
    """Epoch milliseconds as a readable UTC string, for error messages only.

    Never for anything that gets stored or compared. "1722470400000 is before
    1722452400000" is technically an explanation and takes a calculator to read.
    """
    moment = dt.datetime.fromtimestamp(ms / 1000, tz=dt.timezone.utc)
    return moment.strftime("%Y-%m-%d %H:%M")


def _check_ms(value, name):
    """
    Validate an optional epoch-millisecond argument.

    `numbers.Integral` accepts numpy integers, which is what comes back out of a
    DataFrame column. bool is excluded first because it satisfies Integral and
    True == 1, which would mean a timestamp one millisecond after 1970.

    A date string gets its own sentence because `start_ms="2024-08-01"` is
    exactly what someone will type, and the useful reply names the layer that
    does know how to read that.
    """
    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise PriceError(
            f"{name} must be epoch milliseconds as a whole number, got "
            f"{value!r} ({type(value).__name__}). Dates are parsed by the "
            f"command line, not here -- if you have '2024-08-01', convert it "
            f"with collector.settings.parse_start first."
        )

    if value < 0:
        raise PriceError(f"{name} must not be negative, got {value}")

    return int(value)


def _check_aligned(value, name, timeframe_ms, timeframe):
    """
    Require a timestamp to be a real candle open time for this timeframe.

    Every timeframe this project supports divides a day evenly and the epoch
    began at midnight UTC, so alignment is a single modulo -- the same property
    timeframes.py relies on when it refuses weeks and months.

    Rounding would be friendlier and is the wrong trade. Rounding down silently
    includes a candle that was not asked for, rounding up silently excludes one,
    and neither leaves a mark in the output. In practice every start comes from a
    date, which is midnight UTC and therefore aligned for every timeframe here,
    so strictness costs nothing and catches a genuine confusion between naming a
    moment and naming a bar.
    """
    if value is not None and value % timeframe_ms != 0:
        raise PriceError(
            f"{name} {_utc(value)} is not a {timeframe} candle open time. "
            f"Ranges name candles, not moments, so the value has to land on a "
            f"boundary -- for {timeframe} that means every {timeframe_ms // 1000} "
            f"seconds from midnight UTC."
        )


def load(
    data_dir,
    exchange,
    symbol,
    timeframe,
    *,
    start_ms=None,
    end_ms=None,
    require_continuous=True,
):
    """
    Load one symbol's candles, refusing anything an indicator should not see.

    Returns a DataFrame with the store's six columns, sorted oldest first, with
    the index reset to 0..n-1. `timestamp` stays int64: converting it here would
    be convenient and would produce a naive datetime, which pandas treats as
    local time, so a store that is right on this machine would be an hour out in
    Oslo in December with nothing raising anywhere.

    `start_ms` and `end_ms` are both inclusive and both name a candle by its open
    time. Inclusive because the argument names a bar rather than a moment, so
    start == end asking for one candle should return one candle rather than none.
    Either may be omitted, in which case the stored extreme is used and no
    coverage check applies at that end.

    `require_continuous` defaults to True and has to be typed out to turn off,
    so accepting a gap is always a visible decision in the code rather than
    something that happened.

    Raises PriceError for anything about the data or the range. An unsupported
    timeframe or a malformed symbol raises TimeframeError or StoreError from the
    layer below, unwrapped, because those modules already explain themselves at
    length and rephrasing would only lose the explanation.
    """
    start_ms = _check_ms(start_ms, "start_ms")
    end_ms = _check_ms(end_ms, "end_ms")

    # Raises TimeframeError before anything touches the disk, so an unsupported
    # timeframe is not reported as a missing file.
    timeframe_ms = timeframe_to_ms(timeframe)
    _check_aligned(start_ms, "start_ms", timeframe_ms, timeframe)
    _check_aligned(end_ms, "end_ms", timeframe_ms, timeframe)

    if start_ms is not None and end_ms is not None and start_ms > end_ms:
        raise PriceError(
            f"start_ms {_utc(start_ms)} is after end_ms {_utc(end_ms)}"
        )

    # Checked here even though settings.py checks it too, because the failure
    # this prevents is specific to reading. 'BTCUSDT' is a legal directory name,
    # so the store builds a path for it happily and the file is simply missing,
    # and the honest-looking reply is "no candles, run `uv run collect`" -- which
    # cannot work. The collector writes 'BTC_USDT'; no number of re-runs will
    # ever produce the directory being looked in. Better to name the typo than
    # to send someone off to repeat a backfill that cannot fix it.
    if not isinstance(symbol, str) or "/" not in symbol:
        raise PriceError(
            f"symbol {symbol!r} is not in ccxt's BASE/QUOTE form. Write "
            f"'BTC/USDT' rather than 'BTC' or 'BTCUSDT' -- the store is laid "
            f"out by the slashed name, so anything else looks for a directory "
            f"that `collect` never creates."
        )

    # Raises StoreError for a symbol that cannot become a safe directory name.
    path = store.candle_path(data_dir, exchange, symbol, timeframe)
    frame = store.read_candles(path)

    if frame.empty:
        raise PriceError(
            f"no candles for {symbol} {timeframe} on {exchange}: {path} "
            f"{'does not exist' if not path.exists() else 'is empty'}. "
            f"Run `uv run collect` first, or check the exchange, symbol and "
            f"timeframe against config/symbols.json."
        )

    stored_first = int(frame["timestamp"].iloc[0])
    stored_last = int(frame["timestamp"].iloc[-1])

    if start_ms is not None and start_ms < stored_first:
        raise PriceError(
            f"{symbol} {timeframe} begins at {_utc(stored_first)}, but "
            f"start_ms asks for {_utc(start_ms)}. Returning the shorter range "
            f"would answer a different question than the one asked without "
            f"saying so. Either move the start, or backfill further with "
            f"`uv run collect --start`."
        )

    if end_ms is not None and end_ms > stored_last:
        raise PriceError(
            f"{symbol} {timeframe} ends at {_utc(stored_last)}, but end_ms "
            f"asks for {_utc(end_ms)}. Run `uv run collect` to bring the store "
            f"up to date, or move the end back."
        )

    first = stored_first if start_ms is None else start_ms
    last = stored_last if end_ms is None else end_ms

    selected = frame[
        frame["timestamp"].between(first, last, inclusive="both")
    ].reset_index(drop=True)

    timestamps = [int(value) for value in selected["timestamp"]]

    # Checked before continuity and regardless of require_continuous, because a
    # duplicate is not a continuity problem -- gap detection compares sets, and
    # two copies of 10:00 look identical to one. An indicator would weight that
    # bar twice and a backtest could enter the same trade twice. The store's
    # writer deduplicates, so a duplicate here means the file came from
    # somewhere else, which is worth stopping for rather than working around.
    if len(set(timestamps)) != len(timestamps):
        repeated = sorted({ts for ts in timestamps if timestamps.count(ts) > 1})
        shown = ", ".join(_utc(ts) for ts in repeated[:MAX_GAPS_SHOWN])
        raise PriceError(
            f"{symbol} {timeframe} has {len(repeated)} timestamp(s) stored "
            f"twice, at {shown}. This is a duplicate, not a gap, so it survives "
            f"require_continuous=False; delete the file and refetch it."
        )

    if require_continuous:
        found = gaps.find_gaps(first, last, timeframe_ms, timestamps)
        if found:
            missing = gaps.total_missing(found)
            lines = [
                f"  {_utc(gap.start_ms)} to {_utc(gap.end_ms)} "
                f"({gap.count} candle{'s' if gap.count != 1 else ''})"
                for gap in found[:MAX_GAPS_SHOWN]
            ]
            if len(found) > MAX_GAPS_SHOWN:
                lines.append(f"  ... and {len(found) - MAX_GAPS_SHOWN} more")
            detail = "\n".join(lines)
            raise PriceError(
                f"{symbol} {timeframe} is missing {missing} candle(s) in "
                f"{len(found)} gap(s) between {_utc(first)} and {_utc(last)}:\n"
                f"{detail}\n"
                f"Averaging across a hole produces a value spanning a "
                f"discontinuity that looks exactly like an ordinary one. Try "
                f"`uv run collect` to fill it; if the exchange simply does not "
                f"have those candles, narrow the range or pass "
                f"require_continuous=False knowing what it means."
            )

    return selected
