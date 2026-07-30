"""Timeframe parsing and candle-boundary arithmetic.

Everything here is a pure function over integers. No I/O, no clock reads, no
exchange calls. Anything needing the current time takes `now_ms` as an argument,
which is what lets the tests pin time to a fixed value instead of patching it.

All timestamps are epoch milliseconds in UTC. There are no datetime objects in
this module on purpose: a naive datetime that silently means local time is the
single easiest way to corrupt a candle store, so the type that could carry that
mistake is simply absent.

Why candle boundaries can be computed with plain modulo arithmetic: the Unix
epoch begins at midnight UTC, so for any timeframe that divides a day evenly,
`ts - (ts % timeframe)` lands exactly on the same boundaries the exchange uses.
That property is why weeks and months are rejected -- see timeframe_to_ms.
"""

import datetime as dt
import re

SECOND_MS = 1000
MINUTE_MS = 60 * SECOND_MS
HOUR_MS = 60 * MINUTE_MS
DAY_MS = 24 * HOUR_MS

_UNIT_MS = {
    "m": MINUTE_MS,
    "h": HOUR_MS,
    "d": DAY_MS,
}

# Case-sensitive by design. In ccxt notation `1m` is one minute and `1M` is one
# month, so a case-insensitive parser could turn a minute into a month.
_TIMEFRAME_PATTERN = re.compile(r"^(\d+)([mhd])$")

# Timeframes this project supports, for error messages.
_EXAMPLES = "1m, 5m, 15m, 30m, 1h, 4h, 12h, 1d"


class TimeframeError(ValueError):
    """Raised for a timeframe string this project cannot handle correctly.

    A subclass of ValueError so that callers who only care that the input was
    bad can catch the broader type, while the CLI can catch this specifically
    and print something friendlier than a traceback.
    """


def timeframe_to_ms(timeframe):
    """
    Convert a timeframe string such as "1h" into milliseconds.

    Accepts a positive integer followed by `m`, `h` or `d`, and additionally
    requires that the result divides a day evenly.

    That last rule rules out three families of input that would otherwise
    produce quietly wrong candles:

    - Weeks (`1w`). The epoch fell on a Thursday, so flooring by seven-day
      multiples yields Thursday-aligned weeks, while exchanges align weekly
      candles to Monday. The arithmetic would succeed and the data would be
      offset by three days.
    - Months and years (`1M`, `1y`). Not fixed durations, so no multiplier
      exists at all.
    - Multi-day and non-dividing sub-daily values (`3d`, `7h`). Boundaries
      counted from the epoch need not coincide with the exchange's own, and for
      `7h` they land at a different wall-clock time every day.

    Refusing these is better than approximating them. A store whose candle
    boundaries disagree with the exchange's is worse than no store, because
    nothing downstream would notice.

    Raises TimeframeError on anything unsupported.
    """
    if not isinstance(timeframe, str):
        raise TimeframeError(
            f"timeframe must be a string like '1h', got {type(timeframe).__name__}"
        )

    match = _TIMEFRAME_PATTERN.match(timeframe)
    if match is None:
        raise TimeframeError(
            f"cannot parse timeframe {timeframe!r}. "
            f"Expected a positive whole number followed by m, h or d "
            f"(supported: {_EXAMPLES}). "
            f"Weeks and months are not supported because their boundaries "
            f"cannot be derived from the epoch reliably."
        )

    count = int(match.group(1))
    unit = match.group(2)

    if count == 0:
        raise TimeframeError(f"timeframe {timeframe!r} has a zero length")

    if unit == "d" and count != 1:
        raise TimeframeError(
            f"timeframe {timeframe!r} is not supported: multi-day candles "
            f"counted from the epoch need not line up with the exchange's own. "
            f"Use 1d, or a sub-daily timeframe."
        )

    total = count * _UNIT_MS[unit]

    if total < DAY_MS and DAY_MS % total != 0:
        raise TimeframeError(
            f"timeframe {timeframe!r} does not divide a day evenly, so its "
            f"boundaries would drift from one day to the next. "
            f"Supported: {_EXAMPLES}."
        )

    return total


def candle_open_time(ts_ms, timeframe_ms):
    """
    Floor a timestamp to the open time of the candle containing it.

    Because the epoch starts at midnight UTC and every supported timeframe
    divides a day, plain modulo arithmetic gives the same boundaries the
    exchange uses.
    """
    if timeframe_ms <= 0:
        raise ValueError(f"timeframe_ms must be positive, got {timeframe_ms}")
    if ts_ms < 0:
        # A pre-1970 timestamp almost always means seconds were passed where
        # milliseconds were expected, or an uninitialised value leaked through.
        # Failing loudly here is much cheaper than tracing it back later.
        raise ValueError(
            f"timestamp must not be negative, got {ts_ms}. "
            f"This usually means seconds were passed instead of milliseconds."
        )
    return ts_ms - (ts_ms % timeframe_ms)


def is_candle_closed(open_ms, timeframe_ms, now_ms):
    """
    Has the candle that opened at `open_ms` finished forming?

    A candle covering [open, open + timeframe) is final the moment `now` reaches
    its end. The comparison is `<=` rather than `<` because at exactly that
    instant the period has fully elapsed and the next candle has opened.

    This is the single rule behind requirement 4, and it is expressed in terms
    of the candle's own close time rather than its position in a list -- a page
    from the exchange may end before the live candle, and positional logic would
    then drop a perfectly good candle or keep a forming one.
    """
    if timeframe_ms <= 0:
        raise ValueError(f"timeframe_ms must be positive, got {timeframe_ms}")
    return (open_ms + timeframe_ms) <= now_ms


def last_closed_open_time(now_ms, timeframe_ms):
    """
    Open time of the newest candle that has finished forming.

    This is the upper bound of what may be written to the store. Note that it is
    one timeframe below the current candle's open time even when `now_ms` sits
    exactly on a boundary: at 09:00:00.000 the 09:00 candle exists but has zero
    elapsed time, so the newest final candle is the one that opened at 08:00.
    """
    current_open = candle_open_time(now_ms, timeframe_ms)
    result = current_open - timeframe_ms
    if result < 0:
        raise ValueError(
            f"no closed candle exists before {now_ms}; "
            f"the timestamp is within one timeframe of the epoch"
        )
    return result


def drop_incomplete_candles(candles, timeframe_ms, now_ms):
    """
    Return only those candles whose period has fully elapsed.

    Requirement 4. The newest candle from an exchange is still being built: its
    high, low, close and volume will all change before it is final. Writing it
    to the store would mean a backtest reads a close price that never actually
    was, and nothing downstream would flag it -- the row looks entirely normal.
    So it is removed here, once, at the boundary between fetching and storing.

    `candles` is a sequence of ccxt OHLCV rows, each `[ts, open, high, low,
    close, volume]`. Only element 0 is examined. A new list is returned so the
    caller keeps the raw page for logging.
    """
    return [
        candle for candle in candles
        if is_candle_closed(candle[0], timeframe_ms, now_ms)
    ]


def expected_open_times(start_ms, end_ms, timeframe_ms):
    """
    Every candle open time that should exist between two timestamps, inclusive.

    Gap detection compares this against what the store actually holds; whatever
    is missing is a hole.

    Unaligned bounds are pulled inward rather than outward: the start rounds up
    to the first real boundary and the end rounds down to the last one. Rounding
    outward would invent candles beyond the range asked about, which would then
    be reported as gaps -- a false alarm produced entirely by the arithmetic.

    Returns a list rather than a generator; a two-year hourly range is about
    17,500 integers, small enough that being able to len() and index it is worth
    more than the memory saved.
    """
    if timeframe_ms <= 0:
        raise ValueError(f"timeframe_ms must be positive, got {timeframe_ms}")
    if start_ms < 0:
        raise ValueError(f"start_ms must not be negative, got {start_ms}")

    remainder = start_ms % timeframe_ms
    first = start_ms if remainder == 0 else start_ms + (timeframe_ms - remainder)
    last = candle_open_time(end_ms, timeframe_ms)

    if first > last:
        return []

    return list(range(first, last + timeframe_ms, timeframe_ms))


def to_utc_string(ms):
    """
    Render epoch milliseconds as a readable UTC string.

    For log lines and error messages only -- never for storage or comparison.
    Stored timestamps stay as integers, and any code that parses this back into a
    number has taken a wrong turn.

    It lives here, rather than being written separately wherever it is needed, so
    that every timestamp this project shows a human is formatted identically and
    every one of them says UTC out loud. A log that mixes formats, or omits the
    zone, is how you end up debugging an off-by-one-hour problem that was never
    there.
    """
    return dt.datetime.fromtimestamp(ms / 1000, tz=dt.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
