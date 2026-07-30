"""Exchange access: the one place this project talks to a trading venue.

This is the seam. Everything above it deals in plain OHLCV rows, so pointing the
collector at a different venue is a change of one string, not a change of design.

Three responsibilities:

1. Build a public, keyless ccxt client (requirement 1: no API key, ever).
2. Fetch one page of candles, retrying transient failures with exponential
   backoff (requirement 7).
3. Verify that the exchange behaved as a paginated backfill requires -- that it
   honoured `since`, and that successive pages actually progress.

The two guards in part 3 exist because both failure modes are silent. An
exchange that caps its candle endpoint to a recent window will happily answer a
request for two years of history with the last few weeks, and a backfill built on
that looks entirely successful while producing a store that poisons every
backtest reading it. Detecting that at runtime is better than relying on
knowledge of any particular venue's current limits, which go out of date.
"""

import logging
import random
import time

import ccxt

from collector.timeframes import to_utc_string

log = logging.getLogger(__name__)

# Retry policy. Every transient ccxt failure descends from NetworkError --
# RequestTimeout, ExchangeNotAvailable, DDoSProtection, RateLimitExceeded and
# OnMaintenance all included -- while everything meaning "your request was
# wrong" descends from ExchangeError. That inheritance split is the entire rule,
# so there is no hand-maintained list of retryable types to fall out of date.
TRANSIENT_ERRORS = (ccxt.NetworkError,)
PERMANENT_ERRORS = (ccxt.ExchangeError,)

DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 60.0
DEFAULT_TIMEOUT_MS = 30_000

DAY_MS = 24 * 60 * 60 * 1000


class ExchangeError(Exception):
    """Any failure while talking to an exchange, after retries are exhausted."""


class HistoryNotAvailable(ExchangeError):
    """The exchange did not serve history as far back as was requested."""


class PaginationStalled(ExchangeError):
    """Successive pages stopped making progress, so the loop would never end."""


def build_exchange(exchange_id, timeout_ms=DEFAULT_TIMEOUT_MS):
    """
    Construct a public ccxt client for `exchange_id`.

    No credentials are passed, ever. This project reads public market data only,
    so a key is not merely unnecessary but a liability: a script that cannot
    authenticate cannot place an order, however badly it misbehaves.

    `enableRateLimit` lets ccxt space out requests according to each venue's
    published limits, which is the first half of requirement 7. The backoff in
    fetch_page is the second half, covering what the limiter cannot predict.

    Note that no market data is loaded here. Constructing a client makes no
    network call, so this stays cheap and testable; an unknown symbol surfaces on
    the first fetch instead.
    """
    if exchange_id not in ccxt.exchanges:
        raise ExchangeError(
            f"unknown exchange id {exchange_id!r}. "
            f"Expected a ccxt id such as 'kraken', 'binance', 'bybit' or 'okx'."
        )

    client_class = getattr(ccxt, exchange_id)
    return client_class(
        {
            "enableRateLimit": True,
            "timeout": timeout_ms,
        }
    )


def fetch_page(
    exchange,
    symbol,
    timeframe,
    since_ms,
    limit=None,
    *,
    max_attempts=DEFAULT_MAX_ATTEMPTS,
    base_delay=DEFAULT_BASE_DELAY,
    max_delay=DEFAULT_MAX_DELAY,
    sleep=None,
    rng=None,
):
    """
    Fetch one page of candles, retrying transient failures.

    Returns the raw ccxt rows, `[timestamp, open, high, low, close, volume]`, or
    an empty list if the exchange has nothing further. An empty page is a normal
    end condition, not an error.

    Delays grow geometrically -- 1s, 2s, 4s, 8s -- so that a struggling exchange
    is given room to recover rather than being hit at a fixed interval. Each
    delay is then jittered into the upper half of its interval, which stops five
    symbols that failed together from retrying in perfect lockstep and
    reproducing the same overload.

    `sleep` and `rng` are injected so tests can cover four backoff steps
    instantly and deterministically instead of actually waiting fifteen seconds.
    In production they are time.sleep and random.random.

    They default to None and are resolved here rather than in the signature,
    which looks like a pointless detour but is not. `sleep=time.sleep` as a
    default would capture the real function object at import time, so anything
    trying to intercept sleeping from outside -- the test suite's guard against
    accidentally waiting on real backoff -- would silently fail to see it. A
    dependency frozen at import is not injectable, only overridable.

    Permanent errors are not retried. A misspelled symbol will still be
    misspelled after five attempts, so waiting fifteen seconds to confirm it
    wastes time and buries the real message.
    """
    if sleep is None:
        sleep = time.sleep
    if rng is None:
        rng = random.random

    last_error = None

    for attempt in range(max_attempts):
        try:
            candles = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since_ms, limit=limit
            )
            if attempt:
                log.info("%s %s: succeeded on attempt %d", symbol, timeframe, attempt + 1)
            return candles

        except PERMANENT_ERRORS as error:
            # Deliberately checked before the transient case. ccxt's two families
            # are siblings rather than nested, but ordering the handlers makes
            # the intent unmistakable to anyone reading this later.
            raise ExchangeError(
                f"{symbol} {timeframe}: {type(error).__name__} from "
                f"{getattr(exchange, 'id', 'exchange')} -- not retried, because "
                f"this means the request itself was rejected: {error}"
            ) from error

        except TRANSIENT_ERRORS as error:
            last_error = error
            attempts_left = max_attempts - attempt - 1

            if not attempts_left:
                break

            delay = min(base_delay * (2**attempt), max_delay)
            # Jitter into the upper half of the interval: still backing off
            # meaningfully, but no longer synchronised with other symbols.
            jittered = delay * (0.5 + 0.5 * rng())

            log.warning(
                "%s %s: %s (%s), retrying in %.1fs (%d attempt(s) left)",
                symbol,
                timeframe,
                type(error).__name__,
                error,
                jittered,
                attempts_left,
            )
            sleep(jittered)

    raise ExchangeError(
        f"{symbol} {timeframe}: giving up after {max_attempts} attempts. "
        f"Last error was {type(last_error).__name__}: {last_error}"
    ) from last_error


def verify_history_reachable(
    candles, requested_since_ms, timeframe_ms, now_ms, tolerance_ms=None
):
    """
    Check that the exchange served history from roughly where it was asked to.

    Raises HistoryNotAvailable when the returned page looks like a fixed recent
    window rather than an answer to the request. Two conditions must both hold:

    - The oldest candle is meaningfully later than `requested_since_ms`.
    - The page runs up to the present.

    Requiring both is what separates a capped endpoint from a market that was
    simply listed after the requested start date. A late-listed market returns
    its own earliest candles, and that page ends long before now because it is
    page one of many. A capped endpoint returns a window pinned to the live edge.

    That distinction is not perfectly sharp: a market listed only hours ago would
    trip this. The error message therefore names both possible causes rather than
    asserting one, because being told the two candidates is more useful than
    being confidently told the wrong one.
    """
    if not candles:
        return

    if tolerance_ms is None:
        # A day of slack, or two candles for coarse timeframes. Exchanges
        # commonly begin a market a little after any given instant.
        tolerance_ms = max(DAY_MS, 2 * timeframe_ms)

    oldest = candles[0][0]
    newest = candles[-1][0]

    started_late = oldest > requested_since_ms + tolerance_ms
    reaches_present = newest >= now_ms - (2 * timeframe_ms)

    if started_late and reaches_present:
        span_days = round((newest - oldest) / DAY_MS, 1)
        raise HistoryNotAvailable(
            f"asked for candles from {_readable(requested_since_ms)} but the oldest "
            f"returned was {_readable(oldest)}, and the page runs to "
            f"{_readable(newest)} -- a {span_days}-day window ending at the present.\n"
            f"Either this exchange's candle endpoint only serves a recent window "
            f"and ignores an older `since`, or this market did not exist before "
            f"{_readable(oldest)}.\n"
            f"If it is the former, a deep backfill is not possible here: use an "
            f"exchange that serves full history, or move the start date forward."
        )


def verify_pagination_advanced(previous_last_ms, candles):
    """
    Check that a page reaches further than the one before it.

    Raises PaginationStalled if it does not. Without this, an exchange that
    returns the same page for every `since` would leave the backfill loop
    requesting forever, adding nothing, and looking busy while doing so.

    An empty page is not a stall: that is how a backfill finishes. Overlap is
    also fine -- some venues repeat the boundary candle -- provided the page
    extends beyond where the last one ended.
    """
    if not candles or previous_last_ms is None:
        return

    newest = candles[-1][0]
    if newest <= previous_last_ms:
        raise PaginationStalled(
            f"pagination stopped progressing: the previous page ended at "
            f"{_readable(previous_last_ms)} and this one ends at "
            f"{_readable(newest)}, so the loop would repeat forever.\n"
            f"This usually means the exchange is ignoring the `since` parameter."
        )


def _readable(ms):
    """Render epoch milliseconds as a UTC string, for error messages only.

    Delegates to timeframes.to_utc_string so that this project has exactly one
    timestamp format. Kept as a local name because it reads better inside the
    long f-strings above.
    """
    return to_utc_string(ms)
