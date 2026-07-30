"""The backfill loop: fetch what is missing, store it, report what is still not there.

This is the module that joins exchange.py to store.py, and the only one that
holds both at once.

The central idea is that a backfill run is defined by the range you *asked for*,
not by what happens to be in the file already. Work out which candles in that
range are absent, then fetch each absent stretch. A first run, a resume an hour
later, and a repair of a hole left by an outage six months ago are then the same
operation with different inputs, rather than three code paths that can disagree
about the state of the store.

An earlier design resumed only forwards, from the newest stored candle. That was
subtly wrong in a way worth recording: asking for a start date earlier than the
file already held did nothing whatsoever -- no fetch, no error, no warning -- and
the missing range escaped gap detection too, because that only looked *within*
the stored range. Planning against the requested range instead removes the whole
category of problem: earlier data is simply another gap.

Termination is guaranteed rather than hoped for. Each iteration sets the next
`since` to one timeframe past the newest candle actually returned, and
verify_pagination_advanced raises if a page fails to reach further than the one
before it. So `since` strictly increases on every iteration, and the loop cannot
run forever even against a badly behaved venue.
"""

import logging
from typing import NamedTuple, Optional

from collector import gaps, store
from collector.exchange import (  # re-exported so callers need one import
    ExchangeError,
    HistoryNotAvailable,
    PaginationStalled,
    fetch_page,
    verify_history_reachable,
    verify_pagination_advanced,
)
from collector.timeframes import (
    drop_incomplete_candles,
    last_closed_open_time,
    timeframe_to_ms,
    to_utc_string,
)

log = logging.getLogger(__name__)

# How many separate holes to try repairing in a single run.
#
# Unbounded repair is a trap. A genuine multi-day exchange outage leaves candles
# that will never exist, and without a cap every future run would re-request all
# of them forever -- each one a round trip that cannot succeed. Capping keeps the
# cost of an old scar bounded, and keeps a real new problem visible in the log
# instead of buried among hundreds of hopeless retries.
DEFAULT_MAX_GAPS = 20

# Pages to accumulate before writing to disk. Writing every page would rewrite
# the whole Parquet file per request; never writing until the end would lose the
# entire run if it were killed. Five is a compromise: at binance's 500 candles a
# page that is 2,500 candles of work at risk, a few seconds' worth.
DEFAULT_FLUSH_PAGES = 5

__all__ = [
    "DEFAULT_FLUSH_PAGES",
    "DEFAULT_MAX_GAPS",
    "ExchangeError",
    "HistoryNotAvailable",
    "PaginationStalled",
    "Plan",
    "Result",
    "backfill_symbol",
    "fetch_range",
    "gap_count",
    "plan_backfill",
]


class Plan(NamedTuple):
    """What a run intends to do, decided before any request is made.

    trailing
        The missing stretch that runs up to the end of the requested range, if
        any. This is the one that brings the store up to date, so it is always
        attempted and never subject to the cap.
    gaps
        Interior holes to attempt, newest first. Repairs rather than progress.
    skipped
        Interior holes left alone because the cap was reached, chronologically
        ordered because the only thing that reads them is a person.
    """

    trailing: Optional[gaps.Gap]
    gaps: list
    skipped: list


class Result(NamedTuple):
    """What a run actually did, for logging and for the gap report."""

    symbol: str
    added: int
    pages: int
    filled: list
    unfilled: list
    skipped: list


def plan_backfill(stored, start_ms, end_ms, timeframe_ms, max_gaps=DEFAULT_MAX_GAPS):
    """
    Decide which ranges need fetching, without fetching anything.

    Kept separate from the fetching so that the decision can be tested against a
    set of integers, and so that a dry run is possible later by simply not acting
    on the plan.

    The trailing range is singled out because it is qualitatively different: it
    is the run's actual job, whereas the interior gaps are opportunistic repairs.
    That distinction is what lets the cap apply to repairs without ever throttling
    progress.
    """
    found = gaps.find_gaps(start_ms, end_ms, timeframe_ms, stored)
    if not found:
        return Plan(None, [], [])

    trailing = None
    interior = list(found)

    # The trailing range is the one touching the end of what was requested. Only
    # the last gap can qualify, since gaps come back chronologically.
    if interior[-1].end_ms >= end_ms - timeframe_ms + 1:
        trailing = interior.pop()

    # Newest first: an old hole is more likely to be permanent, and recent
    # candles are what a backtest is most likely to be reading.
    by_recency = sorted(interior, key=lambda gap: gap.start_ms, reverse=True)
    attempt = by_recency[:max_gaps]
    skipped = sorted(by_recency[max_gaps:], key=lambda gap: gap.start_ms)

    return Plan(trailing, attempt, skipped)


def gap_count(plan):
    """Total interior gaps in a plan, attempted and skipped together.

    Useful in log lines, where "repairing 2 of 6 holes" is more honest than
    "repairing 2 holes".
    """
    return len(plan.gaps) + len(plan.skipped)


def fetch_range(
    exchange,
    path,
    symbol,
    timeframe,
    start_ms,
    end_ms,
    now_ms,
    *,
    page_limit=None,
    flush_every=DEFAULT_FLUSH_PAGES,
    check_history=False,
    fetch_options=None,
):
    """
    Walk one contiguous range, page by page, storing as it goes.

    Returns `(candles_added, pages_fetched)`.

    The next `since` is always derived from the newest timestamp actually
    returned, never from an assumed page size. Venues differ enormously -- the
    probe measured 500 candles a page on binance, 199 on bybit, 100 on okx -- and
    any of them may return fewer than usual. Advancing by an assumed size would
    skip candles on one venue and re-request them on another.

    Candles outside [start_ms, end_ms] are stored rather than discarded when they
    turn up. A page covering a two-candle hole will mostly contain data already
    held; append_candles ignores what it already has, so filtering first would
    only throw away history that arrived free.

    `check_history` should be true for the range that brings the store up to date
    from a cold start, where `since` is doing real work. It is what catches an
    exchange whose candle endpoint silently serves only a recent window.
    """
    timeframe_ms = timeframe_to_ms(timeframe)
    options = dict(fetch_options or {})

    since = start_ms
    previous_last = None
    pending = []
    pages = 0
    added = 0

    log.info(
        "%s %s: fetching %s to %s",
        symbol,
        timeframe,
        to_utc_string(start_ms),
        to_utc_string(end_ms),
    )

    while since <= end_ms:
        page = fetch_page(exchange, symbol, timeframe, since, limit=page_limit, **options)

        if not page:
            # The venue has nothing further. Normal end of a range.
            break

        if pages == 0 and check_history:
            verify_history_reachable(page, since, timeframe_ms, now_ms)

        verify_pagination_advanced(previous_last, page)
        previous_last = page[-1][0]
        pages += 1

        pending.extend(drop_incomplete_candles(page, timeframe_ms, now_ms))

        if flush_every and pages % flush_every == 0:
            added += store.append_candles(path, pending)
            pending = []

        since = previous_last + timeframe_ms

    if pending:
        added += store.append_candles(path, pending)

    log.info(
        "%s %s: %d candle(s) added from %d page(s)", symbol, timeframe, added, pages
    )
    return added, pages


def backfill_symbol(
    exchange,
    path,
    symbol,
    timeframe,
    start_ms,
    now_ms,
    *,
    max_gaps=DEFAULT_MAX_GAPS,
    page_limit=None,
    flush_every=DEFAULT_FLUSH_PAGES,
    fetch_options=None,
):
    """
    Bring one symbol's stored candles up to date over [start_ms, now].

    Returns a Result describing what happened, including any candles that are
    still missing after the exchange was asked for them.

    The end of the range is the last *closed* candle, so the one currently being
    built is never requested and never stored. Its high, low, close and volume are
    provisional; writing them would put a number in the file that later becomes
    wrong, which is worse than not having it.

    Gaps that remain after being asked for are reported rather than retried
    indefinitely. If the venue returned the candles either side of a hole but not
    the hole itself, the hole is real -- a halted market, or an hour in which
    nothing traded -- and no number of further requests will change that.
    """
    timeframe_ms = timeframe_to_ms(timeframe)
    end_ms = last_closed_open_time(now_ms, timeframe_ms)

    if end_ms < start_ms:
        log.info(
            "%s %s: nothing to do -- no candle has closed since %s",
            symbol,
            timeframe,
            to_utc_string(start_ms),
        )
        return Result(symbol, 0, 0, [], [], [])

    stored = store.stored_timestamps(path)
    plan = plan_backfill(stored, start_ms, end_ms, timeframe_ms, max_gaps=max_gaps)

    if plan.trailing is None and not plan.gaps and not plan.skipped:
        log.info(
            "%s %s: already complete from %s to %s",
            symbol,
            timeframe,
            to_utc_string(start_ms),
            to_utc_string(end_ms),
        )
        return Result(symbol, 0, 0, [], [], [])

    added = 0
    pages = 0

    if plan.trailing is not None:
        gained, walked = fetch_range(
            exchange,
            path,
            symbol,
            timeframe,
            plan.trailing.start_ms,
            plan.trailing.end_ms,
            now_ms,
            page_limit=page_limit,
            flush_every=flush_every,
            check_history=True,
            fetch_options=fetch_options,
        )
        added += gained
        pages += walked

    if plan.gaps:
        log.info(
            "%s %s: attempting %d of %d hole(s)",
            symbol,
            timeframe,
            len(plan.gaps),
            gap_count(plan),
        )

    for gap in plan.gaps:
        log.info("%s %s: repairing %s", symbol, timeframe, gap.describe())
        gained, walked = fetch_range(
            exchange,
            path,
            symbol,
            timeframe,
            gap.start_ms,
            gap.end_ms,
            now_ms,
            page_limit=page_limit,
            flush_every=flush_every,
            check_history=False,
            fetch_options=fetch_options,
        )
        added += gained
        pages += walked

    # Re-derive what is missing from the file itself rather than trusting the
    # plan. After a crash-safe write the file is the only authority on what is
    # stored, and a count kept alongside it could disagree.
    remaining = gaps.find_gaps(
        start_ms, end_ms, timeframe_ms, store.stored_timestamps(path)
    )
    still_missing = set(remaining)
    skipped = set(plan.skipped)

    # Everything still absent that the exchange was actually asked for.
    #
    # Deriving this from the file rather than from plan.gaps matters more than it
    # looks. An earlier version listed only planned interior gaps, which meant
    # holes inside the trailing range went unreported entirely -- and on a first
    # run the trailing range is the whole request, so a two-year backfill with
    # missing candles would have reported nothing at all. Exactly the silent
    # failure requirement 6 exists to prevent.
    #
    # Skipped gaps are excluded because they are reported separately: "we asked
    # and the venue had nothing" is a different fact from "we did not ask".
    unfilled = [gap for gap in remaining if gap not in skipped]
    filled = [gap for gap in plan.gaps if gap not in still_missing]

    for gap in unfilled:
        log.warning(
            "%s %s: still missing after refetch -- %s. The exchange does not "
            "appear to have these candles.",
            symbol,
            timeframe,
            gap.describe(),
        )

    for gap in plan.skipped:
        log.info(
            "%s %s: known hole, not attempted this run (gap cap %d reached) -- %s",
            symbol,
            timeframe,
            max_gaps,
            gap.describe(),
        )

    return Result(symbol, added, pages, filled, unfilled, plan.skipped)
