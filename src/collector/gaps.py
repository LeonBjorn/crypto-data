"""Working out which candles are missing from a stored range.

Every function here is pure: integers in, integers out. No exchange, no files,
no clock. That separation is the main idea of this module -- deciding *what* is
missing has nothing to do with fetching it, and keeping the decision separate
means it can be tested exhaustively without a network or a filesystem.

The word "gap" covers more than it sounds like it should. On a first run the
entire requested range is one large gap. On a re-run an hour later, the gap is
the one or two candles that have closed since. A hole left in the middle by an
outage six months ago is also a gap. Deliberately treating all three the same
way is what lets a single code path serve a fresh backfill, a resume, and a
repair -- rather than three paths that can disagree with each other about what
the store contains.
"""

from typing import NamedTuple

from collector.timeframes import expected_open_times, to_utc_string


class Gap(NamedTuple):
    """A contiguous run of missing candles, inclusive at both ends.

    `end_ms` is a candle that is *missing*, not the first one present after the
    gap. Fetches use it as a stopping condition, so an exclusive end would leave
    the final candle of every gap unfetched -- and therefore missing again on the
    next run, and the one after that, forever.

    A NamedTuple rather than a plain tuple so that reading `gap.start_ms` beats
    remembering what `gap[0]` meant, and rather than a full class because two
    gaps covering the same candles should simply compare equal.
    """

    start_ms: int
    end_ms: int
    count: int

    def describe(self):
        """A one-line summary for logs and reports.

        Unattended runs are read afterwards through their log, so a gap needs to
        be able to say what it is without the reader reconstructing it from
        epoch milliseconds.
        """
        noun = "candle" if self.count == 1 else "candles"
        if self.start_ms == self.end_ms:
            return f"{to_utc_string(self.start_ms)} (1 {noun})"
        return (
            f"{to_utc_string(self.start_ms)} to {to_utc_string(self.end_ms)} "
            f"({self.count} {noun})"
        )


def missing_timestamps(start_ms, end_ms, timeframe_ms, stored):
    """
    Return the candle open times in [start_ms, end_ms] that `stored` lacks.

    `stored` may be any iterable of timestamps -- typically the list that
    store.stored_timestamps hands back. It is converted to a set once, because
    membership testing per candle against a list would turn two years of hourly
    data into tens of millions of comparisons for no benefit.

    Timestamps held outside the requested range are ignored rather than counted
    either way. The question this answers is only ever "is the range I asked
    about complete", so a store containing extra history neither helps nor hurts.

    An end before the start returns an empty list rather than raising: asking for
    a range whose first candle has not closed yet is a normal thing to do, and
    the correct answer is that there is nothing to fetch.
    """
    stored_set = set(stored)
    expected = expected_open_times(start_ms, end_ms, timeframe_ms)
    return [ts for ts in expected if ts not in stored_set]


def group_into_gaps(timestamps, timeframe_ms):
    """
    Collapse individual missing timestamps into contiguous runs.

    This is what turns four missing consecutive candles into one request instead
    of four. Adjacency is defined by the timeframe -- two daily candles a day
    apart are neighbours, two hourly candles a day apart are not -- so the same
    function works for every timeframe without special cases.

    Input may arrive in any order and may repeat; sets have no order, and a
    caller should not have to think about it. Output is always chronological.
    """
    ordered = sorted(set(timestamps))
    if not ordered:
        return []

    found = []
    run_start = ordered[0]
    run_end = ordered[0]
    count = 1

    for ts in ordered[1:]:
        if ts == run_end + timeframe_ms:
            # Still inside the same run.
            run_end = ts
            count += 1
        else:
            found.append(Gap(run_start, run_end, count))
            run_start = ts
            run_end = ts
            count = 1

    found.append(Gap(run_start, run_end, count))
    return found


def find_gaps(start_ms, end_ms, timeframe_ms, stored):
    """
    Convenience wrapper: locate missing candles and group them in one step.

    Almost every caller wants both halves, and doing them separately at each call
    site is an opportunity to pass mismatched arguments to the two.
    """
    return group_into_gaps(
        missing_timestamps(start_ms, end_ms, timeframe_ms, stored), timeframe_ms
    )


def total_missing(found):
    """Total candles across a list of gaps.

    Worth a named function because `sum(gap.count for gap in found)` appears in
    log lines, and the count of *gaps* and the count of *candles* are easy to
    confuse when both are just numbers in a message.
    """
    return sum(gap.count for gap in found)
