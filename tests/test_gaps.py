"""Tests for working out which candles are missing from a stored range.

Pure functions over integers -- no exchange, no files, no clock. That is the
point of separating this from backfill.py: deciding *what* is missing is
independent of fetching it, and the independent half is the half that can be
tested exhaustively and cheaply.
"""

import datetime as dt

from collector import gaps

HOUR = 3_600_000


def utc(year, month, day, hour=0):
    return int(
        dt.datetime(year, month, day, hour, tzinfo=dt.timezone.utc).timestamp() * 1000
    )


class TestMissingTimestamps:
    def test_empty_store_means_the_whole_range_is_missing(self):
        """The first-run case. Nothing stored, so every expected candle is a
        candle to fetch."""
        start = utc(2025, 1, 1)
        end = utc(2025, 1, 1, 5)

        assert gaps.missing_timestamps(start, end, HOUR, set()) == [
            start + n * HOUR for n in range(6)
        ]

    def test_complete_store_means_nothing_is_missing(self):
        """The idempotency case, expressed at the level where it is decided. If
        this returns anything, a re-run would fetch something."""
        start = utc(2025, 1, 1)
        end = utc(2025, 1, 1, 5)
        stored = {start + n * HOUR for n in range(6)}

        assert gaps.missing_timestamps(start, end, HOUR, stored) == []

    def test_finds_an_interior_hole(self):
        start = utc(2025, 1, 1)
        end = utc(2025, 1, 1, 5)
        stored = {start + n * HOUR for n in range(6)} - {
            start + 2 * HOUR,
            start + 3 * HOUR,
        }

        assert gaps.missing_timestamps(start, end, HOUR, stored) == [
            start + 2 * HOUR,
            start + 3 * HOUR,
        ]

    def test_finds_a_missing_tail(self):
        """The resume case: the store is complete up to a point and stops."""
        start = utc(2025, 1, 1)
        end = utc(2025, 1, 1, 5)
        stored = {start + n * HOUR for n in range(4)}

        assert gaps.missing_timestamps(start, end, HOUR, stored) == [
            start + 4 * HOUR,
            start + 5 * HOUR,
        ]

    def test_timestamps_stored_outside_the_range_are_irrelevant(self):
        """A store may hold more than was asked about. The question is only ever
        whether the *requested* range is complete, so data either side of it
        neither counts towards completeness nor against it."""
        start = utc(2025, 1, 1)
        end = utc(2025, 1, 1, 2)
        stored = {start + n * HOUR for n in range(3)} | {
            utc(2024, 1, 1),
            utc(2026, 1, 1),
        }

        assert gaps.missing_timestamps(start, end, HOUR, stored) == []

    def test_range_boundaries_snap_to_candle_opens(self):
        """Inherited from expected_open_times: a start part-way through a candle
        rounds up, so we never treat a candle we are inside of as missing."""
        start = utc(2025, 1, 1) + 30 * 60 * 1000  # 00:30
        end = utc(2025, 1, 1, 3)

        assert gaps.missing_timestamps(start, end, HOUR, set()) == [
            utc(2025, 1, 1, 1),
            utc(2025, 1, 1, 2),
            utc(2025, 1, 1, 3),
        ]

    def test_a_list_of_stored_timestamps_works_as_well_as_a_set(self):
        """store.stored_timestamps returns a list, so a list has to be accepted.
        The implementation converts it to a set once rather than scanning it per
        candle -- two years of hourly candles against a list would be tens of
        millions of comparisons for no reason."""
        start = utc(2025, 1, 1)
        end = utc(2025, 1, 1, 2)

        assert gaps.missing_timestamps(start, end, HOUR, [start, start + HOUR]) == [end]

    def test_end_before_start_is_empty_rather_than_an_error(self):
        """Happens legitimately: ask for a range whose first candle has not
        closed yet and there is simply nothing to do."""
        assert gaps.missing_timestamps(utc(2025, 1, 2), utc(2025, 1, 1), HOUR, set()) == []


class TestGroupIntoGaps:
    def test_no_missing_timestamps_means_no_gaps(self):
        assert gaps.group_into_gaps([], HOUR) == []

    def test_a_single_missing_candle_is_a_gap_of_one(self):
        ts = utc(2025, 1, 1)
        assert gaps.group_into_gaps([ts], HOUR) == [gaps.Gap(ts, ts, 1)]

    def test_consecutive_timestamps_form_a_single_gap(self):
        """The reason grouping exists at all: one fetch can cover a whole run,
        so four missing candles should mean one request, not four."""
        first = utc(2025, 1, 1)
        run = [first + n * HOUR for n in range(4)]

        assert gaps.group_into_gaps(run, HOUR) == [gaps.Gap(first, first + 3 * HOUR, 4)]

    def test_a_present_candle_splits_one_run_into_two_gaps(self):
        first = utc(2025, 1, 1)
        # missing 00:00, 01:00 -- present 02:00 -- missing 03:00, 04:00
        missing = [first, first + HOUR, first + 3 * HOUR, first + 4 * HOUR]

        assert gaps.group_into_gaps(missing, HOUR) == [
            gaps.Gap(first, first + HOUR, 2),
            gaps.Gap(first + 3 * HOUR, first + 4 * HOUR, 2),
        ]

    def test_unsorted_input_is_grouped_correctly(self):
        """Callers should not have to care about order, and a set has none."""
        first = utc(2025, 1, 1)
        shuffled = [first + 2 * HOUR, first, first + HOUR]

        assert gaps.group_into_gaps(shuffled, HOUR) == [
            gaps.Gap(first, first + 2 * HOUR, 3)
        ]

    def test_duplicates_do_not_inflate_the_count(self):
        ts = utc(2025, 1, 1)
        assert gaps.group_into_gaps([ts, ts, ts], HOUR) == [gaps.Gap(ts, ts, 1)]

    def test_the_end_of_a_gap_is_a_missing_candle_not_the_next_present_one(self):
        """end_ms is inclusive, and that matters. Fetches use it as a stopping
        condition, so an exclusive end would leave the last candle of every gap
        unfetched -- and therefore missing again next run, forever."""
        first = utc(2025, 1, 1)
        gap = gaps.group_into_gaps([first, first + HOUR], HOUR)[0]

        assert gap.end_ms == first + HOUR
        assert gap.count == 2

    def test_count_is_consistent_with_the_span(self):
        first = utc(2025, 1, 1)
        run = [first + n * HOUR for n in range(10)]

        gap = gaps.group_into_gaps(run, HOUR)[0]

        assert gap.count == (gap.end_ms - gap.start_ms) // HOUR + 1

    def test_gaps_come_back_in_chronological_order(self):
        first = utc(2025, 1, 1)
        missing = [first + 5 * HOUR, first]

        result = gaps.group_into_gaps(missing, HOUR)

        assert [gap.start_ms for gap in result] == [first, first + 5 * HOUR]

    def test_a_daily_timeframe_groups_by_days_not_hours(self):
        """Adjacency is defined by the timeframe, not by any fixed interval."""
        first = utc(2025, 1, 1)
        day = 24 * HOUR
        run = [first, first + day, first + 2 * day]

        assert gaps.group_into_gaps(run, day) == [gaps.Gap(first, first + 2 * day, 3)]


class TestFindGaps:
    """The wrapper that does both steps, since nearly every caller wants both."""

    def test_empty_store_gives_one_gap_covering_everything(self):
        """The first-run shape: not many gaps, one big one."""
        start = utc(2025, 1, 1)
        end = utc(2025, 1, 1, 5)

        assert gaps.find_gaps(start, end, HOUR, set()) == [gaps.Gap(start, end, 6)]

    def test_complete_store_gives_no_gaps(self):
        start = utc(2025, 1, 1)
        end = utc(2025, 1, 1, 5)
        stored = {start + n * HOUR for n in range(6)}

        assert gaps.find_gaps(start, end, HOUR, stored) == []

    def test_a_hole_and_a_missing_tail_are_separate_gaps(self):
        """The awkward real case: an old outage plus time having passed since the
        last run. One request each, and the fetching code should not have to know
        which is which."""
        start = utc(2025, 1, 1)
        end = utc(2025, 1, 1, 7)
        stored = {start + n * HOUR for n in (0, 1, 4, 5)}

        assert gaps.find_gaps(start, end, HOUR, stored) == [
            gaps.Gap(start + 2 * HOUR, start + 3 * HOUR, 2),
            gaps.Gap(start + 6 * HOUR, start + 7 * HOUR, 2),
        ]

    def test_agrees_with_calling_the_two_steps_separately(self):
        """Guards against the wrapper drifting from what it wraps."""
        start = utc(2025, 1, 1)
        end = utc(2025, 1, 2)
        stored = {start + n * HOUR for n in (0, 3, 4, 9, 20)}

        assert gaps.find_gaps(start, end, HOUR, stored) == gaps.group_into_gaps(
            gaps.missing_timestamps(start, end, HOUR, stored), HOUR
        )


class TestGapDescribe:
    def test_describes_a_gap_in_utc_with_a_candle_count(self):
        """Log lines are how you find out what happened during an unattended
        run, so a gap has to be able to say what it is."""
        gap = gaps.Gap(utc(2025, 1, 1), utc(2025, 1, 1, 2), 3)

        text = gap.describe()

        assert "2025-01-01 00:00 UTC" in text
        assert "2025-01-01 02:00 UTC" in text
        assert "3" in text

    def test_a_single_candle_gap_reads_as_one_candle(self):
        """The assertion is the exact phrase, not a substring.

        `"candle" in text` was the first version, and it passes just as happily
        against "1 candles" -- a substring check cannot tell singular from plural
        when one word contains the other. Only worth a test at all because these
        strings are the interface a person reads at 2am to find out what an
        unattended run did.
        """
        ts = utc(2025, 1, 1)
        text = gaps.Gap(ts, ts, 1).describe()

        assert text == "2025-01-01 00:00 UTC (1 candle)"

    def test_a_two_candle_gap_reads_as_plural(self):
        first = utc(2025, 1, 1)
        text = gaps.Gap(first, first + HOUR, 2).describe()

        assert text == "2025-01-01 00:00 UTC to 2025-01-01 01:00 UTC (2 candles)"


class TestTotalMissing:
    def test_sums_the_candle_counts(self):
        first = utc(2025, 1, 1)
        found = [gaps.Gap(first, first + HOUR, 2), gaps.Gap(first + 5 * HOUR, first + 5 * HOUR, 1)]

        assert gaps.total_missing(found) == 3

    def test_no_gaps_totals_zero(self):
        assert gaps.total_missing([]) == 0
