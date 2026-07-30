"""Tests for the backfill loop -- the part that ties the exchange to the store.

Three layers are tested separately here, because they fail for different reasons:

plan_backfill
    Pure decision-making. Given what is stored and what was asked for, which
    ranges need fetching and in what order. No exchange, no files.
fetch_range
    One range, walked page by page. This is where pagination, flushing, and the
    two exchange guards live.
backfill_symbol
    The whole job for one symbol, which is what the CLI will call.

Requirement 3 -- resumable and idempotent, killable mid-run -- is really a claim
about this file, so several tests below exist purely to hold it up.
"""

import datetime as dt
import logging

import pytest

from collector import backfill, gaps, store

HOUR = 3_600_000
DAY = 24 * HOUR


def utc(year, month, day, hour=0):
    return int(
        dt.datetime(year, month, day, hour, tzinfo=dt.timezone.utc).timestamp() * 1000
    )


@pytest.fixture
def path(tmp_path):
    return store.candle_path(tmp_path / "data", "binance", "BTC/USDT", "1h")


def alternating_store(start, hours=12):
    """Even hours present, odd hours missing, and the final hour present too.

    Keeping the final hour deliberately excludes the trailing range, because
    these particular tests are about interior repairs and the cap that applies to
    them. With `hours=12` this leaves five one-candle holes at 01:00, 03:00,
    05:00, 07:00 and 09:00.
    """
    present = [n for n in range(hours) if n % 2 == 0] + [hours - 1]
    return {start + n * HOUR for n in present}


class TestPlanBackfill:
    """Deciding what to fetch, before fetching any of it."""

    def test_empty_store_plans_one_range_covering_everything(self):
        """A first run is not a special case -- it is simply the case where the
        only missing range happens to be the whole thing."""
        start = utc(2025, 1, 1)
        end = utc(2025, 1, 1, 5)

        plan = backfill.plan_backfill(set(), start, end, HOUR)

        assert plan.trailing == gaps.Gap(start, end, 6)
        assert plan.gaps == []
        assert plan.skipped == []

    def test_complete_store_plans_nothing(self):
        """Requirement 3, decided here: a second run has nothing to do, and
        knows it before making a single request."""
        start = utc(2025, 1, 1)
        end = utc(2025, 1, 1, 5)
        stored = {start + n * HOUR for n in range(6)}

        plan = backfill.plan_backfill(stored, start, end, HOUR)

        assert plan.trailing is None
        assert plan.gaps == []
        assert plan.skipped == []

    def test_a_missing_tail_becomes_the_trailing_range(self):
        """The ordinary resume: the store stops somewhere and time has moved on."""
        start = utc(2025, 1, 1)
        end = utc(2025, 1, 1, 5)
        stored = {start + n * HOUR for n in range(4)}

        plan = backfill.plan_backfill(stored, start, end, HOUR)

        assert plan.trailing == gaps.Gap(start + 4 * HOUR, end, 2)
        assert plan.gaps == []

    def test_an_interior_hole_is_a_gap_not_the_trailing_range(self):
        """Distinguished by whether the range touches the end of the request.
        The trailing range is the one that brings the store up to date; a gap is
        a repair."""
        start = utc(2025, 1, 1)
        end = utc(2025, 1, 1, 5)
        stored = {start + n * HOUR for n in (0, 1, 4, 5)}

        plan = backfill.plan_backfill(stored, start, end, HOUR)

        assert plan.trailing is None
        assert plan.gaps == [gaps.Gap(start + 2 * HOUR, start + 3 * HOUR, 2)]

    def test_a_hole_ending_one_candle_before_the_end_is_still_interior(self):
        """The exact boundary of "touches the end of the request".

        Written because a mutation widening that boundary by one timeframe broke
        nothing: every other interior-hole test leaves at least two present
        candles after the hole, so a one-candle slip in the comparison never
        showed. The single-candle case is the one that pins it down.

        Getting this wrong is not cosmetic. A hole misfiled as trailing escapes
        the gap cap entirely and is fetched with the history guard armed, so an
        old scar would be re-requested on every run forever -- the exact
        behaviour the cap exists to prevent.
        """
        start = utc(2025, 1, 1)
        end = utc(2025, 1, 1, 5)
        # Only 04:00 missing. The 05:00 candle is present, so nothing is absent
        # at the end of the request.
        stored = {start + n * HOUR for n in (0, 1, 2, 3, 5)}

        plan = backfill.plan_backfill(stored, start, end, HOUR)

        assert plan.trailing is None
        assert plan.gaps == [gaps.Gap(start + 4 * HOUR, start + 4 * HOUR, 1)]

    def test_a_hole_ending_at_the_last_candle_is_the_trailing_range(self):
        """The other side of the same boundary, so the pair reads as a decision
        rather than an accident."""
        start = utc(2025, 1, 1)
        end = utc(2025, 1, 1, 5)
        stored = {start + n * HOUR for n in (0, 1, 2, 3)}

        plan = backfill.plan_backfill(stored, start, end, HOUR)

        assert plan.trailing == gaps.Gap(start + 4 * HOUR, end, 2)
        assert plan.gaps == []

    def test_a_hole_and_a_missing_tail_are_planned_separately(self):
        start = utc(2025, 1, 1)
        end = utc(2025, 1, 1, 7)
        stored = {start + n * HOUR for n in (0, 1, 4, 5)}

        plan = backfill.plan_backfill(stored, start, end, HOUR)

        assert plan.trailing == gaps.Gap(start + 6 * HOUR, end, 2)
        assert plan.gaps == [gaps.Gap(start + 2 * HOUR, start + 3 * HOUR, 2)]

    def test_a_start_earlier_than_the_store_extends_backwards(self):
        """The hole I flagged in store.next_start_ms, closed.

        Resuming only forwards meant that asking for an earlier start date did
        nothing at all -- no fetch, no warning, and the missing range fell
        outside gap detection too because that only looked *within* what was
        stored. Planning against the requested range instead of against the
        stored range makes the earlier data just another gap.
        """
        stored = {utc(2025, 6, 1) + n * HOUR for n in range(3)}

        plan = backfill.plan_backfill(stored, utc(2025, 1, 1), utc(2025, 6, 1, 2), HOUR)

        assert plan.trailing is None
        assert len(plan.gaps) == 1
        assert plan.gaps[0].start_ms == utc(2025, 1, 1)
        assert plan.gaps[0].end_ms == utc(2025, 6, 1) - HOUR

    def test_gaps_beyond_the_cap_are_reported_rather_than_attempted(self):
        """Bounded retries. A long outage can leave holes the exchange will never
        fill, and re-requesting all of them on every run forever makes every
        future run slower for no gain."""
        start = utc(2025, 1, 1)
        end = utc(2025, 1, 1, 11)

        plan = backfill.plan_backfill(alternating_store(start), start, end, HOUR, max_gaps=2)

        assert len(plan.gaps) == 2
        assert len(plan.skipped) == 3
        assert backfill.gap_count(plan) == 5

    def test_the_newest_gaps_are_the_ones_attempted(self):
        """When the cap bites, recent history is the more useful half to repair:
        an old hole is more likely to be permanent, and recent candles are what a
        backtest is most likely to be reading."""
        start = utc(2025, 1, 1)
        end = utc(2025, 1, 1, 11)

        plan = backfill.plan_backfill(alternating_store(start), start, end, HOUR, max_gaps=2)

        assert [gap.start_ms for gap in plan.gaps] == [
            start + 9 * HOUR,
            start + 7 * HOUR,
        ]

    def test_skipped_gaps_are_listed_chronologically(self):
        """Attempted gaps are ordered by usefulness; skipped ones are only ever
        read by a human, so they are ordered for reading."""
        start = utc(2025, 1, 1)
        end = utc(2025, 1, 1, 11)

        plan = backfill.plan_backfill(alternating_store(start), start, end, HOUR, max_gaps=2)

        starts = [gap.start_ms for gap in plan.skipped]
        assert starts == sorted(starts)

    def test_end_before_start_plans_nothing(self):
        """Asking for a range whose first candle has not closed yet."""
        plan = backfill.plan_backfill(set(), utc(2025, 1, 2), utc(2025, 1, 1), HOUR)

        assert plan.trailing is None
        assert plan.gaps == []


class TestFetchRange:
    """Walking one contiguous range, page by page."""

    def test_stores_a_whole_range_across_several_pages(self, path, fake_exchange, candles):
        start = utc(2025, 1, 1)
        client = fake_exchange(candles(start, 10, HOUR), page_size=4)

        added, pages = backfill.fetch_range(
            client, path, "BTC/USDT", "1h", start, start + 9 * HOUR,
            now_ms=start + 20 * HOUR,
        )

        assert added == 10
        assert pages == 3
        assert store.stored_timestamps(path) == [start + n * HOUR for n in range(10)]

    def test_advances_from_the_last_timestamp_returned_not_by_page_size(
        self, path, fake_exchange, candles
    ):
        """The probe showed page sizes differ wildly between venues -- 500 on
        binance, 199 on bybit, 100 on okx -- and a venue may return fewer than it
        promises. Advancing by an assumed page size would skip candles on one
        exchange and re-request them on another, so the next `since` is derived
        from what actually arrived.
        """
        start = utc(2025, 1, 1)
        client = fake_exchange(candles(start, 10, HOUR), page_size=4)

        backfill.fetch_range(
            client, path, "BTC/USDT", "1h", start, start + 9 * HOUR,
            now_ms=start + 20 * HOUR,
        )

        assert [call["since"] for call in client.calls] == [
            start,
            start + 4 * HOUR,
            start + 8 * HOUR,
        ]

    def test_stops_when_the_exchange_returns_an_empty_page(
        self, path, fake_exchange, candles
    ):
        """An empty page is how a venue says 'nothing further', and is the normal
        way a backfill ends."""
        start = utc(2025, 1, 1)
        client = fake_exchange(candles(start, 3, HOUR), page_size=10)

        added, pages = backfill.fetch_range(
            client, path, "BTC/USDT", "1h", start, start + 100 * HOUR,
            now_ms=start + 200 * HOUR,
        )

        assert added == 3
        assert pages == 1
        assert len(client.calls) == 2  # one page of data, one empty

    def test_never_stores_the_still_forming_candle(
        self, path, fake_exchange, candles
    ):
        """Requirement 4. The exchange will happily serve the candle currently
        being built; its high, low, close and volume are all provisional, so
        storing it would put a value in the file that later becomes wrong."""
        start = utc(2025, 1, 1)
        client = fake_exchange(candles(start, 5, HOUR), page_size=10)
        # Half past 04:00: the 04:00 candle is still open.
        now = start + 4 * HOUR + 30 * 60 * 1000

        backfill.fetch_range(
            client, path, "BTC/USDT", "1h", start, start + 4 * HOUR, now_ms=now
        )

        assert store.stored_timestamps(path) == [start + n * HOUR for n in range(4)]

    def test_flushing_preserves_progress_when_a_run_dies_partway(
        self, path, fake_exchange, candles
    ):
        """Requirement 3, the 'killable mid-run' half.

        Two pages land, then the exchange goes away permanently. Without periodic
        flushing the whole walk would be held in memory and lost; with it, the
        pages that succeeded are on disk and the next run resumes from there
        rather than starting over.

        The no-op `sleep` matters for more than speed. fail_after raises
        ExchangeNotAvailable, which is a NetworkError, so fetch_page treats it as
        transient and retries through the full backoff -- 1+2+4+8 seconds of real
        waiting inside a unit test. Injecting sleep keeps every retry attempt
        exercised while taking the wall clock out of the suite.
        """
        start = utc(2025, 1, 1)
        client = fake_exchange(candles(start, 20, HOUR), page_size=4, fail_after=2)

        with pytest.raises(backfill.ExchangeError, match="giving up after"):
            backfill.fetch_range(
                client, path, "BTC/USDT", "1h", start, start + 19 * HOUR,
                now_ms=start + 50 * HOUR, flush_every=1,
                fetch_options={"sleep": lambda _seconds: None},
            )

        assert store.stored_timestamps(path) == [start + n * HOUR for n in range(8)]

    def test_a_stalled_exchange_is_caught_instead_of_looping_forever(
        self, path, fake_exchange, candles
    ):
        start = utc(2025, 1, 1)
        client = fake_exchange(candles(start, 20, HOUR), page_size=4, stall=True)

        with pytest.raises(backfill.PaginationStalled):
            backfill.fetch_range(
                client, path, "BTC/USDT", "1h", start, start + 19 * HOUR,
                now_ms=start + 50 * HOUR,
            )

    def test_an_exchange_that_ignores_since_is_caught(
        self, path, fake_exchange, candles
    ):
        """This is Kraken, as measured on 2026-07-30: `since` ignored, a window
        of 721 recent candles returned, and a two-year backfill quietly
        impossible. The guard turns that into an error on page one.

        A live capped endpoint serves candles right up to the present, so `now` is
        set just after the newest candle returned -- that pairing is what makes
        the window recognisable as a window.
        """
        client = fake_exchange(
            candles(utc(2026, 6, 30), 721, HOUR), page_size=721, ignore_since=True
        )

        with pytest.raises(backfill.HistoryNotAvailable):
            backfill.fetch_range(
                client, path, "BTC/USDT", "1h",
                utc(2024, 8, 1), utc(2026, 7, 30),
                now_ms=utc(2026, 7, 30) + 30 * 60 * 1000, check_history=True,
            )

    def test_a_capped_window_with_stale_data_is_caught_by_the_stall_guard(
        self, path, fake_exchange, candles
    ):
        """The second line of defence, and worth knowing exists.

        The history guard needs the returned window to reach the present, because
        that is what distinguishes a cap from a market listed after the requested
        start. A venue whose candles lag by half a day slips past it. The
        pagination guard catches it anyway on page two: the same window comes back
        for the advanced `since`, so nothing progresses.

        Two guards on one failure is not redundancy here -- they cover different
        halves of it. Which one fires depends on where the requested end sits
        relative to the venue's newest candle, and neither position is unusual.
        """
        client = fake_exchange(
            candles(utc(2026, 6, 30), 721, HOUR), page_size=721, ignore_since=True
        )

        with pytest.raises(backfill.PaginationStalled):
            backfill.fetch_range(
                client, path, "BTC/USDT", "1h",
                utc(2024, 8, 1), utc(2026, 7, 30, 11),
                now_ms=utc(2026, 7, 30, 12), check_history=True,
            )

    def test_running_the_same_range_twice_adds_nothing_the_second_time(
        self, path, fake_exchange, candles
    ):
        start = utc(2025, 1, 1)
        universe = candles(start, 10, HOUR)
        now = start + 20 * HOUR

        first_client = fake_exchange(universe, page_size=4)
        backfill.fetch_range(
            first_client, path, "BTC/USDT", "1h", start, start + 9 * HOUR, now_ms=now
        )

        second_client = fake_exchange(universe, page_size=4)
        added, _ = backfill.fetch_range(
            second_client, path, "BTC/USDT", "1h", start, start + 9 * HOUR, now_ms=now
        )

        assert added == 0

    def test_candles_the_exchange_lacks_are_simply_absent(
        self, path, fake_exchange, candles
    ):
        """A venue with a genuine hole -- a halted market, or an hour with no
        trades -- returns the candles either side and nothing for the hole. That
        is not an error, and must not be retried as though it were."""
        start = utc(2025, 1, 1)
        client = fake_exchange(
            candles(start, 6, HOUR), page_size=10, holes=[start + 2 * HOUR]
        )

        added, _ = backfill.fetch_range(
            client, path, "BTC/USDT", "1h", start, start + 5 * HOUR,
            now_ms=start + 20 * HOUR,
        )

        assert added == 5
        assert start + 2 * HOUR not in store.stored_timestamps(path)


class TestBackfillSymbol:
    """The whole job for one symbol, which is what the CLI will call."""

    def test_first_run_stores_the_requested_range(
        self, path, fake_exchange, candles
    ):
        start = utc(2025, 1, 1)
        client = fake_exchange(candles(start, 48, HOUR), page_size=10)

        result = backfill.backfill_symbol(
            client, path, "BTC/USDT", "1h", start, now_ms=start + 48 * HOUR
        )

        assert result.added == 48
        assert result.unfilled == []
        assert len(store.stored_timestamps(path)) == 48

    def test_second_run_immediately_after_adds_nothing_and_errors_on_nothing(
        self, path, fake_exchange, candles
    ):
        """Straight out of the acceptance criteria: re-running immediately should
        add nothing and fail on nothing."""
        start = utc(2025, 1, 1)
        universe = candles(start, 48, HOUR)
        now = start + 48 * HOUR

        backfill.backfill_symbol(
            fake_exchange(universe, page_size=10), path, "BTC/USDT", "1h",
            start, now_ms=now,
        )
        client = fake_exchange(universe, page_size=10)
        result = backfill.backfill_symbol(
            client, path, "BTC/USDT", "1h", start, now_ms=now
        )

        assert result.added == 0
        assert result.unfilled == []
        assert client.calls == []  # knows there is nothing to do without asking

    def test_fills_an_interior_hole_left_by_an_earlier_run(
        self, path, fake_exchange, candles
    ):
        start = utc(2025, 1, 1)
        universe = candles(start, 10, HOUR)
        # Pretend an earlier run stored everything except two candles.
        store.append_candles(path, [row for row in universe if row[0] not in
                                    (start + 4 * HOUR, start + 5 * HOUR)])

        result = backfill.backfill_symbol(
            fake_exchange(universe, page_size=10), path, "BTC/USDT", "1h",
            start, now_ms=start + 10 * HOUR,
        )

        assert result.added == 2
        assert store.stored_timestamps(path) == [start + n * HOUR for n in range(10)]

    def test_extends_backwards_when_asked_for_an_earlier_start(
        self, path, fake_exchange, candles
    ):
        """The flagged hole, end to end. Previously this did nothing at all."""
        universe = candles(utc(2025, 1, 1), 240, HOUR)
        later = utc(2025, 1, 6)
        store.append_candles(path, [row for row in universe if row[0] >= later])

        before = len(store.stored_timestamps(path))
        result = backfill.backfill_symbol(
            fake_exchange(universe, page_size=50), path, "BTC/USDT", "1h",
            utc(2025, 1, 1), now_ms=utc(2025, 1, 11),
        )

        assert result.added == 240 - before
        assert store.stored_timestamps(path)[0] == utc(2025, 1, 1)

    def test_reports_candles_the_exchange_does_not_have(
        self, path, fake_exchange, candles
    ):
        """Requirement 6: report missing candles rather than failing silently.
        After asking for a gap and receiving nothing, the gap is real and the run
        should say so rather than treating the range as complete."""
        start = utc(2025, 1, 1)
        client = fake_exchange(
            candles(start, 6, HOUR), page_size=10, holes=[start + 2 * HOUR]
        )

        result = backfill.backfill_symbol(
            client, path, "BTC/USDT", "1h", start, now_ms=start + 6 * HOUR
        )

        assert result.unfilled == [gaps.Gap(start + 2 * HOUR, start + 2 * HOUR, 1)]

    def test_does_not_re_report_a_gap_it_managed_to_fill(
        self, path, fake_exchange, candles
    ):
        start = utc(2025, 1, 1)
        universe = candles(start, 10, HOUR)
        store.append_candles(path, [row for row in universe if row[0] != start + 3 * HOUR])

        result = backfill.backfill_symbol(
            fake_exchange(universe, page_size=10), path, "BTC/USDT", "1h",
            start, now_ms=start + 10 * HOUR,
        )

        assert result.unfilled == []
        assert result.filled == [gaps.Gap(start + 3 * HOUR, start + 3 * HOUR, 1)]

    def test_respects_the_gap_cap_and_reports_what_it_skipped(
        self, path, fake_exchange, candles
    ):
        start = utc(2025, 1, 1)
        universe = candles(start, 12, HOUR)
        present = alternating_store(start)
        store.append_candles(path, [row for row in universe if row[0] in present])

        result = backfill.backfill_symbol(
            fake_exchange(universe, page_size=10), path, "BTC/USDT", "1h",
            start, now_ms=start + 12 * HOUR, max_gaps=2,
        )

        assert len(result.skipped) == 3
        assert result.added == 2

    def test_a_skipped_gap_is_not_also_reported_as_unfilled(
        self, path, fake_exchange, candles
    ):
        """"We asked and the venue had nothing" and "we did not ask" are different
        facts, and conflating them makes the gap report useless.

        A hole beyond the cap is still missing at the end of the run, so anything
        derived from the file alone would list it as unfilled -- and then every
        report would accuse the exchange of withholding candles nobody requested.
        The two lists have to stay disjoint.
        """
        start = utc(2025, 1, 1)
        universe = candles(start, 12, HOUR)
        present = alternating_store(start)
        store.append_candles(path, [row for row in universe if row[0] in present])

        result = backfill.backfill_symbol(
            fake_exchange(universe, page_size=10), path, "BTC/USDT", "1h",
            start, now_ms=start + 12 * HOUR, max_gaps=2,
        )

        assert result.skipped != []
        assert result.unfilled == []
        for gap in result.skipped:
            assert gap not in result.unfilled

    def test_filled_excludes_a_gap_the_exchange_could_not_supply(
        self, path, fake_exchange, candles
    ):
        """`filled` is a claim about what was repaired, so it must be checked
        against the file rather than against what was attempted.

        Two gaps are attempted; the exchange has the candles for one and not the
        other. Reporting both as filled would be a quiet lie of exactly the kind
        requirement 6 is about -- the run would look like a complete success while
        leaving a hole behind.
        """
        start = utc(2025, 1, 1)
        universe = candles(start, 10, HOUR)
        # Store everything except 03:00 and 06:00, so both are planned as gaps.
        store.append_candles(
            path,
            [row for row in universe if row[0] not in (start + 3 * HOUR, start + 6 * HOUR)],
        )

        result = backfill.backfill_symbol(
            # The venue genuinely lacks 06:00, so asking for it returns nothing.
            fake_exchange(universe, page_size=10, holes=[start + 6 * HOUR]),
            path, "BTC/USDT", "1h", start, now_ms=start + 10 * HOUR,
        )

        assert result.filled == [gaps.Gap(start + 3 * HOUR, start + 3 * HOUR, 1)]
        assert result.unfilled == [gaps.Gap(start + 6 * HOUR, start + 6 * HOUR, 1)]

    def test_a_capped_exchange_is_caught_on_the_production_path(
        self, path, fake_exchange, candles
    ):
        """The history guard has to be armed by backfill_symbol, not just be
        reachable through fetch_range.

        Disabling `check_history=True` on the trailing range broke no test, because
        every test of that guard called fetch_range directly and passed the flag
        itself. Nothing checked that the function a caller actually uses turns it
        on. This is the Kraken case arriving through the front door.
        """
        client = fake_exchange(
            candles(utc(2026, 6, 30), 721, HOUR), page_size=721, ignore_since=True
        )

        with pytest.raises(backfill.HistoryNotAvailable):
            backfill.backfill_symbol(
                client, path, "BTC/USDT", "1h",
                utc(2024, 8, 1), now_ms=utc(2026, 7, 30) + 30 * 60 * 1000,
            )

    def test_nothing_to_do_before_the_first_candle_has_closed(
        self, path, fake_exchange, candles, caplog
    ):
        """The early return, and the log line that is its only visible effect.

        Removing the early return entirely changes no return value -- an empty
        range plans nothing, so the function falls through to the same empty
        Result. The log line is the whole point of it: an unattended run that
        prints nothing is indistinguishable from one that failed to start, and
        requirement 9 exists because that log is how the run gets read afterwards.
        """
        start = utc(2025, 1, 1)
        client = fake_exchange(candles(start, 5, HOUR))

        with caplog.at_level(logging.INFO, logger="collector.backfill"):
            result = backfill.backfill_symbol(
                client, path, "BTC/USDT", "1h", start, now_ms=start + 30 * 60 * 1000
            )

        assert result.added == 0
        assert client.calls == []
        assert "no candle has closed" in caplog.text

    def test_the_symbol_is_named_in_the_result(self, path, fake_exchange, candles):
        """A five-symbol run produces five results, and they need telling apart."""
        start = utc(2025, 1, 1)
        client = fake_exchange(candles(start, 5, HOUR))

        result = backfill.backfill_symbol(
            client, path, "ETH/USDT", "1h", start, now_ms=start + 5 * HOUR
        )

        assert result.symbol == "ETH/USDT"
