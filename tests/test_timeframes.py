"""Tests for timeframe parsing and candle-boundary maths.

Every function under test takes `now_ms` as an argument rather than reading the
clock itself. That is deliberate: it means these tests need no clock patching,
no freezegun, no monkeypatching -- just numbers in and numbers out. The cost is
one extra parameter at the call site; the benefit is that the trickiest logic in
the project is also the easiest part of it to test.
"""

import datetime as dt

import pytest

from collector import timeframes as tf

MINUTE = 60_000
HOUR = 60 * MINUTE
DAY = 24 * HOUR


def utc(year, month, day, hour=0, minute=0):
    """Build an epoch-millisecond timestamp from UTC calendar parts."""
    return int(
        dt.datetime(year, month, day, hour, minute, tzinfo=dt.timezone.utc).timestamp()
        * 1000
    )


class TestTimeframeToMs:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("1m", MINUTE),
            ("5m", 5 * MINUTE),
            ("15m", 15 * MINUTE),
            ("30m", 30 * MINUTE),
            ("1h", HOUR),
            ("4h", 4 * HOUR),
            ("12h", 12 * HOUR),
            ("1d", DAY),
        ],
    )
    def test_parses_supported_timeframes(self, text, expected):
        assert tf.timeframe_to_ms(text) == expected

    @pytest.mark.parametrize(
        "text",
        ["", "h", "1", "1x", "1.5h", "-1h", "0h", "0m", " 1h", "1h ", None, 3600],
    )
    def test_rejects_malformed_input(self, text):
        with pytest.raises(tf.TimeframeError):
            tf.timeframe_to_ms(text)

    @pytest.mark.parametrize("text", ["1w", "2w", "1M", "1y"])
    def test_rejects_units_without_stable_alignment(self, text):
        """Weeks and months cannot be floored by simple epoch arithmetic.

        The epoch began on a Thursday, so flooring by 7-day multiples produces
        Thursday-aligned weeks, while exchanges align to Monday. Months are not
        a fixed duration at all. Rather than silently produce candles offset
        from the exchange's own, we refuse these units outright.
        """
        with pytest.raises(tf.TimeframeError):
            tf.timeframe_to_ms(text)

    def test_rejects_multi_day_timeframes(self):
        """`3d` has the same alignment ambiguity as weeks: 3-day blocks counted
        from the epoch need not match the exchange's own 3-day blocks."""
        with pytest.raises(tf.TimeframeError):
            tf.timeframe_to_ms("3d")

    def test_rejects_sub_daily_that_does_not_divide_a_day(self):
        """A 7h candle would drift: 24 is not a multiple of 7, so boundaries
        land at a different time each day and flooring becomes meaningless."""
        with pytest.raises(tf.TimeframeError):
            tf.timeframe_to_ms("7h")

    def test_is_case_sensitive(self):
        """`1m` is one minute and `1M` is one month in ccxt's notation, so
        accepting either case would make a minute silently mean a month."""
        with pytest.raises(tf.TimeframeError):
            tf.timeframe_to_ms("1H")


class TestCandleOpenTime:
    def test_timestamp_already_on_boundary_is_unchanged(self):
        boundary = utc(2025, 3, 14, 9)
        assert tf.candle_open_time(boundary, HOUR) == boundary

    def test_timestamp_mid_candle_floors_down(self):
        assert tf.candle_open_time(utc(2025, 3, 14, 9, 37), HOUR) == utc(2025, 3, 14, 9)

    def test_one_millisecond_before_next_boundary_still_floors_down(self):
        boundary = utc(2025, 3, 14, 9)
        assert tf.candle_open_time(boundary + HOUR - 1, HOUR) == boundary

    def test_daily_candles_align_to_utc_midnight(self):
        assert tf.candle_open_time(utc(2025, 3, 14, 23, 59), DAY) == utc(2025, 3, 14)

    def test_four_hour_candles_align_to_multiples_of_four_from_midnight(self):
        assert tf.candle_open_time(utc(2025, 3, 14, 7, 30), 4 * HOUR) == utc(
            2025, 3, 14, 4
        )

    def test_rejects_negative_timestamp(self):
        """Pre-1970 timestamps mean something upstream produced garbage, most
        likely seconds mistaken for milliseconds or an uninitialised value."""
        with pytest.raises(ValueError):
            tf.candle_open_time(-1, HOUR)


class TestIsCandleClosed:
    def test_candle_is_closed_the_instant_its_period_elapses(self):
        opened = utc(2025, 3, 14, 9)
        assert tf.is_candle_closed(opened, HOUR, now_ms=opened + HOUR) is True

    def test_candle_is_open_one_millisecond_before_that(self):
        opened = utc(2025, 3, 14, 9)
        assert tf.is_candle_closed(opened, HOUR, now_ms=opened + HOUR - 1) is False

    def test_candle_that_just_opened_is_open(self):
        opened = utc(2025, 3, 14, 9)
        assert tf.is_candle_closed(opened, HOUR, now_ms=opened) is False

    def test_old_candle_is_closed(self):
        assert (
            tf.is_candle_closed(utc(2024, 1, 1), HOUR, now_ms=utc(2025, 1, 1)) is True
        )


class TestLastClosedOpenTime:
    def test_when_now_is_mid_candle(self):
        now = utc(2025, 3, 14, 9, 37)
        assert tf.last_closed_open_time(now, HOUR) == utc(2025, 3, 14, 8)

    def test_when_now_is_exactly_on_a_boundary(self):
        """At 09:00:00.000 the 09:00 candle has just opened with zero elapsed
        time, so the newest *closed* candle is the one that opened at 08:00."""
        now = utc(2025, 3, 14, 9)
        assert tf.last_closed_open_time(now, HOUR) == utc(2025, 3, 14, 8)

    def test_result_is_always_a_closed_candle(self):
        now = utc(2025, 3, 14, 9, 37)
        opened = tf.last_closed_open_time(now, HOUR)
        assert tf.is_candle_closed(opened, HOUR, now) is True
        # ...and the next one along is not.
        assert tf.is_candle_closed(opened + HOUR, HOUR, now) is False


class TestDropIncompleteCandles:
    """Requirement 4. The newest candle is still forming and its high, low,
    close and volume will all change. Storing it as final would silently
    corrupt every backtest that later reads this store.
    """

    @staticmethod
    def candle(ts):
        return [ts, 100.0, 110.0, 90.0, 105.0, 12.5]

    def test_drops_the_still_forming_final_candle(self):
        now = utc(2025, 3, 14, 9, 37)
        rows = [
            self.candle(utc(2025, 3, 14, 7)),
            self.candle(utc(2025, 3, 14, 8)),
            self.candle(utc(2025, 3, 14, 9)),  # opened 37 minutes ago, still forming
        ]
        kept = tf.drop_incomplete_candles(rows, HOUR, now)
        assert [row[0] for row in kept] == [utc(2025, 3, 14, 7), utc(2025, 3, 14, 8)]

    def test_keeps_everything_when_all_candles_have_closed(self):
        now = utc(2025, 3, 14, 12)
        rows = [self.candle(utc(2025, 3, 14, 7)), self.candle(utc(2025, 3, 14, 8))]
        assert tf.drop_incomplete_candles(rows, HOUR, now) == rows

    def test_empty_input_gives_empty_output(self):
        assert tf.drop_incomplete_candles([], HOUR, utc(2025, 3, 14)) == []

    def test_filters_by_close_time_not_by_position(self):
        """The rule is 'has this candle's period elapsed', not 'is this the last
        row'. Given input that is out of order and gapped, every unclosed candle
        must be removed regardless of where it sits in the list.
        """
        now = utc(2025, 3, 14, 9, 30)
        rows = [
            self.candle(utc(2025, 3, 14, 9)),  # forming
            self.candle(utc(2025, 3, 14, 5)),  # closed
            self.candle(utc(2025, 3, 14, 10)),  # entirely in the future
            self.candle(utc(2025, 3, 14, 6)),  # closed
        ]
        kept = tf.drop_incomplete_candles(rows, HOUR, now)
        assert sorted(row[0] for row in kept) == [
            utc(2025, 3, 14, 5),
            utc(2025, 3, 14, 6),
        ]

    def test_does_not_mutate_the_input_list(self):
        """The caller may still want the raw page for logging or debugging."""
        now = utc(2025, 3, 14, 9, 37)
        rows = [self.candle(utc(2025, 3, 14, 8)), self.candle(utc(2025, 3, 14, 9))]
        before = list(rows)
        tf.drop_incomplete_candles(rows, HOUR, now)
        assert rows == before


class TestExpectedOpenTimes:
    """Used by gap detection: the full set of candle open times that *should*
    exist across a range, to be compared against what is actually stored.
    """

    def test_inclusive_of_both_aligned_endpoints(self):
        start = utc(2025, 3, 14, 0)
        end = utc(2025, 3, 14, 3)
        assert tf.expected_open_times(start, end, HOUR) == [
            utc(2025, 3, 14, 0),
            utc(2025, 3, 14, 1),
            utc(2025, 3, 14, 2),
            utc(2025, 3, 14, 3),
        ]

    def test_unaligned_start_rounds_up_to_the_first_real_boundary(self):
        """Rounding up rather than down avoids inventing a candle before the
        range the caller actually asked about."""
        result = tf.expected_open_times(utc(2025, 3, 14, 0, 30), utc(2025, 3, 14, 2), HOUR)
        assert result == [utc(2025, 3, 14, 1), utc(2025, 3, 14, 2)]

    def test_unaligned_end_excludes_the_partial_trailing_candle(self):
        result = tf.expected_open_times(utc(2025, 3, 14, 0), utc(2025, 3, 14, 2, 30), HOUR)
        assert result == [
            utc(2025, 3, 14, 0),
            utc(2025, 3, 14, 1),
            utc(2025, 3, 14, 2),
        ]

    def test_single_candle_range(self):
        boundary = utc(2025, 3, 14, 5)
        assert tf.expected_open_times(boundary, boundary, HOUR) == [boundary]

    def test_end_before_start_gives_empty(self):
        assert tf.expected_open_times(utc(2025, 3, 14, 5), utc(2025, 3, 14, 3), HOUR) == []

    def test_counts_a_full_day_of_hourly_candles(self):
        result = tf.expected_open_times(utc(2025, 3, 14), utc(2025, 3, 15), HOUR)
        assert len(result) == 25  # inclusive of both midnights

    def test_spans_a_dst_change_without_a_gap(self):
        """A sanity check on the UTC-only requirement. 30 March 2025 is when
        European clocks jumped forward. In UTC nothing happens, so the hour
        count is unremarkable -- if this ever returns 23 or 25, something has
        started interpreting these timestamps as local time.
        """
        result = tf.expected_open_times(utc(2025, 3, 30), utc(2025, 3, 31), HOUR)
        assert len(result) == 25
        deltas = {b - a for a, b in zip(result, result[1:])}
        assert deltas == {HOUR}
