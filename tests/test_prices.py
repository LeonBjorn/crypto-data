"""Tests for loading candles out of the store, ready to analyse.

The collector's job was to be tolerant: a missing file is a normal first run, a
short range is something to go and fetch. This module's job is the opposite. By
the time candles reach an indicator, anything ambiguous has to have become an
error, because past this point a missing bar stops looking like a missing bar
and starts looking like a price that did not move.

So most of what is tested here is refusal, and the interesting cases are the
ones where refusing would be wrong -- a hole in 2024 must not block a backtest
that only covers 2026.
"""

import datetime as dt

import pandas as pd
import pytest

from collector import store
from collector.timeframes import TimeframeError
from signals import prices

HOUR = 3_600_000
T0 = 1_722_470_400_000  # 2024-08-01T00:00:00Z, a boundary for every timeframe.


def build_store(root, timestamps, *, exchange="binance", symbol="BTC/USDT", timeframe="1h"):
    """Write a candle file containing exactly `timestamps` and return the data root.

    Prices are derived from the index so that every row is distinguishable in a
    failure message, and so a test can assert it got the rows it meant to.
    """
    data_dir = root / "data"
    path = store.candle_path(data_dir, exchange, symbol, timeframe)
    rows = [
        [ts, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 10.0 + i]
        for i, ts in enumerate(timestamps)
    ]
    store.write_candles(path, store.candles_to_frame(rows))
    return data_dir


def hours(count, start=T0):
    """`count` consecutive hourly candle open times."""
    return [start + i * HOUR for i in range(count)]


def utc(text):
    """Epoch milliseconds from 'YYYY-MM-DD HH:MM'."""
    parsed = dt.datetime.strptime(text, "%Y-%m-%d %H:%M")
    return int(parsed.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


class TestLoading:
    def test_loads_the_stored_candles(self, tmp_path):
        data_dir = build_store(tmp_path, hours(10))
        frame = prices.load(data_dir, "binance", "BTC/USDT", "1h")
        assert len(frame) == 10
        assert frame["timestamp"].iloc[0] == T0
        assert frame["timestamp"].iloc[-1] == T0 + 9 * HOUR

    def test_the_columns_are_the_stores_columns_in_order(self, tmp_path):
        data_dir = build_store(tmp_path, hours(5))
        frame = prices.load(data_dir, "binance", "BTC/USDT", "1h")
        assert list(frame.columns) == store.COLUMNS

    def test_the_timestamp_stays_an_integer(self, tmp_path):
        """Not a datetime, and not by accident.

        Converting on load would be convenient and is the one change most likely
        to be made later by someone tidying up. A pandas datetime with no
        timezone attached means local time, so a store that is correct on this
        machine becomes an hour out on a laptop in Oslo in December, and nothing
        anywhere raises. The conversion belongs in the display layer.
        """
        data_dir = build_store(tmp_path, hours(5))
        frame = prices.load(data_dir, "binance", "BTC/USDT", "1h")
        assert frame["timestamp"].dtype == "int64"

    def test_the_index_is_zero_upwards_after_slicing(self, tmp_path):
        """Indicators are assigned back by index, so the index has to be sane.

        Slicing a frame in pandas keeps the original labels, so a range starting
        at the hundredth candle would otherwise hand back a frame indexed
        100..199. Every indicator would then also be indexed 100..199, which
        works, until something compares two frames loaded over different ranges.
        """
        data_dir = build_store(tmp_path, hours(50))
        frame = prices.load(data_dir, "binance", "BTC/USDT", "1h", start_ms=T0 + 40 * HOUR)
        assert frame.index.tolist() == list(range(10))

    def test_rows_come_back_oldest_first(self, tmp_path):
        data_dir = build_store(tmp_path, hours(20))
        frame = prices.load(data_dir, "binance", "BTC/USDT", "1h")
        assert frame["timestamp"].is_monotonic_increasing


class TestNothingToLoad:
    def test_a_symbol_that_was_never_collected_is_an_error(self, tmp_path):
        """The collector returns an empty frame here; this must not.

        Same situation, opposite correct response. For the collector an absent
        file means "nothing fetched yet, start at the beginning". For analysis it
        means every statistic that follows would be computed over zero rows, and
        a backtest reporting zero trades looks identical to a rule that never
        fired.
        """
        data_dir = build_store(tmp_path, hours(5))
        with pytest.raises(prices.PriceError, match="collect"):
            prices.load(data_dir, "binance", "SOL/USDT", "1h")

    def test_the_message_names_the_file_it_looked_for(self, tmp_path):
        data_dir = build_store(tmp_path, hours(5))
        with pytest.raises(prices.PriceError, match="SOL_USDT"):
            prices.load(data_dir, "binance", "SOL/USDT", "1h")

    def test_a_stored_but_empty_file_is_an_error(self, tmp_path):
        data_dir = tmp_path / "data"
        path = store.candle_path(data_dir, "binance", "BTC/USDT", "1h")
        store.write_candles(path, store.empty_frame())
        with pytest.raises(prices.PriceError, match="no candles"):
            prices.load(data_dir, "binance", "BTC/USDT", "1h")


class TestRangeSelection:
    def test_start_is_inclusive(self, tmp_path):
        data_dir = build_store(tmp_path, hours(10))
        frame = prices.load(data_dir, "binance", "BTC/USDT", "1h", start_ms=T0 + 3 * HOUR)
        assert frame["timestamp"].iloc[0] == T0 + 3 * HOUR
        assert len(frame) == 7

    def test_end_is_inclusive(self, tmp_path):
        """Inclusive because the argument names a candle, not a moment.

        `end_ms=T0 + 3h` means "the candle that opened at 03:00", so that candle
        is in the result. An exclusive end would make `start=X, end=X` return
        nothing, which is a surprising answer to a request for one candle.
        """
        data_dir = build_store(tmp_path, hours(10))
        frame = prices.load(data_dir, "binance", "BTC/USDT", "1h", end_ms=T0 + 3 * HOUR)
        assert frame["timestamp"].iloc[-1] == T0 + 3 * HOUR
        assert len(frame) == 4

    def test_both_ends_together(self, tmp_path):
        data_dir = build_store(tmp_path, hours(10))
        frame = prices.load(
            data_dir, "binance", "BTC/USDT", "1h",
            start_ms=T0 + 2 * HOUR, end_ms=T0 + 5 * HOUR,
        )
        assert frame["timestamp"].tolist() == [T0 + i * HOUR for i in range(2, 6)]

    def test_a_single_candle_range(self, tmp_path):
        data_dir = build_store(tmp_path, hours(10))
        frame = prices.load(
            data_dir, "binance", "BTC/USDT", "1h",
            start_ms=T0 + 4 * HOUR, end_ms=T0 + 4 * HOUR,
        )
        assert len(frame) == 1

    def test_a_start_after_the_end_is_refused(self, tmp_path):
        data_dir = build_store(tmp_path, hours(10))
        with pytest.raises(prices.PriceError, match="after"):
            prices.load(
                data_dir, "binance", "BTC/USDT", "1h",
                start_ms=T0 + 5 * HOUR, end_ms=T0 + 2 * HOUR,
            )


class TestRangeCoverage:
    """Asking for more than the store holds is an error, not a shorter answer.

    This is the check that stops a two-year backtest quietly becoming a
    two-month one. Silently returning what exists is the tempting behaviour and
    it produces a result that is not wrong about anything except the question it
    answered, which no statistic downstream can detect.
    """

    def test_a_start_before_the_first_stored_candle_is_refused(self, tmp_path):
        data_dir = build_store(tmp_path, hours(10))
        with pytest.raises(prices.PriceError, match="begins at"):
            prices.load(data_dir, "binance", "BTC/USDT", "1h", start_ms=T0 - 5 * HOUR)

    def test_an_end_after_the_last_stored_candle_is_refused(self, tmp_path):
        data_dir = build_store(tmp_path, hours(10))
        with pytest.raises(prices.PriceError, match="ends at"):
            prices.load(data_dir, "binance", "BTC/USDT", "1h", end_ms=T0 + 20 * HOUR)

    def test_the_message_gives_readable_dates(self, tmp_path):
        """Epoch milliseconds in an error message are not an explanation.

        "1722470400000 is before 1722452400000" is technically the answer and
        takes a calculator to read. Both numbers go in as UTC timestamps.
        """
        data_dir = build_store(tmp_path, hours(10))
        with pytest.raises(prices.PriceError, match="2024-08-01 00:00"):
            prices.load(data_dir, "binance", "BTC/USDT", "1h", start_ms=T0 - 5 * HOUR)

    def test_a_range_entirely_outside_the_store_is_refused(self, tmp_path):
        data_dir = build_store(tmp_path, hours(10))
        with pytest.raises(prices.PriceError):
            prices.load(
                data_dir, "binance", "BTC/USDT", "1h",
                start_ms=T0 + 100 * HOUR, end_ms=T0 + 110 * HOUR,
            )

    def test_a_range_the_store_fully_covers_is_fine(self, tmp_path):
        data_dir = build_store(tmp_path, hours(100))
        frame = prices.load(
            data_dir, "binance", "BTC/USDT", "1h",
            start_ms=T0 + 10 * HOUR, end_ms=T0 + 20 * HOUR,
        )
        assert len(frame) == 11


class TestBoundaryAlignment:
    def test_a_start_that_is_not_a_candle_open_time_is_refused(self, tmp_path):
        """Half past the hour is not a candle, and guessing which one is meant
        is the kind of quiet correction this project does not do.

        Rounding down would silently include a candle the caller did not ask
        for; rounding up would silently exclude one. Neither is visible in the
        output. In practice every start comes from a date, which is midnight
        UTC and therefore aligned for every timeframe here, so this costs
        nothing and catches a real confusion between "a moment" and "a bar".
        """
        data_dir = build_store(tmp_path, hours(10))
        with pytest.raises(prices.PriceError, match="candle open time"):
            prices.load(data_dir, "binance", "BTC/USDT", "1h", start_ms=T0 + 1800_000)

    def test_an_end_that_is_not_a_candle_open_time_is_refused(self, tmp_path):
        data_dir = build_store(tmp_path, hours(10))
        with pytest.raises(prices.PriceError, match="candle open time"):
            prices.load(data_dir, "binance", "BTC/USDT", "1h", end_ms=T0 + 1800_000)

    def test_alignment_is_judged_against_the_timeframe_asked_for(self, tmp_path):
        """T0 + 1h is a valid 1h open and not a valid 4h one."""
        data_dir = build_store(tmp_path, hours(40), timeframe="4h")
        with pytest.raises(prices.PriceError, match="candle open time"):
            prices.load(data_dir, "binance", "BTC/USDT", "4h", start_ms=T0 + HOUR)


class TestContinuity:
    def test_a_hole_inside_the_requested_range_is_refused(self, tmp_path):
        stamps = hours(20)
        del stamps[10]
        data_dir = build_store(tmp_path, stamps)
        with pytest.raises(prices.PriceError, match="missing"):
            prices.load(data_dir, "binance", "BTC/USDT", "1h")

    def test_the_message_names_when_the_hole_is(self, tmp_path):
        stamps = hours(20)
        del stamps[10]  # 2024-08-01 10:00
        data_dir = build_store(tmp_path, stamps)
        with pytest.raises(prices.PriceError, match="2024-08-01 10:00"):
            prices.load(data_dir, "binance", "BTC/USDT", "1h")

    def test_a_hole_outside_the_requested_range_does_not_matter(self, tmp_path):
        """The case that makes strictness bearable rather than obstructive.

        Exchanges have permanent holes in old history that no amount of
        re-running will fill. If any hole anywhere disqualified the file, one
        bad hour in 2024 would block every backtest over 2026 forever, and the
        only way to work would be to turn the check off entirely -- which is
        how a safety check becomes a habit of passing `False`.
        """
        stamps = hours(100)
        del stamps[5]
        data_dir = build_store(tmp_path, stamps)
        frame = prices.load(
            data_dir, "binance", "BTC/USDT", "1h",
            start_ms=T0 + 50 * HOUR, end_ms=T0 + 90 * HOUR,
        )
        assert len(frame) == 41

    def test_a_hole_can_be_accepted_deliberately(self, tmp_path):
        """Off is allowed, because inspecting a broken file is a real need.

        It is not the default, and the argument has to be typed out at the call
        site, so accepting a gap is always a visible decision in the code rather
        than something that happened.
        """
        stamps = hours(20)
        del stamps[10]
        data_dir = build_store(tmp_path, stamps)
        frame = prices.load(
            data_dir, "binance", "BTC/USDT", "1h", require_continuous=False
        )
        assert len(frame) == 19

    def test_a_run_of_missing_candles_is_reported_as_one_gap(self, tmp_path):
        stamps = hours(30)
        del stamps[10:15]
        data_dir = build_store(tmp_path, stamps)
        with pytest.raises(prices.PriceError, match="1 gap"):
            prices.load(data_dir, "binance", "BTC/USDT", "1h")

    def test_many_gaps_are_summarised_rather_than_all_listed(self, tmp_path):
        """A message longer than the terminal is a message nobody reads."""
        stamps = [ts for i, ts in enumerate(hours(200)) if i % 3 != 0]
        data_dir = build_store(tmp_path, stamps)
        with pytest.raises(prices.PriceError, match="more") as caught:
            prices.load(data_dir, "binance", "BTC/USDT", "1h")
        assert len(str(caught.value).splitlines()) < 15


class TestIntegrity:
    def test_a_duplicated_timestamp_is_refused_even_without_continuity(self, tmp_path):
        """Not a continuity problem, so `require_continuous=False` must not skip it.

        A duplicate is invisible to gap detection, which compares sets -- two
        copies of 10:00 and a set containing 10:00 look the same. But an
        indicator would average that bar twice, and a backtest would be able to
        enter the same trade twice. The store's writer deduplicates, so a
        duplicate on disk means the file was written by something else, which is
        worth knowing about rather than working around.
        """
        data_dir = tmp_path / "data"
        path = store.candle_path(data_dir, "binance", "BTC/USDT", "1h")
        stamps = hours(10)
        frame = pd.DataFrame(
            [[ts, 1.0, 2.0, 0.5, 1.5, 10.0] for ts in stamps + [stamps[4]]],
            columns=store.COLUMNS,
        )
        store.write_candles(path, frame)

        with pytest.raises(prices.PriceError, match="twice|duplicate"):
            prices.load(
                data_dir, "binance", "BTC/USDT", "1h", require_continuous=False
            )


class TestArgumentValidation:
    def test_an_unsupported_timeframe_is_refused(self, tmp_path):
        """Deliberately not wrapped in PriceError.

        timeframes.py explains at length why weeks and months cannot be aligned
        from the epoch, and restating that here would only lose the explanation.
        The same choice settings.py already makes.
        """
        data_dir = build_store(tmp_path, hours(10))
        with pytest.raises(TimeframeError):
            prices.load(data_dir, "binance", "BTC/USDT", "1w")

    def test_a_symbol_without_a_slash_is_refused(self, tmp_path):
        """Caught here rather than left to produce a missing-file message.

        'BTCUSDT' is a perfectly legal directory name, so the store builds a
        path for it happily and the file is simply absent. The natural error is
        then "no candles, run `uv run collect`" -- advice that cannot possibly
        work, because the collector writes 'BTC_USDT' and would never create
        that directory no matter how many times it ran. Sending someone off to
        repeat a backfill that cannot fix their typo is worse than saying
        nothing.
        """
        data_dir = build_store(tmp_path, hours(10))
        with pytest.raises(prices.PriceError, match="BASE/QUOTE"):
            prices.load(data_dir, "binance", "BTCUSDT", "1h")

    def test_a_symbol_the_store_cannot_turn_into_a_filename_is_refused(self, tmp_path):
        data_dir = build_store(tmp_path, hours(10))
        with pytest.raises(store.StoreError):
            prices.load(data_dir, "binance", "BTC/../USDT", "1h")

    @pytest.mark.parametrize("bad", [1.5, "2024-08-01", True])
    def test_a_start_that_is_not_epoch_milliseconds_is_refused(self, tmp_path, bad):
        """A date string is the mistake worth catching by name.

        This layer speaks in integers on purpose -- parsing '2024-08-01' is the
        CLI's job, and doing it in both places is how two parts of one program
        come to disagree about what a date means. But `start_ms="2024-08-01"` is
        exactly what someone will type, so the message says where to convert it.
        """
        data_dir = build_store(tmp_path, hours(10))
        with pytest.raises(prices.PriceError, match="epoch milliseconds"):
            prices.load(data_dir, "binance", "BTC/USDT", "1h", start_ms=bad)

    def test_a_negative_start_is_refused(self, tmp_path):
        """The `match` is the whole test.

        Without it this passes even with the negativity check deleted, because a
        negative start is also before the first stored candle and the coverage
        check a few lines later raises a PriceError of its own. Asserting only
        the exception type asserts almost nothing here -- every wrong value in
        this argument raises something.
        """
        data_dir = build_store(tmp_path, hours(10))
        with pytest.raises(prices.PriceError, match="not be negative"):
            prices.load(data_dir, "binance", "BTC/USDT", "1h", start_ms=-HOUR)


class TestSeveralSymbols:
    def test_symbols_are_kept_apart(self, tmp_path):
        data_dir = build_store(tmp_path, hours(10), symbol="BTC/USDT")
        build_store(tmp_path, hours(20), symbol="ETH/USDT")
        assert len(prices.load(data_dir, "binance", "BTC/USDT", "1h")) == 10
        assert len(prices.load(data_dir, "binance", "ETH/USDT", "1h")) == 20

    def test_timeframes_are_kept_apart(self, tmp_path):
        data_dir = build_store(tmp_path, hours(10), timeframe="1h")
        build_store(tmp_path, [T0 + i * 4 * HOUR for i in range(7)], timeframe="4h")
        assert len(prices.load(data_dir, "binance", "BTC/USDT", "1h")) == 10
        assert len(prices.load(data_dir, "binance", "BTC/USDT", "4h")) == 7
