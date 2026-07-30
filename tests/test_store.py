"""Tests for the Parquet store.

Three properties matter more than the rest, and most of these tests exist to pin
them down:

1. Idempotency. Re-running must add nothing and error on nothing.
2. Crash safety. A kill mid-write must leave the previous file intact.
3. Timestamp fidelity. Epoch milliseconds must survive a round trip as exact
   int64. If pandas ever turns them into floats, candle boundaries stop landing
   on exact values and every comparison downstream becomes subtly unreliable.
"""

import datetime as dt

import pandas as pd
import pytest

from collector import store

HOUR = 3_600_000


def utc(year, month, day, hour=0):
    return int(
        dt.datetime(year, month, day, hour, tzinfo=dt.timezone.utc).timestamp() * 1000
    )


def rows(*timestamps, close=100.0):
    """Build ccxt-shaped OHLCV rows: [ts, open, high, low, close, volume]."""
    return [[ts, 100.0, 110.0, 90.0, close, 5.0] for ts in timestamps]


@pytest.fixture
def path(tmp_path):
    """A candle file path inside a temporary store that does not yet exist."""
    return store.candle_path(tmp_path / "data", "kraken", "BTC/USD", "1h")


class TestSymbolToFilename:
    @pytest.mark.parametrize(
        "symbol,expected",
        [
            ("BTC/USD", "BTC_USD"),
            ("ETH/USDT", "ETH_USDT"),
            ("BTC/USD:USD", "BTC_USD_USD"),  # ccxt notation for a settled contract
        ],
    )
    def test_replaces_separators(self, symbol, expected):
        assert store.symbol_to_filename(symbol) == expected

    def test_normalises_case(self):
        """macOS filesystems are case-insensitive by default, so btc/usd and
        BTC/USD would resolve to one file regardless. Normalising makes that
        deliberate instead of an accident that behaves differently on Linux.
        """
        assert store.symbol_to_filename("btc/usd") == "BTC_USD"

    @pytest.mark.parametrize("symbol", ["", "   ", None, 42, "/", "BTC/"])
    def test_rejects_unusable_symbols(self, symbol):
        with pytest.raises(store.StoreError):
            store.symbol_to_filename(symbol)

    @pytest.mark.parametrize("symbol", ["../../etc/passwd", "..", "BTC/../USD"])
    def test_rejects_path_traversal(self, symbol):
        """Symbols arrive from a config file. A symbol must never be able to
        steer a write outside the store directory."""
        with pytest.raises(store.StoreError):
            store.symbol_to_filename(symbol)


class TestCandlePath:
    def test_layout_is_exchange_symbol_timeframe(self, tmp_path):
        result = store.candle_path(tmp_path / "data", "kraken", "BTC/USD", "1h")
        assert result == tmp_path / "data" / "kraken" / "BTC_USD" / "1h.parquet"

    def test_exchange_name_is_lowercased(self, tmp_path):
        result = store.candle_path(tmp_path / "data", "KRAKEN", "BTC/USD", "1h")
        assert "kraken" in result.parts


class TestReadCandles:
    def test_missing_file_gives_empty_frame_with_correct_schema(self, path):
        """An absent file is a normal state, not an error -- it is what every
        first run looks like. Returning a correctly typed empty frame means the
        caller needs no special case for it.
        """
        frame = store.read_candles(path)
        assert frame.empty
        assert list(frame.columns) == store.COLUMNS
        assert frame["timestamp"].dtype == "int64"

    def test_round_trip_preserves_exact_milliseconds(self, path):
        """A float64 can represent these integers, but any arithmetic done in
        float risks a boundary landing at x.9999. Assert the dtype survives.
        """
        ts = 1_741_942_800_123
        store.append_candles(path, rows(ts))
        frame = store.read_candles(path)
        assert frame["timestamp"].dtype == "int64"
        assert frame["timestamp"].iloc[0] == ts

    def test_returns_rows_sorted_by_timestamp(self, path):
        store.append_candles(path, rows(utc(2025, 3, 14, 3), utc(2025, 3, 14, 1)))
        frame = store.read_candles(path)
        assert frame["timestamp"].is_monotonic_increasing


class TestAppendCandles:
    def test_creates_file_and_parent_directories(self, path):
        assert not path.exists()
        added = store.append_candles(path, rows(utc(2025, 3, 14, 1)))
        assert path.exists()
        assert added == 1

    def test_appending_the_same_rows_twice_adds_nothing(self, path):
        """The core idempotency requirement, stated as plainly as possible."""
        batch = rows(utc(2025, 3, 14, 1), utc(2025, 3, 14, 2))
        assert store.append_candles(path, batch) == 2
        assert store.append_candles(path, batch) == 0
        assert len(store.read_candles(path)) == 2

    def test_second_run_does_not_touch_the_file_at_all(self, path):
        """If there is nothing to add, the file should not be rewritten. Beyond
        saving work, it means an interrupted no-op run cannot corrupt anything,
        because no write was ever started.
        """
        batch = rows(utc(2025, 3, 14, 1))
        store.append_candles(path, batch)
        before = path.stat().st_mtime_ns

        store.append_candles(path, batch)
        assert path.stat().st_mtime_ns == before

    def test_overlapping_batches_keep_one_row_per_timestamp(self, path):
        store.append_candles(path, rows(utc(2025, 3, 14, 1), utc(2025, 3, 14, 2)))
        added = store.append_candles(
            path, rows(utc(2025, 3, 14, 2), utc(2025, 3, 14, 3))
        )
        assert added == 1
        stored = store.read_candles(path)["timestamp"].tolist()
        assert stored == [utc(2025, 3, 14, 1), utc(2025, 3, 14, 2), utc(2025, 3, 14, 3)]

    def test_collision_keeps_the_already_stored_values(self, path):
        """When a timestamp arrives that is already stored, the stored row wins.

        Closed candles do not change, so a differing value means the incoming
        row is the suspect one -- most likely a partially formed candle from a
        re-fetch. Preferring what is already on disk makes repeat runs
        genuinely inert rather than merely non-duplicating.
        """
        ts = utc(2025, 3, 14, 1)
        store.append_candles(path, rows(ts, close=100.0))
        store.append_candles(path, rows(ts, close=999.0))

        frame = store.read_candles(path)
        assert len(frame) == 1
        assert frame["close"].iloc[0] == 100.0

    def test_duplicates_within_a_single_batch_are_collapsed(self, path):
        ts = utc(2025, 3, 14, 1)
        added = store.append_candles(path, rows(ts) + rows(ts))
        assert added == 1
        assert len(store.read_candles(path)) == 1

    def test_empty_batch_is_a_no_op_and_creates_no_file(self, path):
        assert store.append_candles(path, []) == 0
        assert not path.exists()

    def test_rejects_malformed_rows(self, path):
        with pytest.raises(store.StoreError):
            store.append_candles(path, [[utc(2025, 3, 14, 1), 1.0, 2.0]])


class TestCrashSafety:
    def test_failure_before_any_bytes_are_written_leaves_the_file_intact(
        self, path, monkeypatch
    ):
        """The easy case: the write fails before it starts.

        Note that this test alone does NOT prove writes are atomic -- it passes
        even if the store writes straight over the target file, because nothing
        was written. The test below is the one that proves atomicity.
        """
        store.append_candles(path, rows(utc(2025, 3, 14, 1)))
        original = store.read_candles(path)

        def explode(*args, **kwargs):
            raise OSError("disk went away before the write began")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", explode)

        with pytest.raises(OSError):
            store.append_candles(path, rows(utc(2025, 3, 14, 2)))

        pd.testing.assert_frame_equal(store.read_candles(path), original)

    def test_partial_write_cannot_corrupt_the_stored_file(self, path, monkeypatch):
        """The real crash case: bytes land on disk and *then* the process dies.

        This is what actually happens when you press Ctrl-C during a flush. The
        fake below writes junk to whatever path it is handed and then raises, so
        a store that wrote directly over the target would leave an unreadable
        file behind. Because the junk goes to the temporary path instead, the
        previously stored candles survive untouched.

        This test is the reason the temp-file-then-os.replace dance exists. If
        you ever simplify write_candles to a direct to_parquet call, this is the
        test that will stop you.
        """
        store.append_candles(path, rows(utc(2025, 3, 14, 1)))
        original = store.read_candles(path)

        def write_junk_then_die(self, target, *args, **kwargs):
            with open(target, "wb") as handle:
                handle.write(b"PAR1 truncated garbage")
            raise OSError("killed partway through writing")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", write_junk_then_die)

        with pytest.raises(OSError):
            store.append_candles(path, rows(utc(2025, 3, 14, 2)))

        # The original file must still parse and hold exactly what it held before.
        pd.testing.assert_frame_equal(store.read_candles(path), original)

    def test_no_temporary_files_are_left_behind(self, path):
        store.append_candles(path, rows(utc(2025, 3, 14, 1)))
        store.append_candles(path, rows(utc(2025, 3, 14, 2)))
        leftovers = [p.name for p in path.parent.iterdir() if p.suffix != ".parquet"]
        assert leftovers == []

    def test_temporary_file_is_cleaned_up_after_a_failed_write(
        self, path, monkeypatch
    ):
        store.append_candles(path, rows(utc(2025, 3, 14, 1)))

        def explode(*args, **kwargs):
            raise OSError("boom")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", explode)
        with pytest.raises(OSError):
            store.append_candles(path, rows(utc(2025, 3, 14, 2)))

        leftovers = [p.name for p in path.parent.iterdir() if p.suffix != ".parquet"]
        assert leftovers == []


class TestStoredRange:
    def test_empty_store_reports_no_range(self, path):
        assert store.stored_range(path) == (None, None)

    def test_reports_oldest_and_newest_stored_timestamps(self, path):
        store.append_candles(
            path, rows(utc(2025, 3, 14, 5), utc(2025, 3, 14, 1), utc(2025, 3, 14, 3))
        )
        assert store.stored_range(path) == (utc(2025, 3, 14, 1), utc(2025, 3, 14, 5))


class TestNextStart:
    def test_empty_store_starts_from_the_requested_default(self, path):
        default = utc(2024, 8, 1)
        assert store.next_start_ms(path, HOUR, default) == default

    def test_populated_store_resumes_one_timeframe_after_the_last_candle(self, path):
        store.append_candles(path, rows(utc(2025, 3, 14, 1), utc(2025, 3, 14, 2)))
        assert store.next_start_ms(path, HOUR, utc(2024, 8, 1)) == utc(2025, 3, 14, 3)

    def test_resume_point_is_derived_from_the_data_not_a_progress_file(self, path):
        """Resume state has one source of truth: the stored candles themselves.

        A sidecar progress file can disagree with the data after a crash, and
        then there are two answers to 'what do we have' and no way to tell which
        is right. Deleting rows must move the resume point back accordingly.
        """
        store.append_candles(
            path, rows(utc(2025, 3, 14, 1), utc(2025, 3, 14, 2), utc(2025, 3, 14, 3))
        )
        frame = store.read_candles(path)
        store.write_candles(path, frame[frame["timestamp"] < utc(2025, 3, 14, 3)])

        assert store.next_start_ms(path, HOUR, utc(2024, 8, 1)) == utc(2025, 3, 14, 3)


class TestStoredTimestamps:
    def test_empty_store_gives_empty_list(self, path):
        assert store.stored_timestamps(path) == []

    def test_returns_sorted_python_ints(self, path):
        """Gap detection compares these against expected_open_times, which
        produces plain ints. Numpy int64 would compare equal but makes set
        operations and error messages needlessly confusing.
        """
        store.append_candles(path, rows(utc(2025, 3, 14, 3), utc(2025, 3, 14, 1)))
        result = store.stored_timestamps(path)
        assert result == [utc(2025, 3, 14, 1), utc(2025, 3, 14, 3)]
        assert all(type(value) is int for value in result)
