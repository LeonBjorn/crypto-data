"""Tests for the gap report -- the machine-readable record of what is missing.

The report exists because requirement 6 says missing candles must be reported
rather than failing silently, and a log line is not really a report: nothing can
read it, and it scrolls away. This file is what a backtester, or you in six
months, can check before trusting a range of data.

Two properties matter more than the exact shape of the JSON, and most of the
tests below are about them:

  - It must never understate what is missing. Every way of getting that wrong is
    a silent failure, which is the one category of bug this project is most
    concerned with.
  - It must never be left half-written. A truncated JSON file that happens to
    parse is worse than no file, because it looks like an answer.
"""

import datetime as dt
import json

import pytest

from collector import gaps, report
from collector.backfill import Result

HOUR = 3_600_000


def utc(*args):
    return int(dt.datetime(*args, tzinfo=dt.timezone.utc).timestamp() * 1000)


def gap(start, count, timeframe_ms=HOUR):
    """A gap of `count` candles beginning at `start`."""
    return gaps.Gap(start, start + (count - 1) * timeframe_ms, count)


def clean_result(symbol="BTC/USDT", added=100, pages=2):
    return Result(symbol, added, pages, [], [], [])


@pytest.fixture
def log_dir(tmp_path):
    return tmp_path / "logs"


RUN = {
    "exchange": "binance",
    "timeframe": "1h",
    "start_ms": utc(2024, 8, 1),
    "end_ms": utc(2026, 7, 30),
    "generated_at_ms": utc(2026, 7, 30, 12),
}


class TestGapToDict:
    def test_carries_both_a_readable_time_and_epoch_milliseconds(self):
        """Deliberate redundancy, because the file has two different audiences.

        A person opening it needs to know when the hole was without doing
        arithmetic; a program reading it needs a number it can compare without
        parsing a date format. Storing only one forces the other to work for it,
        and the cost here is a few bytes.
        """
        entry = report.gap_to_dict(gap(utc(2025, 3, 4, 7), 2))

        assert entry["start"] == "2025-03-04 07:00 UTC"
        assert entry["end"] == "2025-03-04 08:00 UTC"
        assert entry["start_ms"] == utc(2025, 3, 4, 7)
        assert entry["end_ms"] == utc(2025, 3, 4, 8)
        assert entry["candles"] == 2

    def test_a_single_candle_gap_has_equal_start_and_end(self):
        entry = report.gap_to_dict(gap(utc(2025, 3, 4, 7), 1))

        assert entry["start_ms"] == entry["end_ms"]
        assert entry["candles"] == 1


class TestBuildReport:
    def test_a_clean_run_is_recorded_as_complete(self):
        """The positive statement is the point of writing a report on a clean run:
        "as of this time, nothing was missing" is information, and it is not the
        same as the absence of a file."""
        built = report.build_report([clean_result()], **RUN)

        assert built["complete"] is True
        assert built["totals"]["candles_missing"] == 0
        assert built["symbols"][0]["complete"] is True
        assert built["symbols"][0]["missing"] == []

    def test_run_metadata_is_recorded(self):
        """Without these, a gap list is unreadable a month later -- missing from
        what, over which range, on which venue."""
        built = report.build_report([clean_result()], **RUN)

        assert built["exchange"] == "binance"
        assert built["timeframe"] == "1h"
        assert built["requested_start"] == "2024-08-01 00:00 UTC"
        assert built["requested_end"] == "2026-07-30 00:00 UTC"
        assert built["generated_at"] == "2026-07-30 12:00 UTC"
        assert built["requested_start_ms"] == utc(2024, 8, 1)

    def test_a_schema_version_is_recorded(self):
        """So that a reader written against this shape can refuse a later one
        loudly instead of quietly misreading it."""
        built = report.build_report([clean_result()], **RUN)

        assert built["schema"] == 2

    def test_missing_candles_are_listed_per_symbol(self):
        hole = gap(utc(2025, 3, 4, 7), 2)
        result = Result("ETH/USDT", 50, 1, [], [hole], [])

        built = report.build_report([result], **RUN)
        entry = built["symbols"][0]

        assert entry["symbol"] == "ETH/USDT"
        assert entry["complete"] is False
        assert entry["missing"] == [report.gap_to_dict(hole)]

    def test_a_symbol_with_missing_candles_makes_the_whole_run_incomplete(self):
        """One bad symbol out of five has to be visible at the top of the file.
        Anything else means a reader has to scan every symbol to find out whether
        the run is trustworthy, and eventually one of them will not bother.
        """
        results = [
            clean_result("BTC/USDT"),
            Result("ETH/USDT", 50, 1, [], [gap(utc(2025, 3, 4, 7), 2)], []),
            clean_result("SOL/USDT"),
        ]

        built = report.build_report(results, **RUN)

        assert built["complete"] is False
        assert [s["complete"] for s in built["symbols"]] == [True, False, True]

    def test_gaps_not_attempted_are_kept_separate_from_missing_ones(self):
        """"We asked and the venue had nothing" and "we ran out of budget and did
        not ask" are different facts with different fixes. Merging them would let
        a capped run look like a permanently damaged store.
        """
        asked = gap(utc(2025, 3, 4, 7), 2)
        skipped = gap(utc(2025, 1, 1), 3)
        result = Result("ETH/USDT", 50, 1, [], [asked], [skipped])

        built = report.build_report([result], **RUN)
        entry = built["symbols"][0]

        assert entry["missing"] == [report.gap_to_dict(asked)]
        assert entry["not_attempted"] == [report.gap_to_dict(skipped)]

    def test_a_run_that_only_skipped_gaps_is_still_incomplete(self):
        """Not asking is not the same as being complete. A run that hit the cap
        knows about holes it left alone, and hiding that behind complete: true
        would be the silent failure requirement 6 exists to prevent.
        """
        result = Result("ETH/USDT", 50, 1, [], [], [gap(utc(2025, 1, 1), 3)])

        built = report.build_report([result], **RUN)

        assert built["complete"] is False
        assert built["symbols"][0]["complete"] is False

    def test_totals_count_candles_not_gaps(self):
        """Three holes of one candle and one hole of three candles are very
        different situations, and a count of gaps cannot tell them apart."""
        results = [
            Result("BTC/USDT", 10, 1, [], [gap(utc(2025, 3, 4), 5)], []),
            Result("ETH/USDT", 10, 1, [], [gap(utc(2025, 5, 1), 2)], [gap(utc(2025, 1, 1), 4)]),
        ]

        built = report.build_report(results, **RUN)

        assert built["totals"]["candles_missing"] == 7
        assert built["totals"]["candles_not_attempted"] == 4
        assert built["totals"]["gaps_missing"] == 2
        assert built["totals"]["gaps_not_attempted"] == 1

    def test_totals_sum_the_work_done(self):
        results = [
            Result("BTC/USDT", 500, 3, [gap(utc(2025, 1, 1), 1)], [], []),
            Result("ETH/USDT", 250, 2, [], [], []),
        ]

        built = report.build_report(results, **RUN)

        assert built["totals"]["symbols"] == 2
        assert built["totals"]["candles_added"] == 750
        assert built["totals"]["pages_fetched"] == 5
        assert built["totals"]["gaps_repaired"] == 1

    def test_symbols_appear_in_the_order_they_were_run(self):
        """So the file lines up with the log, which is how the two get read
        together when something has gone wrong.

        The order below is deliberately not alphabetical. The first version of
        this test used BTC, ETH, SOL, which is both the run order and the sorted
        order, so it could not tell the two apart -- sorting the list broke
        nothing. A test whose input satisfies two different rules cannot
        distinguish between them.
        """
        results = [clean_result(s) for s in ("SOL/USDT", "BTC/USDT", "XRP/USDT")]

        built = report.build_report(results, **RUN)

        assert [s["symbol"] for s in built["symbols"]] == [
            "SOL/USDT",
            "BTC/USDT",
            "XRP/USDT",
        ]

    def test_a_run_covering_no_symbols_is_not_claimed_to_be_complete(self):
        """An empty run has verified nothing, so it cannot report success.

        This is the degenerate case that an `all()` over an empty list gets
        wrong: all([]) is True, so the natural implementation would cheerfully
        declare a run that did nothing at all to be complete.
        """
        built = report.build_report([], **RUN)

        assert built["complete"] is False
        assert built["totals"]["symbols"] == 0


class TestFailedSymbols:
    """A symbol whose fetch raised is the largest possible understatement.

    The CLI keeps going when one symbol fails, because a single bad symbol should
    not deny you four good backfills. But that leaves a symbol with no Result at
    all, and if the report simply omitted it, a run where the network died on the
    first of five symbols would produce a file saying four symbols, nothing
    missing, complete: true. Every word of that is accurate and the whole is a
    lie, which is precisely the failure mode requirement 6 exists to prevent.
    """

    def test_a_failure_is_recorded_with_its_reason(self):
        built = report.build_report(
            [clean_result("BTC/USDT")],
            errors=[{"symbol": "ETH/USDT", "error": "connection reset"}],
            **RUN,
        )

        assert built["errors"] == [
            {"symbol": "ETH/USDT", "error": "connection reset"}
        ]
        assert built["totals"]["symbols_failed"] == 1

    def test_a_failure_makes_the_run_incomplete_even_when_every_result_is_clean(self):
        """The one that matters. Every symbol that returned a Result returned a
        clean one, so an implementation that only consults `symbols` reports
        complete: true.
        """
        built = report.build_report(
            [clean_result("BTC/USDT"), clean_result("SOL/USDT")],
            errors=[{"symbol": "ETH/USDT", "error": "connection reset"}],
            **RUN,
        )

        assert built["complete"] is False

    def test_a_failed_symbol_is_not_counted_among_those_that_succeeded(self):
        """`symbols` means "was checked", so a symbol that blew up cannot be in
        it -- otherwise the count of what was verified is inflated by the count
        of what was not.
        """
        built = report.build_report(
            [clean_result("BTC/USDT")],
            errors=[{"symbol": "ETH/USDT", "error": "connection reset"}],
            **RUN,
        )

        assert built["totals"]["symbols"] == 1
        assert [s["symbol"] for s in built["symbols"]] == ["BTC/USDT"]

    def test_no_failures_still_records_an_empty_list(self):
        """Present and empty rather than absent, so a reader can check
        `report["errors"]` without first checking whether the key exists. An
        optional key is a key somebody forgets to look for.
        """
        built = report.build_report([clean_result()], **RUN)

        assert built["errors"] == []
        assert built["totals"]["symbols_failed"] == 0

    def test_a_run_where_everything_failed_is_not_complete(self):
        built = report.build_report(
            [], errors=[{"symbol": "BTC/USDT", "error": "connection reset"}], **RUN
        )

        assert built["complete"] is False
        assert built["totals"]["symbols"] == 0
        assert built["totals"]["symbols_failed"] == 1

    def test_the_summary_names_the_failures(self):
        line = report.summarise(
            report.build_report(
                [clean_result("BTC/USDT")],
                errors=[{"symbol": "ETH/USDT", "error": "connection reset"}],
                **RUN,
            )
        )

        assert "1 symbol(s) failed" in line
        # A failed symbol left everything it holds unverified, so the summary
        # cannot pass the run off as clean.
        assert "nothing missing" not in line


class TestWriteReport:
    def test_writes_valid_json_that_reads_back_identically(self, log_dir):
        built = report.build_report([clean_result()], **RUN)

        path = report.write_report(log_dir, built)

        assert json.loads(path.read_text()) == built

    def test_creates_the_log_directory_if_it_is_missing(self, log_dir):
        assert not log_dir.exists()

        report.write_report(log_dir, report.build_report([clean_result()], **RUN))

        assert log_dir.is_dir()

    def test_the_second_run_replaces_the_first(self, log_dir):
        """The file answers "what is missing now", so a stale entry from a
        previous run would be actively misleading."""
        first = report.build_report(
            [Result("BTC/USDT", 0, 1, [], [gap(utc(2025, 1, 1), 3)], [])], **RUN
        )
        report.write_report(log_dir, first)

        second = report.build_report([clean_result()], **RUN)
        path = report.write_report(log_dir, second)

        written = json.loads(path.read_text())
        assert written["complete"] is True
        assert written["symbols"][0]["missing"] == []

    def test_leaves_no_temporary_file_behind(self, log_dir):
        report.write_report(log_dir, report.build_report([clean_result()], **RUN))

        assert [p.name for p in log_dir.iterdir()] == ["gaps.json"]

    def test_a_failure_partway_through_leaves_the_previous_report_intact(
        self, log_dir, monkeypatch
    ):
        """The reason for writing to a temp file and moving it into place.

        A report is not expensive to regenerate, so durability is not the point;
        atomicity is. A half-written file can still parse as JSON -- just with
        fewer symbols in it -- and would then be read as an authoritative
        statement that less is missing than really is.
        """
        good = report.build_report(
            [Result("BTC/USDT", 0, 1, [], [gap(utc(2025, 1, 1), 3)], [])], **RUN
        )
        path = report.write_report(log_dir, good)

        def explode(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(report.os, "replace", explode)

        with pytest.raises(OSError):
            report.write_report(log_dir, report.build_report([clean_result()], **RUN))

        # The old report is still there, still correct, still parseable.
        survived = json.loads(path.read_text())
        assert survived["symbols"][0]["missing"][0]["candles"] == 3
        assert [p.name for p in log_dir.iterdir()] == ["gaps.json"]

    def test_the_file_is_indented_so_a_person_can_read_it(self, log_dir):
        """It is checked by a person far more often than by a program, and a
        single-line JSON blob of five symbols is unreadable in a terminal.

        `"\\n" in text` was the first assertion and it was worthless: the writer
        appends a trailing newline, so even single-line output contains one. The
        test has to look for an indented key.
        """
        path = report.write_report(log_dir, report.build_report([clean_result()], **RUN))
        text = path.read_text()

        assert '\n  "schema": 2' in text
        assert text.count("\n") > 10


class TestSummarise:
    def test_a_clean_run_says_so_plainly(self):
        built = report.build_report([clean_result()], **RUN)

        assert report.summarise(built) == (
            "1 symbol, 100 candle(s) added, nothing missing."
        )

    def test_missing_candles_are_named_in_the_summary(self):
        """This line goes to stdout at the end of a run, and it is the only thing
        many runs will ever have read. It has to carry the bad news itself rather
        than referring the reader elsewhere."""
        results = [
            clean_result("BTC/USDT"),
            Result("ETH/USDT", 50, 1, [], [gap(utc(2025, 3, 4, 7), 2)], []),
        ]

        line = report.summarise(report.build_report(results, **RUN))

        assert "2 candle(s) missing" in line
        assert "1 gap(s)" in line
        # And it must not also claim the opposite. A line reading "2 candle(s)
        # missing ... nothing missing" is worse than either half alone, and
        # asserting only on what should be present cannot catch that.
        assert "nothing missing" not in line

    def test_skipped_gaps_are_mentioned_separately(self):
        result = Result("ETH/USDT", 50, 1, [], [], [gap(utc(2025, 1, 1), 4)])

        line = report.summarise(report.build_report([result], **RUN))

        assert "4 candle(s) in 1 gap(s) not attempted" in line
        assert "nothing missing" not in line
