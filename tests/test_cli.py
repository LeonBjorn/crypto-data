"""Tests for the command-line entry point -- the layer that makes all of this runnable.

Two things are worth saying about how these tests are written.

First, they drive `run()` rather than a subprocess. A subprocess would prove that
the script starts, but it could not pin the clock, could not substitute the
exchange, and would need the network guard to be trusted rather than enforced.
`run()` takes `now_ms` and an exchange factory for exactly that reason.

Second, most of these are about the *exit code*, which is the one part of a CLI
that a scheduled run actually reads. Everything else -- the log, the report, the
summary -- is for a person, and a person is not watching at 4am. If the exit code
is wrong then a store with holes in it looks like a successful night's work, so
the codes get more tests than the happy path does.
"""

import json
import logging

import ccxt
import pytest

from collector import cli, report, settings, store
from collector.timeframes import HOUR_MS

START = 1735689600000  # 2025-01-01T00:00:00Z
# 05:00, so 04:00 is the newest closed candle and 00:00..04:00 are expected.
NOW = START + 5 * HOUR_MS
EXPECTED_CANDLES = 5


@pytest.fixture(autouse=True)
def _restore_logging():
    """Undo cli's logging setup after every test in this file.

    configure_logging clears the root logger's handlers and attaches its own,
    including a stream handler bound to whatever sys.stdout was at the time. Under
    pytest that is the capture object for the current test, which is torn down
    afterwards -- so without this, a later test that logs anything writes to a
    closed file and fails somewhere completely unrelated to its own subject.
    """
    root = logging.getLogger()
    saved = list(root.handlers)
    saved_level = root.level
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    for handler in saved:
        root.addHandler(handler)
    root.setLevel(saved_level)


@pytest.fixture
def project(tmp_path):
    """A config file plus data and log directories, and the args pointing at them.

    Returned as a small object rather than a tuple so a test can reach for just
    the part it cares about without unpacking four values it does not use.
    """

    class Project:
        def __init__(self):
            self.root = tmp_path
            self.data_dir = tmp_path / "data"
            self.log_dir = tmp_path / "logs"
            self.config_path = tmp_path / "symbols.json"
            self.write_config()

        def write_config(self, **overrides):
            config = {
                "exchange": "binance",
                "timeframe": "1h",
                "start": "2025-01-01",
                "symbols": ["BTC/USDT"],
            }
            config.update(overrides)
            self.config_path.write_text(json.dumps(config), encoding="utf-8")

        def args(self, *extra):
            return [
                "--config",
                str(self.config_path),
                "--data-dir",
                str(self.data_dir),
                "--log-dir",
                str(self.log_dir),
                *extra,
            ]

        def path_for(self, symbol, timeframe="1h"):
            return store.candle_path(self.data_dir, "binance", symbol, timeframe)

        def report(self):
            return json.loads(report.report_path(self.log_dir).read_text())

    return Project()


def factory_for(exchange):
    """An exchange factory that hands back a prepared fake, ignoring the id.

    The id is still recorded, so a test can assert the CLI asked for the exchange
    the config named rather than a hardcoded one.
    """
    seen = []

    def factory(exchange_id, **_kwargs):
        seen.append(exchange_id)
        return exchange

    factory.seen = seen
    return factory


class TestASuccessfulRun:
    def test_candles_are_stored_and_the_run_reports_success(
        self, project, fake_exchange, candles
    ):
        fake = fake_exchange(candles(START, 6, HOUR_MS))

        code = cli.run(
            project.args(), now_ms=NOW, exchange_factory=factory_for(fake)
        )

        assert code == cli.EXIT_OK
        stored = store.read_candles(project.path_for("BTC/USDT"))
        assert len(stored) == EXPECTED_CANDLES

    def test_the_forming_candle_is_not_stored(self, project, fake_exchange, candles):
        """Requirement 4, checked at the outside edge of the program.

        It is tested thoroughly in test_timeframes and test_backfill, but this is
        the one that would catch the CLI passing the wrong `now` and undoing all
        of it.
        """
        fake = fake_exchange(candles(START, 6, HOUR_MS))

        cli.run(project.args(), now_ms=NOW, exchange_factory=factory_for(fake))

        stored = store.read_candles(project.path_for("BTC/USDT"))
        assert stored["timestamp"].max() == START + 4 * HOUR_MS

    def test_a_second_run_adds_nothing_and_still_succeeds(
        self, project, fake_exchange, candles
    ):
        """The acceptance criterion, stated as a test: re-running immediately
        adds nothing and errors on nothing.

        Worth having at this level even though store and backfill both test
        idempotence themselves, because the CLI is where a wrong start date or a
        rebuilt path would break it.
        """
        fake = fake_exchange(candles(START, 6, HOUR_MS))
        factory = factory_for(fake)

        cli.run(project.args(), now_ms=NOW, exchange_factory=factory)
        first = project.report()["totals"]["candles_added"]

        code = cli.run(project.args(), now_ms=NOW, exchange_factory=factory)
        second = project.report()

        assert first == EXPECTED_CANDLES
        assert code == cli.EXIT_OK
        assert second["totals"]["candles_added"] == 0
        assert second["complete"] is True

    def test_every_symbol_gets_its_own_file(self, project, fake_exchange, candles):
        project.write_config(symbols=["BTC/USDT", "ETH/USDT"])
        fake = fake_exchange(candles(START, 6, HOUR_MS))

        cli.run(project.args(), now_ms=NOW, exchange_factory=factory_for(fake))

        assert project.path_for("BTC/USDT").exists()
        assert project.path_for("ETH/USDT").exists()

    def test_the_exchange_named_in_the_config_is_the_one_built(
        self, project, fake_exchange, candles
    ):
        project.write_config(exchange="bybit")
        factory = factory_for(fake_exchange(candles(START, 6, HOUR_MS)))

        cli.run(project.args(), now_ms=NOW, exchange_factory=factory)

        assert factory.seen == ["bybit"]

    def test_the_client_is_built_once_for_the_whole_run(
        self, project, fake_exchange, candles
    ):
        """ccxt's rate limiter keeps its state on the client, so a new client per
        symbol would reset the spacing between requests and quietly undo half of
        requirement 7.
        """
        project.write_config(symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        factory = factory_for(fake_exchange(candles(START, 6, HOUR_MS)))

        cli.run(project.args(), now_ms=NOW, exchange_factory=factory)

        assert len(factory.seen) == 1


class TestTheReportAndTheLog:
    def test_a_report_is_written_even_when_nothing_is_missing(
        self, project, fake_exchange, candles
    ):
        fake = fake_exchange(candles(START, 6, HOUR_MS))

        cli.run(project.args(), now_ms=NOW, exchange_factory=factory_for(fake))

        assert report.report_path(project.log_dir).exists()
        assert project.report()["complete"] is True

    def test_the_report_records_the_range_that_was_actually_covered(
        self, project, fake_exchange, candles
    ):
        """Not the range that was asked for. The request ends at "now", but the
        run can only reach the last closed candle, and a report claiming to have
        verified up to 05:00 when it stopped at 04:00 overstates itself.
        """
        fake = fake_exchange(candles(START, 6, HOUR_MS))

        cli.run(project.args(), now_ms=NOW, exchange_factory=factory_for(fake))

        written = project.report()
        assert written["requested_start_ms"] == START
        assert written["requested_end_ms"] == START + 4 * HOUR_MS
        assert written["generated_at_ms"] == NOW

    def test_a_log_file_is_written_to_the_log_directory(
        self, project, fake_exchange, candles
    ):
        """Requirement 9. Asserting on the file rather than on caplog, because
        caplog would pass even if the file handler were never attached.
        """
        fake = fake_exchange(candles(START, 6, HOUR_MS))

        cli.run(project.args(), now_ms=NOW, exchange_factory=factory_for(fake))

        log_file = project.log_dir / "collector.log"
        assert "BTC/USDT" in log_file.read_text()

    def test_verbose_adds_the_per_write_detail_and_the_default_does_not(
        self, project, fake_exchange, candles
    ):
        """Both halves in one test, because either alone would pass for the wrong
        reason: asserting only that --verbose shows the line cannot tell whether
        the flag did anything, since the line might be showing all the time.

        store.py's DEBUG lines are what this actually buys -- how many candles each
        write added and how many the file then held.

        The two runs use different symbols so that both have real work to do. The
        first version reran the same symbol, which was already complete the second
        time, so nothing was written, nothing was logged, and the test failed for a
        reason that had nothing to do with --verbose.
        """
        fake = fake_exchange(candles(START, 6, HOUR_MS))
        log_file = project.log_dir / "collector.log"

        cli.run(
            project.args("--symbols", "BTC/USDT"),
            now_ms=NOW,
            exchange_factory=factory_for(fake),
        )
        assert "DEBUG" not in log_file.read_text()

        log_file.unlink()
        cli.run(
            project.args("--verbose", "--symbols", "ETH/USDT"),
            now_ms=NOW,
            exchange_factory=factory_for(fake),
        )
        assert "DEBUG" in log_file.read_text()

    def test_the_summary_reaches_stdout(self, project, fake_exchange, candles, capsys):
        """The one line that many runs will have read and nothing else."""
        fake = fake_exchange(candles(START, 6, HOUR_MS))

        cli.run(project.args(), now_ms=NOW, exchange_factory=factory_for(fake))

        out = capsys.readouterr().out
        assert "nothing missing" in out
        assert "1 symbol, 5 candle(s) added" in out

    def test_the_summary_says_what_is_missing_without_being_asked(
        self, project, fake_exchange, candles, capsys
    ):
        fake = fake_exchange(candles(START, 6, HOUR_MS), holes=[START + 2 * HOUR_MS])

        cli.run(project.args(), now_ms=NOW, exchange_factory=factory_for(fake))

        out = capsys.readouterr().out
        assert "1 candle(s) missing" in out
        assert "nothing missing" not in out


class TestExitCodes:
    def test_missing_candles_make_the_run_exit_non_zero(
        self, project, fake_exchange, candles
    ):
        """The most important test in the file.

        A nightly run that leaves holes in the store and exits 0 is the silent
        failure this whole project is built to avoid. It has to be visible to
        something that reads nothing but the exit status.
        """
        fake = fake_exchange(candles(START, 6, HOUR_MS), holes=[START + 2 * HOUR_MS])

        code = cli.run(project.args(), now_ms=NOW, exchange_factory=factory_for(fake))

        assert code == cli.EXIT_INCOMPLETE
        assert project.report()["complete"] is False

    def test_an_incomplete_store_is_distinguished_from_a_broken_run(
        self, project, fake_exchange, candles
    ):
        """Two different exit codes because they need two different responses.

        "The tool worked and the data has holes the venue does not have" calls for
        a look at the report. "The tool could not run" calls for a look at the
        config or the network. One code for both would flatten that distinction
        away at the only moment it is cheap to make.
        """
        holed = fake_exchange(candles(START, 6, HOUR_MS), holes=[START + 2 * HOUR_MS])
        incomplete = cli.run(
            project.args(), now_ms=NOW, exchange_factory=factory_for(holed)
        )

        broken = cli.run(["--config", str(project.root / "gone.json")], now_ms=NOW)

        assert incomplete == cli.EXIT_INCOMPLETE
        assert broken == cli.EXIT_FAILED
        assert incomplete != broken

    def test_a_missing_config_file_fails_cleanly(self, project, capsys):
        code = cli.run(["--config", str(project.root / "gone.json")], now_ms=NOW)

        assert code == cli.EXIT_FAILED
        # A message, not a traceback. A stack trace for a mistyped path teaches
        # you to skim past error output, which is a habit worth not forming.
        assert "gone.json" in capsys.readouterr().err

    def test_an_unsupported_timeframe_fails_before_anything_is_created(
        self, project, capsys
    ):
        project.write_config(timeframe="1w")

        code = cli.run(project.args(), now_ms=NOW)

        assert code == cli.EXIT_FAILED
        assert "1w" in capsys.readouterr().err
        # Nothing was half-built on the way to failing.
        assert not project.data_dir.exists()

    def test_a_start_date_in_the_future_is_refused(self, project, capsys):
        """Otherwise nothing is missing from an empty range, every symbol is
        complete, and a run that did nothing at all exits 0 -- the report would
        be technically true and completely useless.
        """
        project.write_config(start="2099-01-01")

        code = cli.run(project.args(), now_ms=NOW)

        assert code == cli.EXIT_FAILED
        assert "future" in capsys.readouterr().err

    def test_an_unreadable_config_does_not_reach_the_exchange(self, project):
        """No client, no directories, no report. Failing before any side effect
        means a bad config can be fixed and rerun with nothing to clean up.
        """
        project.write_config(timeframe="1w")

        def explode(*_args, **_kwargs):
            raise AssertionError("the exchange must not be built at all")

        cli.run(project.args(), now_ms=NOW, exchange_factory=explode)

        assert not report.report_path(project.log_dir).exists()


class TestOneSymbolFailing:
    def test_the_remaining_symbols_are_still_attempted(
        self, project, fake_exchange, candles
    ):
        """One unavailable symbol should not cost you the other four backfills.

        BadSymbol is used rather than a network error on purpose: it is permanent,
        so fetch_page raises it immediately instead of retrying through real
        backoff, and the suite's sleep guard stays satisfied.
        """
        project.write_config(symbols=["BTC/USDT", "ETH/USDT"])
        fake = fake_exchange(
            candles(START, 6, HOUR_MS),
            fail_times=1,
            error=ccxt.BadSymbol("no such market"),
        )

        cli.run(project.args(), now_ms=NOW, exchange_factory=factory_for(fake))

        # BTC failed on the first call; ETH went through afterwards.
        assert not project.path_for("BTC/USDT").exists()
        assert project.path_for("ETH/USDT").exists()

    def test_the_failure_is_recorded_in_the_report(
        self, project, fake_exchange, candles
    ):
        project.write_config(symbols=["BTC/USDT", "ETH/USDT"])
        fake = fake_exchange(
            candles(START, 6, HOUR_MS),
            fail_times=1,
            error=ccxt.BadSymbol("no such market"),
        )

        cli.run(project.args(), now_ms=NOW, exchange_factory=factory_for(fake))

        written = project.report()
        assert [entry["symbol"] for entry in written["errors"]] == ["BTC/USDT"]
        assert "BadSymbol" in written["errors"][0]["error"]
        assert written["complete"] is False

    def test_a_failure_makes_the_run_exit_non_zero(
        self, project, fake_exchange, candles
    ):
        """Even though the symbol that did run was perfectly clean. Continuing
        past a failure is a convenience, not an acquittal.
        """
        project.write_config(symbols=["BTC/USDT", "ETH/USDT"])
        fake = fake_exchange(
            candles(START, 6, HOUR_MS),
            fail_times=1,
            error=ccxt.BadSymbol("no such market"),
        )

        code = cli.run(project.args(), now_ms=NOW, exchange_factory=factory_for(fake))

        assert code == cli.EXIT_INCOMPLETE

    def test_every_symbol_failing_still_writes_a_report(
        self, project, fake_exchange, candles
    ):
        """The run that most needs a report is the one where nothing worked."""
        fake = fake_exchange(
            candles(START, 6, HOUR_MS),
            fail_times=99,
            error=ccxt.BadSymbol("no such market"),
        )

        code = cli.run(project.args(), now_ms=NOW, exchange_factory=factory_for(fake))

        assert code == cli.EXIT_INCOMPLETE
        assert project.report()["totals"]["symbols_failed"] == 1


class TestInterruption:
    def test_ctrl_c_exits_cleanly_without_a_traceback(
        self, project, fake_exchange, candles, capsys
    ):
        """Requirement 3 says a run must be killable mid-fetch. store.py makes
        that safe for the data; this makes it civil for the person.
        """

        class Interrupting:
            id = "fake"

            def fetch_ohlcv(self, *_args, **_kwargs):
                raise KeyboardInterrupt

        code = cli.run(
            project.args(), now_ms=NOW, exchange_factory=factory_for(Interrupting())
        )

        assert code == cli.EXIT_FAILED
        assert "interrupted" in capsys.readouterr().err.lower()

    def test_an_interrupted_run_leaves_the_previous_report_alone(
        self, project, fake_exchange, candles
    ):
        """A report describes a run that finished. Overwriting a good one with a
        partial one would replace a true statement with a narrower one, and the
        next reader has no way to tell which they are holding.
        """
        fake = fake_exchange(candles(START, 6, HOUR_MS))
        cli.run(project.args(), now_ms=NOW, exchange_factory=factory_for(fake))
        before = report.report_path(project.log_dir).read_text()

        class Interrupting:
            id = "fake"

            def fetch_ohlcv(self, *_args, **_kwargs):
                raise KeyboardInterrupt

        project.write_config(start="2024-06-01")
        cli.run(
            project.args(), now_ms=NOW, exchange_factory=factory_for(Interrupting())
        )

        assert report.report_path(project.log_dir).read_text() == before


class TestOverrides:
    def test_symbols_can_be_overridden_on_the_command_line(
        self, project, fake_exchange, candles
    ):
        project.write_config(symbols=["BTC/USDT"])
        fake = fake_exchange(candles(START, 6, HOUR_MS))

        cli.run(
            project.args("--symbols", "ETH/USDT", "SOL/USDT"),
            now_ms=NOW,
            exchange_factory=factory_for(fake),
        )

        assert project.path_for("ETH/USDT").exists()
        assert project.path_for("SOL/USDT").exists()
        assert not project.path_for("BTC/USDT").exists()

    def test_the_start_can_be_overridden(self, project, fake_exchange, candles):
        fake = fake_exchange(candles(START, 6, HOUR_MS))

        cli.run(
            project.args("--start", "2025-01-01T02:00"),
            now_ms=NOW,
            exchange_factory=factory_for(fake),
        )

        assert project.report()["requested_start_ms"] == START + 2 * HOUR_MS

    def test_the_timeframe_can_be_overridden(self, project, fake_exchange, candles):
        fake = fake_exchange(candles(START, 6, HOUR_MS))

        cli.run(
            project.args("--timeframe", "4h"),
            now_ms=NOW,
            exchange_factory=factory_for(fake),
        )

        assert project.report()["timeframe"] == "4h"
        assert project.path_for("BTC/USDT", timeframe="4h").exists()

    def test_max_gaps_zero_leaves_older_holes_alone(
        self, project, fake_exchange, candles
    ):
        """Zero has to mean "repair nothing", not "unset". The classic version of
        this bug treats 0 as falsey and quietly restores the default of 20.

        Also proves max_gaps is actually threaded through to backfill: with the
        trailing range already stored and the only hole capped away, a working
        implementation makes no requests at all.
        """
        rows = candles(START, 5, HOUR_MS)
        del rows[2]  # a hole at 02:00, with 03:00 and 04:00 present after it
        store.append_candles(project.path_for("BTC/USDT"), rows)

        fake = fake_exchange(candles(START, 6, HOUR_MS))
        code = cli.run(
            project.args("--max-gaps", "0"),
            now_ms=NOW,
            exchange_factory=factory_for(fake),
        )

        written = project.report()
        assert code == cli.EXIT_INCOMPLETE
        assert written["totals"]["candles_not_attempted"] == 1
        assert written["totals"]["candles_missing"] == 0
        assert fake.calls == []

    def test_a_bad_symbol_on_the_command_line_is_refused(self, project, capsys):
        code = cli.run(project.args("--symbols", "BTC"), now_ms=NOW)

        assert code == cli.EXIT_FAILED
        assert "BASE/QUOTE" in capsys.readouterr().err


class TestTheParser:
    def test_symbols_accepts_several_values(self):
        args = cli.build_parser().parse_args(["--symbols", "BTC/USDT", "ETH/USDT"])

        assert args.symbols == ["BTC/USDT", "ETH/USDT"]

    def test_the_config_path_has_a_default_so_the_common_case_needs_no_flags(self):
        args = cli.build_parser().parse_args([])

        assert args.config == settings.DEFAULT_CONFIG_PATH

    def test_unspecified_overrides_are_none_rather_than_a_second_set_of_defaults(self):
        """If the parser supplied its own defaults for these, they would always
        win over the config file and the file would become decorative.
        """
        args = cli.build_parser().parse_args([])

        assert args.symbols is None
        assert args.timeframe is None
        assert args.start is None
        assert args.exchange is None
        assert args.max_gaps is None

    def test_an_unknown_flag_is_rejected(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["--nonsense", "7"])

    def test_an_abbreviated_flag_is_rejected(self):
        """argparse accepts any unambiguous prefix by default, so this test
        originally failed: `--symbol` was silently accepted as `--symbols`.

        Turning that off is worth the extra typing. An abbreviation is unambiguous
        only with respect to the flags that exist today -- adding `--database`
        later would break every script using `--data`, and the breakage would look
        like the script's fault. It also sat oddly beside the config file, which
        refuses the key "symbol" outright.
        """
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["--symbol", "BTC/USDT"])

    def test_credentials_cannot_be_passed_even_by_accident(self):
        """Requirement 1 at the outermost layer. exchange.py never sends a key;
        this is the belt-and-braces check that there is no way to offer one.
        """
        for flag in ("--api-key", "--secret", "--password"):
            with pytest.raises(SystemExit):
                cli.build_parser().parse_args([flag, "whatever"])
