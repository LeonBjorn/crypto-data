"""Tests for the research command -- the layer that makes milestone 2 runnable.

Like the collector's CLI tests these drive `run()` rather than a subprocess, so
the store can be built in a temp directory and the output read back. Unlike
them, they are mostly not about the exit code, because this tool has only two
and neither carries a finding. They are about what gets *printed*, which for a
research tool is the whole product.

Three of these tests are really about design decisions rather than code, and are
written so that reversing the decision breaks them loudly:

    - the bare command prints every comparison, not one
    - a rule sitting at the top of the distribution still exits 0
    - the guard runs before anything is measured

The first two exist because the failure mode of this tool is not a crash, it is
a person reading one flattering number out of twelve and believing it. The third
exists because a rule that reads the future produces beautiful output.
"""

import json

import numpy as np
import pandas as pd
import pytest

from collector import store
from signals import cli, evaluate, rules, trades

HOUR = 3_600_000
T0 = 1_722_470_400_000  # 2024-08-01T00:00:00Z

# Long enough that the longest default hold (168 bars) still leaves a population
# to sample from, and short enough that a whole suite of runs stays quick.
BARS = 600


def candle_rows(count, *, seed, start=T0):
    """A deterministic random walk, as OHLCV rows.

    A straight ramp would be easier to reason about and useless here: no rule
    fires on it in an interesting way and every percentile comes out degenerate.
    What these tests need is a series with enough shape that every rule triggers,
    and a seed so that it is the same series every time.

    Volume is tied to the size of each bar's move, in *either* direction, so the
    volume-confirmed rules fire here -- the downside one included. Spiking only
    on up-bars leaves every down-bar at flat volume, which silently starves
    breakdown-volume of signals, and a rule that never fires is refused by the
    lookahead guard rather than passed. That failure is correct and its cause is
    nothing to do with what any test here is checking.
    """
    rng = np.random.default_rng(seed)
    rows = []
    price = 100.0
    for index in range(count):
        opened = price
        closed = max(1.0, opened + rng.normal(0.0, 1.0))
        rows.append(
            [
                start + index * HOUR,
                opened,
                max(opened, closed) + abs(rng.normal(0.0, 0.3)),
                min(opened, closed) - abs(rng.normal(0.0, 0.3)),
                closed,
                10.0 + 800.0 * abs(closed - opened),
            ]
        )
        price = closed
    return rows


def build_store(root, symbols=("BTC/USDT",), *, bars=BARS, exchange="binance", timeframe="1h"):
    """Write one candle file per symbol and return the data directory."""
    data_dir = root / "data"
    for offset, symbol in enumerate(symbols):
        path = store.candle_path(data_dir, exchange, symbol, timeframe)
        frame = store.candles_to_frame(candle_rows(bars, seed=offset))
        store.write_candles(path, frame)
    return data_dir


@pytest.fixture
def project(tmp_path):
    """A config file and a populated store, plus the args that point at them."""

    class Project:
        def __init__(self):
            self.data_dir = build_store(tmp_path, ("BTC/USDT", "ETH/USDT"))
            self.config_path = tmp_path / "symbols.json"
            self.write_config()

        def write_config(self, **overrides):
            config = {
                "exchange": "binance",
                "timeframe": "1h",
                "start": "2024-08-01",
                "symbols": ["BTC/USDT", "ETH/USDT"],
            }
            config.update(overrides)
            self.config_path.write_text(json.dumps(config), encoding="utf-8")

        def args(self, *extra):
            # --draws is pinned low in most tests because a thousand draws per
            # cell is the right default for a real run and pure waiting here.
            return [
                "--config",
                str(self.config_path),
                "--data-dir",
                str(self.data_dir),
                "--draws",
                "20",
                *extra,
            ]

    return Project()


def rows_for(printed, labels):
    """The indented table rows whose first field is one of `labels`, as fields.

    Matched structurally rather than by searching for the numbers, because a
    hold of 168 is also a substring of a mean of -0.168% and a test that reads
    `"168" not in printed` passes or fails on the price data.

    The indent requirement matters: the per-symbol section's heading starts
    with a rule name too, and it is not a row.
    """
    labels = set(labels)
    rows = []
    for line in printed.splitlines():
        fields = line.split()
        if line.startswith("  ") and fields and fields[0] in labels:
            rows.append(fields)
    return rows


def rule_rows(printed):
    """The data rows of the pooled comparison table."""
    return rows_for(printed, rules.RULES)


# The columns of a row, once split. Named because a test that says `fields[4]`
# is a test nobody can check by reading it.
HOLD = 1
TRADES = 2
RULE_MEAN = 3
EVERY_BAR = 4
PERCENTILE = 5


def percent(field):
    return float(field.rstrip("%"))


def peeking_rule(candles, *, ahead=25):
    """Buys when the price 25 bars from now is higher. Cannot lose, cannot exist."""
    closes = candles["close"].to_numpy()
    fired = np.zeros(len(candles), dtype=bool)
    fired[: len(candles) - ahead] = closes[ahead:] > closes[:-ahead]
    return pd.Series(fired, index=candles.index, name="peeker")


@pytest.fixture
def peeker(monkeypatch):
    """Register a rule that reads the future, for the run's lifetime only."""
    monkeypatch.setitem(
        rules.RULES,
        "peeker",
        rules.Rule(
            name="peeker",
            function=peeking_rule,
            description="reads 25 bars ahead",
            columns=rules.CLOSE_ONLY,
        ),
    )
    return "peeker"


def silent_rule(candles):
    """Never fires. Passes every causality check by having nothing to check."""
    return pd.Series(False, index=candles.index, name="silent")


@pytest.fixture
def silent(monkeypatch):
    monkeypatch.setitem(
        rules.RULES,
        "silent",
        rules.Rule(
            name="silent",
            function=silent_rule,
            description="never fires",
            columns=rules.CLOSE_ONLY,
        ),
    )
    return "silent"


class TestTheArgumentSurface:
    def test_no_flag_can_be_abbreviated(self):
        """Same reasoning as the collector: a prefix is a flag that works by luck."""
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["--rul", "breakout"])

    def test_every_registered_rule_is_offered_by_name(self):
        parser = cli.build_parser()
        for name in rules.names():
            assert parser.parse_args(["--rule", name]).rule == [name]

    def test_a_rule_that_does_not_exist_is_refused(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["--rule", "moon-phase"])

    def test_the_overrides_default_to_none_so_the_config_still_counts(self):
        args = cli.build_parser().parse_args([])
        assert args.exchange is None
        assert args.timeframe is None
        assert args.symbols is None
        assert args.rule is None
        assert args.hold is None

    def test_the_measurement_settings_default_to_the_agreed_values(self):
        args = cli.build_parser().parse_args([])
        assert args.draws == evaluate.DEFAULT_DRAWS
        assert args.seed == evaluate.DEFAULT_SEED


class TestTheWholeTable:
    """The bare command shows every comparison, which is the point of it."""

    def test_it_prints_every_rule_at_every_hold(self, project, capsys):
        assert cli.run(project.args()) == cli.EXIT_OK
        rows = rule_rows(capsys.readouterr().out)
        got = {(fields[0], int(fields[HOLD])) for fields in rows}
        assert got == {
            (name, hold) for name in rules.names() for hold in trades.DEFAULT_HOLDS
        }

    def test_it_shows_the_rule_and_the_benchmark_side_by_side(self, project, capsys):
        cli.run(project.args())
        printed = capsys.readouterr().out
        assert "rule mean" in printed
        assert "every bar" in printed
        assert "percentile" in printed

    def test_it_counts_the_comparisons_it_just_made(self, project, capsys):
        """The line that stops one good number from looking like a discovery.

        Twelve cells is twelve chances for noise to clear the 95th percentile,
        and the tool says so out loud in the same breath as the table, because
        a caveat somewhere else is a caveat nobody reads.
        """
        cli.run(project.args())
        printed = capsys.readouterr().out
        cells = len(rules.names()) * len(trades.DEFAULT_HOLDS)
        assert "above the 95th percentile" in printed
        assert f"of {cells}" in printed
        assert "by chance" in printed

    def test_it_names_the_symbols_that_went_into_the_pool(self, project, capsys):
        cli.run(project.args())
        printed = capsys.readouterr().out
        assert "BTC/USDT" in printed
        assert "ETH/USDT" in printed

    def test_it_says_how_much_history_it_measured(self, project, capsys):
        """The header line specifically.

        Asserting on `"600 candles"` alone passes on the guard's own
        `on BTC/USDT (600 candles)`, which would leave the header free to say
        anything at all.
        """
        cli.run(project.args())
        assert f"{BARS} candles per symbol" in capsys.readouterr().out

    def test_the_pool_is_every_symbol_together(self, project, capsys):
        """A pooled row is the symbols added up, not the first one of them.

        Silently pooling one symbol is invisible in a pooled table -- the
        numbers stay plausible, they are just about BTC. The per-symbol
        breakdown is the only place the arithmetic can be checked.
        """
        cli.run(project.args("--rule", "breakout", "--hold", "24", "--per-symbol"))
        printed = capsys.readouterr().out

        pooled = rule_rows(printed)
        per_symbol = rows_for(printed, ["BTC/USDT", "ETH/USDT"])
        assert len(pooled) == 1
        assert len(per_symbol) == 2
        assert int(pooled[0][TRADES]) == sum(int(row[TRADES]) for row in per_symbol)

    def test_the_rule_and_the_benchmark_are_not_in_each_other_s_column(
        self, project, peeker, capsys
    ):
        """Swapping the two columns is invisible unless one is known to be bigger.

        A rule that reads the future is the one case where the ordering is not
        a matter of opinion: it must beat the every-bar mean, so if the column
        headed `rule mean` holds the smaller number, they are the wrong way
        round.
        """
        cli.run(project.args("--rule", peeker, "--hold", "24", "--no-check"))
        row = rows_for(capsys.readouterr().out, [peeker])[0]
        assert percent(row[RULE_MEAN]) > percent(row[EVERY_BAR])

    def test_the_count_above_the_threshold_matches_the_rows(
        self, project, peeker, capsys
    ):
        """The honesty line has to be counting the table it is printed under."""
        cli.run(project.args("--rule", peeker, "--no-check"))
        printed = capsys.readouterr().out

        rows = rows_for(printed, [peeker])
        above = sum(1 for row in rows if percent(row[PERCENTILE]) > cli.NOTABLE)
        assert above > 0, "a rule that reads the future should clear the threshold"

        line = next(line for line in printed.splitlines() if "above the" in line)
        assert line.split(":")[1].strip() == f"{above} of {len(rows)}"


class TestNarrowing:
    def test_one_rule_only(self, project, capsys):
        assert cli.run(project.args("--rule", "breakout")) == cli.EXIT_OK
        rows = rule_rows(capsys.readouterr().out)
        assert {fields[0] for fields in rows} == {"breakout"}

    def test_one_hold_only(self, project, capsys):
        cli.run(project.args("--rule", "breakout", "--hold", "24"))
        rows = rule_rows(capsys.readouterr().out)
        assert [int(fields[HOLD]) for fields in rows] == [24]

    def test_several_holds_in_the_order_given(self, project, capsys):
        """Asked for out of order, printed out of order.

        `--hold 6 24` cannot tell you this: it is already sorted, so a tool
        that quietly sorts and a tool that preserves the order print the same
        thing. Reading a table against the command that produced it is easier
        when the rows are where you put them.
        """
        cli.run(project.args("--rule", "breakout", "--hold", "24", "6"))
        rows = rule_rows(capsys.readouterr().out)
        assert [int(fields[HOLD]) for fields in rows] == [24, 6]

    def test_the_symbols_flag_overrides_the_config(self, project, capsys):
        cli.run(project.args("--rule", "breakout", "--symbols", "BTC/USDT"))
        printed = capsys.readouterr().out
        assert "BTC/USDT" in printed
        assert "ETH/USDT" not in printed

    def test_a_rule_that_never_fires_is_refused_rather_than_scored(
        self, project, silent, capsys
    ):
        """Zero trades is not a result, and every statistic over it is NaN.

        Left alone this ends in a table of nan% against nan%, which reads like
        a bug in the maths rather than a rule that did nothing.

        The guard is off here on purpose. With it on, a silent rule never
        reaches the measurement at all -- the guard refuses it first, which is
        the test below -- and this check would pass without the code it is
        supposed to be about ever running.
        """
        assert cli.run(project.args("--rule", silent, "--no-check")) == cli.EXIT_FAILED
        assert "fired on no bars" in capsys.readouterr().err


class TestTheGuardRunsFirst:
    """A rule that reads the future produces beautiful, meaningless output."""

    def test_a_clean_rule_is_reported_as_checked(self, project, capsys):
        cli.run(project.args("--rule", "breakout"))
        printed = capsys.readouterr().out
        assert "lookahead" in printed
        assert "breakout" in printed

    def test_a_rule_that_reads_the_future_stops_the_run(self, project, peeker, capsys):
        assert cli.run(project.args("--rule", peeker)) == cli.EXIT_FAILED

    def test_and_nothing_is_measured_after_it_is_caught(self, project, peeker, capsys):
        """The order matters, not just the verdict.

        If the table were printed and the guard reported afterwards, the number
        would already be on the screen and in the terminal history, and the
        warning underneath it would be competing with it.
        """
        cli.run(project.args("--rule", peeker))
        printed = capsys.readouterr().out
        assert "percentile" not in printed

    def test_the_guard_can_be_skipped_and_then_the_run_completes(self, project, peeker, capsys):
        assert cli.run(project.args("--rule", peeker, "--no-check")) == cli.EXIT_OK
        assert "percentile" in capsys.readouterr().out

    def test_the_guard_is_on_when_nobody_says_otherwise(self):
        assert cli.build_parser().parse_args([]).no_check is False

    def test_a_rule_that_never_fires_does_not_pass_by_having_nothing_to_check(
        self, project, silent, capsys
    ):
        """The guard refuses it rather than reporting it clean.

        A rule with no signals gives the truncation checks nothing to compare,
        and "found no problems" and "could not look" are not the same sentence.
        """
        assert cli.run(project.args("--rule", silent)) == cli.EXIT_FAILED
        assert "guard" in capsys.readouterr().err

    def test_it_is_checked_against_the_longest_history_available(
        self, tmp_path, capsys
    ):
        """The truncation checks need history to cut, so the choice is not free.

        With both symbols the same length -- which is every other test here --
        picking the longest and picking the shortest are the same pick, and
        nothing notices the difference.
        """
        data_dir = tmp_path / "data"
        for symbol, bars in (("BTC/USDT", 300), ("ETH/USDT", 600)):
            path = store.candle_path(data_dir, "binance", symbol, "1h")
            store.write_candles(path, store.candles_to_frame(candle_rows(bars, seed=0)))

        config_path = tmp_path / "symbols.json"
        config_path.write_text(
            json.dumps(
                {
                    "exchange": "binance",
                    "timeframe": "1h",
                    "start": "2024-08-01",
                    "symbols": ["BTC/USDT", "ETH/USDT"],
                }
            ),
            encoding="utf-8",
        )

        cli.run(
            [
                "--config", str(config_path),
                "--data-dir", str(data_dir),
                "--draws", "20",
                "--rule", "breakout",
                "--hold", "24",
            ]
        )
        assert "on ETH/USDT (600 candles)" in capsys.readouterr().out


class TestPerSymbol:
    def test_it_is_off_unless_asked(self, project, capsys):
        cli.run(project.args("--rule", "breakout", "--hold", "24"))
        assert "symbol by symbol" not in capsys.readouterr().out

    def test_it_breaks_the_pool_into_its_symbols(self, project, capsys):
        cli.run(project.args("--rule", "breakout", "--hold", "24", "--per-symbol"))
        printed = capsys.readouterr().out
        assert "symbol by symbol" in printed
        # Once in the header, once in its own row.
        assert printed.count("BTC/USDT") >= 2
        assert printed.count("ETH/USDT") >= 2


class TestTheSplit:
    def test_it_is_off_unless_asked(self, project, capsys):
        cli.run(project.args("--rule", "breakout", "--hold", "24"))
        assert "first half" not in capsys.readouterr().out

    def test_it_reports_both_halves(self, project, capsys):
        cli.run(project.args("--rule", "breakout", "--hold", "24", "--split"))
        printed = capsys.readouterr().out
        assert "first half" in printed
        assert "second half" in printed

    def test_each_half_is_measured_against_its_own_benchmark(self, project, capsys):
        """Not the whole sample's benchmark.

        A half compared against the full period's every-bar mean would be
        measuring which half the market went up in, which is a fact about the
        market and not about the rule.
        """
        cli.run(project.args("--rule", "breakout", "--hold", "24", "--split"))
        lines = [
            line
            for line in capsys.readouterr().out.splitlines()
            if "half" in line and "every bar" in line
        ]
        assert len(lines) == 2
        assert lines[0] != lines[1]


class TestTheWalkForward:
    """--walk N generalises --split from two halves to N sequential windows."""

    def test_it_is_off_unless_asked(self, project, capsys):
        cli.run(project.args("--rule", "breakout", "--hold", "6"))
        assert "sequential windows" not in capsys.readouterr().out

    def test_it_reports_one_row_per_window(self, project, capsys):
        cli.run(project.args("--rule", "breakout", "--hold", "6", "--walk", "3"))
        printed = capsys.readouterr().out
        assert "across 3 sequential windows" in printed
        rows = [line for line in printed.splitlines() if "every bar" in line and "/3" in line]
        assert len(rows) == 3

    def test_each_window_is_measured_against_its_own_benchmark(self, project, capsys):
        cli.run(project.args("--rule", "breakout", "--hold", "6", "--walk", "3"))
        rows = [
            line
            for line in capsys.readouterr().out.splitlines()
            if "/3" in line and "every bar" in line
        ]
        assert len(rows) == 3
        assert len(set(rows)) == 3  # three different periods, three different lines

    def test_the_windows_tile_the_history_without_overlap_or_gaps(self):
        """The slicing itself, checked directly: every bar lands in exactly one
        window, so a rule cannot be measured twice or missed between windows.
        """
        f = store.candles_to_frame(candle_rows(50, seed=0))
        pieces = [cli._window_slice(f, i, 4) for i in range(4)]
        assert sum(len(p) for p in pieces) == len(f)
        rebuilt = pd.concat(pieces)["timestamp"].tolist()
        assert rebuilt == f["timestamp"].tolist()

    def test_fewer_than_two_windows_is_refused(self, project, capsys):
        assert cli.run(
            project.args("--rule", "breakout", "--hold", "6", "--walk", "1", "--no-check")
        ) == cli.EXIT_FAILED
        assert "error:" in capsys.readouterr().err

    def test_slicing_so_fine_that_a_window_cannot_trade_is_refused(self, project, capsys):
        """Over-splitting leaves a window too short to hold a single trade, which
        the evaluator refuses rather than reporting a percentile off two trades.
        """
        assert cli.run(
            project.args("--rule", "breakout", "--hold", "100", "--walk", "50", "--no-check")
        ) == cli.EXIT_FAILED
        assert "error:" in capsys.readouterr().err


class TestCosts:
    def test_costs_are_charged_unless_turned_off(self, project, capsys):
        cli.run(project.args("--rule", "breakout", "--hold", "24"))
        charged = capsys.readouterr().out
        cli.run(project.args("--rule", "breakout", "--hold", "24", "--no-costs"))
        free = capsys.readouterr().out
        assert charged != free

    def test_the_run_says_which_it_used(self, project, capsys):
        cli.run(project.args("--rule", "breakout", "--hold", "24"))
        assert "fee" in capsys.readouterr().out
        cli.run(project.args("--rule", "breakout", "--hold", "24", "--no-costs"))
        assert "no costs" in capsys.readouterr().out


class TestTheJsonOutput:
    def test_nothing_is_written_unless_asked(self, project, tmp_path):
        cli.run(project.args("--rule", "breakout", "--hold", "24"))
        assert not (tmp_path / "out.json").exists()

    def test_a_row_per_comparison(self, project, tmp_path):
        target = tmp_path / "out.json"
        cli.run(project.args("--rule", "breakout", "--json", str(target)))
        written = json.loads(target.read_text(encoding="utf-8"))
        assert len(written["comparisons"]) == len(trades.DEFAULT_HOLDS)
        first = written["comparisons"][0]
        assert first["rule"] == "breakout"
        assert "percentile" in first
        assert "rule_mean" in first
        assert "baseline_mean" in first

    def test_it_records_what_produced_the_numbers(self, project, tmp_path):
        """A percentile without its seed and draw count is not reproducible."""
        target = tmp_path / "out.json"
        cli.run(project.args("--rule", "breakout", "--seed", "7", "--json", str(target)))
        written = json.loads(target.read_text(encoding="utf-8"))
        assert written["draws"] == 20
        assert written["seed"] == 7
        assert written["symbols"] == ["BTC/USDT", "ETH/USDT"]
        assert written["costs"]["fee"] == trades.DEFAULT_COSTS.fee

    def test_the_table_is_still_printed(self, project, tmp_path, capsys):
        cli.run(project.args("--rule", "breakout", "--json", str(tmp_path / "out.json")))
        assert "percentile" in capsys.readouterr().out


class TestTheTrailingStopFlag:
    """--trail changes the exit, so it changes the numbers and is recorded."""

    def test_the_header_names_the_exit(self, project, capsys):
        cli.run(project.args("--rule", "breakout", "--hold", "24"))
        assert "held to time" in capsys.readouterr().out
        cli.run(project.args("--rule", "breakout", "--hold", "24", "--trail", "0.05"))
        printed = capsys.readouterr().out
        assert "trailing stop" in printed
        assert "5.0%" in printed

    def test_it_actually_changes_the_result(self, project, tmp_path):
        """A tight trail fires often on this walk, so the rule's mean must move.
        A flag that is wired up but never reaches the fill model would leave the
        two files identical and pass every other test here.
        """
        plain = tmp_path / "plain.json"
        trailed = tmp_path / "trailed.json"
        cli.run(project.args("--rule", "breakout", "--hold", "24", "--json", str(plain)))
        cli.run(project.args("--rule", "breakout", "--hold", "24", "--trail", "0.02",
                             "--json", str(trailed)))
        a = json.loads(plain.read_text(encoding="utf-8"))["comparisons"][0]
        b = json.loads(trailed.read_text(encoding="utf-8"))["comparisons"][0]
        assert a["rule_mean"] != b["rule_mean"]

    def test_it_is_recorded_in_the_json_and_off_by_default(self, project, tmp_path):
        plain = tmp_path / "plain.json"
        trailed = tmp_path / "trailed.json"
        cli.run(project.args("--rule", "breakout", "--hold", "24", "--json", str(plain)))
        cli.run(project.args("--rule", "breakout", "--hold", "24", "--trail", "0.05",
                             "--json", str(trailed)))
        assert json.loads(plain.read_text(encoding="utf-8"))["trail"] is None
        assert json.loads(trailed.read_text(encoding="utf-8"))["trail"] == 0.05

    @pytest.mark.parametrize("bad", ["0", "1", "1.5", "-0.1"])
    def test_a_trail_outside_zero_to_one_is_refused(self, project, capsys, bad):
        assert cli.run(
            project.args("--rule", "breakout", "--hold", "24", "--trail", bad, "--no-check")
        ) == cli.EXIT_FAILED
        assert "error:" in capsys.readouterr().err


class TestTheExitCodes:
    def test_there_are_only_two(self):
        """Deliberately. See the module docstring in cli.py.

        The collector's third code says "the run finished and the data has
        holes", which is a fact about disk. There is no equivalent fact here:
        every number this tool prints is an opinion about one two-year sample,
        and an exit code is an invitation to build something automatic on top
        of one.
        """
        codes = {name for name in dir(cli) if name.startswith("EXIT_")}
        assert codes == {"EXIT_OK", "EXIT_FAILED"}
        assert (cli.EXIT_OK, cli.EXIT_FAILED) == (0, 1)

    def test_a_rule_at_the_top_of_the_distribution_still_exits_zero(
        self, project, peeker, capsys
    ):
        """The strongest result this tool can produce, and it changes nothing."""
        assert cli.run(project.args("--rule", peeker, "--no-check")) == cli.EXIT_OK
        assert "100.0" in capsys.readouterr().out

    def test_a_run_that_cannot_read_the_store_exits_one(self, project, tmp_path, capsys):
        code = cli.run(project.args("--data-dir", str(tmp_path / "nowhere")))
        assert code == cli.EXIT_FAILED
        assert "error:" in capsys.readouterr().err


class TestItRefusesRatherThanGuesses:
    def _fails_with(self, project, capsys, *extra, containing):
        assert cli.run(project.args(*extra)) == cli.EXIT_FAILED
        captured = capsys.readouterr()
        assert containing in captured.err
        return captured.err

    def test_a_hold_longer_than_the_history(self, project, capsys):
        self._fails_with(
            project, capsys, "--rule", "breakout", "--hold", "5000", containing="error:"
        )

    def test_a_hold_of_zero(self, project, capsys):
        self._fails_with(
            project, capsys, "--rule", "breakout", "--hold", "0", containing="error:"
        )

    def test_a_negative_hold(self, project, capsys):
        self._fails_with(
            project, capsys, "--rule", "breakout", "--hold", "-3", containing="error:"
        )

    def test_no_draws_at_all(self, project, capsys):
        self._fails_with(
            project, capsys, "--rule", "breakout", "--draws", "0", containing="error:"
        )

    def test_a_symbol_that_is_not_in_the_store(self, project, capsys):
        message = self._fails_with(
            project, capsys, "--symbols", "DOGE/USDT", containing="error:"
        )
        assert "DOGE" in message

    def test_a_date_in_a_format_that_means_two_things(self, project, capsys):
        self._fails_with(project, capsys, "--start", "01/08/2024", containing="error:")

    def test_a_config_file_that_is_not_there(self, project, capsys):
        assert cli.run(["--config", "no-such-file.json"]) == cli.EXIT_FAILED
        assert "error:" in capsys.readouterr().err

    def test_none_of_these_print_a_traceback(self, project, capsys):
        cli.run(project.args("--rule", "breakout", "--hold", "0"))
        assert "Traceback" not in capsys.readouterr().err


class TestTheWindow:
    def test_a_start_narrows_what_is_measured(self, project, capsys):
        cli.run(project.args("--rule", "breakout", "--hold", "24"))
        whole = capsys.readouterr().out
        cli.run(project.args("--rule", "breakout", "--hold", "24", "--start", "2024-08-10"))
        narrowed = capsys.readouterr().out
        assert whole != narrowed
        assert f"{BARS} candles" not in narrowed

    def test_an_end_narrows_it_too(self, project, capsys):
        cli.run(project.args("--rule", "breakout", "--hold", "24", "--end", "2024-08-15"))
        assert f"{BARS} candles" not in capsys.readouterr().out


class TestReproducibility:
    def test_the_same_seed_gives_the_same_table(self, project, capsys):
        cli.run(project.args("--rule", "breakout", "--seed", "3"))
        first = capsys.readouterr().out
        cli.run(project.args("--rule", "breakout", "--seed", "3"))
        assert capsys.readouterr().out == first

    def test_a_different_seed_moves_the_percentiles(self, project, capsys):
        cli.run(project.args("--rule", "breakout", "--seed", "3"))
        first = capsys.readouterr().out
        cli.run(project.args("--rule", "breakout", "--seed", "4"))
        assert capsys.readouterr().out != first
