"""The paper command, the state it keeps, and the dashboard that reads it.

Two properties carry this file, and everything else is detail around them.

The first is that *stopping and starting changes nothing*. A paper account
advanced in two halves, saved and reloaded in between, must end up in exactly
the state it would have reached in one pass. Without that, an hourly cron and a
catch-up after a laptop closes produce different ledgers, and neither can be
trusted -- so this is tested against the arithmetic rather than argued for in a
docstring.

The second is that *the run refuses to mix strategies*. Change the rule or the
hold, resume, and the trades taken under the new settings would be appended to a
ledger built under the old ones, producing one equity curve describing two
strategies. That is refused, loudly, with the fix named.

The dashboard is tested for what it serves and, more importantly, for what it
does not: there is no route that changes anything, and it binds to localhost.
"""

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from collector import store
from paper import state as state_module
from paper.account import Account
from paper.risk import DrawdownGuard
from paper.cli import build_parser, load_config, run
from paper.portfolio import Portfolio, PortfolioError
from paper.server import Handler, serve
from signals import rules
from signals.trades import Costs

HOUR = 3_600_000
T0 = 1_722_470_400_000
BARS = 600


def candle_rows(count, *, seed, start=T0):
    """A deterministic walk with volume tied to the up-moves, so breakout-volume fires."""
    rng = np.random.default_rng(seed)
    rows = []
    price = 100.0
    for index in range(count):
        opened = price
        closed = max(1.0, opened + rng.normal(0.0, 1.0))
        rows.append([
            start + index * HOUR,
            opened,
            max(opened, closed) + abs(rng.normal(0.0, 0.3)),
            min(opened, closed) - abs(rng.normal(0.0, 0.3)),
            closed,
            10.0 + 800.0 * max(0.0, closed - opened),
        ])
        price = closed
    return rows


@pytest.fixture
def project(tmp_path):
    """A store, a paper config, and the arguments that point at both."""

    class Project:
        def __init__(self):
            self.root = tmp_path
            self.data_dir = tmp_path / "data"
            self.symbols = ["BTC/USDT", "ETH/USDT"]
            for offset, symbol in enumerate(self.symbols):
                path = store.candle_path(self.data_dir, "binance", symbol, "1h")
                store.write_candles(path, store.candles_to_frame(candle_rows(BARS, seed=offset)))
            self.config_path = tmp_path / "paper.json"
            self.state_path = tmp_path / "state" / "paper.json"
            self.write_config()

        def write_config(self, **overrides):
            config = {
                "exchange": "binance", "timeframe": "1h", "symbols": self.symbols,
                "rule": "breakout-volume", "hold": 24,
                "starting_capital": 10_000.0, "size_fraction": 0.25, "max_positions": 4,
            }
            config.update(overrides)
            self.config_path.write_text(json.dumps(config), encoding="utf-8")

        def args(self, *extra):
            return [
                "--config", str(self.config_path),
                "--data-dir", str(self.data_dir),
                "--state", str(self.state_path),
                *extra,
            ]

        def state(self):
            return json.loads(self.state_path.read_text(encoding="utf-8"))

        def snapshot(self):
            path = self.state_path.with_name("snapshot.json")
            return json.loads(path.read_text(encoding="utf-8"))

    return Project()


def frames_for(project, upto=None):
    from signals import prices
    out = {}
    for symbol in project.symbols:
        frame = prices.load(project.data_dir, "binance", symbol, "1h")
        out[symbol] = frame if upto is None else frame.iloc[:upto].copy()
    return out


def signals_for(frames):
    return {symbol: rules.apply("breakout-volume", frame) for symbol, frame in frames.items()}


def build_portfolio(symbols, hold=24):
    return Portfolio(
        symbols, hold=hold,
        costs=Costs(),
        account=Account(10_000.0, size_fraction=0.25, max_positions=4),
    )


class TestStoppingAndStartingChangesNothing:
    """The property that makes a scheduled run trustworthy."""

    def test_two_halves_equal_one_pass(self, project):
        whole = build_portfolio(project.symbols)
        full = frames_for(project)
        whole.advance(full, signals_for(full))

        half = build_portfolio(project.symbols)
        early = frames_for(project, upto=BARS // 2)
        half.advance(early, signals_for(early))

        # Save, throw the object away, and rebuild from the file alone.
        saved = half.to_state()
        resumed = build_portfolio(project.symbols)
        resumed.restore(saved, full)
        resumed.advance(full, signals_for(full))

        assert resumed.cursor == whole.cursor
        assert resumed.account.cash == pytest.approx(whole.account.cash)
        assert len(resumed.open_positions()) == len(whole.open_positions())

        left = whole.ledger().sort_values(["entry_time", "symbol"]).reset_index(drop=True)
        right = resumed.ledger().sort_values(["entry_time", "symbol"]).reset_index(drop=True)
        assert len(left) == len(right)
        for column in ("symbol", "entry_price", "exit_price", "exit_reason", "net_return"):
            assert list(left[column]) == pytest.approx(list(right[column])) if column not in (
                "symbol", "exit_reason"
            ) else list(left[column]) == list(right[column])

    def test_running_again_with_no_new_candles_does_nothing(self, project):
        assert run(project.args()) == 0
        first = project.state()
        assert run(project.args()) == 0
        second = project.state()
        assert first["ledger"] == second["ledger"]
        assert first["positions"] == second["positions"]
        assert first["cash"] == second["cash"]
        assert first["cursor"] == second["cursor"]

    def test_the_refused_count_survives_a_restart(self, project):
        """Only a tail of the rejections is saved, so the total has to be
        remembered separately -- and a resumed run must add to it rather than
        replace it, or every restart would claim the wallet never refused
        anything.
        """
        run(project.args())
        before = project.state()["rejections_total"]
        assert before > 0
        run(project.args())
        assert project.state()["rejections_total"] == before

    def test_a_restored_position_does_not_spend_its_cash_twice(self, project):
        run(project.args())
        cash = project.state()["cash"]
        run(project.args())
        assert project.state()["cash"] == pytest.approx(cash)


class TestItRefusesToMixStrategies:
    def test_changing_the_rule_refuses_to_resume(self, project, capsys):
        run(project.args())
        project.write_config(rule="breakout")
        assert run(project.args()) == 1
        assert "different settings" in capsys.readouterr().err

    def test_changing_the_hold_refuses_to_resume(self, project, capsys):
        run(project.args())
        project.write_config(hold=48)
        assert run(project.args()) == 1
        assert "error:" in capsys.readouterr().err

    def test_reset_starts_a_new_ledger_under_the_new_settings(self, project):
        run(project.args())
        project.write_config(rule="breakout")
        assert run(project.args("--reset")) == 0
        assert project.state()["config"]["rule"] == "breakout"

    def test_a_cosmetic_change_does_not_refuse(self, project):
        """Only settings that change what a trade would have been are
        fingerprinted, or everyone learns to pass --reset by reflex.
        """
        run(project.args())
        assert run(project.args("--json", str(project.root / "elsewhere.json"))) == 0


class TestTheCommand:
    def test_it_advances_and_reports(self, project, capsys):
        assert run(project.args()) == 0
        printed = capsys.readouterr().out
        assert "paper: breakout-volume" in printed
        assert "equity" in printed
        assert "refused" in printed

    def test_status_does_not_advance_or_write_state(self, project):
        assert run(project.args("--status")) == 0
        assert not project.state_path.exists()

    def test_it_writes_a_snapshot_every_run(self, project):
        run(project.args())
        snapshot = project.snapshot()
        assert snapshot["config"]["rule"] == "breakout-volume"
        assert snapshot["stats"]["closed"] >= 0
        assert "equity_curve" in snapshot

    def test_the_snapshot_can_be_redirected(self, project):
        target = project.root / "custom.json"
        run(project.args("--json", str(target)))
        assert json.loads(target.read_text(encoding="utf-8"))["config"]["hold"] == 24

    def test_it_actually_trades_on_this_fixture(self, project):
        """A run that quietly takes no trades would pass every other test here."""
        run(project.args())
        assert project.state()["ledger"]

    def test_no_flag_can_be_abbreviated(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--stat"])

    def test_a_missing_store_fails_cleanly(self, project, capsys):
        assert run(project.args("--data-dir", str(project.root / "nowhere"))) == 1
        assert "error:" in capsys.readouterr().err


class TestTheConfig:
    def test_an_unknown_key_is_refused(self, project):
        project.config_path.write_text(json.dumps({
            "exchange": "binance", "timeframe": "1h", "symbols": ["BTC/USDT"],
            "rule": "breakout", "hold": 24, "symbol": ["BTC/USDT"],
        }), encoding="utf-8")
        with pytest.raises(Exception, match="unrecognised"):
            load_config(project.config_path)

    def test_a_missing_required_key_is_refused(self, project):
        project.config_path.write_text(json.dumps({"exchange": "binance"}), encoding="utf-8")
        with pytest.raises(Exception, match="missing required"):
            load_config(project.config_path)

    def test_a_rule_that_does_not_exist_is_refused(self, project):
        project.write_config(rule="moon-phase")
        with pytest.raises(Exception, match="does not exist"):
            load_config(project.config_path)

    def test_comment_keys_are_allowed(self, project):
        project.write_config(_why="because")
        assert load_config(project.config_path)["rule"] == "breakout-volume"

    def test_the_defaults_are_filled_in(self, project):
        config = load_config(project.config_path)
        assert config["trail"] is None
        assert config["costs"]["fee"] == 0.001
        assert config["one_per_symbol"] is True


class TestTheStateFile:
    def test_a_missing_file_is_not_an_error(self, tmp_path):
        assert state_module.load(tmp_path / "nothing.json") is None

    def test_a_round_trip_keeps_everything(self, tmp_path):
        path = tmp_path / "s.json"
        state_module.save(path, {"cursor": 7, "fingerprint": "x"})
        assert state_module.load(path, expect="x")["cursor"] == 7

    def test_a_corrupt_file_is_refused_rather_than_ignored(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(state_module.StateError, match="not valid JSON"):
            state_module.load(path)

    def test_an_older_version_is_refused(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"version": 0}), encoding="utf-8")
        with pytest.raises(state_module.StateError, match="version"):
            state_module.load(path)

    def test_no_temporary_file_is_left_behind(self, tmp_path):
        path = tmp_path / "s.json"
        state_module.save(path, {"cursor": 1})
        assert not path.with_name(path.name + ".tmp").exists()

    def test_the_fingerprint_ignores_settings_that_do_not_change_a_trade(self):
        base = {"rule": "breakout", "hold": 24, "symbols": ["A/B"]}
        assert state_module.fingerprint(base) == state_module.fingerprint({**base, "port": 9})

    def test_the_fingerprint_notices_settings_that_do(self):
        base = {"rule": "breakout", "hold": 24, "symbols": ["A/B"]}
        assert state_module.fingerprint(base) != state_module.fingerprint({**base, "hold": 48})

    def test_symbol_order_does_not_change_the_fingerprint(self):
        left = state_module.fingerprint({"symbols": ["A/B", "C/D"]})
        right = state_module.fingerprint({"symbols": ["C/D", "A/B"]})
        assert left == right


class TestRestoringRefusesTheImpossible:
    def test_a_position_in_a_symbol_no_longer_configured(self, project):
        full = frames_for(project)
        portfolio = build_portfolio(["BTC/USDT"])
        with pytest.raises(PortfolioError, match="not in the current symbol list"):
            portfolio.restore(
                {"cursor": 1, "cash": 1.0, "positions": [
                    {"symbol": "ETH/USDT", "entry_time": int(full["ETH/USDT"]["timestamp"].iloc[0]),
                     "entry_price": 1.0, "effective_entry": 1.0, "qty": 1.0, "peak": 1.0}
                ]},
                full,
            )

    def test_a_position_whose_entry_candle_is_gone(self, project):
        full = frames_for(project)
        portfolio = build_portfolio(project.symbols)
        with pytest.raises(PortfolioError, match="no longer in the store"):
            portfolio.restore(
                {"cursor": 1, "cash": 1.0, "positions": [
                    {"symbol": "BTC/USDT", "entry_time": 1, "entry_price": 1.0,
                     "effective_entry": 1.0, "qty": 1.0, "peak": 1.0}
                ]},
                full,
            )


class TestTheDashboard:
    @staticmethod
    def running(snapshot):
        """A live server on a spare port, answering on a background thread.

        Driven over a real socket rather than through a hand-built request
        object, because the thing worth testing is the server that ships, and a
        fake request would exercise a different code path than a browser does.
        """
        httpd = serve("127.0.0.1", 0, snapshot, forever=False)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        return httpd

    @pytest.fixture
    def server(self, project):
        run(project.args())
        httpd = self.running(project.state_path.with_name("snapshot.json"))
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
        httpd.shutdown()
        httpd.server_close()

    def get(self, url):
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, response.read().decode("utf-8")

    def test_it_serves_the_page(self, server):
        status, body = self.get(server + "/")
        assert status == 200
        assert "paper account" in body

    def test_it_serves_the_snapshot(self, server):
        status, body = self.get(server + "/api/snapshot")
        assert status == 200
        assert json.loads(body)["config"]["rule"] == "breakout-volume"

    def test_an_unknown_route_is_a_clean_404(self, server):
        with pytest.raises(urllib.error.HTTPError) as caught:
            self.get(server + "/anything-else")
        assert caught.value.code == 404

    def test_there_is_no_route_that_changes_anything(self, server):
        """The property to defend hardest before this is ever pointed at a live
        account: a page that can place an order can place one by accident.
        """
        assert not hasattr(Handler, "do_POST")
        assert not hasattr(Handler, "do_PUT")
        assert not hasattr(Handler, "do_DELETE")
        assert not hasattr(Handler, "do_PATCH")

    def test_a_missing_snapshot_says_so_rather_than_crashing(self, tmp_path):
        httpd = self.running(tmp_path / "absent.json")
        try:
            with pytest.raises(urllib.error.HTTPError) as caught:
                self.get(f"http://127.0.0.1:{httpd.server_address[1]}/api/snapshot")
            assert caught.value.code == 404
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_it_binds_to_localhost_by_default(self):
        from paper.server import DEFAULT_HOST
        assert DEFAULT_HOST == "127.0.0.1"


class TestTheNetworkGuardStillBites:
    """The loopback exception must not have opened the door generally.

    Allowing 127.0.0.1 so the dashboard can be tested is only defensible if
    everything else still fails loudly, so that is asserted here rather than
    assumed -- this is the guarantee that keeps the suite honest on a train.
    """

    def test_a_real_host_is_still_refused(self):
        import socket

        from conftest import NetworkAccessAttempted

        with pytest.raises(NetworkAccessAttempted):
            socket.create_connection(("api.binance.com", 443), timeout=1)

    def test_a_dns_lookup_for_a_real_host_is_still_refused(self):
        import socket

        from conftest import NetworkAccessAttempted

        with pytest.raises(NetworkAccessAttempted):
            socket.getaddrinfo("api.binance.com", 443)


class TestThePageAndTheSnapshotAgree:
    """The dashboard's data contract, checked without a browser.

    The page is a string of JavaScript, so nothing type-checks it against the
    JSON it reads. That is a real gap: rename a field in the snapshot and the
    page silently renders "undefined" everywhere while still returning 200, and
    every test that only asserts the server answers would keep passing.

    So the fields the page reads are extracted from the source and required to
    exist. It is not a substitute for looking at it, but it does catch the
    failure that would otherwise be found by looking at it a week later.
    """

    @staticmethod
    def page_source():
        from paper.page import PAGE
        return PAGE[PAGE.index("<script>"):PAGE.index("</script>")]

    def test_every_top_level_field_the_page_reads_exists(self, project):
        import re
        run(project.args())
        snapshot = project.snapshot()
        read = set(re.findall(r"\bs\.([a-z_]+)", self.page_source()))
        assert read, "found no field references -- the extraction itself broke"
        missing = sorted(field for field in read if field not in snapshot)
        assert not missing, f"the page reads fields the snapshot does not have: {missing}"

    @pytest.mark.parametrize("group,keys", [
        ("config", ["rule", "hold", "exchange", "timeframe", "symbols", "trail"]),
        ("stats", ["closed", "hit_rate", "mean_net_pct", "best_pct", "worst_pct", "refused"]),
        ("risk", ["peak", "max_drawdown_pct", "current_drawdown_pct"]),
        ("refusals", ["total", "recent"]),
    ])
    def test_the_nested_groups_carry_what_the_panels_need(self, project, group, keys):
        run(project.args())
        snapshot = project.snapshot()
        assert all(key in snapshot[group] for key in keys)

    def test_an_open_position_carries_its_hold_progress(self, project):
        """The progress bar needs more than a price: how far through the hold it
        is, and when it leaves. An unrealised number at hour three means
        something different from the same number at hour a hundred and sixty.
        """
        run(project.args())
        for position in project.snapshot()["open_positions"]:
            for key in ("bars_held", "bars_total", "exit_utc", "progress_pct"):
                assert key in position
            assert 0 <= position["progress_pct"] <= 100

    def test_the_whole_ledger_is_shipped_not_a_tail(self, project):
        """The page sorts and filters these rows. Doing that to the most recent
        fifty of four hundred would look exactly like doing it to all of them
        while quietly answering a different question.
        """
        run(project.args())
        snapshot = project.snapshot()
        assert len(snapshot["trades"]) == snapshot["stats"]["closed"]

    def test_the_chart_can_reconcile_with_the_headline_equity(self, project):
        """The realised curve necessarily ends at the last closed trade, so a
        chart of it alone finishes below the equity printed beside it. The gap
        is handed over as a fact rather than left to be puzzled over.
        """
        run(project.args())
        snapshot = project.snapshot()
        now = snapshot["equity_now"]
        assert now["realised"] + now["unrealised"] == pytest.approx(now["equity"], abs=0.02)
        assert now["equity"] == pytest.approx(snapshot["equity"], abs=0.02)

    def test_the_by_symbol_panel_covers_the_whole_ledger(self, project):
        """Not just the recent tail. The research found the edge concentrated in
        some names and absent in others, which a fifty-trade window can invert.
        """
        run(project.args())
        snapshot = project.snapshot()
        assert sum(row["trades"] for row in snapshot["by_symbol"]) == snapshot["stats"]["closed"]

    def test_the_drawdown_is_never_positive_and_the_peak_never_below_start(self, project):
        run(project.args())
        risk = project.snapshot()["risk"]
        assert risk["max_drawdown_pct"] <= 0
        assert risk["peak"] >= project.snapshot()["starting_capital"]

    def test_the_refusal_feed_survives_a_restart(self, project):
        run(project.args())
        first = project.snapshot()["refusals"]["recent"]
        assert first, "this fixture should refuse something"
        run(project.args())
        assert project.snapshot()["refusals"]["recent"] == first

    def test_the_script_has_balanced_delimiters(self):
        """The cheapest possible guard against a syntax error in a string that
        nothing compiles. It cannot prove the JavaScript is valid; it does catch
        the bracket left open by an edit, which is the usual way this breaks.
        """
        source = self.page_source()
        for opener, closer in (("{", "}"), ("(", ")"), ("[", "]")):
            assert source.count(opener) == source.count(closer), f"unbalanced {opener}{closer}"


class TestTheRiskRegimeBoundary:
    """Changing a risk control must neither destroy the ledger nor hide itself.

    Two failure modes sit either side of this. Refusing to resume would mean a
    risk limit can only ever be enabled by throwing away the track record, which
    guarantees nobody turns one on. Silently adopting it would produce one equity
    curve spanning two risk regimes with nothing saying so. The answer is to
    adopt and record.
    """

    def enable_guard(self, project, limit=0.25):
        import json as _json
        config = _json.loads(project.config_path.read_text(encoding="utf-8"))
        config["max_drawdown"] = limit
        project.config_path.write_text(_json.dumps(config), encoding="utf-8")

    def test_changing_a_risk_setting_keeps_the_ledger(self, project):
        run(project.args())
        before = len(project.state()["ledger"])
        assert before > 0

        self.enable_guard(project)
        assert run(project.args()) == 0
        assert len(project.state()["ledger"]) == before

    def test_and_records_when_the_regime_changed(self, project):
        run(project.args())
        self.enable_guard(project)
        run(project.args())
        regimes = project.state()["risk_regimes"]
        assert regimes[-1]["settings"]["max_drawdown"] == 0.25
        assert regimes[-1]["at"] is not None

    def test_changing_the_strategy_still_refuses(self, project, capsys):
        """The distinction being drawn. A different rule is a different
        strategy; a different drawdown limit is the same strategy run more
        carefully from a point in time.
        """
        run(project.args())
        project.write_config(rule="breakout")
        assert run(project.args()) == 1
        assert "different settings" in capsys.readouterr().err

    def test_the_high_water_mark_starts_from_today_not_from_the_peak(self, project):
        """Forward-only. Seeding at the historical peak would apply the limit
        retroactively to a drawdown already lived through, which retires an
        account rather than protecting it.
        """
        run(project.args())
        equity = project.snapshot()["equity"]

        self.enable_guard(project)
        run(project.args())
        guard = project.state()["guard"]
        assert guard["peak"] == pytest.approx(equity, rel=0.05)
        assert not guard["tripped"]

    def test_a_tripped_guard_stops_new_positions(self):
        """Tested against the account directly rather than through a run.

        A guard enabled over an existing ledger seeds its mark at today's equity
        and only observes again when a new bar arrives, so a resumed run with no
        new candles cannot trip however tight the limit -- which is correct, and
        makes the CLI the wrong place to assert this.
        """
        account = Account(10_000, guard=DrawdownGuard(limit=0.25))
        account.guard.observe(10_000)
        account.guard.observe(1_000)
        assert "drawdown limit" in account.refusal("BTC/USDT")

    def test_the_report_shows_the_limit(self, project, capsys):
        run(project.args())
        self.enable_guard(project)
        run(project.args())
        assert "25% limit" in capsys.readouterr().out


class TestTheDrawdownLimitIsLegibleOnThePage:
    """The limit is compared against its own high-water mark, not the record.

    These are two different numbers once a limit is armed forward-only, and
    showing the historical one beside the limit reads as "already far past it,
    and yet not tripped" -- which is worse than showing nothing, because it
    invites the reader to distrust the limit rather than the label.
    """

    def enable(self, project, limit=0.25):
        import json as _json
        config = _json.loads(project.config_path.read_text(encoding="utf-8"))
        config["max_drawdown"] = limit
        project.config_path.write_text(_json.dumps(config), encoding="utf-8")

    def test_the_guard_drawdown_is_reported_separately_from_the_record(self, project):
        run(project.args())
        self.enable(project)
        run(project.args())
        risk = project.snapshot()["risk"]
        assert risk["guard_drawdown_pct"] is not None
        assert risk["guard_peak"] is not None
        # The historical figure is still there and still means what it meant.
        assert risk["max_drawdown_pct"] <= 0

    def test_a_freshly_armed_limit_reads_as_zero_not_as_the_old_drawdown(self, project):
        run(project.args())
        self.enable(project)
        run(project.args())
        risk = project.snapshot()["risk"]
        assert risk["guard_drawdown_pct"] == pytest.approx(0.0, abs=1e-6)
        assert risk["drawdown_tripped"] is False

    def test_the_page_reads_the_guard_figure_and_not_the_record(self):
        from paper.page import PAGE
        script = PAGE[PAGE.index("<script>"):PAGE.index("</script>")]
        cell = script[script.index('"dd limit"'):script.index('"dd limit"') + 400]
        assert "guard_drawdown_pct" in cell
        assert "current_drawdown_pct" not in cell

    def test_a_tripped_limit_gets_a_banner_not_a_table_row(self):
        """It is a state change -- no new positions are being opened at all --
        and anything less than a banner can be scrolled past.
        """
        from paper.page import PAGE
        assert 'id="halted"' in PAGE
        assert "HALTED" in PAGE
