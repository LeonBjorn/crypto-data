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
