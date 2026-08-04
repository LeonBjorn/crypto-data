"""Copies of the ledger, and the pulse that says the schedule is alive.

Both guard the same thing: the forward record is the only out-of-sample evidence
this project has, it cannot be refetched from anywhere, and the two ways to lose
it are destroying the file or quietly failing to add to it.
"""

import json
import time

import pytest

from paper.backup import (
    HEARTBEAT_NAME,
    hours_since,
    keep,
    prune,
    read_heartbeat,
    write_heartbeat,
)


def a_state(path, marker="x"):
    path.write_text(json.dumps({"ledger": [marker]}), encoding="utf-8")
    return path


class TestBackups:
    def test_it_copies_the_state_aside(self, tmp_path):
        state = a_state(tmp_path / "paper.json")
        copy = keep(state)
        assert copy.exists()
        assert json.loads(copy.read_text())["ledger"] == ["x"]

    def test_the_copy_is_of_what_was_there_before_the_save(self, tmp_path):
        """Taken before writing, so a crash midway through the new file leaves
        both the previous state and this copy intact.
        """
        state = a_state(tmp_path / "paper.json", "before")
        copy = keep(state)
        a_state(state, "after")
        assert json.loads(copy.read_text())["ledger"] == ["before"]

    def test_nothing_to_copy_is_not_an_error(self, tmp_path):
        assert keep(tmp_path / "absent.json") is None

    def test_it_keeps_the_oldest_as_well_as_the_newest(self, tmp_path):
        """The rule that matters. A ledger that goes subtly wrong is noticed days
        later, by which point every recent copy has inherited the problem -- so
        the earliest few are kept permanently as a floor.
        """
        folder = tmp_path / "backups"
        folder.mkdir()
        for index in range(30):
            (folder / f"paper-{index:04d}.json").write_text("{}", encoding="utf-8")

        prune(folder, limit=10, oldest=3)
        left = sorted(p.name for p in folder.glob("*.json"))
        assert len(left) == 10
        assert "paper-0000.json" in left        # oldest kept
        assert "paper-0029.json" in left        # newest kept
        assert "paper-0015.json" not in left    # middle discarded

    def test_it_does_not_prune_below_the_limit(self, tmp_path):
        folder = tmp_path / "backups"
        folder.mkdir()
        for index in range(5):
            (folder / f"paper-{index}.json").write_text("{}", encoding="utf-8")
        assert prune(folder, limit=10) == []
        assert len(list(folder.glob("*.json"))) == 5


class TestTheHeartbeat:
    def test_the_first_cycle_has_no_gap_to_report(self, tmp_path):
        beat = write_heartbeat(tmp_path)
        assert beat["gap_hours"] is None
        assert beat["previous"] is None

    def test_the_second_records_the_gap(self, tmp_path):
        write_heartbeat(tmp_path, at=1_000_000_000_000)
        beat = write_heartbeat(tmp_path, at=1_000_000_000_000 + 3 * 3_600_000)
        assert beat["gap_hours"] == pytest.approx(3.0)

    def test_hours_since_measures_from_the_last_cycle(self, tmp_path):
        now = time.time() * 1000
        write_heartbeat(tmp_path, at=now - 5 * 3_600_000)
        assert hours_since(tmp_path, now=now) == pytest.approx(5.0, abs=0.01)

    def test_never_having_run_is_distinguishable_from_having_just_run(self, tmp_path):
        """Not the same thing, and a monitor that conflated them would report a
        machine that has never started as perfectly healthy.
        """
        assert hours_since(tmp_path) is None
        write_heartbeat(tmp_path)
        assert hours_since(tmp_path) is not None

    def test_a_corrupt_heartbeat_does_not_fail_the_run(self, tmp_path):
        """It is a status marker, not the ledger. The next cycle rewrites it."""
        (tmp_path / HEARTBEAT_NAME).write_text("{not json", encoding="utf-8")
        assert read_heartbeat(tmp_path) is None
        assert write_heartbeat(tmp_path)["at"] > 0

    def test_it_is_written_atomically(self, tmp_path):
        write_heartbeat(tmp_path)
        assert not (tmp_path / HEARTBEAT_NAME).with_suffix(".tmp").exists()


class TestTheRunUsesThem:
    def test_a_run_leaves_a_backup_and_a_pulse(self, tmp_path, monkeypatch):
        from test_paper_cli import candle_rows  # reuse the store fixture
        import json as _json
        from collector import store
        from paper.cli import run

        data = tmp_path / "data"
        for offset, symbol in enumerate(["BTC/USDT", "ETH/USDT"]):
            path = store.candle_path(data, "binance", symbol, "1h")
            store.write_candles(path, store.candles_to_frame(candle_rows(600, seed=offset)))

        config = tmp_path / "paper.json"
        config.write_text(_json.dumps({
            "exchange": "binance", "timeframe": "1h",
            "symbols": ["BTC/USDT", "ETH/USDT"], "rule": "breakout-volume", "hold": 24,
        }), encoding="utf-8")
        state = tmp_path / "state" / "paper.json"

        args = ["--config", str(config), "--data-dir", str(data), "--state", str(state)]
        assert run(args) == 0
        assert run(args) == 0     # second run has something to back up

        assert list((state.parent / "backups").glob("*.json")), "no backup written"
        assert read_heartbeat(state.parent) is not None
        assert _json.loads((state.parent / "snapshot.json").read_text())["heartbeat"]["hours_since"] is not None
