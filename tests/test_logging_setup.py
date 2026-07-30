"""Tests for logging configuration.

Mostly wiring, but two behaviours are worth pinning down: that repeat calls do
not duplicate output, and that timestamps are UTC. Both are the kind of thing
that is annoying to notice by eye and easy to break by accident.
"""

import logging

from collector.logging_setup import configure_logging


def test_creates_log_directory_and_file(tmp_path):
    log_dir = tmp_path / "logs"
    log_file = configure_logging(log_dir=log_dir)

    logging.getLogger("test").info("hello")
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert log_file.exists()
    assert "hello" in log_file.read_text(encoding="utf-8")


def test_logs_to_both_stdout_and_file(tmp_path, capsys):
    log_file = configure_logging(log_dir=tmp_path / "logs")

    logging.getLogger("test").info("both places")
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert "both places" in capsys.readouterr().out
    assert "both places" in log_file.read_text(encoding="utf-8")


def test_repeat_calls_do_not_duplicate_handlers(tmp_path):
    """Calling twice previously attached a second pair of handlers, making every
    line appear twice -- a confusing symptom to trace back to its cause."""
    configure_logging(log_dir=tmp_path / "logs")
    first_count = len(logging.getLogger().handlers)

    configure_logging(log_dir=tmp_path / "logs")
    assert len(logging.getLogger().handlers) == first_count == 2


def test_timestamps_are_utc(tmp_path):
    """The store holds UTC candle timestamps. Log lines in a different frame
    would make any timing investigation require constant mental conversion.
    """
    import datetime as dt

    log_file = configure_logging(log_dir=tmp_path / "logs")
    logging.getLogger("test").info("timed")
    for handler in logging.getLogger().handlers:
        handler.flush()

    logged_hour = int(log_file.read_text(encoding="utf-8").split()[1].split(":")[0])
    utc_hour = dt.datetime.now(tz=dt.timezone.utc).hour

    # Allow for the run straddling an hour boundary.
    assert logged_hour in {utc_hour, (utc_hour - 1) % 24}


def test_quiets_the_noisy_third_party_loggers(tmp_path):
    """ccxt at DEBUG level would bury our own records."""
    configure_logging(log_dir=tmp_path / "logs", level=logging.DEBUG)
    assert logging.getLogger("ccxt").level == logging.WARNING
