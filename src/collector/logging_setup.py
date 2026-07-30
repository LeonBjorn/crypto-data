"""Logging configuration: stdout plus a rotating file.

Kept in its own module so that neither the CLI nor the tests have to know how
logging is wired -- they just call configure_logging() once.
"""

import logging
import logging.handlers
import sys
import time
from pathlib import Path

# 5 MB per file, five files kept. At roughly one line per fetched page, a
# two-year backfill of five symbols produces a few thousand lines, so this
# holds many runs' worth of history before anything is discarded.
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 5

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-20s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(log_dir="logs", level=logging.INFO, filename="collector.log"):
    """
    Send log records to both stdout and a rotating file under `log_dir`.

    Timestamps are UTC. This project stores UTC candle timestamps, and having
    log times in a different frame than the data they describe makes debugging
    a timing problem needlessly hard -- you end up mentally converting while
    also trying to reason about the bug.

    Safe to call more than once: existing handlers are cleared first. Without
    that, a second call would attach a second pair of handlers and every line
    would appear twice, which is a confusing thing to chase down.

    Returns the log file path so the caller can mention it to the user.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / filename

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    # time.gmtime is what makes asctime UTC rather than local.
    formatter.converter = time.gmtime

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
        existing.close()

    root.setLevel(level)
    root.addHandler(stream_handler)
    root.addHandler(file_handler)

    # ccxt logs verbosely at DEBUG and would drown out our own records.
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return log_file
