"""The local Parquet candle store.

Layout on disk:

    data/<exchange>/<SYMBOL>/<timeframe>.parquet

    data/kraken/BTC_USD/1h.parquet
    data/kraken/ETH_USD/1h.parquet

One file per symbol and timeframe. Appending rewrites the whole file, which
sounds wasteful but is not at this scale: two years of hourly candles is about
17,500 rows, well under a megabyte. The payoff is that there is exactly one file
to inspect, copy or delete per market, and no month-boundary edge cases.

Two ideas run through this module:

**The store is its own progress record.** There is no state file tracking what
has been fetched. The resume point is derived from the stored candles. A sidecar
file can disagree with the data after a crash, and then there are two answers to
"what do we have" with no way to tell which is right.

**Every write is atomic.** Data is written to a temporary file in the same
directory and then moved into place with os.replace, which is atomic on POSIX.
A process killed at any moment leaves either the old complete file or the new
complete file, never a half-written one. This is what makes it safe for the
backfill loop to flush periodically rather than once at the end.
"""

import logging
import os
from pathlib import Path

import pandas as pd

from collector.timeframes import timeframe_to_ms

log = logging.getLogger(__name__)

# ccxt returns OHLCV rows in exactly this order; the store mirrors it so there
# is no reordering step to get wrong.
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# timestamp is int64 epoch milliseconds, UTC. Deliberately not a datetime: a
# naive datetime silently meaning local time is the easiest way to corrupt this
# store, so the type that could carry that mistake is never created. Convert to
# datetimes at the point of display, never in storage.
DTYPES = {
    "timestamp": "int64",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
}

# Snappy is pyarrow's default and is readable by every Parquet implementation.
# zstd would compress better, but this store is small and broad compatibility
# matters more than a few megabytes.
COMPRESSION = "snappy"

_ALLOWED_FILENAME_CHARS = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."
)


class StoreError(Exception):
    """Raised when the store is asked to do something unsafe or nonsensical."""


def symbol_to_filename(symbol):
    """
    Turn an exchange symbol into a safe directory name: "BTC/USD" -> "BTC_USD".

    Symbols come from a config file, so this doubles as a guard: a symbol must
    never be able to steer a write outside the store directory. Separators are
    replaced rather than stripped, and anything resembling path traversal is
    refused outright.

    The result is upper-cased. macOS filesystems are case-insensitive by
    default, so "btc/usd" and "BTC/USD" would land in the same file there but
    different files on Linux. Normalising makes the behaviour deliberate and
    identical on both.
    """
    if not isinstance(symbol, str):
        raise StoreError(
            f"symbol must be a string like 'BTC/USD', got {type(symbol).__name__}"
        )

    cleaned = symbol.strip()
    if not cleaned:
        raise StoreError("symbol is empty")

    if ".." in cleaned:
        raise StoreError(
            f"symbol {symbol!r} contains '..', which could point outside the store"
        )

    # ccxt uses '/' between base and quote, and ':' before a settlement currency
    # on derivatives markets (for example 'BTC/USD:USD').
    name = cleaned.replace("/", "_").replace(":", "_").upper()

    if any(part == "" for part in name.split("_")):
        raise StoreError(
            f"symbol {symbol!r} has an empty component; expected something like 'BTC/USD'"
        )

    unexpected = set(name) - _ALLOWED_FILENAME_CHARS
    if unexpected:
        raise StoreError(
            f"symbol {symbol!r} contains characters unsafe for a filename: "
            f"{''.join(sorted(unexpected))}"
        )

    return name


def candle_path(root, exchange, symbol, timeframe):
    """
    Build the path to one symbol's candle file.

    The timeframe is validated on the way through, so an unsupported timeframe
    fails here rather than creating a file named after something this project
    cannot compute boundaries for.
    """
    if not isinstance(exchange, str) or not exchange.strip():
        raise StoreError(f"exchange must be a non-empty string, got {exchange!r}")

    exchange_name = exchange.strip().lower()
    if not exchange_name.isalnum():
        raise StoreError(
            f"exchange {exchange!r} should be a plain ccxt id such as 'kraken'"
        )

    timeframe_to_ms(timeframe)  # raises TimeframeError if unsupported

    return Path(root) / exchange_name / symbol_to_filename(symbol) / f"{timeframe}.parquet"


def empty_frame():
    """An empty frame carrying the correct columns and dtypes.

    Built explicitly rather than with pd.DataFrame(columns=COLUMNS), which would
    give every column dtype `object` and cause a later concat to produce a frame
    whose timestamps are no longer integers.
    """
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in DTYPES.items()})


def candles_to_frame(rows):
    """
    Convert ccxt OHLCV rows into a typed frame, dropping duplicate timestamps.

    Each row must be `[timestamp, open, high, low, close, volume]`. A row of the
    wrong width usually means the exchange returned something unexpected, which
    is worth failing on rather than storing a frame with a silently shifted
    column.
    """
    if not rows:
        return empty_frame()

    for index, row in enumerate(rows):
        if len(row) != len(COLUMNS):
            raise StoreError(
                f"candle at position {index} has {len(row)} fields, expected "
                f"{len(COLUMNS)}: {row!r}"
            )

    frame = pd.DataFrame(list(rows), columns=COLUMNS)

    if (frame["timestamp"] < 0).any():
        raise StoreError(
            "candle timestamps must not be negative; this usually means seconds "
            "were returned where milliseconds were expected"
        )

    frame = frame.astype(DTYPES)

    # Within a single batch, keep the first occurrence. Exchanges occasionally
    # repeat a candle across page boundaries.
    return frame.drop_duplicates(subset="timestamp", keep="first")


def read_candles(path):
    """
    Read one symbol's stored candles, sorted oldest first.

    A missing file returns an empty frame rather than raising: that is what
    every first run looks like, so it is a normal state and not an error.
    """
    path = Path(path)
    if not path.exists():
        return empty_frame()

    frame = pd.read_parquet(path)

    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise StoreError(
            f"{path} is missing expected columns {missing}. "
            f"The file may be from an older version or not a candle file at all."
        )

    frame = frame[COLUMNS].astype(DTYPES)
    return frame.sort_values("timestamp").reset_index(drop=True)


def write_candles(path, frame):
    """
    Write a frame to `path` atomically, replacing whatever was there.

    The temporary file is created in the same directory as the target on
    purpose: os.replace is only atomic within a single filesystem, so a temp
    file in /tmp could not be moved into place safely.

    Steps, in order, and each one matters:

    1. Write the full frame to `<name>.parquet.tmp`.
    2. fsync it, so the bytes are on the physical device rather than sitting in
       the operating system's write cache.
    3. os.replace the temp file over the target. This is the atomic moment.
    4. fsync the containing directory, so the rename itself is durable.

    Steps 2 and 4 are best-effort and only matter for a power cut or kernel
    panic; a plain kill -9 is already handled by steps 1 and 3 alone. They are
    cheap here because writes are infrequent.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ordered = frame[COLUMNS].astype(DTYPES).sort_values("timestamp").reset_index(drop=True)
    temp_path = path.with_name(path.name + ".tmp")

    try:
        ordered.to_parquet(
            temp_path, index=False, engine="pyarrow", compression=COMPRESSION
        )
        _fsync_path(temp_path)
        os.replace(temp_path, path)
        _fsync_path(path.parent)
    except BaseException:
        # Includes KeyboardInterrupt, which is exactly the interrupted-run case
        # this module exists to survive. Clear the temp file and let the
        # exception continue; the original file has not been touched.
        temp_path.unlink(missing_ok=True)
        raise

    return path


def _fsync_path(target):
    """Flush an already-written file or directory to disk, best effort.

    Wrapped because directory fsync is not permitted on every platform, and
    failing to flush is not a reason to abandon an otherwise complete write.
    """
    try:
        fd = os.open(target, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as error:
        log.debug("could not fsync %s: %s", target, error)


def append_candles(path, rows):
    """
    Add candles that are not already stored. Returns how many rows were added.

    Idempotency comes from the data model rather than from the caller being
    careful: only timestamps absent from the file are added, so re-running with
    the same input is inert.

    When an incoming timestamp is already present, the stored row wins and the
    incoming one is discarded. Closed candles do not change, so a differing
    value means the incoming row is the suspect one -- most likely a partially
    formed candle that slipped through a re-fetch. The trade-off is that a bad
    row already on disk stays there; correcting one means deleting the file (or
    that range) and refetching.

    If nothing needs adding, the file is not rewritten at all. That saves work,
    and it means an interrupted no-op run cannot corrupt anything because no
    write was ever begun.
    """
    path = Path(path)
    incoming = candles_to_frame(rows)
    if incoming.empty:
        return 0

    existing = read_candles(path)
    fresh = incoming[~incoming["timestamp"].isin(existing["timestamp"])]

    if fresh.empty:
        log.debug("%s: nothing new in %d candles", path.name, len(incoming))
        return 0

    combined = pd.concat([existing, fresh], ignore_index=True)
    write_candles(path, combined)

    log.debug("%s: added %d candles (%d total)", path.name, len(fresh), len(combined))
    return len(fresh)


def stored_range(path):
    """Oldest and newest stored timestamps, or (None, None) for an empty store."""
    frame = read_candles(path)
    if frame.empty:
        return (None, None)
    return (int(frame["timestamp"].iloc[0]), int(frame["timestamp"].iloc[-1]))


def stored_timestamps(path):
    """
    Every stored candle open time, sorted, as plain Python ints.

    Gap detection compares these against expected_open_times, which produces
    plain ints. Returning numpy int64 would compare equal but makes set
    operations and error messages needlessly confusing to read.
    """
    frame = read_candles(path)
    return [int(value) for value in frame["timestamp"]]


def next_start_ms(path, timeframe_ms, default_start_ms):
    """
    Where the next fetch should begin: one timeframe after the newest stored
    candle, or `default_start_ms` if the store is empty.

    Note the direction. This only ever resumes *forwards*. If the store already
    holds data and you ask for an earlier start date than its oldest candle,
    that earlier range will not be fetched by this function alone, and it also
    falls outside what gap detection inspects -- gaps are looked for within the
    stored range, and this would be before it.

    Extending a store backwards is a separate operation from resuming it. The
    backfill layer is where that belongs, and it should warn rather than
    silently ignore the request.
    """
    _, newest = stored_range(path)
    if newest is None:
        return default_start_ms
    return newest + timeframe_ms
