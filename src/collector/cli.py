"""The command-line entry point: the only module that reads the clock or touches the network.

Requirement 8. This is where the program becomes a program rather than a library,
and it is deliberately the thinnest module in the project. Everything with a
decision in it lives elsewhere and is tested against plain data -- what needs
fetching (backfill), what is missing (gaps), what the settings are (settings),
what to say about it (report). What is left here is the wiring: read the config,
build one client, loop, write the report, choose an exit code.

Two things are concentrated here on purpose, because both are untestable wherever
they appear and so belong in as few places as possible.

The first is the clock. `now_ms` is read exactly once, at the top of the run, and
passed down through every call. Nothing below this module ever asks the time. That
is what lets the whole suite pin time to a fixed integer, and it also means the
five symbols in a run share one definition of "now" -- otherwise the last symbol
could see a newer closed candle than the first and the report's range would be a
polite fiction.

The second is failure. A traceback is a fine thing for a developer and a poor
thing for a person who mistyped a date, so the project's own error types are
caught here and printed as messages. Nothing else is caught: an error this module
does not recognise is a bug in this project, and burying a bug under a friendly
message is how it survives to next year.

The exit codes are the part to read carefully if you are skimming -- see below.
"""

import argparse
import logging
import sys

from collector import backfill, report, settings, store
from collector.exchange import ExchangeError, build_exchange
from collector.logging_setup import configure_logging
from collector.store import StoreError
from collector.timeframes import TimeframeError, last_closed_open_time, to_utc_string

log = logging.getLogger(__name__)

# Exit codes. A scheduled run reads nothing else -- not the log, not the report,
# not the summary -- so these carry the whole outcome on their own.
#
# The distinction between 1 and 2 is the one that earns its keep. "The tool could
# not run" and "the tool ran and the venue does not have some candles" need
# completely different responses: the first is a config or network problem to fix,
# the second is a fact about the data to look at and probably accept. Collapsing
# both into 1 would throw that away at the only moment it costs nothing to keep.
#
# The harder call is 2 rather than 0 for a run that completed with known holes. An
# old permanent scar makes every run afterwards exit 2 forever, which is how an
# exit code becomes noise you learn to ignore -- a real risk. It is still the right
# way round: a store with holes that reports success is precisely the silent
# failure requirement 6 exists to prevent, and if the noise does become a problem
# the fix is a list of accepted gaps to compare against, not a quieter exit code.
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INCOMPLETE = 2

# Errors that mean "this run cannot proceed" rather than "this project has a bug".
# Kept as a tuple so the list of what gets a friendly message is visible in one
# place, and so nothing wider than this can be caught by accident.
USER_ERRORS = (settings.ConfigError, TimeframeError, StoreError, ExchangeError)

__all__ = [
    "EXIT_FAILED",
    "EXIT_INCOMPLETE",
    "EXIT_OK",
    "build_parser",
    "main",
    "run",
]


def build_parser():
    """
    The argument surface.

    Every override defaults to None rather than to a sensible value. That looks
    like an oversight and is the opposite: if the parser supplied `--timeframe 1h`
    as its own default, that default would arrive at `settings.resolve` as an
    explicit value and win over the config file, and the config file would quietly
    stop meaning anything. None is how "the user did not say" survives the trip.

    `allow_abbrev=False` turns off argparse's default willingness to accept any
    unambiguous prefix of a flag. That default is convenient and quietly unstable:
    `--data` means `--data-dir` only until some later version adds `--database`,
    at which point a command that worked for a year stops working, and `--symbol`
    means `--symbols` only by luck. A flag either exists or it does not.
    """
    parser = argparse.ArgumentParser(
        prog="collect",
        allow_abbrev=False,
        description=(
            "Backfill historical OHLCV candles into a local Parquet store. "
            "Reads public market data only -- no API credentials are used or "
            "accepted. Safe to run repeatedly: it fetches only what is missing."
        ),
        epilog=(
            "Exit status: 0 nothing missing, 1 the run could not complete, "
            "2 the run completed but candles are missing or a symbol failed "
            "(see logs/gaps.json)."
        ),
    )

    parser.add_argument(
        "--config",
        default=settings.DEFAULT_CONFIG_PATH,
        metavar="PATH",
        help="JSON config file holding the symbol list (default: %(default)s)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        metavar="SYMBOL",
        help="override the config's symbols, e.g. --symbols BTC/USDT ETH/USDT",
    )
    parser.add_argument(
        "--timeframe",
        metavar="TF",
        help="override the config's timeframe, e.g. 1h, 4h, 1d",
    )
    parser.add_argument(
        "--start",
        metavar="DATE",
        help="override the config's start, as UTC YYYY-MM-DD or YYYY-MM-DDTHH:MM",
    )
    parser.add_argument(
        "--exchange",
        metavar="ID",
        help="override the config's exchange, as a ccxt id such as binance",
    )
    parser.add_argument(
        "--data-dir",
        default=settings.DEFAULT_DATA_DIR,
        metavar="DIR",
        help="root of the Parquet store (default: %(default)s)",
    )
    parser.add_argument(
        "--log-dir",
        default=settings.DEFAULT_LOG_DIR,
        metavar="DIR",
        help="where the log and the gap report are written (default: %(default)s)",
    )
    parser.add_argument(
        "--max-gaps",
        type=int,
        metavar="N",
        help=(
            f"older holes to attempt repairing per symbol per run "
            f"(default: {backfill.DEFAULT_MAX_GAPS}). 0 brings the store up to "
            f"date without retrying old holes."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        # Not "log every request", which is what this said until a check of the
        # actual output caught it: logging_setup pins ccxt and urllib3 to WARNING
        # precisely so their DEBUG chatter cannot bury our own records. What DEBUG
        # actually adds is store.py's per-write detail.
        help="log each write to the store at DEBUG level",
    )

    return parser


def _resolve_settings(args, now_ms):
    """
    Load the config, apply the overrides, and check the result against the clock.

    The future-start check is here rather than in settings.py because it is the
    one piece of validation that needs the time, and settings.py is deliberately
    clock-free.

    It is worth doing at all because the failure is so quiet. Asking for a start
    in 2099 makes the requested range empty, an empty range has nothing missing,
    every symbol is therefore complete, and the run exits 0 having done nothing
    whatsoever. A mistyped year would look like a successful backfill.
    """
    config = settings.load_config(args.config)

    resolved = settings.resolve(
        config,
        exchange=args.exchange,
        timeframe=args.timeframe,
        start=args.start,
        symbols=args.symbols,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        max_gaps=args.max_gaps,
    )

    if resolved.start_ms > now_ms:
        raise settings.ConfigError(
            f"start {to_utc_string(resolved.start_ms)} is in the future "
            f"(now is {to_utc_string(now_ms)}). Nothing could be fetched, and the "
            f"run would report success having done nothing. Check the year."
        )

    return resolved


def _collect(client, resolved, now_ms, end_ms):
    """
    Back-fill every symbol, and return `(results, errors)`.

    One failing symbol does not stop the others. A misspelled symbol, or one the
    venue has delisted, should not cost you the other four backfills -- and since
    every run is resumable, the ones that did work keep their progress.

    The failure is collected rather than swallowed. It goes into the report and it
    changes the exit code; continuing past it is a convenience, not an acquittal.
    KeyboardInterrupt is deliberately not caught here, so Ctrl-C stops the run
    instead of being logged once per remaining symbol.
    """
    results = []
    errors = []

    for symbol in resolved.symbols:
        path = store.candle_path(
            resolved.data_dir, resolved.exchange, symbol, resolved.timeframe
        )

        try:
            results.append(
                backfill.backfill_symbol(
                    client,
                    path,
                    symbol,
                    resolved.timeframe,
                    resolved.start_ms,
                    now_ms,
                    max_gaps=resolved.max_gaps,
                )
            )
        except (ExchangeError, StoreError) as failure:
            # Recorded with the type name because the message alone rarely says
            # whether this was a bad symbol, a dead connection or a full disk,
            # and that is the first thing you want to know from the report.
            errors.append(
                {"symbol": symbol, "error": f"{type(failure).__name__}: {failure}"}
            )
            log.error("%s: giving up on this symbol -- %s", symbol, failure)

    return results, errors


def run(argv=None, *, now_ms=None, exchange_factory=None):
    """
    Run one collection and return the exit code.

    Returns rather than exits, so that the whole program can be driven from a test
    with the clock pinned and the exchange substituted. `main` is the two-line
    wrapper that turns the return value into a process exit status.

    `now_ms` and `exchange_factory` are resolved inside the body rather than in
    the signature. For the clock that is required -- a default evaluated at import
    would freeze the time the module was loaded. For the factory it is the lesson
    from fetch_page: a default argument captures the function object at import,
    which makes it overridable but not interceptable.
    """
    args = build_parser().parse_args(argv)

    if now_ms is None:
        import time

        now_ms = int(time.time() * 1000)

    if exchange_factory is None:
        exchange_factory = build_exchange

    # Config errors are reported before logging is configured, because setting up
    # logging would create the log directory -- and a run that could not read its
    # own config should leave nothing behind to clean up.
    try:
        resolved = _resolve_settings(args, now_ms)
    except USER_ERRORS as failure:
        print(f"error: {failure}", file=sys.stderr)
        return EXIT_FAILED

    log_file = configure_logging(
        resolved.log_dir, level=logging.DEBUG if args.verbose else logging.INFO
    )

    end_ms = last_closed_open_time(now_ms, resolved.timeframe_ms)

    log.info(
        "collecting %d symbol(s) of %s from %s on %s",
        len(resolved.symbols),
        resolved.timeframe,
        to_utc_string(resolved.start_ms),
        resolved.exchange,
    )
    # Spelled out rather than abbreviated. Reading a real log showed the terse
    # version, "store data, log logs/collector.log", parses as a sentence with
    # two verbs in it before it parses as two paths.
    log.info("writing candles to %s/ and logs to %s", resolved.data_dir, log_file)

    try:
        client = exchange_factory(resolved.exchange)
        results, errors = _collect(client, resolved, now_ms, end_ms)
    except KeyboardInterrupt:
        # Requirement 3's other half. The store is already safe -- writes are
        # atomic and flushed as the run goes -- so all that is left is to say so
        # plainly instead of printing a stack trace at whatever line it landed on.
        #
        # No report is written. A report states what a finished run established,
        # and an interrupted run established less than the previous one did; the
        # atomic write means the older, truer report survives untouched.
        log.warning("interrupted -- candles already written are intact")
        print(
            "interrupted. Whatever was already written is intact; "
            "run the same command again to carry on.",
            file=sys.stderr,
        )
        return EXIT_FAILED
    except USER_ERRORS as failure:
        # A failure outside any single symbol: an unknown exchange id, a bad
        # timeframe reaching store.candle_path. Per-symbol errors never arrive
        # here; _collect keeps those and carries on.
        log.error("%s", failure)
        print(f"error: {failure}", file=sys.stderr)
        return EXIT_FAILED

    built = report.build_report(
        results,
        exchange=resolved.exchange,
        timeframe=resolved.timeframe,
        start_ms=resolved.start_ms,
        # The range actually covered, not the range asked for. The request runs to
        # "now", but nothing past the last closed candle can be verified, and a
        # report claiming otherwise overstates what it knows.
        end_ms=end_ms,
        generated_at_ms=now_ms,
        errors=errors,
    )
    report_file = report.write_report(resolved.log_dir, built)

    # Through the log rather than print, so the outcome of every run is in the
    # rotating file too. Logging already goes to stdout, so this is still the last
    # line a person sees.
    log.info("%s", report.summarise(built))

    if not built["complete"]:
        log.warning("this run did not verify the full range -- see %s", report_file)

    return EXIT_OK if built["complete"] else EXIT_INCOMPLETE


def main():
    """Console entry point. Turns the exit code into a process exit status."""
    sys.exit(run())


if __name__ == "__main__":
    main()
