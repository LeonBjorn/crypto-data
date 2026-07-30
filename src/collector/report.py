"""The gap report: a machine-readable statement of what is missing and why.

Requirement 6 says missing candles must be reported rather than failing silently.
backfill.py already logs each hole it finds, but a log is not a report -- nothing
can read it, and it scrolls away. This module writes the same information to
`logs/gaps.json`, where a backtester can check it before trusting a range, and
where you can check it without reading a thousand log lines.

One file, rewritten on every run. It answers one question -- what is missing right
now -- and a folder of timestamped reports answers that question badly, because
you have to work out which one is current before you can begin. Run-by-run history
is what the rotating log is for, so keeping it here too would duplicate it.

It is written even when nothing is missing. "As of 12:04 today, all five symbols
were verified complete" is information, and it is not the same as the absence of a
file, which could equally mean nothing has ever run.

The one thing the report must never do is understate what is missing, because
anything reading it will act on it. That shapes two decisions below: `complete` is
false unless positively established, including for a run that covered no symbols
at all; and the file is written atomically, so a run killed mid-write leaves the
previous report rather than a truncated one that still parses.
"""

import json
import logging
import os
from pathlib import Path

from collector.timeframes import to_utc_string

log = logging.getLogger(__name__)

FILENAME = "gaps.json"

# Bumped if the shape of the file changes. A reader written against this shape can
# then refuse a later one loudly instead of quietly misreading it -- silently
# reading the wrong field is the failure mode that costs a day.
#
# 2 added `errors` and `totals.symbols_failed`, when wiring up the CLI showed that
# a symbol whose fetch raised had no representation here at all. It was not merely
# absent from the file: a run that died on the first of five symbols would have
# reported four symbols, nothing missing, complete: true.
SCHEMA_VERSION = 2

__all__ = [
    "FILENAME",
    "SCHEMA_VERSION",
    "build_report",
    "gap_to_dict",
    "report_path",
    "summarise",
    "write_report",
]


def gap_to_dict(gap):
    """
    One gap as plain JSON-serialisable data.

    Both a readable timestamp and epoch milliseconds are stored, which is
    redundant on purpose. The file has two audiences: a person who needs to see
    when the hole was without doing arithmetic, and a program that needs a number
    it can compare without parsing a date format. Recording only one of them makes
    the other do work it should not have to, and the cost is a few bytes.
    """
    return {
        "start": to_utc_string(gap.start_ms),
        "end": to_utc_string(gap.end_ms),
        "start_ms": gap.start_ms,
        "end_ms": gap.end_ms,
        "candles": gap.count,
    }


def _symbol_entry(result):
    """One symbol's outcome.

    `missing` and `not_attempted` are kept apart because they call for different
    responses. Missing means the exchange was asked and had nothing, so the candles
    probably do not exist and no amount of rerunning will produce them.
    Not attempted means the gap cap was reached and nobody asked, so another run
    will make progress. Merging the two would make a merely capped run look like a
    permanently damaged store.
    """
    missing = [gap_to_dict(gap) for gap in result.unfilled]
    not_attempted = [gap_to_dict(gap) for gap in result.skipped]

    return {
        "symbol": result.symbol,
        "complete": not missing and not not_attempted,
        "candles_added": result.added,
        "pages_fetched": result.pages,
        "gaps_repaired": len(result.filled),
        "candles_missing": sum(gap.count for gap in result.unfilled),
        "candles_not_attempted": sum(gap.count for gap in result.skipped),
        "missing": missing,
        "not_attempted": not_attempted,
    }


def build_report(
    results, *, exchange, timeframe, start_ms, end_ms, generated_at_ms, errors=None
):
    """
    Turn a run's Results into the dict that gets written to disk.

    Separated from writing so the decisions -- what counts as complete, what the
    totals are -- can be tested against plain data, with no filesystem involved.

    `complete` is true only if there is at least one symbol, nothing failed, and
    every symbol is complete. The emptiness check is not defensive padding:
    `all([])` is True, so the obvious implementation would declare a run that
    fetched nothing at all to be a clean bill of health. A run that verified
    nothing has established nothing.

    `errors` carries symbols whose fetch raised, as `{"symbol": ..., "error":
    ...}`. They are deliberately not in `symbols`, because that list means "was
    checked" and a symbol that blew up was not. But they must still drag
    `complete` down to false: the CLI continues past a failed symbol so that one
    bad symbol cannot deny you four good backfills, and without this the run
    where the connection died on the first of five would report four symbols,
    nothing missing, complete: true. Every clause of that is accurate and the
    whole of it is a lie.

    Note that the report describes *this run*. If a run covered two of five
    symbols, only those two appear, and `complete` speaks only for them. The list
    of symbols is what tells a reader which ones were actually checked.
    """
    entries = [_symbol_entry(result) for result in results]
    failures = list(errors or [])

    return {
        "schema": SCHEMA_VERSION,
        "generated_at": to_utc_string(generated_at_ms),
        "generated_at_ms": generated_at_ms,
        "exchange": exchange,
        "timeframe": timeframe,
        "requested_start": to_utc_string(start_ms),
        "requested_end": to_utc_string(end_ms),
        "requested_start_ms": start_ms,
        "requested_end_ms": end_ms,
        "complete": (
            bool(entries)
            and not failures
            and all(entry["complete"] for entry in entries)
        ),
        "totals": {
            "symbols": len(entries),
            "symbols_failed": len(failures),
            "candles_added": sum(entry["candles_added"] for entry in entries),
            "pages_fetched": sum(entry["pages_fetched"] for entry in entries),
            "gaps_repaired": sum(entry["gaps_repaired"] for entry in entries),
            # Candles and gaps are counted separately because they answer
            # different questions. Three one-candle holes and one three-candle
            # hole are the same number of candles and very different situations.
            "candles_missing": sum(entry["candles_missing"] for entry in entries),
            "candles_not_attempted": sum(
                entry["candles_not_attempted"] for entry in entries
            ),
            "gaps_missing": sum(len(entry["missing"]) for entry in entries),
            "gaps_not_attempted": sum(len(entry["not_attempted"]) for entry in entries),
        },
        "symbols": entries,
        # Listed last because it is usually empty, and a reader scanning the file
        # by eye should not have to scroll past `"errors": []` to reach the data.
        # Always present, though: an optional key is a key somebody forgets to
        # check for.
        "errors": failures,
    }


def report_path(log_dir):
    """Where the report lives. One function so nothing hardcodes the name twice."""
    return Path(log_dir) / FILENAME


def write_report(log_dir, built):
    """
    Write the report to `<log_dir>/gaps.json`, atomically.

    Returns the path written.

    The temp-file-then-os.replace pattern is the same one store.write_candles
    uses, and for the same reason: os.replace either happens completely or not at
    all, so a reader never sees a partial file. That matters more here than it
    might look, because a truncated JSON file can still parse -- just with fewer
    symbols in it -- and would then be read as an authoritative statement that
    less is missing than really is.

    Unlike the candle store, this deliberately does not fsync. Durability across a
    power cut is not worth buying for a file that one rerun regenerates, whereas
    refetching two years of candles is not free. The asymmetry is intentional
    rather than an oversight.
    """
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)

    target = report_path(directory)
    temp_path = target.with_name(target.name + ".tmp")

    # indent=2 because this file is read by a person far more often than by a
    # program, and five symbols of single-line JSON is unreadable in a terminal.
    temp_path.write_text(json.dumps(built, indent=2) + "\n", encoding="utf-8")

    try:
        os.replace(temp_path, target)
    except OSError:
        # Leave the previous report in place and take the temp file with us, so a
        # failed write cannot be mistaken later for a real report.
        temp_path.unlink(missing_ok=True)
        raise

    return target


def summarise(built):
    """
    One line for stdout at the end of a run.

    Many runs will never have anything but this line read, so it carries the bad
    news itself rather than pointing at the JSON file. A summary that says "see
    logs/gaps.json for details" and nothing else is how a problem goes unnoticed
    for a month.
    """
    totals = built["totals"]
    symbol_word = "symbol" if totals["symbols"] == 1 else "symbols"
    parts = [
        f"{totals['symbols']} {symbol_word}",
        f"{totals['candles_added']} candle(s) added",
    ]

    if totals["gaps_repaired"]:
        parts.append(f"{totals['gaps_repaired']} gap(s) repaired")

    if totals["candles_missing"]:
        parts.append(
            f"{totals['candles_missing']} candle(s) missing "
            f"in {totals['gaps_missing']} gap(s)"
        )

    if totals["candles_not_attempted"]:
        parts.append(
            f"{totals['candles_not_attempted']} candle(s) in "
            f"{totals['gaps_not_attempted']} gap(s) not attempted"
        )

    if totals["symbols_failed"]:
        parts.append(f"{totals['symbols_failed']} symbol(s) failed")

    # "nothing missing" is a claim about the whole run, so it cannot be made while
    # any part of the run is unaccounted for. A failed symbol was never checked at
    # all, which is the least clean outcome there is, not a neutral one.
    if (
        not totals["candles_missing"]
        and not totals["candles_not_attempted"]
        and not totals["symbols_failed"]
    ):
        parts.append("nothing missing")

    return ", ".join(parts) + "."
