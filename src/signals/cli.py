"""The research command: run the rules against the store and print the verdict.

Milestone 2's entry point, and like the collector's it is deliberately thin.
Everything with a decision in it lives elsewhere and is tested against plain
data -- what a rule fires on (rules), whether it could have known (lookahead),
what a signal is worth (trades), and whether that is more than nothing
(evaluate). What is left here is wiring, formatting, and three choices about
what a research tool should make easy.

THE BARE COMMAND SHOWS EVERYTHING
    `signals` with no arguments prints every rule at all four holds, not
    one number. The obvious design is `--rule breakout` printing a single
    verdict, and it is the wrong default: the failure mode of this tool is not
    a crash, it is a person reading one flattering percentile out of twelve and
    believing it. Twelve comparisons on one sample expect roughly one above the
    95th percentile by chance, and the only reliable way to keep that in view
    is to put the other eleven next to it. The flags narrow the table once you
    already know what you are looking at.

THERE ARE ONLY TWO EXIT CODES
    0 the run completed, 1 it could not. The collector has a third meaning "the
    run finished and the data has holes", which is a fact about disk that a
    scheduled job can act on. There is no equivalent here. Every number this
    prints is an opinion about one two-year sample, and the moment "a rule beat
    the benchmark" is an exit code, something automatic gets built on top of
    one. A research tool should be awkward to automate.

THE GUARD RUNS BEFORE ANYTHING IS MEASURED
    A rule that reads the future produces beautiful output, and it produces it
    from code that looks fine line by line. So the causality checks run first,
    and if a rule fails them the run stops without printing a table -- because
    a number on the screen with a warning underneath it is still a number on
    the screen. It costs about a second. `--no-check` skips it.

This module reads the store and never writes to it. It places no orders, holds
no credentials, and has no way of authenticating against anything.
"""

import argparse
import json
import sys

from collector import settings
from collector.store import StoreError
from collector.timeframes import TimeframeError, to_utc_string

from . import evaluate, indicators, lookahead, prices, rules, trades

EXIT_OK = 0
EXIT_FAILED = 1

# Errors that mean "this run cannot proceed" rather than "this project has a
# bug". Everything else is left to raise, because a traceback is how a bug gets
# found and a friendly message is how it survives to next year.
USER_ERRORS = (
    settings.ConfigError,
    TimeframeError,
    StoreError,
    prices.PriceError,
    rules.RuleError,
    indicators.IndicatorError,
    trades.TradeError,
    evaluate.EvalError,
    lookahead.LookaheadError,
)

# Above this, a rule is worth a second look -- and no more than that. It is the
# threshold the summary line counts against, kept as a name so that the number
# in the code and the number in the sentence cannot drift apart.
NOTABLE = 95.0

__all__ = [
    "EXIT_FAILED",
    "EXIT_OK",
    "NOTABLE",
    "build_parser",
    "main",
    "run",
]


def build_parser():
    """
    The argument surface.

    The overrides default to None for the same reason they do in the collector:
    a parser-supplied default arrives at `settings.resolve` as an explicit value
    and beats the config file, which would quietly stop meaning anything.

    `--draws` and `--seed` are the exception and default to real values, because
    they are not config-file settings -- nothing about a stored candle has an
    opinion about how many random selections to compare a rule against.
    """
    parser = argparse.ArgumentParser(
        prog="signals",
        allow_abbrev=False,
        description=(
            "Measure buy rules against the stored candles. Compares each rule "
            "with the trade taken at every eligible bar, and with random "
            "selections of the same size. Reads the store only -- it places no "
            "orders and uses no credentials."
        ),
        epilog=(
            "Exit status: 0 the run completed, 1 it could not. A rule beating "
            "the benchmark does not change it, on purpose."
        ),
    )

    parser.add_argument(
        "--config",
        default=settings.DEFAULT_CONFIG_PATH,
        metavar="PATH",
        help="JSON config file holding the symbol list (default: %(default)s)",
    )
    parser.add_argument(
        "--data-dir",
        default=settings.DEFAULT_DATA_DIR,
        metavar="DIR",
        help="root of the Parquet store to read (default: %(default)s)",
    )
    parser.add_argument(
        "--exchange",
        metavar="ID",
        help="override the config's exchange, as a ccxt id such as binance",
    )
    parser.add_argument(
        "--timeframe",
        metavar="TF",
        help="override the config's timeframe, e.g. 1h, 4h, 1d",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        metavar="SYMBOL",
        help="override the config's symbols, e.g. --symbols BTC/USDT ETH/USDT",
    )

    parser.add_argument(
        "--rule",
        nargs="+",
        choices=rules.names(),
        metavar="NAME",
        help=f"measure only these rules (default: all of {', '.join(rules.names())})",
    )
    parser.add_argument(
        "--hold",
        nargs="+",
        type=int,
        metavar="BARS",
        help=(
            f"bars to hold each trade "
            f"(default: {' '.join(str(hold) for hold in trades.DEFAULT_HOLDS)})"
        ),
    )
    parser.add_argument(
        "--trail",
        type=float,
        metavar="FRAC",
        help="exit early when the price falls this fraction below its peak since "
        "entry, e.g. 0.05 for a 5%% trailing stop (default: off, a fixed-time exit)",
    )
    parser.add_argument(
        "--start",
        metavar="DATE",
        help="measure from this UTC date onwards, as YYYY-MM-DD or YYYY-MM-DDTHH:MM",
    )
    parser.add_argument(
        "--end",
        metavar="DATE",
        help="measure up to this UTC date",
    )

    parser.add_argument(
        "--per-symbol",
        action="store_true",
        help="break the strongest result down by symbol, to see what carried it",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="re-measure the strongest result on each half of the history separately",
    )
    parser.add_argument(
        "--walk",
        type=int,
        metavar="N",
        help="re-measure the strongest result across N sequential windows of the "
        "history, each against its own benchmark -- a generalisation of --split "
        "(which is the N=2 case) for seeing whether an edge holds period to period",
    )

    parser.add_argument(
        "--draws",
        type=int,
        default=evaluate.DEFAULT_DRAWS,
        metavar="N",
        help="random selections to compare each rule against (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=evaluate.DEFAULT_SEED,
        metavar="N",
        help="seed for those selections, so a run is reproducible (default: %(default)s)",
    )
    parser.add_argument(
        "--no-costs",
        action="store_true",
        help="price every trade at zero fees and no slippage, for comparison only",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="skip the lookahead guard (it takes about a second, and it is the "
        "only thing standing between you and a rule that reads the future)",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="also write the comparison table to this file as JSON",
    )

    return parser


def _window(args):
    """The measurement window as `(start_ms, end_ms)`, either possibly None.

    Deliberately separate from the config's `start`. That value says how far
    back the collector should *fetch*, and reusing it here would silently make
    every measurement begin wherever the backfill happened to begin. What you
    want to research over is what is on disk, unless you say otherwise.
    """
    start_ms = settings.parse_start(args.start) if args.start is not None else None
    end_ms = settings.parse_start(args.end) if args.end is not None else None

    if start_ms is not None and end_ms is not None and end_ms <= start_ms:
        raise settings.ConfigError(
            f"--end {args.end} is not after --start {args.start}, which leaves "
            f"no candles to measure."
        )

    return start_ms, end_ms


def _load(resolved, start_ms, end_ms):
    """Every requested symbol's candles, keyed by symbol and in the given order."""
    return {
        symbol: prices.load(
            resolved.data_dir,
            resolved.exchange,
            symbol,
            resolved.timeframe,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        for symbol in resolved.symbols
    }


def _guard(candles, rule_names):
    """Run the causality checks and return `(text, failed)`.

    Checked against one symbol rather than all of them, and specifically the
    longest history loaded. Whether a rule reads the future is a property of
    its code, not of which coin it was handed; running the same structural
    question five times over five correlated series costs five times as much
    and answers it no better. The longest frame is the most informative single
    choice because the guard's truncations need history to cut.
    """
    symbol = max(candles, key=lambda name: len(candles[name]))
    lines = [f"lookahead guard, on {symbol} ({len(candles[symbol])} candles)"]
    failed = []

    for name in rule_names:
        # A rule that never fires does not reach here as a pass: the guard
        # itself refuses it, because a rule with no signals gives the
        # truncation checks nothing to compare and it says so.
        report = lookahead.check(candles[symbol], name)
        lines.append("  " + report.summary().replace("\n", "\n  "))
        if not report.ok:
            failed.append(name)

    return "\n".join(lines), failed


def _sample(candles, name, hold, costs, symbol, trail=None):
    """One symbol's trades for one rule, alongside the population they came from.

    `trail` reaches both sides of the comparison through evaluate.collect, so
    the rule and its every-bar benchmark are always exited the same way -- the
    one thing that would otherwise quietly rig the percentile.
    """
    return evaluate.collect(
        candles,
        rules.apply(name, candles),
        hold=hold,
        costs=costs,
        symbol=symbol,
        trail=trail,
    )


def _verdict(candles, name, hold, *, costs, draws, seed, trail=None):
    """Pool one rule at one hold across every symbol, and judge it."""
    pooled = evaluate.pool(
        [
            _sample(frame, name, hold, costs, symbol, trail=trail)
            for symbol, frame in candles.items()
        ]
    )

    if not len(pooled.taken):
        # Every statistic over an empty selection is NaN, so left alone this
        # ends in a table of nan% against nan% -- which reads like broken
        # arithmetic rather than a rule that did nothing.
        raise evaluate.EvalError(
            f"{name} fired on no bars of {pooled.label()} in this window, so "
            f"there is nothing to compare against the benchmark. Try a longer "
            f"history, or check the rule's parameters."
        )

    return evaluate.judge(pooled, draws=draws, seed=seed)


def _table(candles, rule_names, holds, *, costs, draws, seed, trail=None):
    """Every rule at every hold, in the order asked for."""
    return {
        (name, hold): _verdict(
            candles, name, hold, costs=costs, draws=draws, seed=seed, trail=trail
        )
        for name in rule_names
        for hold in holds
    }


def _bars(frames):
    """How many candles each frame holds: one number, or the range if they differ.

    Printed in the header and again on each half of a split. The split is the
    place it earns itself -- "second half, 300 candles" is the only thing on
    the screen that says the two halves were the same size, and a split that
    quietly cut somewhere other than the middle would otherwise look identical.
    """
    counts = {len(frame) for frame in frames}
    if len(counts) == 1:
        return str(counts.pop())
    return f"{min(counts)}-{max(counts)}"


ROW = "  {:14s} {:>5} {:>7} {:>11} {:>11} {:>11}"


def _row(label, hold, verdict):
    return ROW.format(
        label,
        hold,
        verdict.rule.trades,
        f"{verdict.rule.mean:.3%}",
        f"{verdict.baseline.mean:.3%}",
        f"{verdict.percentile:.1f}",
    )


def _format_table(table):
    """The comparison grid, plus the line that keeps it in proportion."""
    lines = [ROW.format("rule", "hold", "trades", "rule mean", "every bar", "percentile")]
    lines.extend(_row(name, hold, verdict) for (name, hold), verdict in table.items())

    above = sum(1 for verdict in table.values() if verdict.percentile > NOTABLE)
    expected = len(table) * (100.0 - NOTABLE) / 100.0
    lines.append("")
    lines.append(f"  above the {NOTABLE:.0f}th percentile: {above} of {len(table)}")
    lines.append(
        f"  one comparison in twenty clears that by chance, so {len(table)} of "
        f"them expect {expected:.1f}."
    )
    return "\n".join(lines)


def _strongest(table):
    """The cell that looked best, which is the one worth trying to break."""
    return max(table, key=lambda key: table[key].percentile)


def _format_per_symbol(candles, name, hold, *, costs, draws, seed, trail=None):
    """The same rule and hold, one symbol at a time.

    Pooled numbers hide the case where a rule worked on one symbol and was
    noise on the rest, which on crypto is common: the coins move together but
    not equally. It is the difference between a finding and a coincidence.
    """
    lines = [
        f"{name} at a {hold}-bar hold, symbol by symbol",
        ROW.format("symbol", "hold", "trades", "rule mean", "every bar", "percentile"),
    ]
    for symbol, frame in candles.items():
        verdict = evaluate.judge(
            _sample(frame, name, hold, costs, symbol, trail=trail), draws=draws, seed=seed
        )
        lines.append(_row(symbol, hold, verdict))
    return "\n".join(lines)


def _judge_pieces(pieces, name, hold, *, costs, draws, seed, trail):
    """Pool one rule and hold across `(symbol, frame)` pieces and judge it.

    The single place --split and --walk turn a set of time-slices into a
    verdict, so the two cannot drift apart in how they pool or seed.
    """
    return evaluate.judge(
        evaluate.pool(
            [_sample(piece, name, hold, costs, symbol, trail=trail) for symbol, piece in pieces]
        ),
        draws=draws,
        seed=seed,
    )


def _format_split(candles, name, hold, *, costs, draws, seed, trail=None):
    """The same rule and hold on each half of the history.

    The cheapest thing that resembles an out-of-sample test, and not a
    substitute for one -- both halves are the same period and the same coins.
    But a result that appears in one half and vanishes in the other was never a
    result, and finding that out costs one more pass.

    Each half is measured against its own every-bar benchmark. Against the whole
    period's, this would be measuring which half the market went up in, which is
    a fact about the market rather than about the rule.
    """
    lines = [f"{name} at a {hold}-bar hold, each half of the history on its own"]

    for label in ("first half", "second half"):
        pieces = []
        for symbol, frame in candles.items():
            middle = len(frame) // 2
            piece = frame.iloc[:middle] if label == "first half" else frame.iloc[middle:]
            pieces.append((symbol, piece))

        verdict = _judge_pieces(pieces, name, hold, costs=costs, draws=draws, seed=seed, trail=trail)
        lines.append(
            f"  {label:12s} {_bars(piece for _, piece in pieces):>9s} candles   "
            f"{verdict.rule.trades:6d} trades   "
            f"rule {verdict.rule.mean:8.3%}   "
            f"every bar {verdict.baseline.mean:8.3%}   "
            f"percentile {verdict.percentile:6.1f}"
        )

    return "\n".join(lines)


def _window_slice(frame, index, windows):
    """The `index`-th of `windows` equal, contiguous, non-overlapping slices.

    Boundaries are `index * n // windows`, which tiles the frame exactly: no bar
    is dropped between windows and none is counted in two, whatever the length.
    """
    n = len(frame)
    return frame.iloc[index * n // windows:(index + 1) * n // windows]


def _format_walk(candles, name, hold, *, windows, costs, draws, seed, trail=None):
    """The strongest result measured across `windows` sequential slices of time.

    --split generalised from two halves to N, and it carries the same caveat
    writ larger: the cell was chosen on the whole sample, so this shows whether
    an already-picked result holds up period to period, not a clean out-of-sample
    selection. What it is good at is making decay visible -- an edge that lived
    entirely in the first window and was noise or worse in the rest was a fact
    about one stretch of the market, not about the rule. This project has already
    produced exactly that shape once, so the tool to see it earns its place.

    Each window is judged against its own every-bar benchmark. Against the whole
    period's, this would measure which windows the market rose in rather than
    anything about the timing.
    """
    lines = [f"{name} at a {hold}-bar hold, across {windows} sequential windows"]

    for index in range(windows):
        pieces = [(symbol, _window_slice(frame, index, windows)) for symbol, frame in candles.items()]

        try:
            verdict = _judge_pieces(pieces, name, hold, costs=costs, draws=draws, seed=seed, trail=trail)
        except evaluate.EvalError as failure:
            # Over-splitting is the usual cause: a window too short to hold a
            # trade. The underlying message names the candle count and hold, but
            # not that windowing produced it -- so say so, and point at the dial.
            raise evaluate.EvalError(
                f"window {index + 1} of {windows} could not be measured -- "
                f"{failure} Use fewer windows than --walk {windows}."
            ) from failure

        opened = min(int(piece["timestamp"].iloc[0]) for _, piece in pieces)
        closed = max(int(piece["timestamp"].iloc[-1]) for _, piece in pieces)
        span = f"{to_utc_string(opened)[:10]}..{to_utc_string(closed)[:10]}"
        lines.append(
            f"  {index + 1}/{windows}  {span}   "
            f"{verdict.rule.trades:6d} trades   "
            f"rule {verdict.rule.mean:8.3%}   "
            f"every bar {verdict.baseline.mean:8.3%}   "
            f"percentile {verdict.percentile:6.1f}"
        )

    return "\n".join(lines)


def _header(candles, resolved, table_size, costs, args):
    """What was measured, on what, priced how. Printed above every table.

    Worth the five lines. A percentile copied out of a terminal a week later is
    unreadable without them -- the same rule over a different window, or with
    costs switched off, is a different number entirely.
    """
    span = f"{_bars(candles.values())} candles per symbol"

    first = min(int(frame["timestamp"].iloc[0]) for frame in candles.values())
    last = max(int(frame["timestamp"].iloc[-1]) for frame in candles.values())

    priced = (
        "no costs"
        if args.no_costs
        else f"fee {costs.fee:.3%}, slippage {costs.slippage:.3%} per side"
    )

    # The exit is part of what a number means: the same rule with a trailing
    # stop is a different measurement, so it belongs in the header a percentile
    # is read back with, not left to be remembered.
    exit_line = (
        f"held to time, or exited on a {args.trail:.1%} trailing stop"
        if args.trail is not None
        else "held to time"
    )

    return "\n".join(
        [
            f"{table_size} comparison(s) over {', '.join(candles)}",
            f"  {resolved.exchange} {resolved.timeframe}, {span}",
            f"  {to_utc_string(first)} to {to_utc_string(last)}",
            f"  priced with {priced}, {exit_line}",
            f"  {args.draws} random draw(s) per comparison, seed {args.seed}",
        ]
    )


def _as_json(table, candles, resolved, costs, args):
    """The table as data, carrying enough to reproduce it.

    A percentile without its seed and draw count is a number nobody can check,
    including the person who produced it.
    """
    return {
        "exchange": resolved.exchange,
        "timeframe": resolved.timeframe,
        "symbols": list(candles),
        "candles": {symbol: len(frame) for symbol, frame in candles.items()},
        "draws": args.draws,
        "seed": args.seed,
        "trail": args.trail,
        "costs": {"fee": costs.fee, "slippage": costs.slippage},
        "comparisons": [
            {
                "rule": name,
                "hold": hold,
                "trades": verdict.rule.trades,
                "rule_mean": verdict.rule.mean,
                "rule_hit_rate": verdict.rule.hit_rate,
                "baseline_trades": verdict.baseline.trades,
                "baseline_mean": verdict.baseline.mean,
                "percentile": verdict.percentile,
            }
            for (name, hold), verdict in table.items()
        ],
    }


def _measure(args):
    """The whole run, with nothing caught. `run` is the layer that catches."""
    resolved = settings.resolve(
        settings.load_config(args.config),
        exchange=args.exchange,
        timeframe=args.timeframe,
        symbols=args.symbols,
        data_dir=args.data_dir,
    )

    start_ms, end_ms = _window(args)
    candles = _load(resolved, start_ms, end_ms)

    rule_names = args.rule if args.rule is not None else rules.names()
    holds = args.hold if args.hold is not None else list(trades.DEFAULT_HOLDS)
    costs = trades.FREE if args.no_costs else trades.DEFAULT_COSTS

    if not args.no_check:
        text, failed = _guard(candles, rule_names)
        print(text)
        print()
        if failed:
            raise lookahead.LookaheadError(
                f"{', '.join(failed)} did not pass the lookahead guard, so "
                f"nothing was measured. A rule that can see bars it would not "
                f"have had produces results that mean nothing at all."
            )

    table = _table(
        candles, rule_names, holds, costs=costs, draws=args.draws, seed=args.seed,
        trail=args.trail,
    )

    print(_header(candles, resolved, len(table), costs, args))
    print()
    print(_format_table(table))

    name, hold = _strongest(table)

    if args.per_symbol:
        print()
        print(
            _format_per_symbol(
                candles, name, hold, costs=costs, draws=args.draws, seed=args.seed,
                trail=args.trail,
            )
        )

    if args.split:
        print()
        print(
            _format_split(
                candles, name, hold, costs=costs, draws=args.draws, seed=args.seed,
                trail=args.trail,
            )
        )

    if args.walk is not None:
        if args.walk < 2:
            raise evaluate.EvalError(
                f"--walk needs at least 2 windows to compare, got {args.walk}. "
                f"One window is the main table over the whole history, which is "
                f"already printed above."
            )
        print()
        print(
            _format_walk(
                candles, name, hold, windows=args.walk, costs=costs,
                draws=args.draws, seed=args.seed, trail=args.trail,
            )
        )

    if args.json:
        # A plain write rather than the store's atomic one. Nothing reads this
        # file back, nothing is corrupted if a run is interrupted mid-write, and
        # the next run rewrites it -- so the ceremony would buy nothing.
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(_as_json(table, candles, resolved, costs, args), handle, indent=2)
            handle.write("\n")

    return EXIT_OK


def run(argv=None):
    """
    Run one measurement and return the exit code.

    Returns rather than exits, so the whole program can be driven from a test
    and its output read back. `main` is the wrapper that turns the return value
    into a process exit status.
    """
    args = build_parser().parse_args(argv)

    try:
        return _measure(args)
    except USER_ERRORS as failure:
        print(f"error: {failure}", file=sys.stderr)
        return EXIT_FAILED
    except KeyboardInterrupt:
        # Nothing was written and nothing is half-finished; this module only
        # reads. Saying so beats a stack trace at whichever draw it landed on.
        print("interrupted. Nothing was written.", file=sys.stderr)
        return EXIT_FAILED


def main():
    """Console entry point. Turns the exit code into a process exit status."""
    sys.exit(run())


if __name__ == "__main__":
    main()
