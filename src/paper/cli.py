"""The paper command: advance the wallet to the latest closed candle and say what happened.

Milestone 3.2, and like the other two entry points in this project it is
deliberately thin. Everything with a decision in it lives elsewhere and is
tested against plain data -- what a rule fires on (rules), what a trade is worth
(trades), what the wallet allows (account), what happens on a bar (engine), what
survives a restart (state). What is left here is wiring: read the config, load
the store, restore, advance, save, print.

THREE THINGS IT WILL NOT DO
---------------------------
*It does not fetch.* `collect` is the one command in this project that touches
the network, and keeping it that way means this one cannot fail in the middle of
a download and cannot be blamed for a gap. Run `collect` then `paper`; a
scheduler can chain them in that order.

*It does not act on the forming candle.* Only closed candles are loaded, the
same rule the collector enforces when writing them. The bar in progress has a
close that is not yet a close, and a rule computed on it is a rule computed on a
number that will change.

*It does not place an order.* There are no credentials here and no venue. The
Broker seam exists so that adding one later is a new class rather than a
rewrite, and the only implementation fills out of the candle it was handed.

EXIT CODES
----------
Two, matching `signals` rather than `collect`. Zero if the run advanced, one if
it could not. There is deliberately no third code for "the paper account is
losing money": that is an opinion about a strategy, and the moment it becomes an
exit status something automatic gets built on top of it.
"""

import argparse
import json
import sys
from pathlib import Path

from collector import settings
from collector.store import StoreError
from collector.timeframes import TimeframeError, timeframe_to_ms, to_utc_string
from paper import state as state_module
from paper.account import Account, AccountError
from paper.portfolio import Portfolio, PortfolioError
from signals import indicators, prices, rules
from signals.trades import Costs

__all__ = ["EXIT_FAILED", "EXIT_OK", "build_parser", "load_config", "main", "run"]

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_CONFIG_PATH = "config/paper.json"
# Not "paper/", which would sit beside src/paper and invite a directory on the
# import path shadowing the package of the same name.
DEFAULT_STATE_PATH = "state/paper.json"

USER_ERRORS = (
    settings.ConfigError,
    TimeframeError,
    StoreError,
    prices.PriceError,
    rules.RuleError,
    indicators.IndicatorError,
    AccountError,
    PortfolioError,
    state_module.StateError,
)

REQUIRED_KEYS = ("exchange", "timeframe", "symbols", "rule", "hold")

DEFAULTS = {
    "params": {},
    "stop": None,
    "target": None,
    "trail": None,
    "starting_capital": 10_000.0,
    "size_fraction": 0.2,
    "max_positions": 5,
    "one_per_symbol": True,
    "costs": {"fee": 0.001, "slippage": 0.0005},
}


def load_config(path):
    """Read the paper config, filling in what it does not say.

    Unknown keys are refused for the same reason `collector.settings` refuses
    them: `"symbol"` for `"symbols"` produces a file that looks entirely correct
    and a run that does nothing.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise settings.ConfigError(
            f"no paper config at {path}. Expected a JSON file with keys: "
            f"{', '.join(REQUIRED_KEYS)}."
        ) from None

    try:
        config = json.loads(text)
    except json.JSONDecodeError as failure:
        raise settings.ConfigError(f"{path} is not valid JSON: {failure}") from failure

    if not isinstance(config, dict):
        raise settings.ConfigError(f"{path} should contain a JSON object")

    known = set(REQUIRED_KEYS) | set(DEFAULTS)
    unknown = [key for key in config if not key.startswith("_") and key not in known]
    if unknown:
        raise settings.ConfigError(
            f"{path} has unrecognised key(s): {', '.join(sorted(unknown))}. "
            f"Recognised: {', '.join(sorted(known))}, plus any key starting with '_'."
        )

    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise settings.ConfigError(f"{path} is missing required key(s): {', '.join(missing)}")

    resolved = dict(DEFAULTS)
    resolved.update(config)

    if resolved["rule"] not in rules.RULES:
        raise settings.ConfigError(
            f"{path} names rule {resolved['rule']!r}, which does not exist. "
            f"Available: {', '.join(rules.names())}"
        )
    if not isinstance(resolved["symbols"], list) or not resolved["symbols"]:
        raise settings.ConfigError(f"{path}: symbols must be a non-empty list")

    return resolved


def build_parser():
    parser = argparse.ArgumentParser(
        prog="paper",
        allow_abbrev=False,
        description=(
            "Advance a paper-trading account to the latest closed candle in the "
            "local store. Reads the store only -- it never fetches, holds no "
            "credentials and places no orders. Run `collect` first."
        ),
        epilog="Exit status: 0 the run advanced, 1 it could not.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, metavar="PATH",
                        help="paper settings (default: %(default)s)")
    parser.add_argument("--data-dir", default="data", metavar="DIR",
                        help="root of the Parquet store (default: %(default)s)")
    parser.add_argument("--state", default=DEFAULT_STATE_PATH, metavar="PATH",
                        help="where the ledger and open positions live (default: %(default)s)")
    parser.add_argument("--reset", action="store_true",
                        help="discard the saved ledger and start a new one")
    parser.add_argument("--status", action="store_true",
                        help="print the current position without advancing anything")
    parser.add_argument("--json", metavar="PATH",
                        help="also write the dashboard snapshot here (default: beside --state)")
    return parser


def _load_frames(config, data_dir):
    """Every symbol's candles, refusing anything an indicator should not see."""
    frames = {}
    for symbol in config["symbols"]:
        frames[symbol] = prices.load(
            data_dir, config["exchange"], symbol, config["timeframe"]
        )
    return frames


def _build(config):
    account = Account(
        config["starting_capital"],
        size_fraction=config["size_fraction"],
        max_positions=config["max_positions"],
        one_per_symbol=config["one_per_symbol"],
    )
    costs = Costs(fee=config["costs"]["fee"], slippage=config["costs"]["slippage"])
    return Portfolio(
        config["symbols"],
        hold=config["hold"],
        stop=config["stop"],
        target=config["target"],
        trail=config["trail"],
        costs=costs,
        account=account,
    )


def _report(portfolio, frames, config, advanced):
    """What the run did, in the shape a person reads rather than a machine."""
    ledger = portfolio.ledger()
    equity = portfolio.equity(frames)
    capital = portfolio.account.starting_capital
    marks = portfolio.marks(frames)

    lines = [
        f"paper: {config['rule']} on {len(config['symbols'])} symbol(s), "
        f"{config['exchange']} {config['timeframe']}, {config['hold']}-bar hold",
        f"  advanced {advanced} bar(s) to "
        f"{to_utc_string(portfolio.cursor) if portfolio.cursor else 'nothing yet'}",
        "",
        f"  equity      {equity:>12,.2f}   ({equity / capital - 1:+.2%} on {capital:,.0f})",
        f"  cash        {portfolio.account.cash:>12,.2f}",
        f"  open        {len(portfolio.open_positions()):>12d} position(s)"
        f"  worth {sum(marks.values()):,.2f}",
        f"  closed      {len(ledger):>12d} trade(s)",
        f"  refused     {portfolio.rejections_total:>12d} signal(s) the wallet could not take",
    ]

    if len(ledger):
        wins = (ledger["net_return"] > 0).mean()
        lines += [
            "",
            f"  hit rate    {wins:>12.1%}",
            f"  mean net    {ledger['net_return'].mean():>12.3%} per trade",
            f"  best/worst  {ledger['net_return'].max():>+7.2%} /"
            f" {ledger['net_return'].min():+.2%}",
        ]

    if portfolio.open_positions():
        lines += ["", "  open positions"]
        for position in portfolio.open_positions():
            last = float(frames[position.symbol]["close"].iloc[-1])
            move = last / position.entry_price - 1
            lines.append(
                f"    {position.symbol:10s} entered {to_utc_string(position.entry_time)} "
                f"at {position.entry_price:,.4f}  now {last:,.4f}  {move:+.2%}"
            )

    return "\n".join(lines)


def _snapshot(portfolio, frames, config):
    """The dashboard's view of the world, as plain data.

    Written every run so that the page has something to read whether or not
    anything happened, and so that "no snapshot" and "a snapshot saying nothing
    changed" cannot be confused -- the same reasoning the collector applies to
    its gap report.
    """
    ledger = portfolio.ledger()
    marks = portfolio.marks(frames)
    capital = portfolio.account.starting_capital
    equity = portfolio.equity(frames)

    # The realised equity curve, and the drawdown read off it as it is built.
    # Peak-to-trough is the number that says how bad it got on the way, which a
    # final figure hides completely -- an account that ended up twenty percent
    # ahead having been thirty percent behind is not the same account.
    curve = []
    running = capital
    peak = capital
    max_drawdown = 0.0
    for row in ledger.to_dict(orient="records"):
        running += row["cash_out"] - row["cash_in"]
        peak = max(peak, running)
        max_drawdown = min(max_drawdown, running / peak - 1)
        curve.append({"t": int(row["exit_time"]), "equity": round(running, 2)})

    # Per symbol, over the whole ledger rather than the recent tail. The research
    # found the edge concentrated in some names and absent in others, so a
    # dashboard that only ever shows the pooled figure hides the one breakdown
    # most likely to change what you do next.
    by_symbol = []
    if len(ledger):
        for symbol, rows in ledger.groupby("symbol"):
            returns = rows["net_return"]
            by_symbol.append({
                "symbol": symbol,
                "trades": int(len(rows)),
                "hit_rate": round(float((returns > 0).mean()) * 100, 1),
                "mean_pct": round(float(returns.mean()) * 100, 3),
                "pnl": round(float((rows["cash_out"] - rows["cash_in"]).sum()), 2),
            })
        by_symbol.sort(key=lambda entry: entry["pnl"], reverse=True)

    step = timeframe_to_ms(config["timeframe"])
    hold_ms = config["hold"] * step

    return {
        "generated_at": int(portfolio.cursor or 0),
        "config": {
            "rule": config["rule"], "hold": config["hold"], "trail": config["trail"],
            "stop": config["stop"], "target": config["target"],
            "exchange": config["exchange"], "timeframe": config["timeframe"],
            "symbols": config["symbols"],
        },
        "cursor": portfolio.cursor,
        "cursor_utc": to_utc_string(portfolio.cursor) if portfolio.cursor else None,
        "starting_capital": capital,
        "cash": round(portfolio.account.cash, 2),
        "equity": round(equity, 2),
        "return_pct": round((equity / capital - 1) * 100, 4),
        "open_positions": [
            {
                "symbol": position.symbol,
                "entry_time": int(position.entry_time),
                "entry_utc": to_utc_string(position.entry_time),
                "entry_price": position.entry_price,
                "qty": position.qty,
                "mark": float(frames[position.symbol]["close"].iloc[-1]),
                "unrealised_pct": round(
                    (float(frames[position.symbol]["close"].iloc[-1]) / position.entry_price - 1)
                    * 100, 4
                ),
                # How far through its holding period the trade is. Shown because
                # an unrealised number means something different at hour three
                # than at hour a hundred and sixty: one is noise, the other is
                # nearly the result.
                "bars_held": int(((portfolio.cursor or position.entry_time) - position.entry_time) // step),
                "bars_total": config["hold"],
                "exit_utc": to_utc_string(position.entry_time + hold_ms),
                "progress_pct": round(min(100.0, max(0.0,
                    ((portfolio.cursor or position.entry_time) - position.entry_time) / hold_ms * 100)), 1),
            }
            for position in portfolio.open_positions()
        ],
        "open_value": round(sum(marks.values()), 2),
        "stats": {
            "closed": int(len(ledger)),
            "hit_rate": round(float((ledger["net_return"] > 0).mean()) * 100, 2) if len(ledger) else None,
            "mean_net_pct": round(float(ledger["net_return"].mean()) * 100, 4) if len(ledger) else None,
            "best_pct": round(float(ledger["net_return"].max()) * 100, 4) if len(ledger) else None,
            "worst_pct": round(float(ledger["net_return"].min()) * 100, 4) if len(ledger) else None,
            "refused": portfolio.rejections_total,
        },
        "risk": {
            "peak": round(peak, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "current_drawdown_pct": round((running / peak - 1) * 100, 2) if peak else 0.0,
        },
        "by_symbol": by_symbol,
        "refusals": {
            "total": portfolio.rejections_total,
            "recent": portfolio.rejection_feed()[-25:],
        },
        "timeframe_ms": step,
        "next_candle": (portfolio.cursor + step) if portfolio.cursor else None,
        "equity_curve": curve,
        "recent_trades": ledger.tail(50).to_dict(orient="records"),
    }


def _measure(args):
    """The whole run, with nothing caught. `run` is the layer that catches."""
    config = load_config(args.config)
    print_target = Path(args.state)

    frames = _load_frames(config, args.data_dir)
    portfolio = _build(config)

    mark = state_module.fingerprint(config)
    saved = None if args.reset else state_module.load(print_target, expect=mark)
    portfolio.restore(saved, frames)

    signals = {
        symbol: rules.apply(config["rule"], frame, **config["params"])
        for symbol, frame in frames.items()
    }

    advanced = 0 if args.status else portfolio.advance(frames, signals)

    if not args.status:
        payload = portfolio.to_state()
        payload["fingerprint"] = mark
        payload["config"] = config
        state_module.save(print_target, payload)

    snapshot = _snapshot(portfolio, frames, config)
    snapshot_path = Path(args.json) if args.json else print_target.with_name("snapshot.json")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with open(snapshot_path, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, default=str)
        handle.write("\n")

    print(_report(portfolio, frames, config, advanced))
    print()
    print(f"  state    {print_target}")
    print(f"  snapshot {snapshot_path}")
    return EXIT_OK


def run(argv=None):
    """Run one advance and return the exit code."""
    args = build_parser().parse_args(argv)
    try:
        return _measure(args)
    except USER_ERRORS as failure:
        print(f"error: {failure}", file=sys.stderr)
        return EXIT_FAILED
    except KeyboardInterrupt:
        # The state file is written atomically and only at the end, so an
        # interrupted run leaves the previous one intact and complete.
        print("interrupted. The saved state is unchanged.", file=sys.stderr)
        return EXIT_FAILED


def main():
    sys.exit(run())


if __name__ == "__main__":
    main()
