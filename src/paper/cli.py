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
from paper.risk import DrawdownGuard, RiskError, RiskModel, expected_shortfall
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
    RiskError,
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
    # What to measure the account against. Buying one asset and doing
    # nothing is the benchmark every strategy has to clear before any of
    # its cleverness counts for anything, and it is the one comparison a
    # dashboard of your own equity curve cannot make for you.
    "benchmark": "BTC/USDT",
    # The risk layer. All of it is off by default, so an existing ledger keeps
    # its meaning and turning any of it on is a visible decision.
    #
    # sizing      "fixed" -- size_fraction of capital, as before
    #             "inverse-vol" -- weight by 1/sigma, needing no correlations
    #                              and no expected returns
    # target_vol  annualised volatility to run the book at, or null for none
    # max_leverage hard cap on the scale factor. This is the real risk decision:
    #             set it from the drawdown that is survivable, not the return
    #             that is wanted.
    # max_drawdown pre-committed limit; new positions stop when it is breached
    "sizing": "fixed",
    "target_vol": None,
    "max_leverage": 1.0,
    "max_drawdown": None,
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


SIZINGS = ("fixed", "inverse-vol")


def _build(config):
    if config["sizing"] not in SIZINGS:
        raise settings.ConfigError(
            f"sizing must be one of {', '.join(SIZINGS)}, got {config['sizing']!r}"
        )

    step_ms = timeframe_to_ms(config["timeframe"])
    bars_per_day = 86_400_000 / step_ms
    bars_per_year = bars_per_day * 365

    risk = None
    if config["sizing"] == "inverse-vol" or config["target_vol"] is not None:
        risk = RiskModel(
            config["symbols"],
            bars_per_year=bars_per_year,
            bars_per_day=bars_per_day,
            target_vol=config["target_vol"],
            max_leverage=config["max_leverage"],
        )

    guard = None
    if config["max_drawdown"] is not None:
        guard = DrawdownGuard(limit=abs(float(config["max_drawdown"])))

    account = Account(
        config["starting_capital"],
        size_fraction=config["size_fraction"],
        max_positions=config["max_positions"],
        one_per_symbol=config["one_per_symbol"],
        risk=risk,
        guard=guard,
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
        risk=risk,
        guard=guard,
    )


def _report(portfolio, frames, config, advanced, forward_from=None):
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
        shortfall = expected_shortfall(ledger["net_return"].tolist(), 0.95)
        lines.append(f"  ES(95%)     {shortfall * 100:>11.2f}% average loss in the worst 5% of trades")

    if forward_from:
        days = (portfolio.cursor - forward_from) / 86_400_000 if portfolio.cursor else 0
        ahead = ledger[ledger["entry_time"] > forward_from] if len(ledger) else ledger
        lines += [
            "",
            f"  FORWARD RECORD (the only part that is evidence of anything)",
            f"    since     {to_utc_string(forward_from)}  ({days:.1f} days)",
            f"    trades    {len(ahead)}   everything else above is a replay of history",
        ]

    if portfolio.risk is not None:
        lines.append(f"  risk        {portfolio.risk.describe()}")
    if portfolio.guard is not None:
        state = "TRIPPED -- no new positions" if portfolio.guard.tripped else "ok"
        lines.append(
            f"  drawdown    {portfolio.guard.drawdown:>11.1%} of a {portfolio.guard.limit:.0%} limit ({state})"
        )

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


def _count_reasons(feed):
    """How many recent refusals each reason accounts for.

    Only over the kept tail, which is what makes it a shape rather than a total:
    "mostly already-holding" and "mostly at the cap" call for opposite changes,
    and the running total alone cannot tell them apart.
    """
    counts = {}
    for entry in feed:
        counts[entry["reason"]] = counts.get(entry["reason"], 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: -pair[1]))



def _benchmark(frames, config, capital, points=400):
    """Buy the benchmark symbol at the start, hold, and mark it to market.

    The honest yardstick. A strategy that returned twenty percent over two years
    in which its market doubled did not make money by being clever, and an
    equity curve on its own cannot say so -- it has nothing to be compared
    against but itself.

    Anchored at the same instant and the same capital as the account, so the two
    lines start together and any daylight between them is the strategy rather
    than a difference in where the axes begin. Sampled down to roughly `points`
    so the payload stays small however long the history grows; the extremes are
    computed before sampling, so a peak between two samples still counts.
    """
    symbol = config.get("benchmark")

    if symbol == "basket":
        # Equal money into every symbol the strategy trades, held throughout.
        # Arguably the fairer yardstick of the two: the account is allowed to
        # pick among five markets, so comparing it to one of them flatters or
        # punishes it depending on which one was chosen.
        usable = [f for f in frames.values() if not f.empty]
        if not usable:
            return None
        length = min(len(f) for f in usable)
        share = capital / len(usable)
        values = None
        for frame in usable:
            closes = frame["close"].to_numpy(dtype="float64")[:length]
            held = closes * (share / float(closes[0]))
            values = held if values is None else values + held
        times = usable[0]["timestamp"].to_numpy()[:length]
    else:
        frame = frames.get(symbol)
        if frame is None or frame.empty:
            return None
        closes = frame["close"].to_numpy(dtype="float64")
        times = frame["timestamp"].to_numpy()
        values = closes * (capital / float(closes[0]))

    peak = capital
    worst = 0.0
    for value in values:
        peak = max(peak, float(value))
        worst = min(worst, float(value) / peak - 1)

    step = max(1, len(values) // points)
    curve = [
        {"t": int(times[i]), "equity": round(float(values[i]), 2)}
        for i in range(0, len(values), step)
    ]
    if curve[-1]["t"] != int(times[-1]):
        curve.append({"t": int(times[-1]), "equity": round(float(values[-1]), 2)})

    return {
        "symbol": symbol,
        "curve": curve,
        "return_pct": round((float(values[-1]) / capital - 1) * 100, 4),
        "max_drawdown_pct": round(worst * 100, 2),
    }


def _snapshot(portfolio, frames, config, forward_from=None):
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
                "wins": int((returns > 0).sum()),
                "losses": int((returns <= 0).sum()),
                "trades": int(len(rows)),
                "hit_rate": round(float((returns > 0).mean()) * 100, 1),
                "mean_pct": round(float(returns.mean()) * 100, 3),
                "pnl": round(float((rows["cash_out"] - rows["cash_in"]).sum()), 2),
            })
        by_symbol.sort(key=lambda entry: entry["pnl"], reverse=True)

    step = timeframe_to_ms(config["timeframe"])
    hold_ms = config["hold"] * step

    forward = (
        ledger[ledger["entry_time"] > forward_from] if forward_from and len(ledger)
        else ledger.iloc[0:0]
    )
    forward_pnl = float((forward["cash_out"] - forward["cash_in"]).sum()) if len(forward) else 0.0

    return {
        "generated_at": int(portfolio.cursor or 0),
        # Reported apart from everything else on purpose. Blending a replay with
        # a forward record produces one number that is neither.
        "forward": {
            "from": forward_from,
            "from_utc": to_utc_string(forward_from) if forward_from else None,
            "days": round((portfolio.cursor - forward_from) / 86_400_000, 2) if forward_from and portfolio.cursor else 0,
            "trades": int(len(forward)),
            "pnl": round(forward_pnl, 2),
            "mean_pct": round(float(forward["net_return"].mean()) * 100, 4) if len(forward) else None,
            "hit_rate": round(float((forward["net_return"] > 0).mean()) * 100, 2) if len(forward) else None,
        },
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
                # The money, rather than only the percentages. A position that
                # is 19% up tells you nothing about whether it matters to the
                # account; 19% of $340 and 19% of $3,400 are different facts.
                "cost": round(position.cash_in, 2),
                "value": round(position.qty * float(frames[position.symbol]["close"].iloc[-1]), 2),
                "pnl": round(
                    position.qty * float(frames[position.symbol]["close"].iloc[-1]) - position.cash_in, 2
                ),
                "weight_pct": round(
                    position.qty * float(frames[position.symbol]["close"].iloc[-1]) / equity * 100, 2
                ) if equity else None,
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
        "invested_pct": round(sum(marks.values()) / equity * 100, 2) if equity else 0.0,
        "cash_pct": round(portfolio.account.cash / equity * 100, 2) if equity else 100.0,
        "pnl_total": round(equity - capital, 2),
        "pnl_realised": round(running - capital, 2),
        "pnl_unrealised": round(equity - running, 2),
        "quote": "USDT",
        "stats": {
            "closed": int(len(ledger)),
            "hit_rate": round(float((ledger["net_return"] > 0).mean()) * 100, 2) if len(ledger) else None,
            "mean_net_pct": round(float(ledger["net_return"].mean()) * 100, 4) if len(ledger) else None,
            "best_pct": round(float(ledger["net_return"].max()) * 100, 4) if len(ledger) else None,
            "worst_pct": round(float(ledger["net_return"].min()) * 100, 4) if len(ledger) else None,
            "refused": portfolio.rejections_total,
        },
        "risk": {
            # Expected Shortfall over the realised trades: the average loss in
            # the worst tail, not the threshold of it. Monitored rather than
            # optimised -- optimising it needs a scenario set and a linear
            # program, whereas monitoring it needs only what actually happened.
            "expected_shortfall_95": (
                None if not len(ledger)
                else round(expected_shortfall(ledger["net_return"].tolist(), 0.95) * 100, 3)
            ),
            "sizing": config["sizing"],
            "target_vol_pct": None if config["target_vol"] is None else round(config["target_vol"] * 100, 2),
            "max_leverage": config["max_leverage"],
            "scale": None if portfolio.risk is None else round(portfolio.risk.scale(), 3),
            "forecast_vol_pct": (
                None if portfolio.risk is None or portfolio.risk.portfolio_vol() is None
                else round(portfolio.risk.portfolio_vol() * 100, 2)
            ),
            "drawdown_limit_pct": None if portfolio.guard is None else round(portfolio.guard.limit * 100, 1),
            # The guard's own drawdown, which is NOT the one above it. That one
            # is measured from the all-time realised peak; this is measured from
            # the high-water mark the limit was armed with. When a limit is
            # enabled forward-only the two are different numbers describing
            # different things, and showing the historical one next to the limit
            # would read as "already far past it, yet not tripped".
            "guard_drawdown_pct": None if portfolio.guard is None else round(portfolio.guard.drawdown * 100, 2),
            "guard_peak": None if portfolio.guard is None else round(portfolio.guard.peak, 2),
            "drawdown_tripped": None if portfolio.guard is None else portfolio.guard.tripped,
            "peak": round(peak, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "current_drawdown_pct": round((running / peak - 1) * 100, 2) if peak else 0.0,
        },
        "by_symbol": by_symbol,
        "refusals": {
            "total": portfolio.rejections_total,
            "recent": portfolio.rejection_feed(),
            "by_reason": _count_reasons(portfolio.rejection_feed()),
        },
        "timeframe_ms": step,
        "next_candle": (portfolio.cursor + step) if portfolio.cursor else None,
        # Prepended with the account's opening value at the first candle, so the
        # strategy and the benchmark begin at the same point rather than the
        # strategy appearing to start at its first closed trade weeks later.
        "equity_curve": (
            [{"t": int(min(f["timestamp"].iloc[0] for f in frames.values())), "equity": capital}]
            + curve
        ),
        "benchmark": _benchmark(frames, config, capital),
        # Where the account stands *now*, including positions still open. The
        # realised curve above necessarily ends at the last closed trade, which
        # is why a chart of it alone finishes below the equity printed beside it
        # -- so the gap is handed over as a fact rather than left to be noticed.
        "equity_now": {
            "t": portfolio.cursor,
            "equity": round(equity, 2),
            "realised": round(running, 2),
            "unrealised": round(equity - running, 2),
        },
        # The whole ledger, not a tail. The page sorts and filters it, and
        # doing that to the most recent fifty of four hundred rows would look
        # exactly like doing it to all of them while quietly answering a
        # different question. It is about 140 KB and never leaves this machine.
        "trades": ledger.to_dict(orient="records"),
    }


def _measure(args):
    """The whole run, with nothing caught. `run` is the layer that catches."""
    config = load_config(args.config)
    print_target = Path(args.state)

    frames = _load_frames(config, args.data_dir)
    portfolio = _build(config)

    mark = state_module.fingerprint(config)
    risk_mark = state_module.risk_fingerprint(config)
    saved = None if args.reset else state_module.load(print_target, expect=mark)
    portfolio.restore(saved, frames)

    # A change to the risk settings is adopted rather than refused, and the
    # moment of the change is written into the ledger's own record so that a
    # curve spanning two regimes says so instead of quietly averaging them.
    regimes = list((saved or {}).get("risk_regimes") or [])
    if saved is not None and saved.get("risk_fingerprint") not in (None, risk_mark):
        regimes.append({
            "at": portfolio.cursor,
            "at_utc": to_utc_string(portfolio.cursor) if portfolio.cursor else None,
            "settings": {key: config.get(key) for key in state_module.RISK_FINGERPRINTED},
        })
        print(
            f"note: risk settings changed. Adopting them from "
            f"{to_utc_string(portfolio.cursor) if portfolio.cursor else 'the start'} onward; "
            f"the {len(portfolio.ledger())} trade(s) already recorded were taken under the "
            f"previous settings and are kept."
        )
    elif not regimes:
        regimes = [{
            "at": portfolio.cursor,
            "at_utc": to_utc_string(portfolio.cursor) if portfolio.cursor else None,
            "settings": {key: config.get(key) for key in state_module.RISK_FINGERPRINTED},
        }]

    # The boundary between replay and forward testing, fixed once and never
    # moved. Everything before it is history the rules were chosen against, and
    # is evidence of nothing; everything after it is out-of-sample and is the
    # only part that can validate anything. Without the mark the two are
    # indistinguishable in the ledger, and in three months nobody could tell
    # which trades meant something.
    forward_from = (saved or {}).get("forward_from")
    if forward_from is None:
        forward_from = portfolio.cursor

    # A guard switched on over an existing ledger starts its high-water mark at
    # today's equity, not at zero and not at the historical peak. Zero would
    # leave the limit meaningless until the next bar; the historical peak would
    # apply the limit retroactively to a drawdown that already happened and was
    # lived through, which retires the account rather than protecting it. Where
    # the mark starts is a real choice, so it is made here explicitly and
    # recorded in risk_regimes rather than falling out of an initial value.
    if portfolio.guard is not None and portfolio.guard.peak <= 0:
        portfolio.guard.observe(portfolio.equity(frames))

    signals = {
        symbol: rules.apply(config["rule"], frame, **config["params"])
        for symbol, frame in frames.items()
    }

    advanced = 0 if args.status else portfolio.advance(frames, signals)

    if not args.status:
        payload = portfolio.to_state()
        payload["fingerprint"] = mark
        payload["risk_fingerprint"] = risk_mark
        payload["risk_regimes"] = regimes
        payload["forward_from"] = forward_from
        payload["config"] = config
        state_module.save(print_target, payload)

    snapshot = _snapshot(portfolio, frames, config, forward_from)
    snapshot_path = Path(args.json) if args.json else print_target.with_name("snapshot.json")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with open(snapshot_path, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, default=str)
        handle.write("\n")

    print(_report(portfolio, frames, config, advanced, forward_from))
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
