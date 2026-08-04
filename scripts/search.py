"""A walk-forward sweep, for looking at several strategies without fooling yourself.

The point of scoring this way rather than on the whole sample is that the whole
sample has already misled this project once. `breakout-volume` sits at the 99.8th
percentile over two years and made essentially all of it in one four-month
window; five of six windows since were flat or negative. A number computed over
everything cannot see that, and it is the number that looks most convincing.

So a strategy is judged on *consistency*: how many windows it was positive in,
and what its worst window looked like. A mean is reported too, but a mean of six
windows where one is +6% and five are -1% is a fact about the one.

WHAT THIS IS AND IS NOT
-----------------------
It is a way to look at several ideas side by side under the same conditions.

WHAT IT FOUND, SO THE NEXT PERSON DOES NOT REPEAT IT
----------------------------------------------------
Thirty-eight configurations, on 5 symbols of hourly candles over two years, with
Hyperliquid funding charged: nothing survived.

  - Every directional variant -- breakout and breakdown, long and short, faded
    and followed, at holds from 24h to 336h -- came out with percentiles around
    50 and negative means. The ones that won four windows of six did it by
    winning small repeatedly and losing enormously once.
  - Cross-sectional long-strongest/short-weakest, the structural fix for the
    direction dependence, looked like the winner at 5 windows of six. Run over
    the whole sample rather than sliced into windows it returns -0.74% per
    rebalance with a t-statistic of -0.67. It loses money.
  - With 38 configurations, 4.2 of them are *expected* to reach five-of-six
    positive windows by chance alone. Finding one is not evidence of anything.

AND AT ONE-MINUTE RESOLUTION
---------------------------
Tested afterwards, on 5.28M one-minute bars across the same five symbols: does
price revert, or continue, after an extreme one-minute move? This is the
best-documented short-horizon effect in any liquid market -- a large move takes
out resting liquidity, overshoots, and partly reverts as makers replenish -- and
it is mechanical rather than a forecast of trend, which is the one category the
sweeps above had not touched.

The first run showed momentum of +0.19% to +0.37% after 3-to-6 sigma moves,
comfortably clearing the 0.130% round trip at Hyperliquid taker. It was a
lookahead. The trigger return spanned close[j] to close[j+1] and the entry was
placed at open[j+1] -- before the close that defined the trigger -- so the
measurement was capturing part of the trigger move itself.

Moved to open[j+2], the first bar genuinely tradable after the signal is known,
the effect is:

    3 sigma   -0.000%      5 sigma   +0.000%      6 sigma   -0.005%

Zero to three decimal places, on every horizon from 5 to 120 minutes, across all
five symbols, with |t| below 1 everywhere. The entire apparent edge was the
one-bar misalignment.

That is worth recording twice over: as a result, and as a reminder that the
guard in signals/ exists because this mistake is invisible. Nothing about the
first run looked wrong -- the numbers were simply a little too good.

That is the honest state: no edge was found in this data at either frequency
after costs. It is a real answer, and a cheaper one than discovering the same
thing with money.

It is not a way to pick the best one and believe the number attached to it.
Twenty configurations tried against one two-year sample will produce a good
looking one whether or not anything is there, and the more of them are tried the
more certain that becomes -- so the count is printed at the end, and any survivor
is a hypothesis to test on data nobody has looked at, not a result.
"""

import numpy as np

from signals import evaluate, prices, rules, trades

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT"]

# What Hyperliquid is charging per hour right now. Long pays it, short is paid
# it, and over a week-long hold it is two thirds of a round trip in fees.
HL_FUNDING = 1.222e-05

WINDOWS = 6
DRAWS = 200


def load(symbols=SYMBOLS):
    return {s: prices.load("data", "binance", s, "1h") for s in symbols}


def windows_of(frames, count=WINDOWS):
    """`count` equal, contiguous slices of every frame, as a list of dicts."""
    n = min(len(f) for f in frames.values())
    return [
        {s: f.iloc[i * n // count:(i + 1) * n // count] for s, f in frames.items()}
        for i in range(count)
    ]


def measure(pieces, rule, *, side="long", hold=168, funding=HL_FUNDING, params=None,
            costs=trades.HYPERLIQUID_TAKER):
    """One window's pooled mean and percentile, or None if nothing traded."""
    parts = []
    for symbol, frame in pieces.items():
        signal = rules.apply(rule, frame, **(params or {}))
        if not signal.any():
            continue
        try:
            parts.append(
                evaluate.collect(
                    frame, signal, hold=hold, symbol=symbol, side=side, funding=funding,
                    costs=costs
                )
            )
        except evaluate.EvalError:
            continue
    if not parts:
        return None
    try:
        verdict = evaluate.judge(evaluate.pool(parts), draws=DRAWS, seed=0)
    except evaluate.EvalError:
        return None
    return verdict.rule.mean * 100, verdict.percentile, verdict.rule.trades


def sweep(frames, specs, count=WINDOWS):
    """Every spec across every window. Returns rows sorted by consistency."""
    slices = windows_of(frames, count)
    rows = []
    for label, kwargs in specs:
        means, pcts, trades = [], [], 0
        for pieces in slices:
            got = measure(pieces, **kwargs)
            if got is None:
                continue
            mean, pct, n = got
            means.append(mean)
            pcts.append(pct)
            trades += n
        if len(means) < count:
            continue
        rows.append({
            "label": label,
            "positive": sum(1 for m in means if m > 0),
            "mean": float(np.mean(means)),
            "worst": float(np.min(means)),
            "median": float(np.median(means)),
            "pct": float(np.mean(pcts)),
            "trades": trades,
        })
    # Consistency first, then typical size. A strategy positive in five windows
    # of six is a different kind of object from one positive in two, however
    # large the two were.
    rows.sort(key=lambda r: (-r["positive"], -r["median"]))
    return rows


def show(rows, title):
    print(f"\n{title}")
    print(f"  {'strategy':32s}{'win':>5s}{'median':>9s}{'mean':>9s}{'worst':>9s}{'pct':>7s}{'trades':>8s}")
    for r in rows:
        print(f"  {r['label']:32s}{r['positive']:3d}/6{r['median']:8.3f}%{r['mean']:8.3f}%"
              f"{r['worst']:8.3f}%{r['pct']:7.1f}{r['trades']:8d}")


if __name__ == "__main__":
    frames = load()
    specs = []
    for hold in (24, 72, 168, 336):
        specs.append((f"breakout-volume L {hold}h",
                      {"rule": "breakout-volume", "side": "long", "hold": hold}))
        specs.append((f"breakdown-volume S {hold}h",
                      {"rule": "breakdown-volume", "side": "short", "hold": hold}))
    for hold in (24, 72, 168):
        specs.append((f"breakout L {hold}h", {"rule": "breakout", "side": "long", "hold": hold}))
        specs.append((f"rsi-oversold L {hold}h", {"rule": "rsi-oversold", "side": "long", "hold": hold}))
        # Momentum failed in four of six windows on a market that chopped, so the
        # obvious counter-hypothesis is that these breaks were worth fading.
        specs.append((f"breakout FADED S {hold}h", {"rule": "breakout-volume", "side": "short", "hold": hold}))
        specs.append((f"breakdown FADED L {hold}h", {"rule": "breakdown-volume", "side": "long", "hold": hold}))

    rows = sweep(frames, specs)
    show(rows, f"walk-forward over {WINDOWS} windows, Hyperliquid funding charged")
    print(f"\n  {len(specs)} configurations tried against one two-year sample.")
    print("  Expect the best-looking one to look good whether or not anything is there.")


# --------------------------------------------------------------------------
# Cross-sectional: stop betting on direction at all
# --------------------------------------------------------------------------
#
# Every failure measured so far reduces to the same thing. A long-only rule
# makes money when the market rises and loses when it falls; its mirror does the
# opposite; the two are anti-correlated at -0.89 and cancel when run together.
# Filtering by trend did not fix it and made the out-of-sample worse.
#
# So the structural answer is to stop taking a view on direction. Rank the
# symbols against *each other*, buy the strongest and short the weakest in equal
# size, and the market's own move largely cancels between the two legs. What is
# left is whether relative strength persists, which is a different question from
# whether the market goes up -- and it is the only question here that has not
# already been answered no.
#
# It also fixes funding almost for free: the long leg pays it and the short leg
# is paid it, so a balanced book is close to funding-neutral rather than bleeding
# 0.2% a week.

def cross_sectional(frames, *, lookback=168, hold=168, legs=1, costs_per_side=0.0015):
    """Long the strongest symbols and short the weakest, rebalanced every `hold`.

    Returns one net return per rebalance, as a fraction. Deliberately plain:
    equal money per leg, no sizing, no stops, entry and exit at the open of the
    bar *after* the decision, exactly as `trades.py` insists on.
    """
    symbols = list(frames)
    closes = {s: frames[s]["close"].to_numpy(dtype="float64") for s in symbols}
    opens = {s: frames[s]["open"].to_numpy(dtype="float64") for s in symbols}
    n = min(len(c) for c in closes.values())

    out = []
    bar = lookback
    while bar + hold + 1 < n:
        # Ranked on information available at `bar`; traded from `bar + 1`.
        strength = {s: closes[s][bar] / closes[s][bar - lookback] - 1 for s in symbols}
        order = sorted(symbols, key=lambda s: strength[s], reverse=True)
        longs, shorts = order[:legs], order[-legs:]

        entry, exit_ = bar + 1, bar + 1 + hold
        legs_pnl = []
        for s in longs:
            legs_pnl.append(opens[s][exit_] / opens[s][entry] - 1 - 2 * costs_per_side)
        for s in shorts:
            legs_pnl.append(1 - opens[s][exit_] / opens[s][entry] - 2 * costs_per_side)
        out.append(float(np.mean(legs_pnl)))
        bar += hold
    return np.array(out)


def cross_sectional_sweep(frames, count=WINDOWS):
    slices = windows_of(frames, count)
    rows = []
    for lookback in (72, 168, 336):
        for hold in (72, 168, 336):
            for legs in (1, 2):
                means = []
                total = 0
                for pieces in slices:
                    r = cross_sectional(pieces, lookback=lookback, hold=hold, legs=legs)
                    if len(r) < 2:
                        means = []
                        break
                    means.append(float(np.mean(r)) * 100)
                    total += len(r)
                if len(means) < count:
                    continue
                rows.append({
                    "label": f"x-sec look{lookback} hold{hold} legs{legs}",
                    "positive": sum(1 for m in means if m > 0),
                    "mean": float(np.mean(means)),
                    "worst": float(np.min(means)),
                    "median": float(np.median(means)),
                    "pct": 0.0,
                    "trades": total,
                })
    rows.sort(key=lambda r: (-r["positive"], -r["median"]))
    return rows
