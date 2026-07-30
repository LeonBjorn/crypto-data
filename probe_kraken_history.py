# /// script
# requires-python = ">=3.12"
# dependencies = ["ccxt"]
# ///
"""
Throwaway probe: can Kraken's public OHLC endpoint serve deep history?

The whole backfill design assumes we can walk `since` backwards/forwards through
years of candles. Kraken's OHLC endpoint is widely reported to cap at ~720
candles and to ignore a `since` that is older than that window -- meaning 1h
candles would give us ~30 days, not 2 years.

This script answers that empirically before we build anything on the assumption.

Run:  uv run probe_kraken_history.py
      uv run probe_kraken_history.py --compare      # also try other exchanges

Delete this file once the question is answered.
"""

import argparse
import datetime as dt

import ccxt

# One hour in milliseconds. The probe only tests 1h, since that is the timeframe
# the project actually needs.
HOUR_MS = 60 * 60 * 1000


def to_utc(ms):
    """Render an epoch-millisecond timestamp as a readable UTC string."""
    return dt.datetime.fromtimestamp(ms / 1000, tz=dt.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )


def probe(exchange, symbol, since_ms, now_ms):
    """
    Ask one exchange for 1h candles starting at `since_ms` and describe what
    came back.

    Returns a dict of findings, or None if the request failed. Takes the
    exchange object as an argument rather than building it, so this can be
    driven by a fake in a test.
    """
    candles = exchange.fetch_ohlcv(symbol, timeframe="1h", since=since_ms)

    if not candles:
        return {"symbol": symbol, "count": 0}

    first_ts = candles[0][0]
    last_ts = candles[-1][0]

    # The decisive question: did the exchange honour our `since`, or did it
    # quietly hand back its most recent window instead? We allow a day of slack
    # because the oldest available candle for a market may legitimately be a
    # little later than what we asked for.
    honoured = first_ts <= since_ms + (24 * HOUR_MS)

    return {
        "symbol": symbol,
        "count": len(candles),
        "first_ts": first_ts,
        "last_ts": last_ts,
        "requested_since": since_ms,
        "honoured_since": honoured,
        # How far back from *now* the oldest returned candle sits. If `since`
        # was ignored, this reveals the true size of the fixed window.
        "window_days": round((now_ms - first_ts) / (24 * HOUR_MS), 1),
    }


def probe_pagination(exchange, symbol, first_page_last_ts, now_ms):
    """
    Second question: does asking for candles *after* page one actually advance?

    A backfill loop is only possible if handing back the last timestamp we saw
    yields the next chunk rather than the same chunk again. If this returns the
    same starting timestamp, the loop would spin forever.

    Only meaningful when page one ended well before now. If page one already
    runs up to the present there is nothing further to fetch, and a "did not
    advance" result would be normal rather than a problem -- so we skip it
    instead of reporting a false alarm.
    """
    if first_page_last_ts > now_ms - (2 * HOUR_MS):
        return {"skipped": True}

    next_since = first_page_last_ts + HOUR_MS
    candles = exchange.fetch_ohlcv(symbol, timeframe="1h", since=next_since)
    if not candles:
        return {"advanced": False, "count": 0}
    return {
        "advanced": candles[0][0] > first_page_last_ts,
        "count": len(candles),
        "first_ts": candles[0][0],
    }


def report(name, findings, pagination):
    """Print one exchange's results and a plain-language verdict."""
    print(f"\n=== {name} ===")

    if findings is None:
        print("  request failed (see error above)")
        return

    if findings["count"] == 0:
        print("  returned zero candles -- symbol may not exist on this exchange")
        return

    print(f"  requested since : {to_utc(findings['requested_since'])}")
    print(f"  oldest returned : {to_utc(findings['first_ts'])}")
    print(f"  newest returned : {to_utc(findings['last_ts'])}")
    print(f"  candles per page: {findings['count']}")
    print(f"  oldest is {findings['window_days']} days before now")

    if findings["honoured_since"]:
        print("  VERDICT: `since` was honoured -- deep history looks reachable.")
    else:
        print("  VERDICT: `since` was IGNORED. This exchange returned only its")
        print(f"           most recent ~{findings['window_days']} day window.")
        print("           Backfilling 2 years from here is not possible.")

    if pagination is not None:
        if pagination.get("skipped"):
            print("  pagination: not tested (page one already reaches the present)")
        elif pagination["advanced"]:
            print(f"  pagination: advances correctly ({pagination['count']} more candles)")
        else:
            print("  pagination: did NOT advance -- a naive loop would spin forever")


def build(exchange_id):
    """
    Construct a public, keyless ccxt client.

    enableRateLimit is on because it is on in the real collector too; there is
    no reason to probe under different conditions than we will run under.
    """
    return getattr(ccxt, exchange_id)({"enableRateLimit": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTC/USD")
    parser.add_argument("--years", type=float, default=2.0)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="also probe binance, bybit and okx as fallback candidates",
    )
    args = parser.parse_args()

    now_ms = int(dt.datetime.now(tz=dt.timezone.utc).timestamp() * 1000)
    since_ms = now_ms - int(args.years * 365 * 24 * HOUR_MS)

    print(f"Probing for 1h candles going back {args.years} years")
    print(f"Requested start: {to_utc(since_ms)}")

    targets = [("kraken", args.symbol)]
    if args.compare:
        # These quote in USDT rather than USD, hence the different symbol.
        targets += [
            ("binance", "BTC/USDT"),
            ("bybit", "BTC/USDT"),
            ("okx", "BTC/USDT"),
        ]

    for exchange_id, symbol in targets:
        try:
            exchange = build(exchange_id)
            findings = probe(exchange, symbol, since_ms, now_ms)
            pagination = None
            if findings and findings["count"] > 0:
                pagination = probe_pagination(
                    exchange, symbol, findings["last_ts"], now_ms
                )
        except Exception as exc:
            # A probe should never crash on one bad exchange; we want the other
            # results. Print the class name so rate limits are distinguishable
            # from geo-blocks and bad symbols.
            print(f"\n=== {exchange_id} ===")
            print(f"  {type(exc).__name__}: {exc}")
            continue

        report(f"{exchange_id}  {symbol}", findings, pagination)

    print("\nDone. The verdict above decides which exchange milestone 1 targets.")


if __name__ == "__main__":
    main()
