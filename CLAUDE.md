# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A systematic crypto research and paper-trading system, built in milestones that
each hold a property the next must not break. `README.md` covers the collector
alone; `GUIDE.md` covers the whole thing end to end and is worth reading before
changing anything substantial.

**Nothing in this repository can place an order except `paper/hyperliquid.py`,
which defaults to testnet and to sending nothing.** Everything else is keyless.

## Commands

```bash
uv sync                                  # install (first time, or after dep changes)
uv run pytest                            # full suite, no network, ~20s
uv run pytest tests/test_trades.py       # one file
uv run pytest -k drawdown                # by keyword

uv run collect                           # fetch closed candles (ONLY networked command)
uv run collect --timeframe 1m --start 2024-08-01

uv run python -m signals.cli             # research: every rule at every hold
uv run python -m signals.cli --walk 6    # rolling out-of-sample, the honest view
uv run python -m signals.cli --per-symbol --split
uv run python scripts/search.py          # the strategy sweep (read its docstring first)

uv run paper                             # advance the wallet to the latest closed candle
uv run paper --status                    # report without advancing
uv run paper-serve                       # dashboard, http://127.0.0.1:8787
./scripts/hourly.sh                       # one scheduled cycle: collect then paper
```

Two launchd agents run continuously: `…crypto-data.hourly` (collect + paper at
:02 past each hour) and `…crypto-data.dashboard` (KeepAlive). See
`scripts/README.md`.

## Architecture

Four packages in `src/`, strictly one-way. The direction is the point: the
collector records what the exchange said, which is fact; everything downstream
records what we think it means, which is not.

```
collector → signals → paper → (hyperliquid)
   M1         M2       M3         M4
```

- **`collector/`** — fetches candles into an atomic Parquet store. Only networked code, no credentials.
- **`signals/`** — rules, indicators, fill model, benchmarks, causality guard. Reads the store, never writes.
- **`paper/`** — online engine, one finite wallet, risk layer, dashboard. Reads the store.
- **`paper/hyperliquid.py`** — the only code that can trade. Testnet + dry-run by default.

### The invariants that hold it together

These are the things to check you haven't broken. Each is enforced by tests.

1. **Paper == backtest.** Replaying stored candles one at a time through
   `paper.engine.PaperBook` reproduces `signals.trades.round_trips` *exactly* —
   same trades, fills and returns. If this breaks, paper results cannot be
   compared to anything and mean nothing. (`tests/test_paper_engine.py`)

2. **Nothing may look forward.** `signals/lookahead.py` truncates and perturbs
   history to prove a rule cannot see bars it wouldn't have had. It runs before
   any measurement is printed, and a rule that *never fires* is reported as
   inconclusive rather than passing.

3. **Stopping and starting changes nothing.** The paper account resumes from a
   timestamp cursor; running hourly and running once after three days produce
   identical ledgers.

4. **Every write is atomic.** Store and state both write to `.tmp` then
   `os.replace`. A killed process leaves the old complete file or the new one.

### Conventions that will bite you if ignored

- **Timestamps are `int64` epoch milliseconds, never datetimes.** A naive
  datetime silently means local time and corrupts the store.
- **Entry is the *next* bar's open**, never the close that produced the signal.
  This is the most common way a backtest lies (`trades.py` docstring).
- **Injected collaborators default to `None` and resolve in the body**, not in
  the signature — a default evaluated at import can be overridden but not
  intercepted (`now_ms`, `sleep`, `exchange_factory`).
- **Costs default to real numbers.** `trades.VENUES` carries Binance spot and
  Hyperliquid taker/maker. Being in a hurry should not be what makes results
  look good.
- **Rules fire on the bar a condition *turns on*,** not every bar it holds.

## Research findings — read before proposing a strategy

This matters more than any code detail. The searches are done and they came back
negative; re-running them makes things worse, not better.

- **38 directional configurations** (breakout/breakdown, long/short, faded and
  followed, holds 24h–336h, Hyperliquid costs): all around the **50th
  percentile** — no better than random timing.
- **Cross-sectional market-neutral**: −0.74% per rebalance, **t = −0.67**.
- **1-minute microstructure**: exactly zero, once a lookahead in the measurement
  was corrected. The uncorrected version showed a convincing +0.19% to +0.37%.
- **Walk-forward**: the edge sits in **1 window of 6**, and is *below random* in
  another. The +20.7% on the paper account is one bull window plus XRP alone.

**Every additional configuration tried raises the bar any future survivor must
clear** (Deflated Sharpe; the trial count is already ~40). Adding variants is
actively counterproductive. If a new strategy is wanted, it should be
*structurally* different — funding-rate positioning, a wider universe — and it
should go through `signals/rules.py` so the causality guard applies.

**Do not write ad-hoc research scripts that bypass the guard.** The 1m lookahead
above was made exactly that way, and nothing about it looked wrong.

## Paper account state

- `state/` is **gitignored and has no off-machine backup**. Local rollback
  copies exist, but the ledger still cannot be refetched.
- `forward_from` in `state/paper.json` marks where **replay ends and genuine
  out-of-sample testing begins** (2026-08-04). Everything before it is history
  the rules were selected against and is evidence of nothing. The dashboard says
  so at the top.
- `--reset` is guarded behind a second explicit flag once forward testing has
  begun, because it would silently discard that record.
- Two fingerprints: **strategy** settings refuse to resume when changed;
  **risk** settings are adopted and recorded as a regime change in
  `risk_regimes`.

## Going live

`docs/going-live-criteria.md` sets five pre-committed conditions. **Status: not
met.** The strategy has no demonstrated edge, and the forward record is days old.

`paper/hyperliquid.py` guards, in the order they fire: dry-run default; testnet
default (mainnet needs an argument **and** `HYPERLIQUID_ALLOW_MAINNET=yes`); a
kill-switch file; per-order and exposure caps that **refuse rather than
truncate**; an allow-list; and **no withdrawal method at all** — absent, not
disabled. Credentials come from the environment only; there is no constructor
argument for a key. See `docs/hyperliquid-setup.md`.

## Reference docs

| file | what |
|---|---|
| `GUIDE.md` | the whole system, how to run and watch it, what the evidence says |
| `docs/going-live-criteria.md` | what must be true before real money |
| `docs/hyperliquid-setup.md` | testnet setup and the safety rails |
| `docs/portfolio-risk-research.md` | risk/portfolio literature review, cited |
| `scripts/README.md` | the launchd agents |
| `scripts/search.py` | strategy sweep — its docstring records what it found |
