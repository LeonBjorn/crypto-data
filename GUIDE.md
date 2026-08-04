# A guide to this project

What it does, how to run and watch it, what the algorithm actually is, what the
evidence says about whether it works, and what it is genuinely useful for.

`README.md` covers the data collector on its own. This is the whole thing.

---

## 1. What the project is

Three core packages, with one tightly guarded execution adapter, and data only
ever flows one way:

```
    the exchange
         │  public candles, no key
         ▼
  ┌─────────────┐
  │  collector  │   fetches closed 1h candles into Parquet          MILESTONE 1
  └─────────────┘   data/binance/BTC_USDT/1h.parquet
         │
         ▼
  ┌─────────────┐
  │   signals   │   rules, fills, statistics, benchmarks            MILESTONE 2
  └─────────────┘   "would this rule have worked?"
         │
         ▼
  ┌─────────────┐
  │    paper    │   the same rules run forward, one wallet          MILESTONE 3
  └─────────────┘   "what does it do on candles nobody has seen?"
         │
         ▼
    dashboard (read-only, localhost)
         │
         ▼
  Hyperliquid broker (explicit, testnet + dry-run defaults)       MILESTONE 4
```

The direction matters. `collector` records what the exchange said, which is
fact. `signals` and `paper` record what we think it means, which is not. Keeping
them apart means a mistake in the thinking cannot damage two years of candles.

**`collector` is the only routine networked command, and it holds no
credentials.** The optional Hyperliquid broker is the sole exception: it reads
credentials from the environment only, defaults to testnet and dry-run, and
refuses orders unless its safety gates all pass. The paper workflow itself stays
keyless and cannot place orders.

Currently stored: **5 symbols × 17,589 hourly candles**, 2024-08-01 to now.

---

## 2. How the process runs

### The hourly cycle

A macOS launchd agent runs `scripts/hourly.sh` at **two minutes past every
hour**, and once at login.

```
:02  ┌─ collect ──┐   fetch whatever candles have closed since last time
     │            │   → data/binance/<SYMBOL>/1h.parquet
     └─ paper ────┘   advance the wallet over those new candles
                      → state/paper.json  +  state/snapshot.json
```

Three decisions in that script are worth knowing:

**The order is fixed.** `paper` only ever *reads* the store; it never fetches.
Running it before `collect` would advance over candles that were already there
and quietly do nothing.

**:02 rather than :00.** The candle closes exactly on the hour, and asking for
it at that instant is a race with the exchange publishing it. Two minutes costs
nothing and removes the race.

**A failed `collect` does not skip `paper`.** A sleeping laptop or a bad minute
at the exchange should not also cost an hour of paper trading. The store still
holds every candle it held before, and advancing over those is both correct and
idempotent.

### Why it is safe to run repeatedly

Everything in the chain is **idempotent and resumable**:

- `collect` works out what is already on disk and fetches only what is missing.
  Interrupt it and run again; writes are atomic, so a file on disk is always a
  complete, valid, sorted file.
- `paper` keeps a **cursor** — the timestamp of the last bar it acted on — and
  only advances past it. Run it hourly, or once after three days away, and you
  get the *identical* ledger. That is not a hope; it is a test.

If the Mac is asleep at :02 the job does not run then. It fires shortly after
wake and catches up over every missed bar. A closed laptop costs a delay, not
data.

### Running it by hand

```bash
uv run collect        # fetch (the only networked command)
uv run paper          # advance the wallet
uv run paper-serve    # dashboard at http://127.0.0.1:8787

./scripts/hourly.sh   # one full cycle, exactly what the scheduler runs
```

### Cost

Measured, not estimated: **1.7 s wall, ~1.3 s CPU, 180 MB peak** per cycle, and
the process does not exist between runs. That is a duty cycle of 0.047%. Disk
grows about **1.9 MB/year** for the candles; the logs are capped and rotate.

---

## 3. What the algorithm actually does

The rule the account trades is **`breakout-volume`**. It is deliberately
conventional — the point was to measure something someone else designed, not to
invent something and tune it until the backtest looked good.

### The signal

On each closed hourly candle, for each symbol, buy if **both** hold:

1. **Price breakout** — the close is above the highest *high* of the previous
   20 bars.
2. **Volume confirmation** — this bar's volume is above **1.5×** the average
   volume of the previous 20 bars.

The reasoning behind the second condition is old and conventional: a breakout on
thin volume is the market drifting through a level nobody was defending, and is
the kind that fails. A breakout the crowd turned up for is the kind that is
supposed to hold.

Three details do real work:

- **The 20-bar window excludes the current bar** (`.shift(1)`). Without that,
  the bar that breaks out is by definition the highest bar around, so the level
  rises to meet the price and the rule never fires — silently, forever, while
  looking perfectly sensible.
- **Highs set the level, the close breaks it.** Using the high on both sides
  would fire on a price touched at an unknown moment inside the hour; a close is
  something that had actually happened when the bar ended.
- **It fires only on the bar the condition turns on**, not on every bar it
  happens to hold. Otherwise one breakout that stays true for a week becomes
  forty "opportunities", and every statistic downstream describes the same event
  forty times while looking like a healthy sample.

### The trade

| | |
|---|---|
| **Entry** | the **open of the next bar** — never the close that produced the signal |
| **Exit** | after **168 bars (one week)**, at that bar's open |
| **Costs** | 0.10% fee + 0.05% slippage, **per side** (~0.30% round trip) |
| **Stop / target / trail** | off by default |

The entry rule is the single most important line in the project. The close that
produced the signal is not a price you can trade — it is a price you watched go
past. Scoring a trade at that close pays the strategy for noticing the thing
that already happened, and it is the most common way a backtest lies.

Costs default to real numbers rather than zero. A round trip costs about 0.30%,
which is in the same range as the entire edge of most hourly rules — so a
zero-cost backtest is often the difference between a winner and a loser.

### The wallet

This is where paper trading diverges from the backtest on purpose. The research
layer scores every signal independently and lets trades overlap, because that
measures *the rule*. A real wallet cannot do that:

| | |
|---|---|
| starting capital | 10,000 |
| position size | 20% of starting capital (~2,000), capped by free cash |
| max positions | 5 |
| one position per symbol | yes |

**Signals the wallet cannot afford are recorded as refusals, not silence.** This
matters more than it sounds: the account has taken 399 trades and **refused
1,621** — roughly four refusals for every fill. A page that only counted fills
would make a constrained account look like a quiet market.

### Other rules in the registry

`signals` also carries `breakout` (no volume filter), `ma-cross`,
`rsi-oversold`, and `breakout-volume-trend`. They are kept — including the ones
that measured badly — so the table shows what does *not* work next to what does.
`breakout-volume-trend` in particular reads well on the full sample and is
below-random in two of six walk-forward windows; it stays as a documented
negative.

---

## 4. How to watch it

### The dashboard

The dashboard runs as its own launchd agent, so it is already listening at
**http://127.0.0.1:8787** whenever the machine is on -- there is nothing to
start. `uv run paper-serve` runs one in the foreground if you want a second copy
on another port.

Read-only, binds to localhost, has no route that changes anything, holds no
credentials. It re-reads the snapshot every 15 seconds.

| panel | what it tells you |
|---|---|
| headline strip | equity, return, **max drawdown**, cash, open, closed, refused |
| equity curve | realised equity, with starting capital dashed and the peak marked |
| risk | peak, max drawdown, distance from peak now, hit rate, mean/trade, best/worst |
| open positions | entry, mark, unrealised, **progress through the 168h hold** |
| by symbol | trades, hit rate, mean, P&L over the *whole* ledger |
| recent trades / refused | what closed, and what the wallet had to turn down |
| liveness dot | green → amber past 3h → red past 6h, plus a countdown to the next candle |

The dot is the one to glance at. If it goes amber, the collector is not running
and every number on the page is stale while still looking live.

**Drawdown sits next to return at the same size, deliberately.** The return says
the strategy made money; the drawdown says what holding it felt like. A page
showing only the first repeats the mistake this project has been avoiding since
the first backtest.

### The logs

```bash
tail -f logs/hourly.log     # one timestamped block per cycle
tail -f logs/collector.log  # every fetch, rotates at 5 MB
cat  logs/gaps.json         # what the last collect established
```

### The scheduler

```bash
launchctl list | grep crypto      # PID and last exit status ("-  0" = idle, last run OK)
launchctl kickstart gui/$(id -u)/com.leonselvig.crypto-data.hourly   # run now
```

Editing `scripts/hourly.sh` takes effect immediately — the plist just runs bash
on it. Only editing the **plist** needs a reload (`bootout` then `bootstrap`).

### The research tool

```bash
uv run python -m signals.cli                         # every rule at every hold
uv run python -m signals.cli --walk 6                # rolling out-of-sample
uv run python -m signals.cli --per-symbol --split    # where the result came from
```

---

## 5. What the evidence actually says

This is the section to read before doing anything with real money.

### The paper account

| | |
|---|---|
| equity | **12,074** (+20.74% on 10,000) |
| peak | **19,571** |
| **max drawdown** | **−40.99%** |
| distance from peak now | −40.31% |
| closed / refused | 399 / 1,621 |
| hit rate | 44.4% |
| mean per trade | +0.23% |

The headline is +20.7%. The account also nearly doubled and gave most of it
back. Both are true; the second is the more useful one.

### Walk-forward: the edge is not stable

Six sequential four-month windows, each judged against its own every-bar
benchmark (percentile = where the rule's mean sat among 1,000 random selections
of the same size):

| window | period | rule | benchmark | percentile |
|---|---|---|---|---|
| 1 | Aug–Dec 2024 | +6.49% | +4.67% | **99.7** |
| 2 | Dec 24–Apr 25 | −3.28% | −2.11% | **4.0** |
| 3 | Apr–Aug 2025 | +2.90% | +2.93% | 39.3 |
| 4 | Aug–Dec 2025 | −1.39% | −1.82% | 81.7 |
| 5 | Dec 25–Apr 26 | −2.36% | −2.67% | 77.5 |
| 6 | Apr–Aug 2026 | −0.82% | −1.11% | 75.8 |

**The edge lived almost entirely in window 1.** In window 2 the rule was *worse
than random timing* (4th percentile). Windows 4–6 are mildly above random but
negative in absolute terms — the rule loses less than buying at random, which is
not a business.

This is the most important table in the project, and it is why the pooled
"99.8th percentile" figure the full-sample view produces is a mirage: one bull
market wearing a trenchcoat.

### The edge is concentrated, not broad

Paper account, by symbol:

| symbol | trades | hit | mean | P&L |
|---|---|---|---|---|
| XRP/USDT | 79 | 41.8% | +1.03% | **+1,609** |
| SOL/USDT | 77 | 45.5% | +0.30% | +459 |
| ADA/USDT | 78 | 44.9% | +0.33% | +372 |
| BTC/USDT | 82 | 47.6% | −0.01% | −10 |
| ETH/USDT | 83 | 42.2% | −0.45% | **−749** |

Almost all the profit is XRP. BTC — the market you would most want to trade — is
a flat null. A strategy whose result depends on one small-cap is a strategy with
one data point.

### Things that were tried and did not work

- **A trend filter** (only buy above the 200-bar average) made it *worse*
  out-of-sample: it clustered entries into bear-market rallies. Kept in the
  registry as a documented negative.
- **A trailing stop** lowered absolute return but tightened the out-of-sample
  percentile. It is a downside-control tool, not a source of edge.

### The honest summary

The signal has **real entry-timing skill in trending markets** and **no
demonstrated ability to make money across regimes**. It is long-only, so it can
only profit when things go up, and the walk-forward shows exactly that.

---

## 6. How this may be used

### What it is genuinely good for now

**A research instrument.** The most valuable thing here is not the rule, it is
the apparatus around it: a fill model that cannot look ahead, a benchmark that
says whether a result beat random timing, a walk-forward view that catches an
edge that only existed in one period, and a causality guard that refuses to
measure a rule that reads the future. Any new rule you write inherits all of it
for free.

**A forward test that cannot lie to you.** The account now accumulates on
candles nobody has looked at. Every month it runs is a month of genuinely
out-of-sample evidence — which is the only kind that was ever going to settle
this.

**A template for the next strategy.** Adding a rule is one function and one
registry entry; the guard, the benchmark, the trade model, the wallet, the
dashboard and the schedule all apply to it immediately.

**A dashboard you can leave open.** It is a read-only instrument panel that
costs nothing to run and cannot do anything dangerous.

### What it is not ready for

**Trading real money on `breakout-volume` as it stands.** The walk-forward says
the edge is not stable, the profit is concentrated in one small-cap, and BTC
shows nothing. Deploying this would be betting on a regime returning, not on a
strategy working.

### Sensible next steps, roughly in order

1. **Let it run.** Months of forward paper results cost nothing and are the only
   honest evidence. Watch whether the recent windows' "beats random but loses
   money" pattern holds or breaks.
2. **Test new rules with the same apparatus.** Mean reversion, longer
   timeframes, volatility filters — the harness is the asset, use it.
3. **Consider shorting.** Every measured weakness traces back to being
   long-only. This is the strongest argument for a perpetuals venue such as
   Hyperliquid — but it means researching a market model this project has not
   touched yet.
4. **Only then, execution.** The `Broker` seam and a guarded Hyperliquid
   implementation already exist: the engine issues intents and reads fills back
   rather than assuming them, and `side` is explicit so shorts are expressible.
   Its testnet proof remains a gate, not permission to use mainnet.

### If you do eventually go live

The property this codebase defends hardest is that ordinary workflows cannot
place an order. Enabling the guarded broker is still the riskiest operation in
the project, so the path is staged:

```
paper (now) → manual execution → tiny auto size → real size
```

and the live broker needs, from day one: trade-only API keys with **withdrawals
disabled**, kept outside the repository; hard caps on order size and total
exposure; an allowed-symbol list; a kill-switch file; testnet first; and
reconciliation against the venue's own positions each run.

---

## 7. Where things live

| path | what |
|---|---|
| `config/symbols.json` | what to collect |
| `config/paper.json` | rule, universe, wallet, costs |
| `src/collector/` | fetching and the Parquet store |
| `src/signals/` | rules, indicators, fills, benchmarks, causality guard |
| `src/paper/` | engine, wallet, broker seam, state, dashboard |
| `scripts/hourly.sh` | one scheduled cycle |
| `data/` `logs/` `state/` | generated, not committed |

The test suite has no network access. The load-bearing one: replaying stored
candles through the paper engine one at a time reproduces the backtest *exactly*.
If that ever fails, paper results cannot be compared to anything.
