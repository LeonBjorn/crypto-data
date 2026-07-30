# crypto-data

A local store of historical OHLCV candles, kept in Parquet files on this machine.

It fetches public market data from an exchange and writes it to disk in a layout
that a backtester can read directly. It does not trade, does not place orders,
does not know what a strategy is, and never uses an API key. Everything it
touches is public data.

This is milestone 1 of a larger project. The point of doing it separately is
that a backtest is only as trustworthy as the candles under it, and it is much
easier to trust the data when nothing else is going on in the same codebase.


## Setup, once

You need [uv](https://docs.astral.sh/uv/). Then, from the project directory:

```
uv sync
```

That reads `pyproject.toml` and `uv.lock` and builds `.venv/` with the exact
dependency versions this project was built against. It takes a minute the first
time and is instant afterwards.

Check it worked:

```
uv run pytest
```

You should see roughly 300 tests pass in about two seconds. The suite never
touches the network — it is blocked outright while the tests run — so a green
result here says nothing about your internet connection, only about the code.


## Everyday use

```
uv run collect
```

That reads `config/symbols.json`, works out what is already on disk, fetches
only what is missing, and writes a report. Run it as often as you like. It is
safe to run twice in a row, safe to interrupt with Ctrl-C, and safe to run again
after interrupting.

The useful flags, all of which override the config file for one run only:

```
uv run collect --symbols BTC/USDT ETH/USDT     # just these two
uv run collect --timeframe 4h                  # a different timeframe
uv run collect --start 2023-01-01              # go further back
uv run collect --verbose                       # log every request
uv run collect --help                          # the full list
```

The division of labour is deliberate: the config file holds what you want
backfilled routinely, and the flags are for the one-off question you are asking
right now. Nothing you type on the command line changes the config file.

`python -m collector.cli` does exactly the same thing as `collect` and works
without installing anything, which is occasionally useful when something is
wrong with the environment.


## What the exit code means

Worth knowing, because this is how you find out something went wrong without
reading the whole log.

| Code | Meaning | What to do |
|------|---------|------------|
| 0 | The full requested range is on disk. | Nothing. |
| 1 | The run could not complete. | Read the error — it is almost always the config or the network. |
| 2 | The run finished, but candles are missing or a symbol failed. | Read `logs/gaps.json`. |

In a shell, `echo $?` after the command shows it.

The split between 1 and 2 exists because they want different responses from
you. A 1 means fix something and run again. A 2 means the run did what it could
and you should look at what it could not do — which is sometimes fine, because
exchanges genuinely have holes in their history that no amount of re-running
will fill.

One consequence worth knowing in advance: if the exchange is permanently missing
a candle in your range, every future run will exit 2 forever. That is honest but
noisy. The eventual fix is a list of holes you have looked at and accepted, not
a quieter exit code.


## How the store is laid out

```
data/
└── binance/                  the exchange id
    ├── BTC_USDT/             the symbol, with / replaced by _
    │   └── 1h.parquet        one file per timeframe
    ├── ETH_USDT/
    │   └── 1h.parquet
    └── ...
```

One file per exchange, symbol and timeframe. Nothing is ever mixed into one
large file, which means a corrupted or interrupted write can only ever affect
one symbol's one timeframe, and you can delete any single file to force it to be
refetched.

Each file has six columns:

| Column | Type | Meaning |
|--------|------|---------|
| `timestamp` | int64 | The candle's **open** time, epoch milliseconds, UTC |
| `open` | float64 | |
| `high` | float64 | |
| `low` | float64 | |
| `close` | float64 | |
| `volume` | float64 | |

Two things about `timestamp` are load-bearing. It is the moment the candle
*opened*, not closed — so the row stamped 14:00 covers 14:00 to 15:00. And it is
a plain integer rather than a datetime, on purpose: a datetime with no timezone
attached silently means local time, and that is the single easiest way to
corrupt a store like this. Convert to dates when you display them, never when
you store them.

Rows are sorted by timestamp, never duplicated, and the currently-forming candle
is always excluded — only closed candles are stored, so a row never changes
after it is written.


## Reading it back

```python
import pandas as pd

candles = pd.read_parquet("data/binance/BTC_USDT/1h.parquet")

# Only for looking at. Don't write this back to the store.
candles["time"] = pd.to_datetime(candles["timestamp"], unit="ms", utc=True)
print(candles.tail())
```

Two years of hourly candles is about 17,500 rows per symbol, which is small
enough that reading a whole file into memory is entirely reasonable.


## The config file

`config/symbols.json`:

```json
{
  "_comment": "free text, ignored -- any key starting with _ is a comment",
  "exchange": "binance",
  "timeframe": "1h",
  "start": "2024-08-01",
  "symbols": ["BTC/USDT", "ETH/USDT"]
}
```

JSON has no comment syntax, so any key beginning with `_` is treated as prose
for a human and ignored. The reasoning behind the exchange choice lives in there
rather than in a separate file nobody opens.

Supported timeframes are `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `12h` and `1d` —
anything that divides a day evenly. Weeks and months are refused, because their
boundaries cannot be derived from the epoch reliably and the result would be
data that is silently offset rather than obviously broken.

Dates are `YYYY-MM-DD`, optionally with a time (`2024-08-01T06:00`), always
interpreted as UTC. Timezone offsets are refused rather than converted.

The file is strict on purpose. An unrecognised key is an error, not something
quietly ignored — because `"symbol"` instead of `"symbols"` would otherwise
produce a config file that looks completely correct, a run that does nothing at
all, and an exit code of zero. It now produces a one-line message naming the
typo instead. The same applies to duplicate symbols, symbols without a slash,
and dates in a format that means different things in different countries. The
rule throughout is that no input is ever quietly corrected.

The exchange is a [ccxt](https://docs.ccxt.com) id — `binance`, `bybit`, `okx`
and so on. Not all of them are usable here: Kraken's public endpoint ignores the
`since` parameter entirely and serves only its recent history, which makes a
long backfill impossible there. If you change exchanges, check that one thing
first.


## Logs and the gap report

```
logs/
├── collector.log       everything, rotating at 5 MB, 5 old files kept
└── gaps.json           what the last run established
```

The log also goes to your terminal as the run happens. `--verbose` adds a line
each time candles are actually written to disk, saying how many were new and how
many the file now holds. It does **not** show you individual HTTP requests —
ccxt's own logging is deliberately kept quiet even in verbose mode, because at
DEBUG it produces enough output to bury everything this project logs.

`gaps.json` is the machine-readable version and is rewritten every run, clean or
not. It always exists, which matters: "no report" and "a report saying nothing
is missing" have to look different, or an absent file becomes indistinguishable
from a passing one.

The parts to look at first:

- `complete` — whether the full requested range is on disk for every symbol.
- `totals` — added, missing, repaired, and how many symbols failed outright.
- `symbols[].missing` — timestamp ranges the exchange does not appear to have.
- `errors` — symbols whose fetch failed, with the reason.

A symbol that fails does not stop the run; the remaining symbols are still
fetched, because one delisted pair should not cost you four good backfills. But
the failure goes into `errors`, sets `complete` to false, and changes the exit
code, so it cannot be mistaken for success.


## When something goes wrong mid-run

Nothing needs cleaning up. Candles are written every few pages as the run goes,
rather than all at the end, and each write is atomic — the file is written
elsewhere and then moved into place, so a file on disk is always a complete,
valid, sorted file. There is no such thing as a half-written Parquet file here,
and so there is nothing to repair.

To be precise about what you *do* lose: candles fetched since the last write are
held in memory, so an interrupted run discards up to a few pages' worth of
fetching. They are not damaged, they were simply never written, and the next run
fetches them again. The cost of an interruption is some repeated downloading,
never a corrupt file or a store you have to go and inspect.

So if the run dies, the network drops, or you hit Ctrl-C, the answer is always
the same: run it again. It will pick up from what is already stored.


## The tests

```
uv run pytest                    # all of them, quietly
uv run pytest -v                 # one line per test
uv run pytest tests/test_cli.py  # just one file
uv run pytest -k gap             # anything with "gap" in the name
```

The suite runs against a fake exchange held in memory and cannot reach the
network — outbound connections are blocked while it runs, so a test that forgets
its mock fails loudly instead of quietly making a real request and passing
slowly.

The tests have themselves been checked, using a scratch script that deliberately
breaks the source one line at a time and confirms the suite notices. All 109 of
those deliberate breakages are currently caught. This matters more than the test
count: a test that has never been seen to fail is a hypothesis, not a test, and
several tests in this project turned out to be passing for reasons that had
nothing to do with what they claimed to check.


## Git

```
git status                          # what has changed
git add -A && git commit -m "..."   # save a checkpoint
git log --oneline                   # the history
git diff                            # what you changed since the last commit
```

`data/` and `logs/` are deliberately not committed. They are generated, they are
large, they change every run, and git handles binary files that change
constantly very badly — once a large file is in the history it is there
permanently. Everything in them is reproducible from the code, which is the
whole point.

`uv.lock` **is** committed, deliberately. It records the exact version every
dependency resolved to, and committing it is the difference between `uv sync`
reproducing this environment and reproducing a similar one.


## What is deliberately not here

No orders, no strategy logic, no indicators, no backtesting, no live WebSocket
feeds, no scheduling, no dashboard. Those are later milestones and they are
separate on purpose — the value of this piece is that it is small enough to read
end to end and be sure of.

There is also no support for API credentials anywhere in the codebase, including
no flag to pass one. That is not an oversight. Public endpoints are all this
needs, and the safest way to guarantee a data collector cannot place a trade is
for it to have no means of authenticating at all.
