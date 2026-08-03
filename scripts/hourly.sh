#!/bin/bash
#
# One scheduled cycle: fetch whatever candles have closed, then advance the
# paper account over them. Run by launchd a few minutes past each hour; also
# perfectly safe to run by hand at any time.
#
# The order is not negotiable. `paper` reads the store and never fetches, so
# running it before `collect` would advance the account over candles that are
# already there and quietly do nothing new.
#
# WHY collect FAILING DOES NOT STOP paper
# ---------------------------------------
# A laptop asleep, a flaky connection or an exchange having a bad minute all
# make `collect` exit non-zero. None of them is a reason to skip `paper`: the
# store still holds every candle it held before, and advancing over those is
# both correct and idempotent. Skipping would mean one missed fetch silently
# costing an hour of paper trading as well.
#
# `collect` exits 2 when it ran but the venue is missing candles, which is a
# fact about the exchange rather than a failure here -- so it is logged and
# passed over.
#
# Everything is appended to logs/hourly.log with a timestamp, because the whole
# point of a scheduled job is that nobody is watching it when it runs.

set -uo pipefail

PROJECT="/Users/leonselvigbjornaraa/crypto-data"
UV="/Users/leonselvigbjornaraa/.local/bin/uv"
LOG="$PROJECT/logs/hourly.log"

cd "$PROJECT" || exit 1
mkdir -p "$PROJECT/logs"

stamp() { date -u "+%Y-%m-%d %H:%M:%S UTC"; }
say()   { echo "[$(stamp)] $*" >>"$LOG"; }

say "cycle start"

# --- fetch -----------------------------------------------------------------
collect_out=$("$UV" run collect 2>&1)
collect_code=$?
# Only the summary line; the full run log already lives in logs/collector.log,
# and duplicating it here would make this file unreadable within a week.
say "collect exit=$collect_code $(echo "$collect_out" | tail -1 | sed 's/^.*INFO *//')"

case "$collect_code" in
  0) ;;
  2) say "collect: ran, but candles are missing -- see logs/gaps.json" ;;
  *) say "collect: FAILED -- carrying on with the candles already stored" ;;
esac

# --- advance ---------------------------------------------------------------
paper_out=$("$UV" run paper 2>&1)
paper_code=$?

if [ "$paper_code" -ne 0 ]; then
  say "paper: FAILED (exit=$paper_code)"
  echo "$paper_out" | sed 's/^/    /' >>"$LOG"
  say "cycle end"
  exit "$paper_code"
fi

# The lines worth keeping: how far it got, and where the account stands.
echo "$paper_out" \
  | grep -E "advanced|equity|open +[0-9]|closed|refused" \
  | sed 's/^ */    /' >>"$LOG"

say "cycle end"
