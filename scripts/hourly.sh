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

# Rotate before writing, matching what logging_setup.py already does for
# collector.log. Roughly half a kilobyte per cycle is nothing per day and about
# four megabytes a year, which is the kind of growth that is invisible until it
# is a decade old. One megabyte holds about two thousand cycles -- three months
# of hourly runs -- and two old files are kept.
rotate() {
  local file="$1" limit=1048576
  [ -f "$file" ] || return 0
  local size
  size=$(wc -c <"$file" 2>/dev/null || echo 0)
  if [ "$size" -gt "$limit" ]; then
    [ -f "$file.1" ] && mv -f "$file.1" "$file.2"
    mv -f "$file" "$file.1"
  fi
}

rotate "$LOG"
# launchd appends to these itself and never truncates them. They should stay
# empty -- everything the job says goes to hourly.log -- but if the job ever
# starts failing before it can log, this is what stops that filling the disk.
rotate "$PROJECT/logs/launchd.out.log"
rotate "$PROJECT/logs/launchd.err.log"

# How long since the previous cycle, from the heartbeat the paper run writes.
# An hourly job should never be much above 1.0; past 2 a cycle was missed, and
# the forward record has a hole unless the catch-up below fills it.
if [ -f "$PROJECT/state/heartbeat.json" ]; then
  gap=$(python3 -c "
import json,time
try:
    b=json.load(open('$PROJECT/state/heartbeat.json'))
    print(f\"{(time.time()*1000-b['at'])/3600000:.1f}\")
except Exception:
    print('?')
" 2>/dev/null)
  case "$gap" in
    ?|"") ;;
    *) awk -v g="$gap" 'BEGIN{exit !(g>2)}' 2>/dev/null &&        say "WARNING: ${gap}h since the last cycle -- the schedule may have stopped" ;;
  esac
fi

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
