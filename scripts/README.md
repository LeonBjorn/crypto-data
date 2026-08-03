# Scheduling

`hourly.sh` runs one cycle: `collect` to fetch whatever candles have closed,
then `paper` to advance the account over them. That order matters -- `paper`
only ever reads the store, so fetching second would advance over stale candles.

A failed `collect` does not stop `paper`. A sleeping laptop or a bad minute at
the exchange should not also cost an hour of paper trading, and advancing over
the candles already stored is both correct and idempotent.

## Installing the schedule (macOS)

```
cp scripts/com.leonselvig.crypto-data.hourly.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.leonselvig.crypto-data.hourly.plist
```

It fires at two minutes past every hour, and once at login so a machine that was
asleep catches up rather than waiting. The paths inside the plist are absolute,
because launchd starts with almost no environment -- if the project moves, they
have to be edited.

## Watching it

```
tail -f logs/hourly.log          # one block per cycle, timestamped
launchctl list | grep crypto     # PID and last exit status
launchctl kickstart gui/$(id -u)/com.leonselvig.crypto-data.hourly   # run it now
```

## Stopping it

```
launchctl bootout gui/$(id -u)/com.leonselvig.crypto-data.hourly
rm ~/Library/LaunchAgents/com.leonselvig.crypto-data.hourly.plist
```

Nothing else needs undoing: the job only writes to `data/`, `logs/` and
`state/`, all of which are generated and none of which are committed.
