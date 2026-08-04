"""Copies of the ledger, and a pulse saying the schedule is still running.

Two small things guarding the same asset. The paper account's ledger is the only
out-of-sample evidence this project has, and unlike the candle store it cannot
be refetched from anywhere -- if it is lost, the record of those trades stops
having existed. It also lives in `state/`, which is not committed, so until now
there was exactly one copy of it on one disk.

BACKUPS
-------
A timestamped copy is written before each save and a fixed number are kept. That
is not a substitute for a real backup on another machine and does not pretend to
be; it defends against the likely failures rather than the dramatic one -- a bad
edit, a mistaken `--reset`, a bug in this project writing something wrong -- and
those are the ones that actually happen.

Rotation keeps the oldest as well as the newest. Keeping only recent copies is
worse than useless when the corruption is subtle: by the time anyone notices,
every recent copy already contains it.

THE HEARTBEAT
-------------
A scheduled job that stops running is the failure this project is least equipped
to notice, because everything downstream keeps working. The dashboard would show
stale data, the ledger would simply stop growing, and the three-month forward
record would quietly acquire a hole that nothing in it records.

So each cycle writes the time it ran, and the next one reports how long it has
been. The gap is then a number that appears in the log and on the page rather
than something a person has to infer from an absence.
"""

import json
import os
import time
from pathlib import Path

__all__ = ["HEARTBEAT_NAME", "keep", "prune", "read_heartbeat", "write_heartbeat"]

HEARTBEAT_NAME = "heartbeat.json"

# Enough to cover a few days of hourly runs, plus the oldest few kept forever by
# `prune` so that a slow corruption cannot push every clean copy out.
DEFAULT_KEEP = 48
KEEP_OLDEST = 4


def keep(state_path, *, directory=None, limit=DEFAULT_KEEP):
    """Copy the current state file aside, timestamped. Returns the copy's path.

    Called before a save rather than after, so the copy is of the last state
    known to be complete. A crash midway through writing the new one then leaves
    both the previous file and this copy intact.
    """
    state_path = Path(state_path)
    if not state_path.exists():
        return None

    folder = Path(directory) if directory else state_path.parent / "backups"
    folder.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    target = folder / f"{state_path.stem}-{stamp}{state_path.suffix}"
    target.write_bytes(state_path.read_bytes())
    prune(folder, limit=limit)
    return target


def prune(folder, *, limit=DEFAULT_KEEP, oldest=KEEP_OLDEST):
    """Delete the middle of the history, keeping the newest and the oldest few.

    Dropping purely by age is the obvious rule and the wrong one. A ledger that
    goes subtly wrong is usually noticed days later, by which point every recent
    copy has inherited the problem -- so a handful of the earliest copies are
    kept permanently as a floor to fall back to.
    """
    folder = Path(folder)
    copies = sorted(folder.glob("*.json"))
    if len(copies) <= limit:
        return []

    protected = set(copies[:oldest]) | set(copies[-(limit - oldest):])
    removed = []
    for path in copies:
        if path not in protected:
            path.unlink(missing_ok=True)
            removed.append(path)
    return removed


def write_heartbeat(folder, *, at=None, cycle=None):
    """Record that a cycle completed. Returns what was written."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / HEARTBEAT_NAME

    previous = read_heartbeat(folder)
    now = int(at if at is not None else time.time() * 1000)

    beat = {
        "at": now,
        "previous": (previous or {}).get("at"),
        "gap_hours": (
            round((now - previous["at"]) / 3_600_000, 2)
            if previous and previous.get("at") else None
        ),
        "cycle": cycle,
    }

    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(beat, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)
    return beat


def read_heartbeat(folder):
    """The last recorded cycle, or None if there has never been one."""
    path = Path(folder) / HEARTBEAT_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # A corrupt heartbeat is not worth failing a run over. It is a status
        # marker, not the ledger, and the next cycle rewrites it.
        return None


def hours_since(folder, *, now=None):
    """Hours since the last recorded cycle, or None if there has never been one.

    The number the schedule is judged by. An hourly job should never return much
    above one; anything past two means a cycle was missed, and past a day means
    the record has a hole in it.
    """
    beat = read_heartbeat(folder)
    if not beat or not beat.get("at"):
        return None
    now = now if now is not None else time.time() * 1000
    return round((now - beat["at"]) / 3_600_000, 2)
