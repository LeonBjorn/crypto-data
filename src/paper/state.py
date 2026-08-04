"""What a paper run remembers between runs, and how it survives being killed.

A backtest starts from nothing every time. A paper run cannot: it has open
positions, a ledger, and a wallet whose cash is the sum of everything it has
done so far, and losing any of that means the run silently becomes a different
run. So the state is written to disk, and it is written the same way the
collector writes candles -- to a temporary file that is then moved into place,
which is atomic on POSIX. A process killed mid-write leaves either the old
complete state or the new complete state, never half of either.

That matters more here than it does for candles. A truncated candle file can be
refetched from the exchange; a truncated ledger cannot be refetched from
anywhere, because it is the only record that those trades were ever taken.

THE FINGERPRINT
---------------
The awkward case is not a crash, it is a quiet edit. Change the rule, the hold
or the starting capital, run again, and the engine would happily append trades
taken under the new settings to a ledger built under the old ones -- producing
one equity curve that describes two different strategies and says so nowhere.

So the settings that define what a run *is* are fingerprinted, and a run whose
fingerprint no longer matches its state is refused rather than resumed. The fix
is `--reset`, which is one word and deliberately explicit: starting over is a
decision, and it should look like one.

Settings that do not change what a trade would have been -- where the dashboard
listens, how much is printed -- are not in the fingerprint, because refusing to
resume over a port number would just teach everyone to pass --reset by reflex.
"""

import json
import os
from contextlib import contextmanager
from pathlib import Path

import fcntl

__all__ = [
    "MAX_REJECTIONS_KEPT",
    "STATE_VERSION",
    "StateError",
    "exclusive_lock",
    "fingerprint",
    "load",
    "save",
    "save_json",
]

# Bumped when the shape of the file changes incompatibly. A state file from an
# older version is refused rather than guessed at: the alternative is code that
# quietly reinterprets an old ledger under new rules.
STATE_VERSION = 1

# Rejections are unbounded -- a busy account can refuse thousands -- and the
# whole list is worth nothing to anyone. The recent ones say what is being
# refused now, and the count says how much has been refused in total, which is
# the number that actually matters when reading a ledger.
MAX_REJECTIONS_KEPT = 200

# Two fingerprints, because two different things can change and they deserve
# different answers.
#
# STRATEGY settings define what a trade *would have been*. Changing the rule or
# the hold and then resuming would append trades taken under new rules to a
# ledger built under old ones -- one equity curve describing two strategies.
# That is refused.
#
# RISK settings define how much is traded and when to stop. Changing those on a
# running account is a legitimate operational act, not a mistake: a risk control
# you may never adjust without destroying your track record is a risk control
# nobody will turn on. So a risk change is *adopted*, and recorded in the state
# with the moment it happened, and the curve is honestly marked as spanning more
# than one regime rather than being silently thrown away or silently mixed.
FINGERPRINTED = (
    "exchange",
    "timeframe",
    "symbols",
    "rule",
    "params",
    "hold",
    "stop",
    "target",
    "trail",
    "starting_capital",
    "size_fraction",
    "max_positions",
    "one_per_symbol",
    "costs",
)

RISK_FINGERPRINTED = (
    "sizing",
    "target_vol",
    "max_leverage",
    "max_drawdown",
)


class StateError(Exception):
    """Raised when saved state cannot be used as it stands."""


@contextmanager
def exclusive_lock(path):
    """Hold the per-ledger lock for one complete paper run.

    An atomic rename prevents a torn state file. It does not serialize two
    processes that both load the same state, advance independently, and then
    each atomically replace it. The latter writer would silently discard the
    former's work. The lock deliberately spans load, advance, backup, and save
    in ``paper.cli`` so manual runs and launchd cannot race the forward record.

    The lock file remains after a crash, but ``flock`` is released by the kernel
    with its process, so its presence is not itself a stop signal.
    """
    path = Path(path)
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise StateError(
                f"another paper run already holds {lock_path}; refusing to run "
                f"a second writer against the same ledger"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def risk_fingerprint(config) -> str:
    """A stable string over the settings that decide size and when to stop."""
    return json.dumps(
        {key: config.get(key) for key in RISK_FINGERPRINTED}, sort_keys=True, default=str
    )


def fingerprint(config) -> str:
    """A stable string over the settings that define the run.

    Sorted and JSON-encoded rather than hashed. It is longer, and it means that
    when a run is refused the file itself says what changed -- which is the
    question anyone asks next.
    """
    material = {key: config.get(key) for key in FINGERPRINTED}
    if isinstance(material.get("symbols"), (list, tuple)):
        material["symbols"] = sorted(material["symbols"])
    return json.dumps(material, sort_keys=True, default=str)


def save_json(path, payload):
    """Write JSON atomically, replacing whatever was there.

    The temporary file is created beside the target on purpose: os.replace is
    only atomic within one filesystem, so a temp file in /tmp could not be moved
    into place safely.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")

    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        # Includes KeyboardInterrupt, which is the case this exists for. Clear
        # the temp file and let the exception continue; the previous state file
        # has not been touched.
        temp_path.unlink(missing_ok=True)
        raise

    return path


def save(path, payload):
    """Write state atomically, attaching its format version first."""
    body = dict(payload)
    body["version"] = STATE_VERSION
    return save_json(path, body)


def load(path, *, expect=None):
    """Read saved state, or return None if there is none.

    A missing file is not an error -- that is what every first run looks like.
    A file that exists and cannot be used *is* an error, because the alternative
    is silently starting over and calling it a fresh start.
    """
    path = Path(path)
    if not path.exists():
        return None

    try:
        with open(path, encoding="utf-8") as handle:
            state = json.load(handle)
    except json.JSONDecodeError as failure:
        raise StateError(
            f"{path} is not valid JSON: {failure}. It may have been edited by "
            f"hand. Move it aside and run again with --reset to start over."
        ) from failure

    if not isinstance(state, dict):
        raise StateError(f"{path} should contain a JSON object, got {type(state).__name__}")

    version = state.get("version")
    if version != STATE_VERSION:
        raise StateError(
            f"{path} was written by state version {version!r}, and this is "
            f"version {STATE_VERSION}. The shapes are not compatible; run with "
            f"--reset to start a new ledger."
        )

    if expect is not None and state.get("fingerprint") != expect:
        raise StateError(
            f"{path} was written under different settings, so resuming would "
            f"append trades taken under new rules to a ledger built under old "
            f"ones -- one equity curve describing two strategies.\n"
            f"  saved: {state.get('fingerprint')}\n"
            f"  now:   {expect}\n"
            f"Run with --reset to start a new ledger under the current settings."
        )

    return state
