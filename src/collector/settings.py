"""Turning a config file and a command line into one validated set of settings.

Requirement 8 asks for a CLI that takes symbols, timeframe and start, with the
symbol list coming from a config file. That is two sources for the same values,
which needs a rule: the file holds the intent, the command line overrides it for
one run. `config/symbols.json` is what you want backfilled every day; `--symbols
BTC/USDT` is what you want right now while investigating something.

Everything here is validation, and it is deliberately unforgiving. This is the
only layer between a typed mistake and the rest of the system, and past that
point mistakes stop looking like mistakes. A misspelled key becomes an empty run
that exits cleanly. A date in the wrong format becomes a start time that is not
the one you meant. A symbol without a slash becomes a directory named after
nothing. None of those announce themselves later, so all of them are refused
here.

The one rule worth stating outright: no input is ever quietly corrected. If a
value cannot be used exactly as written, this module raises rather than guessing
what was meant. Guessing is how a two-year backfill silently becomes a two-week
one.

No logging and no clock reads. This module is a pure function of a file's
contents and some arguments, which is what makes it cheap to test exhaustively.
"""

import datetime as dt
import difflib
import json
from pathlib import Path
from typing import NamedTuple

from collector.backfill import DEFAULT_MAX_GAPS
from collector.timeframes import timeframe_to_ms

# Relative on purpose. The store and the logs belong to the project directory you
# run from; an absolute default would write somewhere else the moment you moved.
DEFAULT_DATA_DIR = "data"
DEFAULT_LOG_DIR = "logs"
DEFAULT_CONFIG_PATH = "config/symbols.json"

REQUIRED_KEYS = ("exchange", "timeframe", "start", "symbols")

# JSON has no comment syntax, and the reasoning behind the exchange choice is
# worth storing beside the choice rather than in a separate file nobody opens.
# Any key beginning with an underscore is treated as prose for a human.
COMMENT_PREFIX = "_"

_DATE_ONLY = "%Y-%m-%d"
_DATE_AND_TIME = "%Y-%m-%dT%H:%M"

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_DATA_DIR",
    "DEFAULT_LOG_DIR",
    "ConfigError",
    "Settings",
    "load_config",
    "parse_start",
    "resolve",
]


class ConfigError(ValueError):
    """Raised for input this project will not act on.

    A ValueError subclass so callers who only care that the input was bad can
    catch the broader type, while the CLI catches this specifically and prints
    the message instead of a traceback.
    """


class Settings(NamedTuple):
    """One run's fully resolved, fully validated settings.

    Everything is in its final form: the timeframe is parsed, the start is epoch
    milliseconds, the directories are Paths. Nothing downstream needs to
    re-validate or re-parse, which means nothing downstream can disagree with
    this about what was asked for.

    `timeframe_ms` is carried alongside `timeframe` rather than recomputed where
    needed. It is derived from the string that was already validated, so keeping
    it here makes it impossible for one caller to parse it successfully and
    another to parse it differently.
    """

    exchange: str
    timeframe: str
    timeframe_ms: int
    start_ms: int
    symbols: list
    data_dir: Path
    log_dir: Path
    max_gaps: int


def parse_start(text):
    """
    Parse a start date into epoch milliseconds, interpreting it as UTC.

    Accepts `2024-08-01`, `2024-08-01T06:00`, `2024-08-01 06:00`, and any of
    those with a trailing `Z`.

    Explicit offsets such as `+02:00` are refused. There is no good way to handle
    one: honouring it stores a start time other than the one that was typed,
    while ignoring it is silently wrong by the size of the offset. Since the
    whole store is UTC, the useful response is to say so and stop.

    A bare date means midnight UTC, which is also a candle boundary for every
    timeframe this project supports -- so a plain date never lands mid-candle.
    """
    if not isinstance(text, str):
        raise ConfigError(
            f"start must be a string like '2024-08-01', got {type(text).__name__}"
        )

    cleaned = text.strip()
    if not cleaned:
        raise ConfigError("start is empty. Expected a date like '2024-08-01'.")

    # A trailing Z states the timezone we already assume, so it is redundant
    # rather than wrong, and rejecting it would be pedantic.
    if cleaned.endswith(("Z", "z")):
        cleaned = cleaned[:-1]

    cleaned = cleaned.replace(" ", "T")

    # Any sign after the date portion is an offset. Checking from index 10 avoids
    # tripping over the hyphens inside the date itself.
    if "+" in cleaned[10:] or "-" in cleaned[10:]:
        raise ConfigError(
            f"start {text!r} carries a timezone offset. This project stores UTC "
            f"only, so write the time in UTC without an offset, "
            f"e.g. '2024-08-01T06:00'."
        )

    parsed = None
    for pattern in (_DATE_ONLY, _DATE_AND_TIME):
        try:
            parsed = dt.datetime.strptime(cleaned, pattern)
            break
        except ValueError:
            continue

    if parsed is None:
        raise ConfigError(
            f"cannot read start {text!r}. Expected YYYY-MM-DD, optionally with a "
            f"time, e.g. '2024-08-01' or '2024-08-01T06:00'. Note the order: "
            f"year first, and no day/month formats, because '01/08/2024' means "
            f"different dates in different countries."
        )

    ms = int(parsed.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)

    if ms < 0:
        raise ConfigError(
            f"start {text!r} is before 1970, which epoch milliseconds cannot "
            f"represent as a positive number. No exchange has candles that old "
            f"in any case."
        )

    return ms


def _validate_symbols(symbols, source):
    """
    Check a symbol list and return it as a plain list.

    `source` names where the list came from, so the message can point at the file
    or at the command line rather than leaving you to work it out.

    A bare string is refused explicitly because it is the one wrong type that
    would otherwise work: "BTC/USDT" is iterable, so a lenient loader would treat
    it as eight one-character symbols and fail much later with something
    inscrutable.

    Duplicates are refused rather than removed. Removing them silently would be
    convenient and is exactly the kind of quiet correction this module avoids;
    a repeated symbol means the list is not what its author thinks it is, and
    that is worth one second of their attention.
    """
    if isinstance(symbols, str):
        raise ConfigError(
            f"symbols in {source} must be a list, not a single string. "
            f'Write ["{symbols}"] rather than "{symbols}".'
        )

    if not isinstance(symbols, list):
        raise ConfigError(
            f"symbols in {source} must be a list, got {type(symbols).__name__}"
        )

    if not symbols:
        raise ConfigError(f"symbols in {source} is empty; name at least one symbol")

    seen = set()
    for symbol in symbols:
        if not isinstance(symbol, str):
            raise ConfigError(
                f"every symbol in {source} must be a string, got "
                f"{symbol!r} ({type(symbol).__name__})"
            )
        if "/" not in symbol:
            raise ConfigError(
                f"symbol {symbol!r} in {source} is not in ccxt's BASE/QUOTE form. "
                f"Write 'BTC/USDT' rather than 'BTC' or 'BTCUSDT'."
            )
        if symbol in seen:
            raise ConfigError(
                f"symbol {symbol!r} appears twice in {source}. Fetching it twice "
                f"would only repeat the same requests."
            )
        seen.add(symbol)

    return list(symbols)


def load_config(path):
    """
    Read and validate the JSON config file.

    Returns the parsed dict. Does not resolve anything against command-line
    arguments -- that is `resolve`'s job -- so this can be tested purely against
    file contents.

    Unknown keys are an error, which is the most valuable rule in this module.
    The natural alternative is to read the keys you recognise and ignore the
    rest, and that turns `"symbol"` for `"symbols"` into a config file that looks
    entirely correct, a run that does nothing, and an exit code of zero. Refusing
    unknown keys converts that into a one-line message naming the typo.
    """
    path = Path(path)

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(
            f"no config file at {path}. Expected a JSON file with keys: "
            f"{', '.join(REQUIRED_KEYS)}."
        ) from None
    except OSError as failure:
        raise ConfigError(f"cannot read config file {path}: {failure}") from failure

    try:
        config = json.loads(text)
    except json.JSONDecodeError as failure:
        raise ConfigError(
            f"config file {path} is not valid JSON: {failure}. A common cause is "
            f"a trailing comma after the last item, which JSON does not allow."
        ) from failure

    if not isinstance(config, dict):
        raise ConfigError(
            f"config file {path} must contain a JSON object (curly braces), "
            f"got {type(config).__name__}"
        )

    keys = [key for key in config if not key.startswith(COMMENT_PREFIX)]

    unknown = [key for key in keys if key not in REQUIRED_KEYS]
    if unknown:
        # A near-miss is nearly always a typo, and naming the intended key saves
        # staring at two almost identical words.
        hints = []
        for key in unknown:
            close = difflib.get_close_matches(key, REQUIRED_KEYS, n=1)
            hints.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise ConfigError(
            f"config file {path} has unrecognised key(s): {', '.join(hints)}. "
            f"Recognised keys are {', '.join(REQUIRED_KEYS)}, plus any key "
            f"starting with '{COMMENT_PREFIX}' for comments."
        )

    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ConfigError(
            f"config file {path} is missing required key(s): {', '.join(missing)}"
        )

    _validate_symbols(config["symbols"], f"config file {path}")

    return config


def resolve(
    config,
    *,
    exchange=None,
    timeframe=None,
    start=None,
    symbols=None,
    data_dir=None,
    log_dir=None,
    max_gaps=None,
):
    """
    Combine a loaded config with command-line overrides into final Settings.

    Every override is None when the flag was not passed, because that is what
    argparse produces. None therefore has to mean "not specified" rather than
    "set this to nothing" -- and `max_gaps=0` has to survive, since repairing no
    gaps is a real thing to ask for. Treating zero as unset is the classic
    version of this bug, so it is tested for.

    Overrides are validated by exactly the same code as file values. A symbol
    typed on the command line can do just as much damage as one in the file, so
    arriving by a different route must not skip any check.
    """
    resolved_exchange = exchange if exchange is not None else config["exchange"]
    resolved_timeframe = timeframe if timeframe is not None else config["timeframe"]
    resolved_start = start if start is not None else config["start"]

    if symbols is not None:
        resolved_symbols = _validate_symbols(symbols, "the --symbols argument")
    else:
        resolved_symbols = _validate_symbols(config["symbols"], "the config file")

    if not isinstance(resolved_exchange, str) or not resolved_exchange.strip():
        raise ConfigError(
            f"exchange must be a non-empty string such as 'binance', "
            f"got {resolved_exchange!r}"
        )

    resolved_max_gaps = max_gaps if max_gaps is not None else DEFAULT_MAX_GAPS
    if not isinstance(resolved_max_gaps, int) or isinstance(resolved_max_gaps, bool):
        raise ConfigError(f"max_gaps must be a whole number, got {resolved_max_gaps!r}")
    if resolved_max_gaps < 0:
        raise ConfigError(
            f"max_gaps cannot be negative, got {resolved_max_gaps}. Use 0 to "
            f"bring the store up to date without repairing older holes."
        )

    return Settings(
        exchange=resolved_exchange.strip(),
        timeframe=resolved_timeframe,
        # Raises TimeframeError, deliberately not wrapped: timeframes.py already
        # explains at length why a given timeframe is unsupported, and rephrasing
        # it here would only lose that.
        timeframe_ms=timeframe_to_ms(resolved_timeframe),
        start_ms=parse_start(resolved_start),
        symbols=resolved_symbols,
        data_dir=Path(data_dir if data_dir is not None else DEFAULT_DATA_DIR),
        log_dir=Path(log_dir if log_dir is not None else DEFAULT_LOG_DIR),
        max_gaps=resolved_max_gaps,
    )
