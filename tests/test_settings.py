"""Tests for config loading, date parsing, and settling on the final settings.

The theme running through these is that bad input must be rejected loudly. This
is the layer where a typo enters the system, and a typo that survives this layer
becomes a wrong file on disk, or a two-year backfill that quietly asks for two
weeks. Almost every test below is therefore about something being refused.
"""

import json
import time

import pytest

from collector import settings
from collector.timeframes import HOUR_MS, TimeframeError

# 2024-08-01T00:00:00Z, the start date in the shipped config.
AUG_2024 = 1722470400000


def write_config(directory, data):
    """Write a config file and return its path."""
    path = directory / "symbols.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def valid_config(**overrides):
    """A config that loads cleanly, so each test can spoil one field."""
    base = {
        "exchange": "binance",
        "timeframe": "1h",
        "start": "2024-08-01",
        "symbols": ["BTC/USDT", "ETH/USDT"],
    }
    base.update(overrides)
    return base


@pytest.fixture
def machine_in_tokyo(monkeypatch):
    """Pretend the machine's local timezone is nine hours ahead of UTC.

    This exists because the obvious test of "a date is midnight UTC" cannot
    actually detect the bug it is meant to catch. If `parse_start` forgot to
    attach a UTC timezone, the resulting naive datetime would be interpreted in
    local time -- and on a machine already set to UTC that is the same number, so
    the test passes and the bug ships. It would then show up only on a laptop in
    Oslo, as a start date two hours out.

    Forcing a known non-UTC zone makes the assertion mean what it says on any
    machine. tzset has to be called by hand: the C library caches the zone, so
    changing the environment variable alone does nothing.
    """
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


class TestParseStart:
    def test_a_plain_date_is_midnight_utc(self):
        assert settings.parse_start("2024-08-01") == AUG_2024

    def test_a_date_is_utc_even_when_the_machine_is_not(self, machine_in_tokyo):
        """The store is UTC, so the start date has to be UTC regardless of where
        the laptop happens to be. Nine hours of drift would put every candle
        boundary in the wrong place."""
        assert settings.parse_start("2024-08-01") == AUG_2024
        assert settings.parse_start("2024-08-01T06:00") == AUG_2024 + 6 * HOUR_MS

    def test_a_date_and_time_is_accepted(self):
        assert settings.parse_start("2024-08-01T06:00") == AUG_2024 + 6 * HOUR_MS

    def test_a_space_may_separate_date_from_time(self):
        # Typing an ISO 'T' by hand is awkward; a space means the same thing.
        assert settings.parse_start("2024-08-01 06:00") == AUG_2024 + 6 * HOUR_MS

    def test_a_trailing_z_is_accepted_since_it_says_what_we_already_assume(self):
        assert settings.parse_start("2024-08-01T00:00Z") == AUG_2024

    def test_a_named_offset_is_refused_rather_than_silently_shifted(self):
        # The whole store is UTC. Accepting +02:00 would mean either honouring it
        # (and storing a start the user did not type) or ignoring it (and being
        # two hours out). Refusing is the only honest option.
        with pytest.raises(settings.ConfigError, match="UTC"):
            settings.parse_start("2024-08-01T00:00+02:00")

    def test_a_negative_offset_is_refused_too(self):
        # Only the + case was tested at first, and a check written as
        # `if "+" in ...` passed happily. Half a guard looks exactly like a whole
        # one until someone west of Greenwich uses it.
        with pytest.raises(settings.ConfigError, match="UTC"):
            settings.parse_start("2024-08-01T00:00-05:00")

    def test_a_local_looking_date_format_is_refused(self):
        with pytest.raises(settings.ConfigError, match="YYYY-MM-DD"):
            settings.parse_start("01/08/2024")

    def test_an_impossible_date_is_refused(self):
        with pytest.raises(settings.ConfigError):
            settings.parse_start("2024-02-30")

    def test_junk_is_refused(self):
        with pytest.raises(settings.ConfigError):
            settings.parse_start("last tuesday")

    def test_an_empty_string_is_refused(self):
        # The match matters. Without it this test passes even if the emptiness
        # check is deleted, because "" also fails to match either date pattern a
        # few lines later and raises a ConfigError anyway. Asserting only the
        # exception type would leave the guard untested and the message unhelpful.
        with pytest.raises(settings.ConfigError, match="empty"):
            settings.parse_start("")

    def test_whitespace_alone_is_refused_as_empty(self):
        with pytest.raises(settings.ConfigError, match="empty"):
            settings.parse_start("   ")

    def test_a_non_string_start_is_refused_rather_than_crashing(self):
        # JSON allows `"start": 20240801` without quotes, and that is an easy
        # thing to type. Without the type check the first thing that happens is
        # `.strip()` on an int, which raises AttributeError -- a traceback about
        # a method, not a message about a date.
        with pytest.raises(settings.ConfigError, match="must be a string"):
            settings.parse_start(20240801)

    def test_a_pre_epoch_date_is_refused(self):
        # Negative milliseconds break candle_open_time deliberately; catching it
        # here gives a message about the date rather than about arithmetic.
        with pytest.raises(settings.ConfigError, match="1970"):
            settings.parse_start("1965-01-01")

    def test_the_error_names_the_value_that_was_wrong(self):
        # An error saying only "bad date" makes you hunt for which one.
        with pytest.raises(settings.ConfigError, match="nonsense"):
            settings.parse_start("nonsense")


class TestLoadConfig:
    def test_a_valid_config_round_trips(self, tmp_path):
        path = write_config(tmp_path, valid_config())
        loaded = settings.load_config(path)
        assert loaded["exchange"] == "binance"
        assert loaded["symbols"] == ["BTC/USDT", "ETH/USDT"]

    def test_a_missing_file_says_which_path_it_looked_at(self, tmp_path):
        with pytest.raises(settings.ConfigError, match="nowhere.json"):
            settings.load_config(tmp_path / "nowhere.json")

    def test_malformed_json_is_reported_as_such(self, tmp_path):
        path = tmp_path / "symbols.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(settings.ConfigError, match="not valid JSON"):
            settings.load_config(path)

    def test_a_json_list_is_refused_because_the_file_must_be_an_object(self, tmp_path):
        path = tmp_path / "symbols.json"
        path.write_text('["BTC/USDT"]', encoding="utf-8")
        with pytest.raises(settings.ConfigError, match="object"):
            settings.load_config(path)

    def test_a_misspelled_key_is_refused_rather_than_ignored(self, tmp_path):
        # This is the test this class exists for. Writing "symbol" instead of
        # "symbols" must not leave the real key unset and the run empty; the file
        # would look correct and the tool would do nothing.
        #
        # The match is on the suggestion, not on the word "symbol". Matching
        # loosely was a mistake here for a while: deleting the unknown-key check
        # entirely still left this passing, because the missing-key check further
        # down then said "missing required key(s): symbols" -- which also
        # contains "symbol". The test looked like it proved typos are caught and
        # proved nothing of the sort.
        config = valid_config()
        config["symbol"] = config.pop("symbols")
        path = write_config(tmp_path, config)
        with pytest.raises(settings.ConfigError, match="did you mean 'symbols'"):
            settings.load_config(path)

    def test_an_extra_key_is_refused_even_when_nothing_is_missing(self, tmp_path):
        # The companion to the test above, with every required key present, so
        # only the unknown-key check can possibly produce an error. Nothing else
        # in the file has anything to complain about.
        path = write_config(tmp_path, valid_config(rate_limit=200))
        with pytest.raises(settings.ConfigError, match="unrecognised key"):
            settings.load_config(path)

    def test_an_underscore_key_is_allowed_as_a_comment(self, tmp_path):
        # JSON has no comments, and the reasoning behind the exchange choice is
        # worth keeping next to the choice.
        path = write_config(tmp_path, valid_config(_comment="why binance"))
        assert settings.load_config(path)["exchange"] == "binance"

    def test_every_required_key_is_required(self, tmp_path):
        for key in ("exchange", "timeframe", "start", "symbols"):
            config = valid_config()
            del config[key]
            path = write_config(tmp_path, config)
            with pytest.raises(settings.ConfigError, match=key):
                settings.load_config(path)

    def test_an_empty_symbol_list_is_refused(self, tmp_path):
        path = write_config(tmp_path, valid_config(symbols=[]))
        with pytest.raises(settings.ConfigError, match="at least one"):
            settings.load_config(path)

    def test_a_symbol_list_that_is_a_bare_string_is_refused(self, tmp_path):
        # "BTC/USDT" is iterable, so a lenient loader would happily treat it as
        # eight single-character symbols.
        #
        # The assertion is on the specific advice rather than just the word
        # "list", because the generic wrong-type branch below also says "list".
        # Matching loosely would let the string-specific message be deleted
        # without any test noticing, and that message is the whole point: it
        # tells you what to type instead.
        path = write_config(tmp_path, valid_config(symbols="BTC/USDT"))
        with pytest.raises(settings.ConfigError, match="not a single string"):
            settings.load_config(path)

    def test_a_symbol_list_that_is_a_number_is_refused(self, tmp_path):
        path = write_config(tmp_path, valid_config(symbols=7))
        with pytest.raises(settings.ConfigError, match="must be a list"):
            settings.load_config(path)

    def test_a_symbol_without_a_slash_is_refused(self, tmp_path):
        path = write_config(tmp_path, valid_config(symbols=["BTC"]))
        with pytest.raises(settings.ConfigError, match="BASE/QUOTE"):
            settings.load_config(path)

    def test_a_duplicated_symbol_is_refused(self, tmp_path):
        path = write_config(tmp_path, valid_config(symbols=["BTC/USDT", "BTC/USDT"]))
        with pytest.raises(settings.ConfigError, match="twice"):
            settings.load_config(path)

    def test_a_non_string_symbol_is_refused(self, tmp_path):
        path = write_config(tmp_path, valid_config(symbols=["BTC/USDT", 7]))
        with pytest.raises(settings.ConfigError):
            settings.load_config(path)


class TestResolve:
    def test_the_config_supplies_everything_when_nothing_is_overridden(self):
        result = settings.resolve(valid_config())
        assert result.exchange == "binance"
        assert result.timeframe == "1h"
        assert result.start_ms == AUG_2024
        assert result.symbols == ["BTC/USDT", "ETH/USDT"]

    def test_each_field_can_be_overridden(self):
        result = settings.resolve(
            valid_config(),
            exchange="bybit",
            timeframe="4h",
            start="2025-01-01",
            symbols=["SOL/USDT"],
        )
        assert result.exchange == "bybit"
        assert result.timeframe == "4h"
        assert result.start_ms == settings.parse_start("2025-01-01")
        assert result.symbols == ["SOL/USDT"]

    def test_an_override_of_none_leaves_the_config_value_alone(self):
        # argparse gives None for a flag nobody passed, so None must mean
        # "not specified" rather than "set it to nothing".
        result = settings.resolve(valid_config(), exchange=None, symbols=None)
        assert result.exchange == "binance"
        assert result.symbols == ["BTC/USDT", "ETH/USDT"]

    def test_overridden_symbols_are_validated_too(self):
        # A bad symbol on the command line is exactly as damaging as a bad one in
        # the file, so it cannot skip the checks by arriving a different way.
        with pytest.raises(settings.ConfigError, match="BASE/QUOTE"):
            settings.resolve(valid_config(), symbols=["BTC"])

    def test_an_overridden_start_is_validated_too(self):
        with pytest.raises(settings.ConfigError):
            settings.resolve(valid_config(), start="yesterday")

    def test_an_empty_exchange_name_is_refused(self):
        # An empty exchange would reach store.candle_path and be refused there,
        # but only after the run had already started. Worse, whitespace alone
        # gets stripped to nothing on the way into Settings, so a config with
        # "exchange": " " would produce a store path with an empty directory
        # component and no complaint from anyone.
        with pytest.raises(settings.ConfigError, match="non-empty"):
            settings.resolve(valid_config(exchange="   "))

    def test_a_non_string_exchange_is_refused(self):
        with pytest.raises(settings.ConfigError, match="non-empty"):
            settings.resolve(valid_config(exchange=3))

    def test_surrounding_whitespace_is_trimmed_from_the_exchange(self):
        # A trailing space after "binance" in the config is invisible and would
        # otherwise become part of a directory name.
        assert settings.resolve(valid_config(exchange=" binance ")).exchange == "binance"

    def test_an_unsupported_timeframe_is_refused_here_not_later(self):
        # Better to fail before any directory is created or request made.
        with pytest.raises(TimeframeError):
            settings.resolve(valid_config(), timeframe="1w")

    def test_the_timeframe_in_milliseconds_is_carried_along(self):
        # Computed once, from the validated string, so nothing downstream has to
        # re-parse it and no two callers can disagree.
        assert settings.resolve(valid_config()).timeframe_ms == HOUR_MS

    def test_directories_are_paths_not_strings(self):
        result = settings.resolve(valid_config(), data_dir="somewhere/data")
        assert result.data_dir.name == "data"

    def test_the_directory_defaults_are_relative_to_nothing(self):
        # Plain relative defaults, resolved against the working directory by the
        # OS. An absolute default would write outside the project.
        result = settings.resolve(valid_config())
        assert not result.data_dir.is_absolute()
        assert not result.log_dir.is_absolute()

    def test_max_gaps_defaults_to_the_backfill_module_value(self):
        from collector import backfill

        assert settings.resolve(valid_config()).max_gaps == backfill.DEFAULT_MAX_GAPS

    def test_max_gaps_can_be_set_to_zero_to_mean_repair_nothing(self):
        # Zero is meaningful -- bring the store up to date and leave old scars
        # alone -- so it must not be confused with "not specified".
        assert settings.resolve(valid_config(), max_gaps=0).max_gaps == 0

    def test_a_negative_max_gaps_is_refused(self):
        with pytest.raises(settings.ConfigError, match="negative"):
            settings.resolve(valid_config(), max_gaps=-1)
