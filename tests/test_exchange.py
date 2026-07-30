"""Tests for exchange access: client construction, retries, and the two guards.

No test here touches the network. conftest blocks outbound connections outright,
so if any of these ever start making real calls they fail rather than quietly
slowing down.

Retry delays are exercised with an injected sleep function, so a test covering
four backoff steps still finishes instantly instead of waiting fifteen seconds.
"""

import datetime as dt

import ccxt
import pytest

from collector import exchange as ex

HOUR = 3_600_000
DAY = 24 * HOUR


def utc(year, month, day, hour=0):
    return int(
        dt.datetime(year, month, day, hour, tzinfo=dt.timezone.utc).timestamp() * 1000
    )


class RecordingSleep:
    """Stands in for time.sleep, recording what it was asked to wait."""

    def __init__(self):
        self.delays = []

    def __call__(self, seconds):
        self.delays.append(seconds)


def no_jitter():
    """Injected in place of random.random to make backoff delays deterministic.

    Returning 1.0 selects the top of the jitter band, so an asserted delay is
    exactly base * 2 ** attempt.
    """
    return 1.0


class TestBuildExchange:
    def test_returns_a_client_with_rate_limiting_enabled(self):
        client = ex.build_exchange("kraken")
        assert client.enableRateLimit is True

    def test_never_carries_credentials(self):
        """Requirement 1: this project must never need an API key. If a key ever
        appears here, something has gone wrong that is worth failing over."""
        client = ex.build_exchange("kraken")
        assert not client.apiKey
        assert not client.secret

    def test_constructing_a_client_makes_no_network_call(self):
        """Relies on the suite-wide connection block: if build_exchange ever
        started calling load_markets, this test would raise."""
        ex.build_exchange("kraken")

    def test_unknown_exchange_id_fails_with_a_useful_message(self):
        with pytest.raises(ex.ExchangeError) as caught:
            ex.build_exchange("not_a_real_exchange")
        assert "not_a_real_exchange" in str(caught.value)

    def test_exchange_id_is_the_only_thing_that_changes_between_venues(self):
        """The seam. Swapping venue should be a parameter, not a code change."""
        assert ex.build_exchange("binance").id == "binance"
        assert ex.build_exchange("kraken").id == "kraken"


class TestFetchPage:
    def test_returns_candles_from_the_exchange(self, fake_exchange, candles):
        client = fake_exchange(candles(utc(2025, 1, 1), 10, HOUR))
        result = ex.fetch_page(client, "BTC/USD", "1h", utc(2025, 1, 1))
        assert len(result) == 10

    def test_passes_symbol_timeframe_since_and_limit_through(
        self, fake_exchange, candles
    ):
        client = fake_exchange(candles(utc(2025, 1, 1), 5, HOUR))
        ex.fetch_page(client, "BTC/USD", "1h", utc(2025, 1, 1), limit=500)

        assert client.calls == [
            {
                "symbol": "BTC/USD",
                "timeframe": "1h",
                "since": utc(2025, 1, 1),
                "limit": 500,
            }
        ]

    def test_empty_response_is_returned_as_is(self, fake_exchange):
        """An empty page is how the exchange says 'nothing further', which is a
        normal end condition rather than a failure."""
        client = fake_exchange([])
        assert ex.fetch_page(client, "BTC/USD", "1h", utc(2025, 1, 1)) == []

    def test_successful_call_does_not_sleep(self, fake_exchange, candles):
        client = fake_exchange(candles(utc(2025, 1, 1), 3, HOUR))
        sleep = RecordingSleep()
        ex.fetch_page(client, "BTC/USD", "1h", utc(2025, 1, 1), sleep=sleep)
        assert sleep.delays == []


class TestRetryAndBackoff:
    def test_retries_a_network_error_and_then_succeeds(self, fake_exchange, candles):
        client = fake_exchange(candles(utc(2025, 1, 1), 3, HOUR), fail_times=2)
        sleep = RecordingSleep()

        result = ex.fetch_page(
            client, "BTC/USD", "1h", utc(2025, 1, 1), sleep=sleep, rng=no_jitter
        )

        assert len(result) == 3
        assert len(client.calls) == 3

    def test_delays_grow_exponentially(self, fake_exchange, candles):
        """Backing off geometrically gives a struggling exchange room to recover
        instead of hammering it at a fixed interval."""
        client = fake_exchange(candles(utc(2025, 1, 1), 1, HOUR), fail_times=3)
        sleep = RecordingSleep()

        ex.fetch_page(
            client,
            "BTC/USD",
            "1h",
            utc(2025, 1, 1),
            sleep=sleep,
            rng=no_jitter,
            base_delay=1.0,
        )

        assert sleep.delays == [1.0, 2.0, 4.0]

    def test_delays_are_capped(self, fake_exchange, candles):
        """Without a ceiling, a long outage would eventually sleep for hours."""
        client = fake_exchange(candles(utc(2025, 1, 1), 1, HOUR), fail_times=4)
        sleep = RecordingSleep()

        ex.fetch_page(
            client,
            "BTC/USD",
            "1h",
            utc(2025, 1, 1),
            sleep=sleep,
            rng=no_jitter,
            base_delay=1.0,
            max_delay=3.0,
            max_attempts=6,
        )

        assert sleep.delays == [1.0, 2.0, 3.0, 3.0]

    def test_jitter_keeps_delays_inside_the_expected_band(
        self, fake_exchange, candles
    ):
        """Real jitter stops several symbols retrying in lockstep after a shared
        outage. The delay should be somewhere in the upper half of the interval.
        """
        client = fake_exchange(candles(utc(2025, 1, 1), 1, HOUR), fail_times=2)
        sleep = RecordingSleep()

        ex.fetch_page(
            client, "BTC/USD", "1h", utc(2025, 1, 1), sleep=sleep, base_delay=1.0
        )

        assert 0.5 <= sleep.delays[0] <= 1.0
        assert 1.0 <= sleep.delays[1] <= 2.0

    def test_jitter_never_uses_the_whole_interval(self, fake_exchange, candles):
        """Pins the lower edge of the jitter band deterministically.

        The test above draws on real randomness, which means it cannot tell
        jitter from no jitter at all: an unjittered delay sits exactly on its
        upper bound and satisfies the assertion. Feeding rng the smallest value
        it can produce removes the ambiguity -- the delay must land on half the
        interval, so removing the jitter arithmetic now fails here.
        """
        client = fake_exchange(candles(utc(2025, 1, 1), 1, HOUR), fail_times=2)
        sleep = RecordingSleep()

        ex.fetch_page(
            client,
            "BTC/USD",
            "1h",
            utc(2025, 1, 1),
            sleep=sleep,
            rng=lambda: 0.0,
            base_delay=1.0,
        )

        assert sleep.delays == [0.5, 1.0]

    def test_gives_up_after_max_attempts(self, fake_exchange, candles):
        client = fake_exchange(candles(utc(2025, 1, 1), 1, HOUR), fail_times=99)
        sleep = RecordingSleep()

        with pytest.raises(ex.ExchangeError) as caught:
            ex.fetch_page(
                client,
                "BTC/USD",
                "1h",
                utc(2025, 1, 1),
                sleep=sleep,
                rng=no_jitter,
                max_attempts=3,
            )

        assert len(client.calls) == 3
        assert len(sleep.delays) == 2  # one fewer sleep than attempts
        assert isinstance(caught.value.__cause__, ccxt.NetworkError)

    @pytest.mark.parametrize(
        "error",
        [
            ccxt.RequestTimeout("timed out"),
            ccxt.ExchangeNotAvailable("maintenance"),
            ccxt.DDoSProtection("slow down"),
            ccxt.RateLimitExceeded("too many requests"),
        ],
    )
    def test_all_network_family_errors_are_retried(
        self, fake_exchange, candles, error
    ):
        """These all descend from ccxt.NetworkError, which is the whole rule --
        no hand-maintained list of retryable error types to fall out of date.
        """
        client = fake_exchange(
            candles(utc(2025, 1, 1), 1, HOUR), fail_times=1, error=error
        )
        result = ex.fetch_page(
            client, "BTC/USD", "1h", utc(2025, 1, 1), sleep=RecordingSleep()
        )
        assert len(result) == 1

    @pytest.mark.parametrize(
        "error",
        [
            ccxt.BadSymbol("no such market"),
            ccxt.AuthenticationError("bad key"),
            ccxt.NotSupported("no ohlcv here"),
        ],
    )
    def test_permanent_errors_are_not_retried(self, fake_exchange, candles, error):
        """Retrying a misspelled symbol five times with backoff wastes fifteen
        seconds and still fails. These descend from ccxt.ExchangeError and mean
        the request itself is wrong, so they fail immediately.
        """
        client = fake_exchange(
            candles(utc(2025, 1, 1), 1, HOUR), fail_times=99, error=error
        )
        sleep = RecordingSleep()

        with pytest.raises(ex.ExchangeError):
            ex.fetch_page(
                client, "BTC/USD", "1h", utc(2025, 1, 1), sleep=sleep
            )

        assert len(client.calls) == 1
        assert sleep.delays == []

    def test_failure_message_names_the_symbol(self, fake_exchange, candles):
        """A run covering five symbols should not leave you guessing which one
        failed."""
        client = fake_exchange(
            candles(utc(2025, 1, 1), 1, HOUR),
            fail_times=99,
            error=ccxt.BadSymbol("nope"),
        )
        with pytest.raises(ex.ExchangeError) as caught:
            ex.fetch_page(client, "DOGE/USD", "1h", utc(2025, 1, 1))
        assert "DOGE/USD" in str(caught.value)


class TestVerifyHistoryReachable:
    """Guard one: did the exchange actually honour `since`?

    Some venues cap their candle endpoint to a recent window and ignore an older
    `since` entirely. A backfill against such an endpoint appears to succeed
    while storing only the last few weeks -- the worst kind of failure, because
    nothing looks wrong.
    """

    def test_accepts_a_page_that_starts_where_asked(self, candles):
        since = utc(2024, 8, 1)
        page = candles(since, 720, HOUR)
        ex.verify_history_reachable(page, since, HOUR, now_ms=utc(2026, 7, 30))

    def test_rejects_a_recent_window_when_old_history_was_requested(self, candles):
        """The capped-window signature: nothing near the requested start, and the
        page runs right up to the present."""
        now = utc(2026, 7, 30)
        page = candles(now - 720 * HOUR, 720, HOUR)

        with pytest.raises(ex.HistoryNotAvailable):
            ex.verify_history_reachable(page, utc(2024, 8, 1), HOUR, now_ms=now)

    def test_error_names_both_the_requested_and_returned_dates(self, candles):
        now = utc(2026, 7, 30)
        page = candles(now - 720 * HOUR, 720, HOUR)

        with pytest.raises(ex.HistoryNotAvailable) as caught:
            ex.verify_history_reachable(page, utc(2024, 8, 1), HOUR, now_ms=now)

        message = str(caught.value)
        assert "2024-08-01" in message
        assert "2026-06-30" in message

    def test_accepts_a_late_listed_market(self, candles):
        """A market that simply did not exist at the requested start returns its
        own first candle, which is legitimate. The distinguishing signal is that
        such a page ends long before now -- it is page one of many, not a window
        pinned to the live edge.
        """
        now = utc(2026, 7, 30)
        page = candles(utc(2025, 1, 1), 720, HOUR)  # ends Jan 2025, far from now
        ex.verify_history_reachable(page, utc(2024, 8, 1), HOUR, now_ms=now)

    def test_empty_page_is_not_treated_as_a_history_problem(self):
        """Nothing came back, so there is nothing to conclude about `since`."""
        ex.verify_history_reachable([], utc(2024, 8, 1), HOUR, now_ms=utc(2026, 7, 30))

    def test_small_drift_within_tolerance_is_accepted(self, candles):
        """Exchanges commonly start a market a few candles after any given
        instant. That is not a capped window."""
        since = utc(2024, 8, 1)
        page = candles(since + 3 * HOUR, 720, HOUR)
        ex.verify_history_reachable(page, since, HOUR, now_ms=utc(2026, 7, 30))

    def test_accepts_a_current_page_that_also_reaches_the_present(self, candles):
        """The healthy steady state, and the case every other accepting test
        above happens to miss.

        Once a store is up to date, a fetch returns a page that both starts
        where it was asked to and runs right up to now. A guard that keyed only
        on "runs up to now" would reject that -- which is to say it would reject
        every normal fetch of a current store. Both conditions are load-bearing,
        and until this test existed only one of them was.
        """
        now = utc(2026, 7, 30)
        since = now - 720 * HOUR
        page = candles(since, 721, HOUR)  # starts exactly where asked, ends at now
        ex.verify_history_reachable(page, since, HOUR, now_ms=now)

    def test_small_drift_is_still_accepted_on_a_page_reaching_the_present(
        self, candles
    ):
        """Pins the tolerance itself.

        The drift test above passes for the wrong reason: its page ends in 2024,
        so `reaches_present` is false and the tolerance is never consulted. Here
        the page does reach the present, so a few hours of drift is the only
        thing standing between acceptance and a false alarm.
        """
        now = utc(2026, 7, 30)
        since = now - 720 * HOUR
        page = candles(since + 3 * HOUR, 718, HOUR)  # ends at now
        ex.verify_history_reachable(page, since, HOUR, now_ms=now)


class TestVerifyPaginationAdvanced:
    """Guard two: is the loop making progress?

    An exchange that returns the same page for every `since` would make a
    pagination loop run forever, fetching the same candles and storing nothing
    new. Cheap to detect, expensive to discover by watching a log.
    """

    def test_accepts_a_page_beyond_the_previous_one(self, candles):
        previous_last = utc(2025, 1, 1, 5)
        page = candles(utc(2025, 1, 1, 6), 10, HOUR)
        ex.verify_pagination_advanced(previous_last, page)

    def test_rejects_a_repeat_of_the_same_page(self, candles):
        page = candles(utc(2025, 1, 1), 10, HOUR)
        previous_last = page[-1][0]

        with pytest.raises(ex.PaginationStalled):
            ex.verify_pagination_advanced(previous_last, page)

    def test_rejects_a_page_entirely_behind_the_previous_one(self, candles):
        page = candles(utc(2025, 1, 1), 10, HOUR)
        with pytest.raises(ex.PaginationStalled):
            ex.verify_pagination_advanced(utc(2025, 6, 1), page)

    def test_accepts_an_overlapping_page_that_still_progresses(self, candles):
        """Some exchanges repeat the boundary candle. Overlap is fine as long as
        the page reaches further than the last one did."""
        previous_last = utc(2025, 1, 1, 9)
        page = candles(utc(2025, 1, 1, 9), 10, HOUR)
        ex.verify_pagination_advanced(previous_last, page)

    def test_empty_page_is_not_a_stall(self, candles):
        """An empty page means the range is exhausted, which is how a backfill
        finishes normally."""
        ex.verify_pagination_advanced(utc(2025, 1, 1), [])

    def test_no_previous_page_is_always_acceptable(self, candles):
        page = candles(utc(2025, 1, 1), 10, HOUR)
        ex.verify_pagination_advanced(None, page)
