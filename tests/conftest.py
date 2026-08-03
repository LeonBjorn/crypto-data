"""Shared test fixtures, plus a hard guarantee that the suite never uses the network.

"Zero network calls in the suite" is one of the project's acceptance criteria.
Rather than trusting that every future test remembers to mock the exchange, the
fixture below makes outbound connections impossible. A test that forgets its
mock fails with an obvious message instead of quietly making a real request and
passing slowly -- or worse, passing at home and failing on a train.
"""

import socket
import time

import pytest


class NetworkAccessAttempted(RuntimeError):
    """Raised when test code tries to reach the network."""


class SleepAttempted(RuntimeError):
    """Raised when test code tries to actually wait."""


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Block outbound connections for the duration of every test.

    autouse means this applies without any test opting in. monkeypatch undoes it
    afterwards, so nothing leaks into other processes or tooling.

    Note carefully what is patched and what is not. An earlier version of this
    replaced socket.socket itself, and that broke unrelated libraries: the
    stdlib `ssl` module declares `class SSLSocket(socket)`, so once socket.socket
    was a plain function, merely importing ssl raised

        TypeError: function() argument 'code' must be code, not str

    pyarrow imports ssl lazily, which made the Parquet tests fail for a reason
    that had nothing whatsoever to do with Parquet -- an hour of debugging in the
    wrong file, if you are unlucky.

    So the socket *class* is left alone and only the methods that actually reach
    out are blocked. Creating a socket is harmless; connecting is not.
    getaddrinfo is included so a DNS lookup also fails immediately rather than
    hanging until it times out.

    LOOPBACK IS ALLOWED, DELIBERATELY
    ---------------------------------
    The one exception is 127.0.0.1 and ::1. The property being defended here is
    that no test can reach an *exchange* -- that is what makes the suite pass on
    a train and fail loudly when a mock is forgotten. A connection to this
    machine cannot reach an exchange by definition, and the paper dashboard is a
    local HTTP server whose whole job is to be connected to; testing it against a
    hand-rolled fake request object would be testing a different thing from the
    one that ships.

    So loopback is let through and everything else still fails. The guarantee
    that matters is unchanged, and the one that was never the point is not
    pretended to.
    """
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create_connection = socket.create_connection
    real_getaddrinfo = socket.getaddrinfo

    LOOPBACK = {"127.0.0.1", "::1", "localhost", ""}

    def _is_local(address):
        if isinstance(address, tuple) and address:
            return str(address[0]) in LOOPBACK
        return False

    def refuse():
        raise NetworkAccessAttempted(
            "This test tried to open a network connection. Exchange access must "
            "be mocked -- see the FakeExchange fixture. If you genuinely need a "
            "live call, put it in a separate script, not in the test suite. "
            "(Connections to 127.0.0.1 are allowed, for the local dashboard.)"
        )

    def guarded_connect(self, address, *args, **kwargs):
        if not _is_local(address):
            refuse()
        return real_connect(self, address, *args, **kwargs)

    def guarded_connect_ex(self, address, *args, **kwargs):
        if not _is_local(address):
            refuse()
        return real_connect_ex(self, address, *args, **kwargs)

    def guarded_create_connection(address, *args, **kwargs):
        if not _is_local(address):
            refuse()
        return real_create_connection(address, *args, **kwargs)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if str(host) not in LOOPBACK:
            refuse()
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch):
    """Make a real wait fail the test rather than slow the suite down.

    The same reasoning as the network guard, applied to the clock. A test that
    forgets to inject `sleep` does not fail -- it passes, having quietly waited
    on production backoff. That happened once here: one test spent 9.2 seconds
    inside 1+2+4+8s of retry delay and took the whole suite from 1.4s to 12.6s.
    Nothing was wrong with the code, and nothing said anything was wrong; it was
    only noticed by comparing timings by hand.

    A suite that takes ten seconds gets run less often than one that takes one,
    and a suite that is run less often is worth less. So the wait is made loud.

    This only works because fetch_page resolves `sleep` from the time module when
    called rather than capturing it in its signature -- see the note there.
    """

    def blocked(seconds):
        raise SleepAttempted(
            f"This test tried to sleep for {seconds}s. Unit tests must not wait "
            "on real backoff -- pass a no-op sleep, e.g. "
            "fetch_options={'sleep': lambda _: None}, or the RecordingSleep "
            "helper if you want to assert on the delays."
        )

    monkeypatch.setattr(time, "sleep", blocked)


def make_candles(first_ts, count, timeframe_ms, close=100.0):
    """Build a run of consecutive OHLCV rows in ccxt's shape."""
    return [
        [first_ts + index * timeframe_ms, 100.0, 110.0, 90.0, close, 5.0]
        for index in range(count)
    ]


class FakeExchange:
    """A stand-in for a ccxt exchange, serving candles from memory.

    Only fetch_ohlcv is implemented, because that is the entire surface this
    project uses -- no credentials, no order methods, no market loading.

    The keyword arguments each reproduce a specific real-world misbehaviour, so
    that the guards protecting against them can be proven rather than assumed:

    ignore_since
        Returns its most recent page regardless of what `since` was asked for.
        This is the behaviour that would silently limit a two-year backfill to
        the last few weeks.
    stall
        Always returns the same first page, so a naive pagination loop would
        request forever without progressing.
    fail_times
        Raises `error` for the first N calls, then behaves normally. Used to
        exercise retry and backoff without waiting for real delays.
    fail_after
        Succeeds for the first N calls and fails permanently from then on. This
        is the mid-run death a long backfill has to survive: several pages land
        successfully and then the connection goes for good. Distinct from
        fail_times, which fails first and recovers.
    holes
        Timestamps to withhold from the universe, simulating candles the exchange
        genuinely does not have. Asking for them politely returns nothing, which
        is different from an error and has to be reported differently.

    Every call is recorded in `self.calls`, so tests can assert on what was
    requested rather than only on what came back.
    """

    def __init__(
        self,
        candles=None,
        *,
        page_size=720,
        ignore_since=False,
        stall=False,
        fail_times=0,
        fail_after=None,
        error=None,
        holes=(),
        exchange_id="fake",
    ):
        self.id = exchange_id
        withheld = set(holes)
        self.universe = [row for row in (candles or []) if row[0] not in withheld]
        self.page_size = page_size
        self.ignore_since = ignore_since
        self.stall = stall
        self.fail_times = fail_times
        self.fail_after = fail_after
        self.error = error
        self.calls = []

    def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None):
        self.calls.append(
            {"symbol": symbol, "timeframe": timeframe, "since": since, "limit": limit}
        )

        if self.fail_times > 0:
            self.fail_times -= 1
            import ccxt

            raise self.error or ccxt.NetworkError("simulated connection reset")

        if self.fail_after is not None and len(self.calls) > self.fail_after:
            import ccxt

            raise self.error or ccxt.ExchangeNotAvailable("simulated mid-run outage")

        if not self.universe:
            return []

        size = limit or self.page_size

        if self.ignore_since:
            selected = self.universe[-size:]
        elif self.stall:
            selected = self.universe[:size]
        else:
            available = [row for row in self.universe if since is None or row[0] >= since]
            selected = available[:size]

        # Copy each row so a test mutating a result cannot corrupt the universe.
        return [list(row) for row in selected]


@pytest.fixture
def fake_exchange():
    """The FakeExchange class itself, used as a factory in tests.

    Returned rather than instantiated because each test wants different
    behaviour: `fake_exchange(candles, stall=True)` and so on.
    """
    return FakeExchange


@pytest.fixture
def candles():
    """The make_candles helper, for building a run of consecutive candles."""
    return make_candles
