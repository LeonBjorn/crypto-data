"""The live runner, which is mostly a set of refusals.

Every test here is about something the runner must *not* do. The trade logic is
the paper engine's and is already proven elsewhere; what is new is that a
mistake costs money, so what is tested is the braking.

The first class is the one that matters. A live runner handed a fresh state and
a full store would, without it, replay two years of candles into a real venue --
roughly two thousand market orders at today's prices for signals that fired
months ago.
"""

import numpy as np
import pandas as pd
import pytest

from paper.account import Account
from paper.broker import BUY, PaperBroker
from paper.live import LiveError, LiveRunner
from signals import rules, trades

HOUR = 3_600_000
T0 = 1_722_470_400_000


def frame(count, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + rng.standard_normal(count).cumsum()
    spread = abs(rng.standard_normal(count)) + 0.1
    opens = np.concatenate([[100.0], close[:-1]])
    return pd.DataFrame({
        "timestamp": [T0 + i * HOUR for i in range(count)],
        "open": opens,
        "high": np.maximum(opens, close) + spread,
        "low": np.minimum(opens, close) - spread,
        "close": close,
        "volume": 10.0 + 300.0 * (rng.random(count) < 0.5),
    })


class CountingBroker(PaperBroker):
    """A paper broker that also remembers how many orders it was asked for."""

    def __init__(self, *args, positions=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.order_count = 0
        self._forced = positions

    def market(self, symbol, side, qty, reference_price, timestamp, **kwargs):
        self.order_count += 1
        return super().market(symbol, side, qty, reference_price, timestamp)

    def positions(self):
        return dict(self._forced) if self._forced is not None else super().positions()

    def describe(self):
        return "counting broker"


def runner(symbols=("BTC/USDT",), *, broker=None, hold=6, **kwargs):
    broker = broker if broker is not None else CountingBroker(trades.FREE)
    return LiveRunner(list(symbols), hold=hold, broker=broker,
                      account=Account.unlimited(), costs=trades.FREE, **kwargs)


def signals_for(frames, window=5):
    return {s: rules.apply("breakout", f, window=window) for s, f in frames.items()}


class TestItNeverReplaysHistory:
    """The property that separates a rehearsal from a disaster."""

    def test_an_unarmed_runner_refuses_to_advance(self):
        frames = {"BTC/USDT": frame(400)}
        live = runner()
        with pytest.raises(LiveError, match="never been armed"):
            live.advance(frames, signals_for(frames))

    def test_and_places_no_orders_while_refusing(self):
        frames = {"BTC/USDT": frame(400)}
        broker = CountingBroker(trades.FREE)
        live = runner(broker=broker)
        with pytest.raises(LiveError):
            live.advance(frames, signals_for(frames))
        assert broker.order_count == 0

    def test_arming_trades_nothing_at_all(self):
        """Two years of stored candles, every one of them already history."""
        frames = {"BTC/USDT": frame(400)}
        broker = CountingBroker(trades.FREE)
        live = runner(broker=broker)
        live.arm(frames)
        assert broker.order_count == 0
        assert live.armed

    def test_arming_puts_the_cursor_at_the_present(self):
        frames = {"BTC/USDT": frame(400)}
        live = runner()
        live.arm(frames)
        assert live.cursor == int(frames["BTC/USDT"]["timestamp"].iloc[-1])

    def test_an_armed_runner_ignores_everything_before_it_was_armed(self):
        frames = {"BTC/USDT": frame(400)}
        broker = CountingBroker(trades.FREE)
        live = runner(broker=broker)
        live.arm(frames)
        assert live.advance(frames, signals_for(frames)) == 0
        assert broker.order_count == 0


class TestCatchingUpIsNotTrading:
    def test_a_handful_of_new_bars_is_acted_on(self):
        full = frame(400)
        early = {"BTC/USDT": full.iloc[:-2].copy()}
        live = runner()
        live.arm(early)
        assert live.advance({"BTC/USDT": full}, signals_for({"BTC/USDT": full})) == 2

    def test_too_many_missed_bars_is_refused(self):
        """Their prices are gone. Filling them now fires a queue of stale
        signals into a market that has already moved.
        """
        full = frame(400)
        early = {"BTC/USDT": full.iloc[:-40].copy()}
        live = runner(max_catchup=3)
        live.arm(early)
        with pytest.raises(LiveError, match="bars have passed"):
            live.advance({"BTC/USDT": full}, signals_for({"BTC/USDT": full}))

    def test_nothing_is_ordered_while_refusing_to_catch_up(self):
        full = frame(400)
        broker = CountingBroker(trades.FREE)
        live = runner(broker=broker, max_catchup=3)
        live.arm({"BTC/USDT": full.iloc[:-40].copy()})
        with pytest.raises(LiveError):
            live.advance({"BTC/USDT": full}, signals_for({"BTC/USDT": full}))
        assert broker.order_count == 0


class TestTheVenueIsTheAuthority:
    def test_agreement_when_both_are_flat(self):
        live = runner()
        assert live.reconcile() == {}
        assert live.require_agreement()

    def test_a_position_the_venue_has_and_this_process_does_not(self):
        """The dangerous direction: trading as though flat while holding."""
        broker = CountingBroker(trades.FREE, positions={"BTC/USDT": 0.5})
        live = runner(broker=broker)
        assert "BTC/USDT" in live.reconcile()

    def test_a_disagreement_stops_the_run(self):
        frames = {"BTC/USDT": frame(400)}
        broker = CountingBroker(trades.FREE, positions={"BTC/USDT": 0.5})
        live = runner(broker=broker)
        live.arm(frames)
        with pytest.raises(LiveError, match="disagree"):
            live.advance(frames, signals_for(frames))

    def test_the_refusal_names_both_numbers(self):
        broker = CountingBroker(trades.FREE, positions={"BTC/USDT": 0.5})
        live = runner(broker=broker)
        with pytest.raises(LiveError) as caught:
            live.require_agreement()
        assert "0.5" in str(caught.value)
        assert "why they diverged" in str(caught.value)

    def test_it_does_not_quietly_adopt_the_venues_view(self):
        """Self-healing would paper over the interesting question. The answers
        include a rejected order, an unnoticed partial fill, and a liquidation.
        """
        broker = CountingBroker(trades.FREE, positions={"BTC/USDT": 0.5})
        live = runner(broker=broker)
        with pytest.raises(LiveError):
            live.require_agreement()
        assert live.books["BTC/USDT"].positions == []

    def test_lot_size_rounding_is_not_a_disagreement(self):
        broker = CountingBroker(trades.FREE, positions={"BTC/USDT": 1e-13})
        assert runner(broker=broker).reconcile() == {}


class TestItKeepsItsOwnBooks:
    def test_the_state_is_separate_from_the_paper_account(self):
        """The forward record is the only out-of-sample evidence this project
        has. Rehearsal fills do not belong anywhere near it.
        """
        live = runner()
        live.arm({"BTC/USDT": frame(50)})
        state = live.to_state()
        assert set(state) == {"armed_at", "cursor", "positions", "ledger", "venue"}
        assert state["ledger"] == []

    def test_it_says_whether_it_is_armed(self):
        live = runner()
        assert "NOT ARMED" in live.describe()
        live.arm({"BTC/USDT": frame(50)})
        assert "armed" in live.describe()

    def test_it_needs_at_least_one_symbol(self):
        with pytest.raises(LiveError):
            LiveRunner([], hold=6, broker=PaperBroker())
