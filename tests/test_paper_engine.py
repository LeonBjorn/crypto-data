"""The online engine, and the one property everything else rests on.

Most of this file is a single idea checked from several angles: fed the same
candles and the same signals, the engine and `signals.trades.round_trips` must
produce the same trades. One of them sees the whole file at once and the other
sees one bar at a time, and if those two ever disagree then a paper run cannot
be compared to a backtest -- which would leave the paper numbers meaning nothing
at all, since a backtest is the only thing there is to compare them to.

So the invariant tests come first and are deliberately unglamorous: build a
frame, run both, assert the ledgers match column for column. They are run across
several rules, several holds and both exit styles, because the interesting
failures are all off-by-one and an off-by-one hides easily in one configuration.

The rest of the file is the bar-order arithmetic on frames small enough to check
by hand, in the style `test_trades.py` established -- and then the cases the
backtest has no opinion about, because they are about a wallet rather than a
market.
"""

import numpy as np
import pandas as pd
import pytest

from paper.account import Account
from paper.broker import PaperBroker
from paper.engine import Bar, PaperBook
from signals import rules, trades

FIRST_MS = 1_700_000_000_000
HOUR_MS = 3_600_000

# The columns both sides must agree on. `symbol`, `qty` and the cash figures are
# the engine's own and have no counterpart in a backtest that never held money.
SHARED = [
    "entry_bar",
    "entry_time",
    "entry_price",
    "exit_bar",
    "exit_time",
    "exit_price",
    "exit_reason",
    "bars_held",
    "gross_return",
    "net_return",
]


def frame(rows):
    """Candles from a list of (open, high, low, close) tuples."""
    rows = list(rows)
    return pd.DataFrame(
        {
            "timestamp": [FIRST_MS + i * HOUR_MS for i in range(len(rows))],
            "open": [float(row[0]) for row in rows],
            "high": [float(row[1]) for row in rows],
            "low": [float(row[2]) for row in rows],
            "close": [float(row[3]) for row in rows],
            "volume": [1.0] * len(rows),
        }
    )


def flat(price, count):
    return [(price, price, price, price)] * count


def signal_at(candles, *bars):
    fired = np.zeros(len(candles), dtype=bool)
    for bar in bars:
        fired[bar] = True
    return pd.Series(fired, index=candles.index)


def walk(count, seed=0, start=100.0):
    """A random walk with sane OHLC and volume that varies with the up-moves."""
    rng = np.random.default_rng(seed)
    close = start + rng.standard_normal(count).cumsum()
    spread = abs(rng.standard_normal(count)) + 0.1
    opens = np.concatenate([[start], close[:-1]])
    return pd.DataFrame(
        {
            "timestamp": [FIRST_MS + i * HOUR_MS for i in range(count)],
            "open": opens,
            "high": np.maximum(opens, close) + spread,
            "low": np.minimum(opens, close) - spread,
            "close": close,
            "volume": 10.0 + 300.0 * (rng.random(count) < 0.5),
        }
    )


def book(symbol="BTC/USDT", **kwargs):
    """A book with an account that refuses nothing, so only fills are tested."""
    kwargs.setdefault("costs", trades.FREE)
    return PaperBook(symbol, account=Account.unlimited(), **kwargs)


def one_trade(candles, signals, **kwargs):
    ledger = book(**kwargs).run(candles, signals)
    assert len(ledger) == 1, f"expected exactly one trade, got {len(ledger)}"
    return ledger.iloc[0]


RISER = frame([(50, 50, 50, 50), (100, 100, 100, 100), (110, 110, 110, 110)] + flat(110, 4))


class TestItAgreesWithTheBacktest:
    """The invariant. Everything else in this package depends on it holding."""

    def compare(self, candles, signals, **kwargs):
        """Run both engines over the same inputs and return their two ledgers.

        Only trades that could have *completed* are compared. `round_trips`
        drops a signal whose holding period runs off the end of the file,
        because it never became a round trip; the engine, which does not know
        the file has an end, opens it and either closes it early or is still
        holding it. Comparing the completable ones is comparing like with like,
        and the count of the others is checked separately below.

        Both sides are priced with the same Costs -- the real ones by default,
        not FREE. Fees are applied at a different point on each side (the
        backtest adjusts the entry and exit prices, the engine goes through a
        broker's effective price), so charging them is what proves the two
        arrive at the same number rather than merely agreeing about zero.
        """
        kwargs.setdefault("costs", trades.DEFAULT_COSTS)
        expected = trades.round_trips(candles, signals, **kwargs).trades
        got = PaperBook("BTC/USDT", account=Account.unlimited(), **kwargs).run(candles, signals)
        completable = got[got["entry_bar"] + kwargs["hold"] < len(candles)]

        # Both sorted by entry, because the two orders are legitimately
        # different and neither is wrong. The backtest walks the signals, so its
        # rows come out in the order the trades were *entered*; the engine
        # appends a row when a trade *finishes*, which is the only order a live
        # ledger can have -- and with overlapping trades a later entry can close
        # first. The trades are the same; only the sequence differs.
        by_entry = lambda table: table.sort_values("entry_bar").reset_index(drop=True)
        return by_entry(expected), by_entry(completable)

    @pytest.mark.parametrize("hold", [1, 3, 6, 24])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_a_plain_time_exit_matches(self, hold, seed):
        candles = walk(300, seed=seed)
        signals = rules.apply("breakout", candles, window=5)
        expected, got = self.compare(candles, signals, hold=hold)
        assert len(expected) == len(got)
        pd.testing.assert_frame_equal(expected[SHARED], got[SHARED], check_dtype=False)

    @pytest.mark.parametrize("stop,target", [(0.02, None), (None, 0.03), (0.02, 0.03)])
    def test_stops_and_targets_match(self, stop, target):
        candles = walk(300, seed=3)
        signals = rules.apply("breakout", candles, window=5)
        expected, got = self.compare(candles, signals, hold=12, stop=stop, target=target)
        assert len(expected) == len(got)
        pd.testing.assert_frame_equal(expected[SHARED], got[SHARED], check_dtype=False)

    @pytest.mark.parametrize("trail", [0.01, 0.03, 0.08])
    def test_the_trailing_stop_matches(self, trail):
        """The exit with memory, and so the one most likely to drift between a
        batch computation and an incremental one.
        """
        candles = walk(300, seed=4)
        signals = rules.apply("breakout", candles, window=5)
        expected, got = self.compare(candles, signals, hold=24, trail=trail)
        assert len(expected) == len(got)
        pd.testing.assert_frame_equal(expected[SHARED], got[SHARED], check_dtype=False)

    def test_every_exit_style_at_once_matches(self):
        candles = walk(400, seed=5)
        signals = rules.apply("breakout-volume", candles, window=5)
        expected, got = self.compare(
            candles, signals, hold=24, stop=0.05, target=0.05, trail=0.02
        )
        assert len(expected) == len(got)
        pd.testing.assert_frame_equal(expected[SHARED], got[SHARED], check_dtype=False)

    @pytest.mark.parametrize("name", sorted(rules.RULES))
    def test_it_matches_for_every_registered_rule(self, name):
        """A rule added later cannot quietly break the correspondence."""
        candles = walk(400, seed=6)
        signals = rules.apply(name, candles)
        if not signals.any():
            pytest.skip(f"{name} did not fire on this fixture")
        expected, got = self.compare(candles, signals, hold=12)
        pd.testing.assert_frame_equal(expected[SHARED], got[SHARED], check_dtype=False)

    def test_it_also_matches_with_costs_switched_off(self):
        """The other end of the pricing. Everywhere else charges the real fee;
        this proves the agreement is not an artefact of one particular one.
        """
        candles = walk(300, seed=7)
        signals = rules.apply("breakout", candles, window=5)
        expected, got = self.compare(candles, signals, hold=12, costs=trades.FREE)
        assert len(expected) == len(got)
        assert (got["net_return"] == got["gross_return"]).all()
        pd.testing.assert_frame_equal(expected[SHARED], got[SHARED], check_dtype=False)

    def test_what_the_backtest_dropped_is_what_the_engine_is_still_holding(self):
        """The other half of the correspondence. A signal too late to finish is
        a dropped trade on one side and an open position on the other, and the
        two counts have to add up or something has gone missing in between.
        """
        candles = walk(200, seed=8)
        signals = rules.apply("breakout", candles, window=5)
        expected = trades.round_trips(candles, signals, hold=24)

        engine = book(hold=24)
        engine.run(candles, signals)
        late = [row for row in engine.closed if row["entry_bar"] + 24 >= len(candles)]

        assert len(engine.positions) + len(late) == expected.dropped


class TestTheOrderInsideOneBar:
    """The four steps of `advance`, on frames checkable by hand."""

    def test_a_signal_enters_at_the_next_bars_open(self):
        row = one_trade(RISER, signal_at(RISER, 0), hold=1)
        assert row.entry_bar == 1
        assert row.entry_price == pytest.approx(100.0)

    def test_and_not_at_the_close_that_produced_it(self):
        """Bar 0 closes at 50. Entering there would pay the price the rule was
        computed from, which had already gone by the time it could be computed.
        """
        row = one_trade(RISER, signal_at(RISER, 0), hold=1)
        assert row.entry_price != pytest.approx(50.0)

    def test_the_time_exit_is_an_open_too(self):
        row = one_trade(RISER, signal_at(RISER, 0), hold=1)
        assert row.exit_bar == 2
        assert row.exit_price == pytest.approx(110.0)
        assert row.gross_return == pytest.approx(0.10)

    def test_the_bar_a_position_is_sold_on_cannot_stop_it(self):
        """Sold at bar 4's open; what bar 4 does after that is not its business."""
        candles = frame([(50, 50, 50, 50)] + flat(100, 3) + [(100, 100, 90, 100)] + flat(100, 5))
        row = one_trade(candles, signal_at(candles, 0), hold=3, stop=0.02)
        assert row.exit_reason == "hold"
        assert row.exit_price == pytest.approx(100.0)

    def test_the_bar_a_position_is_opened_on_can_stop_it(self):
        """You have held it since that bar's open, so its low is yours."""
        candles = frame([(50, 50, 50, 50), (100, 100, 96, 100)] + flat(100, 5))
        row = one_trade(candles, signal_at(candles, 0), hold=4, stop=0.02)
        assert row.exit_reason == "stop"
        assert row.exit_bar == 1
        assert row.bars_held == 0

    def test_the_trail_starts_at_the_entry_not_at_the_entry_bars_high(self):
        """Bar 1 opens at 100 with a high of 110 and a low of 98. A 2% trail
        must be measured from the entry of 100 -- level 98, so this bar takes it
        -- and not from the same bar's high of 110, which would put the level at
        107.8 and exit at a price the trade never had a chance to reach.
        """
        candles = frame([(50, 50, 50, 50), (100, 110, 98, 104)] + flat(104, 5))
        row = one_trade(candles, signal_at(candles, 0), hold=4, trail=0.02)
        assert row.exit_reason == "trail"
        assert row.exit_bar == 1
        assert row.exit_price == pytest.approx(98.0)

    def test_a_signal_on_the_final_bar_never_fills(self):
        """There is no next bar for it to open on, so it stays armed forever."""
        engine = book(hold=1)
        engine.run(RISER, signal_at(RISER, len(RISER) - 1))
        assert len(engine.positions) == 0
        assert len(engine.closed) == 0

    def test_advancing_bar_by_bar_equals_advancing_all_at_once(self):
        """`run` is a loop over `advance` and must have no other effect, or a
        catch-up run would differ from an hourly one.
        """
        candles = walk(120, seed=9)
        signals = rules.apply("breakout", candles, window=5)

        whole = book(hold=6).run(candles, signals)

        piecemeal = book(hold=6)
        for index in range(len(candles)):
            row = candles.iloc[index]
            piecemeal.advance(
                Bar(index, int(row.timestamp), row.open, row.high, row.low, row.close),
                signal=bool(signals.iloc[index]),
            )
        pd.testing.assert_frame_equal(whole, piecemeal.ledger())


class TestTheWalletChangesTheAnswer:
    """What a finite account does that a backtest never had to think about."""

    def test_a_full_account_turns_a_signal_into_a_rejection(self):
        candles = frame([(50, 50, 50, 50)] + flat(100, 20))
        account = Account(1000, size_fraction=1.0, max_positions=1, one_per_symbol=False)
        engine = PaperBook("BTC/USDT", hold=10, costs=trades.FREE, account=account)
        engine.run(candles, signal_at(candles, 0, 2, 4))

        assert len(engine.closed) + len(engine.positions) == 1
        assert len(account.rejections) == 2
        assert all("cap" in r.reason or "already" in r.reason for r in account.rejections)

    def test_one_position_per_symbol_is_the_default(self):
        candles = frame([(50, 50, 50, 50)] + flat(100, 20))
        account = Account(10_000, size_fraction=0.2, max_positions=5)
        engine = PaperBook("BTC/USDT", hold=10, costs=trades.FREE, account=account)
        engine.run(candles, signal_at(candles, 0, 2))

        assert len(account.rejections) == 1
        assert "already holding" in account.rejections[0].reason

    def test_cash_leaves_on_entry_and_comes_back_on_exit(self):
        account = Account(10_000, size_fraction=0.5, max_positions=2)
        engine = PaperBook("BTC/USDT", hold=1, costs=trades.FREE, account=account)
        engine.run(RISER, signal_at(RISER, 0))

        # 5,000 in at 100, out at 110: 5,500 back, so 10,500 in the end.
        assert account.cash == pytest.approx(10_500.0)
        assert account.open_positions == 0

    def test_a_rejected_signal_is_recorded_with_the_bar_that_caused_it(self):
        candles = frame([(50, 50, 50, 50)] + flat(100, 20))
        account = Account(10_000, max_positions=1)
        engine = PaperBook("BTC/USDT", hold=10, costs=trades.FREE, account=account)
        engine.run(candles, signal_at(candles, 0, 5))

        assert [r.bar for r in account.rejections] == [6]
        assert account.rejections[0].symbol == "BTC/USDT"

    def test_the_account_does_not_change_the_prices_it_pays(self):
        """A wallet decides *whether* and *how much*, never *at what price*. The
        trades a constrained account does take must be identical to the ones an
        unconstrained one took at the same bars.
        """
        # A long hold against closely spaced signals, so the trades genuinely
        # overlap and the one-per-symbol rule has something to refuse.
        candles = walk(300, seed=10)
        signals = rules.apply("breakout", candles, window=3)

        free = book(hold=48).run(candles, signals)
        limited = PaperBook(
            "BTC/USDT", hold=48, costs=trades.FREE, account=Account(10_000, max_positions=2)
        ).run(candles, signals)

        assert len(limited) < len(free)
        overlap = free[free["entry_bar"].isin(limited["entry_bar"])].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            overlap[SHARED], limited[SHARED].reset_index(drop=True), check_dtype=False
        )


class TestTheLedger:
    def test_it_is_empty_but_shaped_before_anything_trades(self):
        ledger = book(hold=1).ledger()
        assert len(ledger) == 0
        assert list(ledger.columns) == list(ledger.columns)
        assert "net_return" in ledger.columns

    def test_it_names_the_symbol_on_every_row(self):
        ledger = book(symbol="ETH/USDT", hold=1).run(RISER, signal_at(RISER, 0))
        assert set(ledger["symbol"]) == {"ETH/USDT"}

    def test_costs_make_the_net_return_worse_than_the_gross(self):
        engine = PaperBook("BTC/USDT", hold=1, account=Account.unlimited())
        row = engine.run(RISER, signal_at(RISER, 0)).iloc[0]
        assert row.net_return < row.gross_return

    def test_the_broker_saw_one_fill_per_side(self):
        broker = PaperBroker(trades.FREE)
        engine = PaperBook(
            "BTC/USDT", hold=1, costs=trades.FREE, account=Account.unlimited(), broker=broker
        )
        engine.run(RISER, signal_at(RISER, 0))
        assert [fill.side for fill in broker.fills] == ["buy", "sell"]
        assert broker.positions() == {}
