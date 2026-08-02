"""Trades: does a signal become a round trip anyone could actually have taken?

Every frame in this file is built by hand, small enough that the right answer
can be worked out on paper before the code is asked for it. That is the whole
method here. An indicator can be checked against a random walk because the
question is "does this match a formula"; a fill model cannot, because the
question is "would this price have been available to you at that moment", and
the only way to answer it is to know what every bar was.

So the numbers are deliberately round. Entry 100, exit 110, gross ten percent.
A stop at two percent sitting at exactly 98. If a test here needs a calculator
to check, it is testing the wrong thing.

WHAT THESE TESTS ARE GUARDING AGAINST
-------------------------------------
One mistake, mostly, wearing different hats: using a price that had not happened
yet. Entering at the close of the bar that produced the signal. Exiting at a
stop price on a bar that opened straight through it. Letting a trade near the
end of the file quietly shorten its holding period rather than admitting it
never finished. Each of those is a small lie, each one flatters the result, and
each one concentrates in exactly the trades that matter most.
"""

import numpy as np
import pandas as pd
import pytest

from signals import trades

FIRST_MS = 1_700_000_000_000
HOUR_MS = 3_600_000


def frame(rows):
    """Candles from a list of (open, high, low, close) tuples.

    Volume is a constant because nothing here reads it, and a column of ones is
    easier to skim past than a column of plausible-looking noise.
    """
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
    """`count` bars that neither move nor wobble. Useful padding."""
    return [(price, price, price, price)] * count


def signal_at(candles, *bars):
    fired = np.zeros(len(candles), dtype=bool)
    for bar in bars:
        fired[bar] = True
    return pd.Series(fired, index=candles.index)


def one_trade(candles, signals, **kwargs):
    """The single row a one-signal frame should have produced."""
    book = trades.round_trips(candles, signals, **kwargs)
    assert len(book.trades) == 1, f"expected exactly one trade, got {len(book.trades)}"
    return book.trades.iloc[0]


# A frame where the interesting bars are the openings: 100 then 110, so a
# one-bar hold entered on bar 1 and left on bar 2 makes exactly ten percent.
RISER = frame([(50, 50, 50, 50), (100, 100, 100, 100), (110, 110, 110, 110)] + flat(110, 4))


class TestTheEntryIsTheNextBarsOpen:
    """The single most important line in the module.

    A signal is computed from a bar's close, which you do not know until the bar
    has ended -- by which time that close is not a price you can trade, it is a
    price you watched go past. The next bar's open is the first number in the
    file you could actually have paid.
    """

    def test_a_signal_on_bar_zero_enters_on_bar_one(self):
        row = one_trade(RISER, signal_at(RISER, 0), hold=1, costs=trades.FREE)
        assert row.entry_bar == 1
        assert row.entry_price == 100.0

    def test_and_not_at_the_close_of_the_bar_that_signalled(self):
        """The mistake this whole module exists to prevent.

        Bar 0 closes at 50 and bar 1 opens at 100. If the entry ever comes back
        as 50, someone has wired the entry to the signal bar and every backtest
        downstream is worth nothing.
        """
        row = one_trade(RISER, signal_at(RISER, 0), hold=1, costs=trades.FREE)
        assert row.entry_price != 50.0

    def test_the_entry_time_is_the_bar_it_entered_on(self):
        row = one_trade(RISER, signal_at(RISER, 0), hold=1, costs=trades.FREE)
        assert row.entry_time == RISER["timestamp"].iloc[1]

    def test_a_later_signal_enters_later(self):
        row = one_trade(RISER, signal_at(RISER, 3), hold=1, costs=trades.FREE)
        assert row.entry_bar == 4

    def test_two_signals_make_two_trades_in_order(self):
        book = trades.round_trips(RISER, signal_at(RISER, 0, 3), hold=1, costs=trades.FREE)
        assert list(book.trades.entry_bar) == [1, 4]

    def test_a_signal_on_the_very_last_bar_has_nowhere_to_enter(self):
        candles = frame(flat(100, 5))
        book = trades.round_trips(candles, signal_at(candles, 4), hold=1, costs=trades.FREE)
        assert len(book.trades) == 0
        assert book.dropped == 1


class TestTheExitIsWhereItWasPromised:
    def test_a_one_bar_hold_leaves_on_the_bar_after_entry(self):
        row = one_trade(RISER, signal_at(RISER, 0), hold=1, costs=trades.FREE)
        assert row.exit_bar == 2
        assert row.exit_price == 110.0

    def test_a_three_bar_hold_leaves_three_bars_after_entry(self):
        row = one_trade(RISER, signal_at(RISER, 0), hold=3, costs=trades.FREE)
        assert row.exit_bar == 4

    def test_bars_held_matches_the_hold_that_was_asked_for(self):
        row = one_trade(RISER, signal_at(RISER, 0), hold=3, costs=trades.FREE)
        assert row.bars_held == 3

    def test_and_says_so(self):
        row = one_trade(RISER, signal_at(RISER, 0), hold=3, costs=trades.FREE)
        assert row.exit_reason == "hold"

    def test_the_exit_time_is_the_bar_it_left_on(self):
        row = one_trade(RISER, signal_at(RISER, 0), hold=1, costs=trades.FREE)
        assert row.exit_time == RISER["timestamp"].iloc[2]

    def test_the_exit_is_an_open_not_a_close(self):
        """Symmetry with the entry, and for the same reason.

        Bar 2 opens at 110 and closes at 999 here. Exiting at the close would
        mean selling at a price that only existed after the decision to sell.
        """
        candles = frame(
            [(50, 50, 50, 50), (100, 100, 100, 100), (110, 999, 110, 999)] + flat(999, 3)
        )
        row = one_trade(candles, signal_at(candles, 0), hold=1, costs=trades.FREE)
        assert row.exit_price == 110.0


class TestTheReturns:
    def test_gross_is_the_move_between_the_two_fills(self):
        row = one_trade(RISER, signal_at(RISER, 0), hold=1, costs=trades.FREE)
        assert row.gross_return == pytest.approx(0.10)

    def test_free_costs_leave_net_equal_to_gross(self):
        row = one_trade(RISER, signal_at(RISER, 0), hold=1, costs=trades.FREE)
        assert row.net_return == pytest.approx(row.gross_return)

    def test_costs_are_paid_on_both_sides(self):
        """Buy 0.15% worse than the open, sell 0.15% worse than the open.

        100 becomes 100.15 going in and 110 becomes 109.835 coming out. The
        expected value is written as that arithmetic rather than as a formula,
        so that a wrong formula in the module cannot be matched by the same
        wrong formula here.
        """
        row = one_trade(RISER, signal_at(RISER, 0), hold=1)
        assert row.net_return == pytest.approx(109.835 / 100.15 - 1)

    def test_and_gross_does_not_notice_them(self):
        row = one_trade(RISER, signal_at(RISER, 0), hold=1)
        assert row.gross_return == pytest.approx(0.10)

    def test_a_trade_that_goes_nowhere_still_loses_money(self):
        """The reason the default is not zero.

        Nothing moved. A costless backtest calls this a scratch; a real account
        calls it about a third of a percent gone.
        """
        candles = frame(flat(100, 6))
        row = one_trade(candles, signal_at(candles, 0), hold=1)
        assert row.gross_return == pytest.approx(0.0)
        assert row.net_return < -0.002

    def test_a_losing_trade_loses_more_after_costs(self):
        candles = frame([(50, 50, 50, 50), (100, 100, 100, 100)] + flat(90, 5))
        row = one_trade(candles, signal_at(candles, 0), hold=1)
        assert row.gross_return == pytest.approx(-0.10)
        assert row.net_return < row.gross_return


class TestSignalsTooLateToFinish:
    """A signal that cannot complete its hold is not a short trade. It is not a
    trade. Quietly shortening it would make the most recent trades look
    different from all the others, which is the worst place to introduce a bias
    because it is the part of the file you will be most tempted to trust."""

    def test_it_is_dropped_rather_than_shortened(self):
        candles = frame(flat(100, 6))
        book = trades.round_trips(candles, signal_at(candles, 3), hold=5, costs=trades.FREE)
        assert len(book.trades) == 0

    def test_and_counted(self):
        candles = frame(flat(100, 6))
        book = trades.round_trips(candles, signal_at(candles, 3), hold=5, costs=trades.FREE)
        assert book.dropped == 1

    def test_the_signal_itself_is_still_counted(self):
        """Otherwise a rule that fires plenty but always too late reads as a
        rule that never fires, and those need different fixes."""
        candles = frame(flat(100, 6))
        book = trades.round_trips(candles, signal_at(candles, 3), hold=5, costs=trades.FREE)
        assert book.signals == 1

    def test_the_last_completable_signal_is_kept(self):
        """Bar 3 signals, enters on 4, and a one-bar hold leaves on 5, which is
        the final bar. Off by one here loses a real trade from every backtest."""
        candles = frame(flat(100, 6))
        book = trades.round_trips(candles, signal_at(candles, 3), hold=1, costs=trades.FREE)
        assert len(book.trades) == 1
        assert book.dropped == 0

    def test_and_the_first_uncompletable_one_is_not(self):
        candles = frame(flat(100, 6))
        book = trades.round_trips(candles, signal_at(candles, 4), hold=1, costs=trades.FREE)
        assert len(book.trades) == 0
        assert book.dropped == 1

    def test_a_longer_hold_drops_more(self):
        candles = frame(flat(100, 20))
        signals = signal_at(candles, *range(20))
        short = trades.round_trips(candles, signals, hold=1, costs=trades.FREE)
        long = trades.round_trips(candles, signals, hold=5, costs=trades.FREE)
        assert long.dropped > short.dropped

    def test_no_signal_at_all_is_not_a_drop(self):
        candles = frame(flat(100, 6))
        book = trades.round_trips(candles, signal_at(candles), hold=1, costs=trades.FREE)
        assert book.signals == 0
        assert book.dropped == 0
        assert len(book.trades) == 0


# Entry on bar 1 at 100. A 2% stop sits at 98, a 3% target at 103.
def stopping(bar_two):
    return frame([(50, 50, 50, 50), (100, 100, 100, 100), bar_two] + flat(100, 5))


class TestTheStop:
    def test_a_touch_of_the_level_ends_the_trade(self):
        candles = stopping((99, 99, 97, 99))
        row = one_trade(candles, signal_at(candles, 0), hold=4, stop=0.02, costs=trades.FREE)
        assert row.exit_reason == "stop"
        assert row.exit_bar == 2

    def test_and_fills_at_the_level(self):
        candles = stopping((99, 99, 97, 99))
        row = one_trade(candles, signal_at(candles, 0), hold=4, stop=0.02, costs=trades.FREE)
        assert row.exit_price == pytest.approx(98.0)
        assert row.gross_return == pytest.approx(-0.02)

    def test_exactly_touching_the_level_counts_as_hit(self):
        """A low of exactly 98.0 against a stop at 98.0. Strict-versus-loose
        comparison decides this one, and it is the kind of thing that is chosen
        by accident unless a test chooses it."""
        candles = stopping((99, 99, 98, 99))
        row = one_trade(candles, signal_at(candles, 0), hold=4, stop=0.02, costs=trades.FREE)
        assert row.exit_reason == "stop"

    def test_stopping_just_short_of_the_level_does_not(self):
        candles = stopping((99, 99, 98.01, 99))
        row = one_trade(candles, signal_at(candles, 0), hold=4, stop=0.02, costs=trades.FREE)
        assert row.exit_reason == "hold"

    def test_a_bar_that_opens_through_the_stop_fills_at_the_open(self):
        """The detail that decides whether the worst trades are honest.

        The bar opens at 95, five percent below entry, having leapt over the
        stop at 98 while nobody could trade. Filling at 98 would be inventing a
        counterparty who was not there, and it would do so precisely on the
        trades that hurt most.
        """
        candles = stopping((95, 95, 94, 95))
        row = one_trade(candles, signal_at(candles, 0), hold=4, stop=0.02, costs=trades.FREE)
        assert row.exit_price == pytest.approx(95.0)
        assert row.gross_return == pytest.approx(-0.05)

    def test_bars_held_is_shorter_when_stopped_early(self):
        candles = stopping((99, 99, 97, 99))
        row = one_trade(candles, signal_at(candles, 0), hold=4, stop=0.02, costs=trades.FREE)
        assert row.bars_held == 1

    def test_the_entry_bar_itself_can_stop_the_trade(self):
        """You are in the position from that bar's open, so its low is yours."""
        candles = frame([(50, 50, 50, 50), (100, 100, 96, 100)] + flat(100, 5))
        row = one_trade(candles, signal_at(candles, 0), hold=4, stop=0.02, costs=trades.FREE)
        assert row.exit_reason == "stop"
        assert row.exit_bar == 1
        assert row.bars_held == 0

    def test_a_stop_that_is_never_touched_changes_nothing(self):
        row = one_trade(RISER, signal_at(RISER, 0), hold=3, stop=0.5, costs=trades.FREE)
        assert row.exit_reason == "hold"
        assert row.exit_bar == 4

    def test_the_first_touch_is_the_one_that_counts(self):
        """Two bars in the window breach the stop. You were out on the first.

        Nothing in the module reads obviously wrong if it takes the last one,
        which is why this is here: the mutation that changed `touched[0]` to
        `touched[-1]` passed the entire rest of this file.
        """
        candles = frame(
            [(50, 50, 50, 50), (100, 100, 100, 100), (99, 99, 97, 99), (99, 99, 95, 99)]
            + flat(100, 5)
        )
        row = one_trade(candles, signal_at(candles, 0), hold=4, stop=0.02, costs=trades.FREE)
        assert row.exit_bar == 2
        assert row.exit_price == pytest.approx(98.0)

    def test_the_bar_the_trade_was_sold_on_cannot_stop_it(self):
        """The exit bar's low belongs to somebody else.

        A time exit sells at bar 4's open. What bar 4 does afterwards is not
        the trade's business, and letting it count would be the same error the
        lookahead guard exists to catch, one layer lower down: reacting to
        prices that arrived after the decision.
        """
        candles = frame([(50, 50, 50, 50)] + flat(100, 3) + [(100, 100, 90, 100)] + flat(100, 5))
        row = one_trade(candles, signal_at(candles, 0), hold=3, stop=0.02, costs=trades.FREE)
        assert row.exit_reason == "hold"
        assert row.exit_price == pytest.approx(100.0)

    def test_the_stop_is_still_paid_for(self):
        candles = stopping((99, 99, 97, 99))
        row = one_trade(candles, signal_at(candles, 0), hold=4, stop=0.02)
        assert row.net_return < row.gross_return


class TestTheTarget:
    def test_a_touch_of_the_level_ends_the_trade(self):
        candles = stopping((101, 104, 101, 101))
        row = one_trade(candles, signal_at(candles, 0), hold=4, target=0.03, costs=trades.FREE)
        assert row.exit_reason == "target"
        assert row.exit_bar == 2

    def test_and_fills_at_the_level(self):
        candles = stopping((101, 104, 101, 101))
        row = one_trade(candles, signal_at(candles, 0), hold=4, target=0.03, costs=trades.FREE)
        assert row.exit_price == pytest.approx(103.0)
        assert row.gross_return == pytest.approx(0.03)

    def test_exactly_touching_the_level_counts_as_hit(self):
        candles = stopping((101, 103, 101, 101))
        row = one_trade(candles, signal_at(candles, 0), hold=4, target=0.03, costs=trades.FREE)
        assert row.exit_reason == "target"

    def test_a_bar_that_opens_through_the_target_fills_at_the_open(self):
        """The same rule as the gapped stop, pointing the other way.

        Gaps are not a courtesy extended only to losses. A bar that opens at 105
        filled you at 105, and pretending it filled at your 103 target would
        understate the good trades exactly as filling a gapped stop at the stop
        would understate the bad ones. The rule is "you got the open", not "you
        got the worse of the two".
        """
        candles = stopping((105, 106, 105, 105))
        row = one_trade(candles, signal_at(candles, 0), hold=4, target=0.03, costs=trades.FREE)
        assert row.exit_price == pytest.approx(105.0)
        assert row.gross_return == pytest.approx(0.05)

    def test_the_first_touch_is_the_one_that_counts(self):
        candles = frame(
            [(50, 50, 50, 50), (100, 100, 100, 100), (101, 104, 101, 101), (101, 108, 101, 101)]
            + flat(100, 5)
        )
        row = one_trade(candles, signal_at(candles, 0), hold=4, target=0.03, costs=trades.FREE)
        assert row.exit_bar == 2
        assert row.exit_price == pytest.approx(103.0)

    def test_a_target_that_is_never_touched_changes_nothing(self):
        candles = frame([(50, 50, 50, 50)] + flat(100, 6))
        row = one_trade(candles, signal_at(candles, 0), hold=3, target=0.5, costs=trades.FREE)
        assert row.exit_reason == "hold"


class TestWhenBothAreTouchedInTheSameBar:
    """An hourly candle records a high and a low and not one word about which
    came first. The data cannot answer, so the answer has to be chosen, and the
    choice is the pessimistic one: assume the stop."""

    def test_the_stop_wins(self):
        candles = stopping((100, 104, 97, 100))
        row = one_trade(
            candles, signal_at(candles, 0), hold=4, stop=0.02, target=0.03, costs=trades.FREE
        )
        assert row.exit_reason == "stop"
        assert row.exit_price == pytest.approx(98.0)

    def test_even_when_the_bar_closes_up(self):
        candles = stopping((100, 104, 97, 104))
        row = one_trade(
            candles, signal_at(candles, 0), hold=4, stop=0.02, target=0.03, costs=trades.FREE
        )
        assert row.exit_reason == "stop"

    def test_a_target_in_an_earlier_bar_still_wins_over_a_later_stop(self):
        """Pessimism applies within a bar, not across them. Order is known
        between bars, and ignoring it would be a different lie."""
        candles = frame(
            [(50, 50, 50, 50), (100, 100, 100, 100), (101, 104, 101, 101), (99, 99, 90, 99)]
            + flat(99, 4)
        )
        row = one_trade(
            candles, signal_at(candles, 0), hold=4, stop=0.02, target=0.03, costs=trades.FREE
        )
        assert row.exit_reason == "target"
        assert row.exit_bar == 2


class TestTheTrailingStop:
    """A stop that starts at the entry and ratchets up under the running peak.

    Same paper-checkable frames as the fixed stop: entry on bar 1 at an open of
    100, and every level a round number. The extra thing being tested is only
    the ratchet -- that the level follows the highest high and never falls back.
    """

    def test_with_no_new_high_it_is_just_a_stop_at_the_entry(self):
        """Nothing rises, so the peak stays at the entry and a 2% trail sits at
        98 -- the same place a 2% fixed stop would. A low of 97 takes it.
        """
        candles = stopping((99, 99, 97, 99))
        row = one_trade(candles, signal_at(candles, 0), hold=4, trail=0.02, costs=trades.FREE)
        assert row.exit_reason == "trail"
        assert row.exit_bar == 2
        assert row.exit_price == pytest.approx(98.0)

    def test_it_ratchets_up_under_a_new_high_and_locks_in_a_gain(self):
        """Bar 2 makes a high of 110, so the 2% trail rises to 107.8. Bar 3
        drops to 105, through that level, and the trade leaves at 106 for a gain
        -- the whole point of a trailing stop over a fixed one.
        """
        candles = frame(
            [(50, 50, 50, 50), (100, 100, 100, 100), (105, 110, 104, 108), (106, 106, 105, 106)]
            + flat(106, 5)
        )
        row = one_trade(candles, signal_at(candles, 0), hold=5, trail=0.02, costs=trades.FREE)
        assert row.exit_reason == "trail"
        assert row.exit_bar == 3
        assert row.exit_price == pytest.approx(106.0)
        assert row.gross_return == pytest.approx(0.06)

    def test_a_bar_that_gaps_through_the_trailing_level_fills_at_the_open(self):
        """The same honesty the fixed stop keeps. The peak is 110 and the level
        107.8, but bar 3 opens at 105, already below it -- nobody was at 107.8,
        so the fill is the open.
        """
        candles = frame(
            [(50, 50, 50, 50), (100, 100, 100, 100), (105, 110, 104, 108), (105, 105, 104, 105)]
            + flat(105, 5)
        )
        row = one_trade(candles, signal_at(candles, 0), hold=5, trail=0.02, costs=trades.FREE)
        assert row.exit_reason == "trail"
        assert row.exit_price == pytest.approx(105.0)

    def test_the_peak_does_not_include_the_bar_being_tested(self):
        """The pessimistic reading of one candle. Bar 2 has a high of 110 and a
        low of 98. If its own high lifted the level to 107.8 before its low was
        checked, the low would trip it inside the same bar. The peak it trails is
        the one locked in before this bar -- still 100, level 98 -- so a low of
        98 exactly is what takes it, and it fills at 98, not at 107.8.
        """
        candles = frame(
            [(50, 50, 50, 50), (100, 100, 100, 100), (104, 110, 98, 104)] + flat(104, 5)
        )
        row = one_trade(candles, signal_at(candles, 0), hold=4, trail=0.02, costs=trades.FREE)
        assert row.exit_reason == "trail"
        assert row.exit_bar == 2
        assert row.exit_price == pytest.approx(98.0)

    def test_a_trail_that_is_never_breached_leaves_at_time(self):
        """Price rises to 110 and holds there. The level trails at 107.8 and is
        never reached, so the trade runs to its time exit like any other.
        """
        candles = frame(
            [(50, 50, 50, 50), (100, 100, 100, 100), (110, 110, 110, 110)] + flat(110, 4)
        )
        row = one_trade(candles, signal_at(candles, 0), hold=3, trail=0.02, costs=trades.FREE)
        assert row.exit_reason == "hold"
        assert row.exit_bar == 4

    def test_the_entry_bar_itself_can_trail_out(self):
        candles = frame([(50, 50, 50, 50), (100, 100, 96, 100)] + flat(100, 5))
        row = one_trade(candles, signal_at(candles, 0), hold=4, trail=0.02, costs=trades.FREE)
        assert row.exit_reason == "trail"
        assert row.exit_bar == 1
        assert row.bars_held == 0

    def test_a_tighter_trail_pre_empts_a_looser_fixed_stop(self):
        """Both are set. The fixed stop at 10% sits at 90 and is never touched;
        the 2% trail at 98 is, so the trade leaves on the trail rather than
        running to time.
        """
        candles = stopping((99, 99, 97, 99))
        row = one_trade(
            candles, signal_at(candles, 0), hold=4, stop=0.10, trail=0.02, costs=trades.FREE
        )
        assert row.exit_reason == "trail"
        assert row.exit_bar == 2

    def test_a_fixed_stop_hit_first_still_wins(self):
        """The other direction. A 2% fixed stop at 98 is hit on bar 2, while the
        5% trail at 95 is not, so the exit is the stop -- the trailing layer
        never overrides an exit that already happened at least as early.
        """
        candles = stopping((99, 99, 97, 99))
        row = one_trade(
            candles, signal_at(candles, 0), hold=4, stop=0.02, trail=0.05, costs=trades.FREE
        )
        assert row.exit_reason == "stop"
        assert row.exit_bar == 2

    def test_a_trail_pre_empts_a_later_target(self):
        """The other exit family. The peak reaches 110 and the 2% trail at 107.8
        is hit on bar 3, while a 20% target at 120 is never printed. The earlier
        trailing exit is the one taken, over both the target and the time hold.
        """
        candles = frame(
            [(50, 50, 50, 50), (100, 100, 100, 100), (105, 110, 104, 108), (106, 106, 105, 106)]
            + flat(106, 5)
        )
        row = one_trade(
            candles, signal_at(candles, 0), hold=5, target=0.20, trail=0.02, costs=trades.FREE
        )
        assert row.exit_reason == "trail"
        assert row.exit_bar == 3

    def test_it_cannot_see_past_the_exit(self):
        """Rewriting the bars after a trailing exit must not change the trade,
        the same causality the fixed fill model keeps.
        """
        candles = frame(
            [(50, 50, 50, 50), (100, 100, 100, 100), (105, 110, 104, 108), (106, 106, 105, 106)]
            + flat(106, 5)
        )
        signals = signal_at(candles, 0)
        before = trades.round_trips(candles, signals, hold=5, trail=0.02, costs=trades.FREE)

        meddled = candles.copy()
        meddled.loc[meddled.index[5:], ["open", "high", "low", "close"]] = 1_000.0
        after = trades.round_trips(meddled, signals, hold=5, trail=0.02, costs=trades.FREE)

        pd.testing.assert_frame_equal(before.trades, after.trades)

    def test_it_is_paid_for_like_any_other_exit(self):
        candles = stopping((99, 99, 97, 99))
        row = one_trade(candles, signal_at(candles, 0), hold=4, trail=0.02)
        assert row.net_return < row.gross_return

    @pytest.mark.parametrize("trail", [0, 1, 1.5, -0.1, "x", True])
    def test_a_trail_that_is_not_a_fraction_between_zero_and_one_is_refused(self, trail):
        with pytest.raises(trades.TradeError, match="trail"):
            trades.round_trips(RISER, signal_at(RISER, 0), hold=3, trail=trail)


class TestOverlappingTradesAreAllKept:
    def test_consecutive_signals_each_get_a_trade(self):
        candles = frame(flat(100, 20))
        book = trades.round_trips(candles, signal_at(candles, 0, 1, 2), hold=5, costs=trades.FREE)
        assert list(book.trades.entry_bar) == [1, 2, 3]

    def test_a_trade_is_not_suppressed_by_an_open_one(self):
        candles = frame(flat(100, 20))
        book = trades.round_trips(candles, signal_at(candles, 0, 1), hold=10, costs=trades.FREE)
        assert len(book.trades) == 2


class TestTheFillModelCannotSeeTheFuture:
    """The lookahead guard from increment 3 watches the signal. Nothing was
    watching the fill, and the fill is where a plausible off-by-one turns into
    an exit at a price that had not happened yet."""

    def test_rewriting_the_bars_after_the_exit_changes_nothing(self):
        candles = frame(flat(100, 20))
        signals = signal_at(candles, 0)
        before = trades.round_trips(candles, signals, hold=3, costs=trades.FREE)
        after = int(before.trades.exit_bar.iloc[0])
        changed = candles.copy()
        for column in ("open", "high", "low", "close"):
            changed.loc[after + 1 :, column] *= 1.5
        again = trades.round_trips(changed, signals, hold=3, costs=trades.FREE)
        assert again.trades.equals(before.trades)

    def test_but_rewriting_a_bar_inside_the_trade_does(self):
        """Without this the test above passes for a rule that ignores prices
        entirely, which is the kind of green that hides a broken module."""
        candles = frame(flat(100, 20))
        signals = signal_at(candles, 0)
        before = trades.round_trips(candles, signals, hold=3, stop=0.02, costs=trades.FREE)
        changed = candles.copy()
        changed.loc[2, "low"] = 90.0
        again = trades.round_trips(changed, signals, hold=3, stop=0.02, costs=trades.FREE)
        assert not again.trades.equals(before.trades)
        assert again.trades.exit_reason.iloc[0] == "stop"

    def test_the_candles_are_not_modified(self):
        candles = frame(flat(100, 20))
        pristine = candles.copy()
        trades.round_trips(candles, signal_at(candles, 0), hold=3, stop=0.02)
        assert candles.equals(pristine)


class TestItRefusesRatherThanGuesses:
    def test_a_signal_series_of_the_wrong_length(self):
        candles = frame(flat(100, 10))
        with pytest.raises(trades.TradeError):
            trades.round_trips(candles, pd.Series([True, False]), hold=1)

    def test_and_says_so_as_a_length_rather_than_as_an_index(self):
        """The index check would catch this too, so the length check is
        redundant as a *check*. It is not redundant as a sentence: "got 2
        signal(s) for 10 candle(s)" tells you what to go and fix, and "the
        signals are indexed differently" tells you almost nothing. The check
        earns its place by being the one that speaks first.
        """
        candles = frame(flat(100, 10))
        with pytest.raises(trades.TradeError, match=r"2 signal\(s\) for 10 candle\(s\)"):
            trades.round_trips(candles, pd.Series([True, False]), hold=1)

    def test_a_signal_series_with_a_different_index(self):
        candles = frame(flat(100, 10))
        signals = pd.Series(np.zeros(10, dtype=bool), index=candles.index + 5)
        with pytest.raises(trades.TradeError):
            trades.round_trips(candles, signals, hold=1)

    def test_a_signal_series_that_is_not_boolean(self):
        candles = frame(flat(100, 10))
        signals = pd.Series(np.zeros(10), index=candles.index)
        with pytest.raises(trades.TradeError):
            trades.round_trips(candles, signals, hold=1)

    def test_something_that_is_not_a_candle_frame(self):
        with pytest.raises(trades.TradeError):
            trades.round_trips([1, 2, 3], pd.Series([True]), hold=1)

    def test_a_frame_missing_a_price_column(self):
        candles = frame(flat(100, 10)).drop(columns=["low"])
        with pytest.raises(trades.TradeError):
            trades.round_trips(candles, signal_at(candles, 0), hold=1, stop=0.02)

    @pytest.mark.parametrize("hold", [0, -1, 1.5, "3", None])
    def test_a_hold_that_is_not_a_positive_whole_number(self, hold):
        candles = frame(flat(100, 10))
        with pytest.raises(trades.TradeError):
            trades.round_trips(candles, signal_at(candles, 0), hold=hold)

    @pytest.mark.parametrize("stop", [0.0, -0.1, 1.0, 1.5, "0.02"])
    def test_a_stop_that_is_not_a_fraction_between_zero_and_one(self, stop):
        candles = frame(flat(100, 10))
        with pytest.raises(trades.TradeError):
            trades.round_trips(candles, signal_at(candles, 0), hold=1, stop=stop)

    @pytest.mark.parametrize("target", [0.0, -0.1, "0.03"])
    def test_a_target_that_is_not_a_positive_fraction(self, target):
        candles = frame(flat(100, 10))
        with pytest.raises(trades.TradeError):
            trades.round_trips(candles, signal_at(candles, 0), hold=1, target=target)

    def test_a_target_above_one_is_allowed(self):
        """Doubling is a strange target, not an impossible one. Stops are
        capped at 100% because a stop below zero has no meaning; targets are
        not, because prices have no ceiling."""
        candles = frame(flat(100, 10))
        book = trades.round_trips(candles, signal_at(candles, 0), hold=1, target=2.0)
        assert len(book.trades) == 1

    def test_a_nan_price_inside_a_holding_window(self):
        """prices.load already refuses holes, so reaching here means something
        upstream changed. Refusing beats averaging over a hole."""
        candles = frame(flat(100, 10))
        candles.loc[3, "low"] = np.nan
        with pytest.raises(trades.TradeError):
            trades.round_trips(candles, signal_at(candles, 0), hold=5, stop=0.02)

    def test_a_nan_price_on_the_exit_bar_itself(self):
        """The exit bar is the last bar the trade touches, so it is inside the
        window, not past its edge. Checking one bar short would let the exit
        fill at NaN and return a NaN quietly -- and a NaN average looks like a
        bug in whatever prints it, not like a hole in the data.
        """
        candles = frame(flat(100, 10))
        candles.loc[4, "open"] = np.nan
        with pytest.raises(trades.TradeError):
            trades.round_trips(candles, signal_at(candles, 0), hold=3, costs=trades.FREE)


class TestTheCosts:
    def test_the_default_is_not_free(self):
        """The whole argument for the default. If this ever becomes zero, every
        backtest in the project silently gets better."""
        assert trades.DEFAULT_COSTS.fee > 0
        assert trades.DEFAULT_COSTS.slippage > 0

    def test_free_is_free(self):
        assert trades.FREE.fee == 0
        assert trades.FREE.slippage == 0

    def test_the_default_is_the_binance_taker_fee(self):
        assert trades.DEFAULT_COSTS.fee == 0.001

    def test_a_negative_fee_is_refused(self):
        candles = frame(flat(100, 10))
        with pytest.raises(trades.TradeError):
            trades.round_trips(
                candles, signal_at(candles, 0), hold=1, costs=trades.Costs(fee=-0.001)
            )

    def test_bigger_costs_make_smaller_returns(self):
        row_cheap = one_trade(
            RISER, signal_at(RISER, 0), hold=1, costs=trades.Costs(fee=0.001, slippage=0.0)
        )
        row_dear = one_trade(
            RISER, signal_at(RISER, 0), hold=1, costs=trades.Costs(fee=0.01, slippage=0.0)
        )
        assert row_dear.net_return < row_cheap.net_return


class TestTheBook:
    def test_it_remembers_what_it_was_asked(self):
        candles = frame(flat(100, 10))
        book = trades.round_trips(candles, signal_at(candles, 0), hold=3, costs=trades.FREE)
        assert book.hold == 3
        assert book.costs == trades.FREE

    def test_the_summary_names_the_counts(self):
        candles = frame(flat(100, 10))
        book = trades.round_trips(candles, signal_at(candles, 0, 8), hold=3, costs=trades.FREE)
        text = book.summary()
        assert "2 signal" in text
        assert "1 trade" in text

    def test_the_summary_admits_what_it_dropped(self):
        candles = frame(flat(100, 10))
        book = trades.round_trips(candles, signal_at(candles, 0, 8), hold=3, costs=trades.FREE)
        assert "1 dropped" in book.summary()

    def test_the_summary_states_the_cost_assumption(self):
        """A return with no stated cost is not a result, and the number most
        worth double-checking is the one you did not type."""
        candles = frame(flat(100, 10))
        book = trades.round_trips(candles, signal_at(candles, 0), hold=3)
        text = book.summary()
        assert "0.1" in text and "0.05" in text

    def test_and_says_so_loudly_when_there_were_none(self):
        candles = frame(flat(100, 10))
        book = trades.round_trips(candles, signal_at(candles, 0), hold=3, costs=trades.FREE)
        assert "no costs" in book.summary().lower()

    def test_an_empty_book_still_has_the_right_columns(self):
        """Downstream code should not need to ask whether anything traded."""
        candles = frame(flat(100, 10))
        book = trades.round_trips(candles, signal_at(candles), hold=3, costs=trades.FREE)
        assert len(book.trades) == 0
        assert "net_return" in book.trades.columns
        assert "exit_reason" in book.trades.columns


class TestAcrossHolds:
    def test_it_returns_one_book_per_hold(self):
        candles = frame(flat(100, 50))
        books = trades.across_holds(
            candles, signal_at(candles, 0, 5, 10), holds=(1, 5, 20), costs=trades.FREE
        )
        assert sorted(books) == [1, 5, 20]

    def test_each_book_used_its_own_hold(self):
        candles = frame(flat(100, 50))
        books = trades.across_holds(
            candles, signal_at(candles, 0, 5, 10), holds=(1, 5, 20), costs=trades.FREE
        )
        assert all(hold == book.hold for hold, book in books.items())

    def test_the_default_holds_are_the_ones_that_were_agreed(self):
        assert trades.DEFAULT_HOLDS == (6, 24, 72, 168)

    def test_a_stop_reaches_every_book(self):
        candles = frame([(50, 50, 50, 50), (100, 100, 90, 100)] + flat(100, 48))
        books = trades.across_holds(
            candles, signal_at(candles, 0), holds=(1, 5), stop=0.02, costs=trades.FREE
        )
        assert all(book.trades.exit_reason.iloc[0] == "stop" for book in books.values())
