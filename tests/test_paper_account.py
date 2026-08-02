"""The wallet: cash, the position cap, and what it refuses.

Tested apart from the engine and against plain numbers, because this is the one
part of the paper package that has no counterpart in the backtest and therefore
nothing to be checked against. The engine can be proved correct by comparing it
to `round_trips`; the account can only be proved correct by being read, so it is
written to be readable and checked a value at a time.

The rejections get as much attention as the fills. A signal the account could
not afford is the difference between what a rule found and what a wallet could
have done with it, and that difference is the entire subject of this package.
"""

import pytest

from paper.account import Account, AccountError, Rejection


class TestSizing:
    def test_a_position_is_a_fixed_slice_of_the_starting_capital(self):
        account = Account(10_000, size_fraction=0.2)
        assert account.notional_for("BTC/USDT") == pytest.approx(2_000.0)

    def test_the_slice_does_not_grow_as_the_account_does(self):
        """Fixed fraction of *starting* capital, not of current cash or equity.

        A size that moves with the account makes trade number four depend on an
        unrealised mark that will have changed again before it is realised. This
        keeps a position size a number that can be checked by hand.
        """
        account = Account(10_000, size_fraction=0.2)
        account.cash = 50_000
        assert account.notional_for("BTC/USDT") == pytest.approx(2_000.0)

    def test_but_it_is_capped_by_the_cash_actually_free(self):
        """A nearly-empty account takes a smaller last position rather than
        spending money it does not have.
        """
        account = Account(10_000, size_fraction=0.5)
        account.cash = 1_200
        assert account.notional_for("BTC/USDT") == pytest.approx(1_200.0)

    def test_an_empty_account_can_size_nothing(self):
        account = Account(10_000, size_fraction=0.5)
        account.cash = 0
        assert account.notional_for("BTC/USDT") == 0
        assert account.refusal("BTC/USDT") == "no free cash"

    def test_overdrawn_cash_never_produces_a_negative_size(self):
        """Cash should not go below zero, but a size derived from it must not be
        negative even if it somehow did -- a negative notional would become a
        negative quantity and a position that makes money when the price falls.
        """
        account = Account(10_000, size_fraction=0.5)
        account.cash = -50
        assert account.notional_for("BTC/USDT") == 0


class TestWhatItRefuses:
    def test_a_second_position_in_the_same_symbol(self):
        account = Account(10_000)
        account.opened("BTC/USDT", 2_000)
        assert "already holding BTC/USDT" == account.refusal("BTC/USDT")

    def test_but_a_different_symbol_is_fine(self):
        account = Account(10_000)
        account.opened("BTC/USDT", 2_000)
        assert account.refusal("ETH/USDT") is None

    def test_one_per_symbol_can_be_turned_off(self):
        account = Account(10_000, one_per_symbol=False)
        account.opened("BTC/USDT", 2_000)
        assert account.refusal("BTC/USDT") is None

    def test_the_position_cap(self):
        account = Account(100_000, size_fraction=0.1, max_positions=3)
        for symbol in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
            assert account.refusal(symbol) is None
            account.opened(symbol, 10_000)
        assert account.refusal("XRP/USDT") == "at the 3-position cap"

    def test_the_cap_frees_up_again_when_a_position_closes(self):
        account = Account(100_000, size_fraction=0.1, max_positions=1)
        account.opened("BTC/USDT", 10_000)
        assert account.refusal("ETH/USDT") is not None
        account.closed("BTC/USDT", 11_000)
        assert account.refusal("ETH/USDT") is None

    def test_the_reason_says_which_constraint_bound(self):
        """A count of rejections says the account was busy. A reason says
        whether it was out of money or out of slots, and those two call for
        opposite changes.
        """
        full = Account(100_000, size_fraction=0.1, max_positions=1)
        full.opened("BTC/USDT", 10_000)
        assert "cap" in full.refusal("ETH/USDT")

        broke = Account(10_000, size_fraction=1.0, max_positions=5)
        broke.opened("BTC/USDT", 10_000)
        assert "cash" in broke.refusal("ETH/USDT")


class TestCash:
    def test_opening_spends_and_closing_returns(self):
        account = Account(10_000)
        account.opened("BTC/USDT", 2_000)
        assert account.cash == pytest.approx(8_000)
        account.closed("BTC/USDT", 2_400)
        assert account.cash == pytest.approx(10_400)

    def test_a_losing_round_trip_leaves_less_than_it_started_with(self):
        account = Account(10_000)
        account.opened("BTC/USDT", 2_000)
        account.closed("BTC/USDT", 1_600)
        assert account.cash == pytest.approx(9_600)

    def test_equity_is_cash_plus_what_is_open(self):
        account = Account(10_000)
        account.opened("BTC/USDT", 2_000)
        assert account.equity({"BTC/USDT": 2_500}) == pytest.approx(10_500)

    def test_equity_with_nothing_open_is_just_cash(self):
        assert Account(10_000).equity() == pytest.approx(10_000)


class TestRejectionsAreRecorded:
    def test_a_rejection_keeps_the_bar_the_symbol_and_the_reason(self):
        account = Account(10_000)
        account.reject(41, "BTC/USDT", "at the 5-position cap")
        assert account.rejections == [
            Rejection(bar=41, symbol="BTC/USDT", reason="at the 5-position cap")
        ]

    def test_there_are_none_to_begin_with(self):
        assert Account(10_000).rejections == []


class TestTheUnlimitedAccount:
    """What makes the backtest invariant testable at all."""

    def test_it_refuses_nothing(self):
        account = Account.unlimited()
        for _ in range(50):
            assert account.refusal("BTC/USDT") is None
            account.opened("BTC/USDT", 1.0)

    def test_it_allows_overlapping_positions_in_one_symbol(self):
        """Which is exactly what `round_trips` does, and why reproducing it
        needs an account that imposes nothing.
        """
        account = Account.unlimited()
        account.opened("BTC/USDT", 1.0)
        assert account.refusal("BTC/USDT") is None

    def test_its_cash_does_not_move(self):
        account = Account.unlimited()
        account.opened("BTC/USDT", 1.0)
        account.closed("BTC/USDT", 2.0)
        assert account.cash == float("inf")

    def test_it_sizes_every_position_the_same(self):
        account = Account.unlimited()
        assert account.notional_for("BTC/USDT") == 1.0


class TestItRefusesToBeConfiguredBadly:
    @pytest.mark.parametrize("capital", [0, -1, "10000", None, True])
    def test_a_starting_capital_that_is_not_a_positive_number(self, capital):
        with pytest.raises(AccountError, match="starting_capital"):
            Account(capital)

    @pytest.mark.parametrize("fraction", [0, -0.1, 1.5, "0.2", None, True])
    def test_a_size_fraction_outside_zero_to_one(self, fraction):
        with pytest.raises(AccountError, match="size_fraction"):
            Account(10_000, size_fraction=fraction)

    def test_a_size_fraction_of_exactly_one_is_allowed(self):
        """Everything in one position is a strange choice, not an impossible
        one, and refusing it would be this module having an opinion about
        strategy rather than about arithmetic.
        """
        assert Account(10_000, size_fraction=1.0).notional_for("BTC/USDT") == 10_000

    @pytest.mark.parametrize("cap", [0, -1, 2.5, "3", True])
    def test_a_position_cap_that_is_not_a_positive_whole_number(self, cap):
        with pytest.raises(AccountError, match="max_positions"):
            Account(10_000, max_positions=cap)

    def test_no_cap_at_all_is_allowed(self):
        account = Account(1_000_000, size_fraction=0.01, max_positions=None)
        for index in range(50):
            account.opened(f"SYM{index}/USDT", 10_000)
        assert account.refusal("LAST/USDT") is None
