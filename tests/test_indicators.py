"""Tests for the indicator arithmetic.

Every expected value in this file was computed by hand or in exact rational
arithmetic *before* the implementation existed. That matters more here than
anywhere else in the project. An indicator bug does not crash and does not raise;
it returns a number that is slightly wrong, forever, and every conclusion drawn
downstream inherits the error without any sign that something happened. The only
defence is to know the answer independently of the code.

The recurring theme is off-by-one. A moving average that includes one bar too
many is still a plausible-looking moving average. It is also, if the extra bar
is the *next* one, a machine for producing backtests that cannot be traded.
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from signals import indicators as ind


def series(values, name=None):
    """A float Series with a plain 0..n-1 index."""
    return pd.Series(values, dtype="float64", name=name)


def expect(values):
    """The expected result: float, unnamed, default index."""
    return pd.Series(values, dtype="float64")


# Every indicator, as (callable, kwargs). Used by the shared property tests
# below so that a new indicator added later cannot quietly skip them.
ALL_INDICATORS = [
    pytest.param(ind.sma, {"window": 3}, id="sma"),
    pytest.param(ind.ema, {"span": 3}, id="ema"),
    pytest.param(ind.rsi, {"period": 3}, id="rsi"),
    pytest.param(ind.rolling_high, {"window": 3}, id="rolling_high"),
    pytest.param(ind.rolling_low, {"window": 3}, id="rolling_low"),
]


class TestSma:
    def test_matches_a_hand_computed_average(self):
        # (1+2+3)/3 = 2, (2+3+4)/3 = 3, (3+4+5)/3 = 4.
        result = ind.sma(series([1, 2, 3, 4, 5]), 3)
        assert_series_equal(result, expect([np.nan, np.nan, 2.0, 3.0, 4.0]))

    def test_the_first_value_appears_only_once_the_window_is_full(self):
        """The warm-up must be NaN, not a partial average.

        pandas will happily give you `min_periods=1` behaviour if asked, and the
        result looks entirely reasonable: a "20-period average" whose first value
        is the mean of one bar. Nothing downstream can tell that apart from a
        real average, so the rule fires on it and the backtest counts the trade.
        """
        result = ind.sma(series([1, 2, 3, 4, 5]), 3)
        assert result.isna().tolist() == [True, True, False, False, False]

    def test_a_window_of_one_is_the_series_itself(self):
        result = ind.sma(series([1, 2, 3]), 1)
        assert_series_equal(result, expect([1.0, 2.0, 3.0]))

    def test_a_constant_series_averages_to_that_constant(self):
        result = ind.sma(series([7, 7, 7, 7]), 3)
        assert result.dropna().tolist() == [7.0, 7.0]

    def test_a_window_longer_than_the_data_is_all_nan_rather_than_an_error(self):
        """Not enough history yet is a fact about the data, not a mistake.

        A 200-period average of 50 bars is a legitimate thing to ask for while
        the store is still filling up. Returning NaN says "no answer yet", which
        every rule downstream already has to handle. Raising here would mean the
        CLI could not show a long average and a short one side by side.
        """
        result = ind.sma(series([1, 2, 3]), 10)
        assert result.isna().all()
        assert len(result) == 3


class TestEma:
    def test_matches_a_hand_computed_exponential_average(self):
        # span=3 gives alpha = 2/(3+1) = 0.5, and the recursion seeds on the
        # first bar: 1, then 1.5, 2.25, 3.125, 4.0625. The first two are masked
        # by the warm-up rule below, so 2.25 is the first value we report.
        result = ind.ema(series([1, 2, 3, 4, 5]), 3)
        assert_series_equal(result, expect([np.nan, np.nan, 2.25, 3.125, 4.0625]))

    def test_the_warm_up_is_masked(self):
        """An EMA has a value from bar one unless you stop it.

        This is the difference between the two moving averages and the reason
        this test exists separately. `rolling` refuses to produce a number until
        the window is full; `ewm` produces one immediately, and that first number
        is simply the first price. A crossover rule comparing a 12-EMA with a
        26-EMA at bar 3 is comparing two lightly-smoothed copies of the same
        price, and it will cross constantly.
        """
        result = ind.ema(series([1, 2, 3, 4, 5]), 3)
        assert result.isna().tolist() == [True, True, False, False, False]

    def test_a_constant_series_stays_at_that_constant(self):
        result = ind.ema(series([7, 7, 7, 7, 7]), 3)
        assert result.dropna().tolist() == [7.0, 7.0, 7.0]

    def test_it_reacts_faster_than_the_simple_average(self):
        """The one property that distinguishes the two implementations.

        Both tests above would still pass if `ema` were secretly `sma`, because
        both are checked against their own constants. This one would not: after a
        step change, the exponential average must be closer to the new level than
        the simple one, which is the entire reason anyone uses it.
        """
        prices = series([10, 10, 10, 10, 20, 20])
        fast = ind.ema(prices, 3).iloc[-1]
        slow = ind.sma(prices, 3).iloc[-1]
        assert fast > slow


class TestRsi:
    def test_matches_a_hand_computed_value_over_two_periods(self):
        # closes 10, 11, 10, 12 -> deltas +1, -1, +2.
        # Seed at index 2: avg gain (1+0)/2 = 0.5, avg loss (0+1)/2 = 0.5,
        #   RS = 1, RSI = 100 - 100/2 = 50.
        # Index 3:        avg gain (0.5 + 2)/2 = 1.25, avg loss (0.5 + 0)/2 = 0.25,
        #   RS = 5, RSI = 100 - 100/6 = 83.333...
        result = ind.rsi(series([10, 11, 10, 12]), period=2)
        assert_series_equal(
            result, expect([np.nan, np.nan, 50.0, 250 / 3]), rtol=1e-12
        )

    def test_matches_a_hand_computed_value_over_three_periods(self):
        """Period 3, specifically, because period 2 cannot tell two seedings apart.

        Wilder seeds the averages with the simple mean of the first `period`
        deltas and only then applies the smoothing. The tempting one-liner --
        `deltas.ewm(alpha=1/period, adjust=False).mean()` -- seeds with the first
        delta instead. At period 2 those two happen to coincide, so the test
        above passes either way. At period 3 they diverge: the shortcut gives
        83.33 here where Wilder gives 75.0, and it also reports a value at index
        1 while Wilder is still warming up.

        Computed in exact fractions: 75, 600/11, 3900/49.
        """
        result = ind.rsi(series([10, 11, 10, 12, 11, 14]), period=3)
        assert_series_equal(
            result,
            expect([np.nan, np.nan, np.nan, 75.0, 600 / 11, 3900 / 49]),
            rtol=1e-12,
        )

    def test_the_first_value_needs_period_deltas_not_period_bars(self):
        """Off-by-one, in the place it is easiest to make.

        N deltas need N+1 bars. A 14-period RSI therefore has its first value at
        index 14, not index 13. Getting this wrong shifts every RSI value by one
        bar for the entire history, which is precisely the lookahead this project
        is trying not to have.
        """
        result = ind.rsi(series(range(20)), period=14)
        assert result.isna().tolist()[:14] == [True] * 14
        assert not np.isnan(result.iloc[14])

    def test_an_unbroken_rise_is_one_hundred(self):
        """No losses at all means dividing by zero, and it is not a rare case.

        Fourteen consecutive up bars is unusual but entirely possible, and the
        honest answer is 100 -- maximally overbought. The implementation has to
        say so deliberately; left alone, the division produces inf, and
        100 - 100/(1+inf) is nan, so the indicator would go blank at exactly the
        moment it is most extreme.
        """
        result = ind.rsi(series([1, 2, 3, 4, 5, 6]), period=3)
        assert result.dropna().tolist() == [100.0, 100.0, 100.0]

    def test_an_unbroken_fall_is_zero(self):
        result = ind.rsi(series([6, 5, 4, 3, 2, 1]), period=3)
        assert result.dropna().tolist() == [0.0, 0.0, 0.0]

    def test_a_flat_series_is_fifty(self):
        """Zero gains and zero losses: 0/0, which needs a decision rather than a
        default. Neither 0 nor 100 is defensible for a price that has not moved,
        so this returns the neutral 50 and the docstring says it is a convention.
        """
        result = ind.rsi(series([5, 5, 5, 5, 5]), period=2)
        assert result.dropna().tolist() == [50.0, 50.0, 50.0]

    def test_it_stays_within_zero_and_one_hundred(self):
        rng = np.random.default_rng(0)
        prices = series(100 + rng.standard_normal(500).cumsum())
        result = ind.rsi(prices, period=14).dropna()
        assert len(result) == 486
        assert result.between(0, 100).all()


class TestRollingHighAndLow:
    def test_high_matches_a_hand_computed_maximum(self):
        result = ind.rolling_high(series([1, 5, 3, 2, 8]), 3)
        assert_series_equal(result, expect([np.nan, np.nan, 5.0, 5.0, 8.0]))

    def test_low_matches_a_hand_computed_minimum(self):
        result = ind.rolling_low(series([1, 5, 3, 2, 8]), 3)
        assert_series_equal(result, expect([np.nan, np.nan, 1.0, 2.0, 2.0]))

    def test_the_window_includes_the_current_bar(self):
        """Stated as a test because it is the assumption a breakout rule breaks.

        "Price broke above the 20-bar high" cannot mean the 20-bar high that
        includes the bar doing the breaking -- the current bar's high is part of
        that maximum, so price can never exceed it and the rule never fires. The
        prior high needs an explicit `.shift(1)` at the call site. These
        functions use the ordinary convention; increment 2 does the shifting
        where it belongs, in the rule.
        """
        result = ind.rolling_high(series([1, 2, 9]), 3)
        assert result.iloc[-1] == 9.0


class TestCausality:
    """The property the whole project rests on: no indicator may look forward.

    Mechanically: compute over the full series, compute again over the series
    truncated at bar k, and require the first k values to be identical. If any
    indicator peeked at a later bar, the truncated run could not reproduce it.

    This is the same idea as milestone 1's "the suite cannot reach the network".
    Rather than reviewing each function and believing the review, make the thing
    that must not happen impossible to do unnoticed. Increment 3 applies the
    identical test one level up, to the rules.
    """

    @pytest.mark.parametrize("func,kwargs", ALL_INDICATORS)
    @pytest.mark.parametrize("cut", [4, 7, 11, 20])
    def test_truncating_the_future_does_not_change_the_past(self, func, kwargs, cut):
        rng = np.random.default_rng(1)
        prices = series(100 + rng.standard_normal(40).cumsum())

        full = func(prices, **kwargs).iloc[:cut]
        truncated = func(prices.iloc[:cut], **kwargs)

        assert_series_equal(full, truncated)

    @pytest.mark.parametrize("func,kwargs", ALL_INDICATORS)
    def test_changing_a_future_bar_does_not_change_an_earlier_value(
        self, func, kwargs
    ):
        """The same guarantee from the other direction, and a stronger check.

        Truncation catches an indicator that reads ahead by index. This catches
        one that reads ahead by *value* -- anything centred, anything normalised
        against the whole series, anything using a full-series mean. Move the
        last bar a long way and every earlier value must be untouched.
        """
        rng = np.random.default_rng(2)
        prices = series(100 + rng.standard_normal(40).cumsum())
        meddled = prices.copy()
        meddled.iloc[-1] = 10_000.0

        before = func(prices, **kwargs).iloc[:-1]
        after = func(meddled, **kwargs).iloc[:-1]

        assert_series_equal(before, after)


class TestSharedBehaviour:
    """Rules every indicator obeys, applied to all of them by parametrisation.

    Written once and applied to the list rather than repeated per function, so
    that adding an indicator in a later increment cannot quietly skip them.
    """

    @pytest.mark.parametrize("func,kwargs", ALL_INDICATORS)
    def test_the_input_is_not_modified(self, func, kwargs):
        """In-place modification of a caller's frame is the bug you find last.

        The intended usage is `candles["sma20"] = sma(candles["close"], 20)`,
        which hands the indicator a live column of the caller's DataFrame. If it
        writes through that reference, the price history itself changes, and
        every indicator computed afterwards is computed on different data.
        """
        prices = series([1, 5, 3, 2, 8, 4, 9, 6])
        before = prices.copy()
        func(prices, **kwargs)
        assert_series_equal(prices, before)

    @pytest.mark.parametrize("func,kwargs", ALL_INDICATORS)
    def test_the_index_is_preserved(self, func, kwargs):
        """Alignment is what makes `candles["x"] = indicator(...)` safe.

        pandas aligns on the index when assigning a Series into a DataFrame. An
        indicator that reset the index would appear to work on a freshly-loaded
        frame -- where the index is 0..n-1 anyway -- and silently misalign the
        moment anyone passed a slice.
        """
        prices = pd.Series(
            [1, 5, 3, 2, 8, 4, 9, 6], dtype="float64", index=range(100, 108)
        )
        result = func(prices, **kwargs)
        assert result.index.tolist() == list(range(100, 108))

    @pytest.mark.parametrize("func,kwargs", ALL_INDICATORS)
    def test_an_empty_series_gives_an_empty_series(self, func, kwargs):
        result = func(series([]), **kwargs)
        assert len(result) == 0

    @pytest.mark.parametrize("func,kwargs", ALL_INDICATORS)
    def test_leading_nan_is_accepted_so_indicators_can_be_chained(
        self, func, kwargs
    ):
        """`sma(ema(close, 12), 3)` must work -- that is how MACD is built.

        The inner indicator's warm-up arrives as leading NaN. Rejecting NaN
        outright would make chaining impossible, so leading NaN is skipped and
        the warm-up simply starts later.
        """
        prices = series([np.nan, np.nan, 1, 5, 3, 2, 8, 4, 9, 6])
        result = func(prices, **kwargs)
        assert result.iloc[:2].isna().all()
        assert result.notna().any()

    @pytest.mark.parametrize("func,kwargs", ALL_INDICATORS)
    def test_a_hole_in_the_middle_is_refused(self, func, kwargs):
        """A NaN surrounded by numbers means missing data, not warm-up.

        The two cases look identical to pandas and mean opposite things. Leading
        NaN is an indicator that has not started; interior NaN is a bar the store
        does not have, and averaging across it produces a number that spans a
        discontinuity while looking perfectly ordinary. `prices.py` refuses gaps
        already, so this should be unreachable -- which is exactly the argument
        that gets a check removed and then, two milestones later, regretted.
        """
        prices = series([1, 5, 3, np.nan, 8, 4, 9, 6])
        with pytest.raises(ind.IndicatorError, match="gap"):
            func(prices, **kwargs)

    @pytest.mark.parametrize("func,kwargs", ALL_INDICATORS)
    def test_a_non_series_input_is_refused(self, func, kwargs):
        """A plain list is the obvious mistake and it half-works.

        `pd.Series([1,2,3]).rolling(3)` and `[1,2,3].rolling(3)` differ by an
        AttributeError deep inside the function, which is a worse message than
        one sentence here.
        """
        with pytest.raises(ind.IndicatorError, match="Series"):
            func([1, 5, 3, 2, 8], **kwargs)

    @pytest.mark.parametrize("func,kwargs", ALL_INDICATORS)
    def test_a_non_numeric_series_is_refused(self, func, kwargs):
        with pytest.raises(ind.IndicatorError, match="numeric"):
            func(pd.Series(["a", "b", "c", "d"]), **kwargs)


class TestPeriodValidation:
    """The window/span/period argument, checked the same way for each function.

    Named differently per indicator on purpose -- `window` for the rolling ones,
    `span` for the EMA, `period` for RSI -- because those are the words the
    documentation for each uses, and a shared name would be a small lie about
    the EMA in particular, whose span is not a count of bars averaged.
    """

    @pytest.mark.parametrize("func,kwargs", ALL_INDICATORS)
    @pytest.mark.parametrize("bad", [0, -1, 2.5, "3", None, True])
    def test_a_period_that_is_not_a_positive_whole_number_is_refused(
        self, func, kwargs, bad
    ):
        # `True` is in the list because bool is a subclass of int and True == 1,
        # so a plain isinstance(x, int) check accepts it. A window of True is
        # never what anyone meant.
        (name,) = kwargs
        with pytest.raises(ind.IndicatorError):
            func(series([1, 5, 3, 2, 8]), **{name: bad})

    def test_rsi_needs_at_least_one_period(self):
        with pytest.raises(ind.IndicatorError):
            ind.rsi(series([1, 2, 3]), period=0)

    def test_the_default_period_is_fourteen(self):
        """Pinned so a later tidy-up cannot change what "RSI" means here.

        14 is Wilder's original and what every chart shows by default. If it
        moved to 10, nothing would fail, every stored result would shift, and
        comparisons against earlier runs would be quietly meaningless.

        The first version of this test compared `rsi(s)` against `rsi(s,
        period=14)` on a three-element series -- and passed with the constant
        changed to 10, because three bars are not enough for either period to
        produce a value, so it was comparing two all-NaN Series and `.equals`
        says those are the same. Sixty bars, so the two genuinely differ, and
        the constant asserted outright so the wiring and the value are both
        pinned rather than only their agreement.
        """
        rng = np.random.default_rng(3)
        prices = series(100 + rng.standard_normal(60).cumsum())

        assert ind.DEFAULT_RSI_PERIOD == 14
        assert_series_equal(ind.rsi(prices), ind.rsi(prices, period=14))
        assert not ind.rsi(prices).equals(ind.rsi(prices, period=10))
