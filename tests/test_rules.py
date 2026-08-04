"""Tests for the rules -- turning indicator values into a decision to buy.

Two ideas are being tested here, and it is worth keeping them apart.

The first is the threshold: does "RSI oversold" mean what the name says. Those
tests use series short enough that the indicator values were computed in exact
rational arithmetic beforehand, so the expected firing bars are known
independently of any code in this project.

The second is the *edge*, and it is the one that will do damage if it is wrong.
A condition like "RSI below 30" can hold for ten bars in a row. A rule that
reports all ten reports one dip as ten opportunities, and every statistic
downstream -- trade count, hit rate, mean return -- then measures the same event
ten times while looking exactly like a larger sample. So a rule fires only where
the condition turns on, and turning on requires an earlier bar where the
condition was known and false.

That last clause is the subtle half. During warm-up the condition is not false,
it is unknown, and the difference matters at exactly one bar per series. If a
20/50 crossover rule fires on bar 49 merely because the fast average happens to
sit above the slow one the first time both exist, that is not a crossing; it is
an artefact of where the data begins, and it produces one spurious trade per
symbol that no amount of extra history will remove.
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from signals import indicators as ind
from signals import rules

HOUR = 3_600_000
T0 = 1_722_470_400_000  # 2024-08-01T00:00:00Z


def candles(closes, highs=None, lows=None, volumes=None):
    """A candle frame shaped like the store's, built around the closes given.

    `volume` defaults to a flat 10.0 on every bar, which is deliberately inert:
    the volume-reading rule cannot fire on a constant series, so the price rules
    can be tested without a volume column that means anything. The tests that
    care about volume pass their own with `volumes=`.
    """
    closes = list(closes)
    highs = list(highs) if highs is not None else [c + 1.0 for c in closes]
    lows = list(lows) if lows is not None else [c - 1.0 for c in closes]
    vols = list(volumes) if volumes is not None else [10.0] * len(closes)
    return pd.DataFrame(
        {
            "timestamp": [T0 + i * HOUR for i in range(len(closes))],
            "open": [float(c) for c in closes],
            "high": [float(h) for h in highs],
            "low": [float(low) for low in lows],
            "close": [float(c) for c in closes],
            "volume": [float(v) for v in vols],
        }
    )


def fired_at(result):
    """The integer positions where a rule fired, which is what tests assert on."""
    return [int(i) for i in np.flatnonzero(result.to_numpy())]


def random_walk(count, seed=0, start=100.0):
    """A price path with no structure, for the property tests.

    Volume tracks the size of each up-move so that the volume-confirmed breakout
    fires here rather than sitting silent on a flat series -- which is what lets
    the shared causality and shape checks actually exercise it. The price rules
    ignore volume, so their behaviour on these fixtures is unchanged.
    """
    rng = np.random.default_rng(seed)
    closes = start + rng.standard_normal(count).cumsum()
    highs = closes + abs(rng.standard_normal(count))
    # Volume is spiky and independent of the price move, so only some breakouts
    # land on a heavy bar. That keeps the volume-confirmed rule a strict subset
    # of plain breakout rather than a copy of it, which is what the rules-differ
    # test needs to see. The price rules ignore this column entirely.
    volumes = 10.0 + 300.0 * (rng.random(count) < 0.5)
    return candles(closes, highs=highs, volumes=volumes)


# Every rule, as (name, kwargs). The shared property tests iterate this, so a
# rule added later cannot quietly opt out of the causality and shape checks.
ALL_RULES = [
    pytest.param("ma-cross", {"fast": 2, "slow": 3}, id="ma-cross"),
    pytest.param("rsi-oversold", {"period": 2, "level": 50}, id="rsi-oversold"),
    pytest.param("breakout", {"window": 3}, id="breakout"),
    pytest.param("breakout-volume", {"window": 3, "volume_mult": 1.5}, id="breakout-volume"),
    pytest.param("breakdown-volume", {"window": 3, "volume_mult": 1.5}, id="breakdown-volume"),
    pytest.param(
        "breakout-volume-trend",
        {"window": 3, "volume_mult": 1.5, "trend": 4},
        id="breakout-volume-trend",
    ),
]


class TestMaCross:
    """Fast average rising through slow, with fast=2 and slow=3.

    Hand-computed. For [5, 4, 3, 4, 6] the two-bar average is
    [-, 4.5, 3.5, 3.5, 5.0] and the three-bar is [-, -, 4.0, 3.667, 4.333], so
    the fast average is below the slow one on the first two bars where both
    exist and above it on the last.
    """

    def test_it_fires_on_the_bar_the_fast_average_rises_through_the_slow(self):
        result = rules.apply("ma-cross", candles([5, 4, 3, 4, 6]), fast=2, slow=3)
        assert fired_at(result) == [4]

    def test_a_series_that_only_rises_never_crosses(self):
        """[1, 2, 3, 4, 5]: the fast average is above the slow one from the
        first bar where both exist, and stays there. There is no crossing --
        nothing rose through anything -- so a monotonic ramp must produce no
        signals at all. A rule that fires here is reporting the start of the
        data as an event.
        """
        result = rules.apply("ma-cross", candles([1, 2, 3, 4, 5]), fast=2, slow=3)
        assert fired_at(result) == []

    def test_it_fires_again_only_after_crossing_back_down(self):
        """[5, 4, 3, 4, 6, 6, 3, 2, 3, 5] crosses up at 4, back down at 6, and
        up again at 9. Two signals, not the six bars on which fast > slow.
        """
        prices = [5, 4, 3, 4, 6, 6, 3, 2, 3, 5]
        result = rules.apply("ma-cross", candles(prices), fast=2, slow=3)
        assert fired_at(result) == [4, 9]

    def test_the_averages_merely_meeting_is_not_a_cross(self):
        """[20, 20, 10, 8, 12]: at bar 4 the two-bar average is (8+12)/2 = 10
        and the three-bar average is (10+8+12)/3 = 10 as well. They touch
        exactly, having been apart on bar 3. Touching is not rising through --
        the fast average has not got above anything -- so nothing may fire.
        """
        prices = candles([20, 20, 10, 8, 12])
        close = prices["close"]
        assert ind.sma(close, 2).iloc[4] == ind.sma(close, 3).iloc[4] == 10.0
        assert ind.sma(close, 2).iloc[3] < ind.sma(close, 3).iloc[3]
        assert fired_at(rules.apply("ma-cross", prices, fast=2, slow=3)) == []

    def test_it_is_silent_through_the_warm_up(self):
        result = rules.apply("ma-cross", candles([5, 4, 3, 4, 6]), fast=2, slow=3)
        assert not result.iloc[:3].any()

    def test_the_averages_are_taken_on_the_close(self):
        """Not on the high, not on the open.

        The highs here are a different *shape*, not the closes plus a constant.
        That distinction is the whole test, and I got it wrong first time
        round: adding 50 to every high shifts both moving averages by 50, so
        their crossings land in exactly the same places and a rule reading the
        wrong column passes happily. The mutation harness is what noticed.
        """
        prices = [5, 4, 3, 4, 6, 6, 3, 2, 3, 5]
        shaped = candles(prices, highs=list(reversed(prices)))
        plain = rules.apply("ma-cross", candles(prices), fast=2, slow=3)
        assert fired_at(plain) == [4, 9]
        assert_series_equal(plain, rules.apply("ma-cross", shaped, fast=2, slow=3))

    def test_a_constant_shift_in_the_highs_is_not_enough_to_prove_that(self):
        """Kept as a warning next to the test above. The averages of the highs
        cross wherever the averages of the closes do when the two differ by a
        constant, so this passes whichever column the rule reads.
        """
        prices = [5, 4, 3, 4, 6, 6, 3, 2, 3, 5]
        highs = [p + 50 for p in prices]
        assert fired_at(rules.apply("ma-cross", candles(highs), fast=2, slow=3)) == [
            4,
            9,
        ]

    def test_a_fast_window_that_is_not_faster_is_refused(self):
        """fast == slow compares a series with itself, which is never greater,
        so the rule silently never fires -- the worst possible failure, because
        a backtest of zero trades looks like a rule that had no opportunities.
        """
        with pytest.raises(rules.RuleError, match="shorter"):
            rules.apply("ma-cross", candles([1, 2, 3, 4, 5]), fast=3, slow=3)

    def test_a_fast_window_longer_than_the_slow_one_is_refused(self):
        with pytest.raises(rules.RuleError, match="shorter"):
            rules.apply("ma-cross", candles([1, 2, 3, 4, 5]), fast=5, slow=2)


class TestRsiOversold:
    """RSI crossing down through a level, with period=2 and level=50.

    Expected RSI values below were computed in exact fractions before this rule
    existed. For [10, 11, 12, 11, 10, 12, 13] the RSI is
    [-, -, 100, 50, 25, 75, 85].
    """

    def test_it_fires_on_the_bar_the_rsi_drops_below_the_level(self):
        result = rules.apply(
            "rsi-oversold", candles([10, 11, 12, 11, 10, 12, 13]), period=2, level=50
        )
        assert fired_at(result) == [4]

    def test_the_level_is_strict(self):
        """Bar 3 of that series has an RSI of exactly 50.0. "Oversold" means
        below the level, not at it, so bar 3 must not count as the condition
        holding -- if it did, the signal would move from bar 4 to bar 3.
        """
        prices = candles([10, 11, 12, 11, 10, 12, 13])
        assert ind.rsi(prices["close"], period=2).iloc[3] == 50.0
        assert fired_at(rules.apply("rsi-oversold", prices, period=2, level=50)) == [4]

    def test_it_does_not_fire_when_the_first_readable_bar_is_already_below(self):
        """[12, 11, 10, 9, 12, 13, 11, 10] has an RSI of
        [-, -, 0, 0, 75, 83.3, 35.7, 22.7]. It is already under 50 on bar 2,
        the first bar where the RSI exists at all, but nothing dropped through
        anything there -- the series simply starts in a downtrend. The only
        real crossing is at bar 6.
        """
        result = rules.apply(
            "rsi-oversold",
            candles([12, 11, 10, 9, 12, 13, 11, 10]),
            period=2,
            level=50,
        )
        assert fired_at(result) == [6]

    def test_staying_oversold_is_not_a_second_signal(self):
        """Bars 6 and 7 of that series are both below 50. One dip, one signal."""
        result = rules.apply(
            "rsi-oversold",
            candles([12, 11, 10, 9, 12, 13, 11, 10]),
            period=2,
            level=50,
        )
        assert result.iloc[6]
        assert not result.iloc[7]

    def test_the_level_changes_the_answer(self):
        """A level of 30 is not reached by that series at bar 6 (RSI 35.7) but
        is at bar 7 (22.7), so lowering the level moves the signal later.
        """
        prices = candles([12, 11, 10, 9, 12, 13, 11, 10])
        assert fired_at(rules.apply("rsi-oversold", prices, period=2, level=30)) == [7]

    @pytest.mark.parametrize("level", [0, 100, -1, 101, 150.0])
    def test_a_level_outside_the_rsi_range_is_refused(self, level):
        """RSI cannot leave 0-100, so a level at or beyond either end is a rule
        that always fires or never does. Both are configuration mistakes that
        produce a running backtest rather than an error.
        """
        with pytest.raises(rules.RuleError, match="between 0 and 100"):
            rules.apply("rsi-oversold", candles([1, 2, 3, 4, 5]), period=2, level=level)

    @pytest.mark.parametrize("level", ["30", None, True])
    def test_a_level_that_is_not_a_number_is_refused(self, level):
        with pytest.raises(rules.RuleError, match="number"):
            rules.apply("rsi-oversold", candles([1, 2, 3, 4, 5]), period=2, level=level)


class TestBreakout:
    """Close above the highest high of the *previous* `window` bars."""

    def test_it_fires_when_the_close_clears_the_prior_high(self):
        """Highs [11, 11, 11, 11, 21] with a three-bar window: the prior-three
        high at bar 4 is 11, and the close of 20 clears it.
        """
        prices = candles([10, 10, 10, 10, 20], highs=[11, 11, 11, 11, 21])
        assert fired_at(rules.apply("breakout", prices, window=3)) == [4]

    def test_the_breaking_bar_does_not_help_set_the_level_it_has_to_clear(self):
        """The trap this rule exists to avoid. The bar that breaks out is,
        by definition, the highest bar around -- so if the rolling high includes
        the current bar, the level always rises to meet the price and the rule
        can never fire. Here the breaking bar's own high of 21 is above its
        close of 20, so an unshifted window gives zero signals forever while
        looking completely reasonable.
        """
        prices = candles([10, 10, 10, 10, 20], highs=[11, 11, 11, 11, 21])
        window = ind.rolling_high(prices["high"], 3)
        assert window.iloc[4] == 21.0  # includes the breaking bar
        assert prices["close"].iloc[4] < window.iloc[4]  # so this comparison fails
        assert fired_at(rules.apply("breakout", prices, window=3)) == [4]

    def test_continuing_higher_is_not_a_second_breakout(self):
        """Bar 5 closes at 30, above the prior-three high of 21, so the raw
        condition holds on bars 4 and 5. One breakout, one signal.
        """
        prices = candles([10, 10, 10, 10, 20, 30], highs=[11, 11, 11, 11, 21, 31])
        assert fired_at(rules.apply("breakout", prices, window=3)) == [4]

    def test_a_close_that_only_matches_the_prior_high_is_not_a_breakout(self):
        """Equal is not above. A close that stops exactly at resistance is the
        case the rule is meant to distinguish from one that goes through it.
        """
        prices = candles([10, 10, 10, 10, 11], highs=[11, 11, 11, 11, 12])
        assert fired_at(rules.apply("breakout", prices, window=3)) == []

    def test_it_does_not_fire_on_the_first_readable_bar(self):
        """Bar 3 of [10, 10, 10, 99, 5, 5, 5, 20] with highs
        [11, 11, 11, 100, 6, 6, 6, 21] is a real breakout -- 99 clears the
        prior high of 11 -- and it is deliberately discarded, because it is
        also the first bar where the rule can say anything, so there is no
        earlier bar it can be said to have changed from.

        This costs one genuine signal per symbol, at bar 20 of seventeen
        thousand. That is worth paying to have every rule in the project mean
        exactly the same thing by "fired", rather than having breakout carry a
        footnote that has to be remembered every time a result is read.
        """
        prices = candles([10, 10, 10, 99, 5, 5, 5, 20], highs=[11, 11, 11, 100, 6, 6, 6, 21])
        assert fired_at(rules.apply("breakout", prices, window=3)) == [7]

    def test_the_level_is_the_highest_high_and_not_the_lowest(self):
        """Every earlier fixture here has a flat run of highs in the window,
        where the rolling maximum and the rolling minimum are the same number,
        so swapping one for the other changed nothing. Here the prior window
        spans 30, 12 and 11, and a close of 20 sits between the two: above the
        low, below the high. A breakout must not fire.
        """
        prices = candles([9, 29, 11, 9, 20], highs=[10, 30, 12, 11, 21])
        assert ind.rolling_high(prices["high"], 3).shift(1).iloc[4] == 30.0
        assert ind.rolling_low(prices["high"], 3).shift(1).iloc[4] == 11.0
        assert fired_at(rules.apply("breakout", prices, window=3)) == []

    def test_it_is_silent_until_a_full_prior_window_exists(self):
        """A three-bar window needs three earlier bars, so the earliest bar that
        can possibly fire is bar 3, not bar 2.
        """
        prices = candles([10, 10, 10, 99, 99], highs=[11, 11, 11, 100, 100])
        assert not rules.apply("breakout", prices, window=3).iloc[:3].any()

    def test_the_level_is_taken_from_the_highs_and_the_break_from_the_close(self):
        """Raising the highs alone makes the level harder to clear, so the same
        closes must stop firing. This pins down which column feeds which side.
        """
        closes = [10, 10, 10, 10, 20]
        assert fired_at(
            rules.apply("breakout", candles(closes, highs=[11, 11, 11, 11, 21]), window=3)
        ) == [4]
        assert (
            fired_at(
                rules.apply(
                    "breakout", candles(closes, highs=[50, 50, 50, 50, 21]), window=3
                )
            )
            == []
        )


class TestBreakoutVolume:
    """The same breakout, gated by volume above a multiple of its own average.

    Every fixture below is the breakout fixture with a volume column bolted on,
    so what is being tested is only the extra condition: the price side is
    already pinned down by TestBreakout.
    """

    CLOSES = [10, 10, 10, 10, 20]
    HIGHS = [11, 11, 11, 11, 21]

    def frame(self, volumes):
        return candles(self.CLOSES, highs=self.HIGHS, volumes=volumes)

    def test_it_fires_when_the_breakout_bar_has_heavy_volume(self):
        """The prior-three volume average at bar 4 is 10, so a bar trading 100
        clears 1.5x it easily, and the breakout is confirmed.
        """
        prices = self.frame([10, 10, 10, 10, 100])
        assert fired_at(rules.apply("breakout-volume", prices, window=3)) == [4]

    def test_a_breakout_on_thin_volume_is_suppressed(self):
        """The whole point of the rule. Plain breakout fires here; the volume
        gate is what removes it, so the two rules must disagree on this bar.
        """
        prices = self.frame([10, 10, 10, 10, 10])
        assert fired_at(rules.apply("breakout", prices, window=3)) == [4]
        assert fired_at(rules.apply("breakout-volume", prices, window=3)) == []

    def test_volume_merely_equal_to_the_threshold_is_not_enough(self):
        """The threshold is 1.5 x 10 = 15, and the gate is strict. A bar trading
        exactly 15 has not exceeded average by half, it has reached the line.
        """
        prices = self.frame([10, 10, 10, 10, 15])
        assert fired_at(rules.apply("breakout-volume", prices, window=3)) == []

    def test_the_average_is_taken_from_the_bars_before_the_breakout(self):
        """A bar trading 16 clears 1.5x the prior average of 10. It would not
        clear 1.5x an average that counted itself -- mean(10, 10, 16) is 12 and
        1.5x that is 18 -- so this fixture fires only if the breaking bar is kept
        out of its own baseline, which is what the .shift(1) does.
        """
        prices = self.frame([10, 10, 10, 10, 16])
        assert ind.sma(prices["volume"], 3).shift(1).iloc[4] == 10.0
        assert fired_at(rules.apply("breakout-volume", prices, window=3)) == [4]

    def test_heavy_volume_without_a_breakout_is_not_a_signal(self):
        """Volume confirms a breakout; it does not manufacture one. A flat price
        that never clears its prior high must stay silent however much trades.
        """
        prices = candles([10, 10, 10, 10, 10], highs=[11, 11, 11, 11, 11],
                         volumes=[10, 10, 10, 10, 100])
        assert fired_at(rules.apply("breakout-volume", prices, window=3)) == []

    def test_it_fires_and_stays_causal_on_a_real_looking_series(self):
        """The shared causality tests run this rule on flat volume, where it
        never fires, so truncation proves nothing. Here volume tracks the size
        of each up-move, so the rule actually triggers -- and only then does
        checking that the past ignores the future have any teeth.
        """
        rng = np.random.default_rng(3)
        count = 80
        closes = 100 + rng.standard_normal(count).cumsum()
        highs = closes + abs(rng.standard_normal(count))
        moves = np.diff(closes, prepend=closes[0])
        volumes = 10.0 + 500.0 * np.clip(moves, 0.0, None)
        prices = candles(closes, highs=highs, volumes=volumes)

        full = rules.apply("breakout-volume", prices, window=5)
        assert full.any()
        for cut in (20, 40, 60):
            short = rules.apply("breakout-volume", prices.iloc[:cut].copy(), window=5)
            assert_series_equal(full.iloc[:cut], short)

    def test_a_volume_multiple_that_is_not_a_positive_number_is_refused(self):
        prices = self.frame([10, 10, 10, 10, 100])
        for bad in (0, -1, -0.5):
            with pytest.raises(rules.RuleError, match="greater than zero"):
                rules.apply("breakout-volume", prices, window=3, volume_mult=bad)
        for bad in ("1.5", None, True):
            with pytest.raises(rules.RuleError, match="must be a number"):
                rules.apply("breakout-volume", prices, window=3, volume_mult=bad)


class TestBreakoutVolumeTrend:
    """A volume-confirmed breakout, taken only while the close is in an uptrend.

    The filter is `close > sma(close, trend)`. The fixtures below break out on
    volume identically; what differs is whether the closes average out below the
    breaking bar (an uptrend, allowed) or above it (a downtrend, blocked).
    """

    # Breakout on a two-bar window at bar 5: the prior-two high is 7 and the
    # close of 8 clears it, on volume 100 against a prior average of 10. Whether
    # the trend filter lets it through is decided entirely by the early closes.
    HIGHS = [9, 9, 6, 6, 7, 9]
    VOLUMES = [10, 10, 10, 10, 10, 100]

    def frame(self, closes):
        return candles(closes, highs=self.HIGHS, volumes=self.VOLUMES)

    def test_the_underlying_breakout_fires_regardless_of_trend(self):
        """Both fixtures below share this breakout; the trend filter is the only
        thing that ever changes the answer.
        """
        for closes in ([8, 8, 5, 5, 6, 8], [50, 50, 5, 5, 6, 8]):
            got = fired_at(rules.apply("breakout-volume", self.frame(closes), window=2))
            assert got == [5]

    def test_a_breakout_in_an_uptrend_is_taken(self):
        """mean(8, 5, 5, 6, 8) = 6.4, and the close of 8 is above it, so the
        market counts as trending up and the breakout stands.
        """
        prices = self.frame([8, 8, 5, 5, 6, 8])
        assert ind.sma(prices["close"], 5).iloc[5] == pytest.approx(6.4)
        assert fired_at(rules.apply("breakout-volume-trend", prices, window=2, trend=5)) == [5]

    def test_the_same_breakout_in_a_downtrend_is_skipped(self):
        """The identical breakout, but the two early closes of 50 lift the
        five-bar average to 14.8, above the breaking close of 8. Price is below
        its trend, so the breakout is bought into a falling market and refused.
        """
        prices = self.frame([50, 50, 5, 5, 6, 8])
        assert ind.sma(prices["close"], 5).iloc[5] == pytest.approx(14.8)
        assert fired_at(rules.apply("breakout-volume", prices, window=2)) == [5]
        assert fired_at(rules.apply("breakout-volume-trend", prices, window=2, trend=5)) == []

    def test_it_is_a_subset_of_the_unfiltered_volume_breakout(self):
        """The filter can only ever remove signals, never add one. On any series,
        every trend-filtered breakout is also a plain volume breakout.
        """
        prices = random_walk(400, seed=21)
        filtered = rules.apply("breakout-volume-trend", prices, window=5, trend=20)
        unfiltered = rules.apply("breakout-volume", prices, window=5)
        assert filtered.any()
        assert set(fired_at(filtered)).issubset(fired_at(unfiltered))


class TestTheEdge:
    """The turns-on behaviour, stated directly rather than through one rule."""

    @pytest.mark.parametrize("name,kwargs", ALL_RULES)
    def test_a_rule_never_fires_on_two_consecutive_bars_without_relief(
        self, name, kwargs
    ):
        """Not a universal law of markets -- two genuine signals can be one bar
        apart -- but it is a law of *this* series: prices that only rise leave
        every condition either permanently on or permanently off, so nothing may
        fire more than once, and a rule returning the raw condition would light
        up on nearly every bar.
        """
        ramp = candles(np.arange(1, 41, dtype="float64"))
        result = rules.apply(name, ramp, **kwargs)
        assert result.sum() <= 1

    def test_a_flat_price_never_fires(self):
        """Nothing crosses, nothing drops, nothing breaks out. Every rule has
        to be silent on a price that does not move -- and a flat line is where
        an off-by-one in the edge logic shows up as a signal out of nowhere.
        """
        for name, kwargs in [p.values for p in ALL_RULES]:
            flat = candles([100.0] * 60)
            assert not rules.apply(name, flat, **kwargs).any(), name


class TestTheEdgeHelper:
    """The turns-on logic on its own, fed conditions by hand.

    Normally a private helper would be left to be tested through the callers,
    and I would rather not name an underscore function in a test. This one earns
    the exception: it is the single place the project's most important
    invariant is written down, every rule delegates to it, and the rules that
    exist today all warm up in a leading block -- so the cases where
    "unknown" appears anywhere else are unreachable through them, and stayed
    untested until the mutation harness pointed at the lines.

    The condition it takes is 1.0 for true, 0.0 for false, NaN for not yet
    known.
    """

    def condition(self, values):
        return pd.Series(values, dtype="float64")

    def test_a_condition_true_from_the_very_first_bar_never_fires(self):
        """There is no earlier bar for it to have become true from."""
        assert fired_at(rules._turns_true(self.condition([1, 1, 1]), "x")) == []

    def test_a_bar_whose_predecessor_is_unknown_does_not_fire(self):
        """Unknown is not false. The condition may well have been true through
        the whole warm-up; nothing changed at the moment we started looking.
        """
        assert fired_at(rules._turns_true(self.condition([0, np.nan, 1, 1]), "x")) == []

    def test_an_unknown_bar_does_not_itself_fire(self):
        assert (
            fired_at(rules._turns_true(self.condition([0, np.nan, 0, 1]), "x")) == [3]
        )

    def test_a_plain_false_then_true_fires_once(self):
        result = rules._turns_true(self.condition([0, 0, 1, 1, 0, 1]), "x")
        assert fired_at(result) == [2, 5]


class TestCausality:
    """A signal on bar t may depend on bars up to t and on nothing after."""

    @pytest.mark.parametrize("name,kwargs", ALL_RULES)
    @pytest.mark.parametrize("cut", [6, 11, 25, 40])
    def test_truncating_the_future_does_not_change_the_past(self, name, kwargs, cut):
        prices = random_walk(60, seed=7)
        full = rules.apply(name, prices, **kwargs)
        short = rules.apply(name, prices.iloc[:cut].copy(), **kwargs)
        assert_series_equal(full.iloc[:cut], short)

    @pytest.mark.parametrize("name,kwargs", ALL_RULES)
    def test_changing_a_future_bar_does_not_change_an_earlier_signal(self, name, kwargs):
        """The other half of causality, and the half truncation cannot see. A
        rule that normalised against the whole series -- "RSI below its own
        average" -- survives truncation happily and still cheats, because the
        average it compares against was computed with the future in it.
        """
        prices = random_walk(60, seed=11)
        meddled = prices.copy()
        meddled.loc[meddled.index[-1], ["open", "high", "low", "close"]] = 10_000.0
        assert_series_equal(
            rules.apply(name, prices, **kwargs).iloc[:-1],
            rules.apply(name, meddled, **kwargs).iloc[:-1],
        )


class TestSharedShape:
    """Everything a caller may assume about any rule's return value."""

    @pytest.mark.parametrize("name,kwargs", ALL_RULES)
    def test_it_returns_one_boolean_per_candle(self, name, kwargs):
        prices = random_walk(50, seed=3)
        result = rules.apply(name, prices, **kwargs)
        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert len(result) == len(prices)

    @pytest.mark.parametrize("name,kwargs", ALL_RULES)
    def test_it_keeps_the_index_it_was_given(self, name, kwargs):
        """Signals get lined up against candles later. If the index drifts, the
        alignment is off by however many bars and still looks like a Series.
        """
        prices = random_walk(50, seed=4)
        prices.index = pd.RangeIndex(500, 550)
        result = rules.apply(name, prices, **kwargs)
        assert result.index.equals(prices.index)

    @pytest.mark.parametrize("name,kwargs", ALL_RULES)
    def test_there_is_no_missing_value_anywhere(self, name, kwargs):
        """Warm-up is False, not NaN. A caller writing `if signal:` on a NaN
        gets True, which is precisely backwards.
        """
        result = rules.apply(name, random_walk(50, seed=5), **kwargs)
        assert not result.isna().any()

    @pytest.mark.parametrize("name,kwargs", ALL_RULES)
    def test_it_is_named_after_the_rule(self, name, kwargs):
        result = rules.apply(name, random_walk(30, seed=6), **kwargs)
        assert result.name == name

    @pytest.mark.parametrize("name,kwargs", ALL_RULES)
    def test_it_does_not_modify_the_candles_it_was_given(self, name, kwargs):
        prices = random_walk(50, seed=8)
        before = prices.copy()
        rules.apply(name, prices, **kwargs)
        assert_series_equal(prices["close"], before["close"])
        assert list(prices.columns) == list(before.columns)

    @pytest.mark.parametrize("name,kwargs", ALL_RULES)
    def test_a_frame_too_short_to_warm_up_gives_no_signals(self, name, kwargs):
        """Not an error. Asking for a rule over four bars is a reasonable thing
        for a caller to do near the start of a range; the honest answer is that
        nothing fired, and the caller finds out from the count.
        """
        result = rules.apply(name, candles([1, 2, 3, 4]), **kwargs)
        assert len(result) == 4
        assert not result.any()

    @pytest.mark.parametrize("name,kwargs", ALL_RULES)
    def test_an_empty_frame_gives_an_empty_result(self, name, kwargs):
        result = rules.apply(name, candles([]), **kwargs)
        assert len(result) == 0
        assert result.dtype == bool


class TestTheRegistry:
    """Looking a rule up by the name the command line will use."""

    def test_every_rule_is_registered_under_a_command_line_name(self):
        assert set(rules.names()) == {
            "ma-cross",
            "rsi-oversold",
            "breakout",
            "breakout-volume",
            "breakout-volume-trend",
            "breakdown-volume",
        }

    def test_the_names_are_sorted_so_help_text_is_stable(self):
        assert rules.names() == sorted(rules.names())

    def test_the_defaults_are_the_textbook_values(self):
        """Pinned deliberately. These are the conventional settings, which is
        the point: they are what someone else would have picked, so measuring
        them is a fair test rather than a number chosen to look good.
        """
        assert rules.get("ma-cross").defaults == {"fast": 20, "slow": 50}
        assert rules.get("rsi-oversold").defaults == {"period": 14, "level": 30}
        assert rules.get("breakout").defaults == {"window": 20}
        # 1.5 is a screening convention rather than a universal constant, but it
        # is still pinned: a change to it silently reprices every earlier result.
        assert rules.get("breakout-volume").defaults == {"window": 20, "volume_mult": 1.5}
        assert rules.get("breakout-volume-trend").defaults == {
            "window": 20,
            "volume_mult": 1.5,
            "trend": 200,
        }

    def test_the_defaults_are_what_apply_uses_when_nothing_is_passed(self):
        prices = random_walk(300, seed=2)
        for name in rules.names():
            assert_series_equal(
                rules.apply(name, prices),
                rules.apply(name, prices, **rules.get(name).defaults),
            )

    def test_calling_the_function_directly_uses_those_same_defaults(self):
        """The registry reads the defaults out of each signature, so this
        should be trivially true -- and it is exactly the kind of thing that
        stops being true the moment someone reintroduces a second copy.
        """
        prices = random_walk(300, seed=13)
        for name in rules.names():
            rule = rules.get(name)
            assert_series_equal(rule.function(prices), rules.apply(name, prices))

    def test_the_declared_columns_are_enough_to_run_the_rule(self):
        """The other half of the column contract. The test below proves each
        declared column is needed; this proves the list is complete, by handing
        the rule a frame with nothing else in it. A rule that quietly read
        `open` while declaring only `close` would pass the other test forever.
        """
        prices = random_walk(300, seed=14)
        for name in rules.names():
            rule = rules.get(name)
            trimmed = prices[list(rule.columns)].copy()
            assert_series_equal(rules.apply(name, trimmed), rules.apply(name, prices))

    def test_a_default_can_be_overridden_one_at_a_time(self):
        """Passing `fast` alone must leave `slow` at its default rather than
        dropping it, which is the usual way a partial-override helper breaks.
        """
        prices = random_walk(300, seed=9)
        assert_series_equal(
            rules.apply("ma-cross", prices, fast=10),
            rules.apply("ma-cross", prices, fast=10, slow=50),
        )
        assert not rules.apply("ma-cross", prices, fast=10).equals(
            rules.apply("ma-cross", prices)
        )

    def test_every_rule_has_a_one_line_description(self):
        for name in rules.names():
            description = rules.get(name).description
            assert description and not description.endswith(".")
            assert "\n" not in description

    def test_the_rules_actually_differ(self):
        """A registry is two names and one function away from a silent bug: the
        CLI would offer several rules, run one, and print several sets of results
        that agree suspiciously well.
        """
        prices = random_walk(400, seed=12)
        results = [rules.apply(name, prices) for name in rules.names()]
        for i, left in enumerate(results):
            for right in results[i + 1 :]:
                assert not left.to_numpy().tolist() == right.to_numpy().tolist()

    def test_an_unknown_rule_name_is_refused_and_lists_the_real_ones(self):
        with pytest.raises(rules.RuleError, match="ma-cross"):
            rules.apply("moving-average-crossover", candles([1, 2, 3]))

    def test_an_unknown_parameter_is_refused(self):
        """A typo in `--param windwo=10` must not be silently ignored, which
        would run the default and report it as the tuned result.
        """
        with pytest.raises(rules.RuleError, match="windwo"):
            rules.apply("breakout", candles([1, 2, 3, 4, 5]), windwo=3)

    def test_the_refusal_names_the_parameters_that_do_exist(self):
        with pytest.raises(rules.RuleError, match="window"):
            rules.apply("breakout", candles([1, 2, 3, 4, 5]), windwo=3)

    def test_the_function_can_also_be_called_directly(self):
        """The registry is for the command line. Nothing should require going
        through it, because a test that has to name a rule as a string cannot be
        found by a rename.
        """
        prices = random_walk(300, seed=10)
        assert_series_equal(
            rules.ma_cross(prices, fast=2, slow=3),
            rules.apply("ma-cross", prices, fast=2, slow=3),
        )


class TestCandleValidation:
    """What a rule refuses to be handed."""

    @pytest.mark.parametrize("name,kwargs", ALL_RULES)
    def test_something_that_is_not_a_frame_is_refused(self, name, kwargs):
        with pytest.raises(rules.RuleError, match="DataFrame"):
            rules.apply(name, [1, 2, 3, 4, 5], **kwargs)

    @pytest.mark.parametrize("name,kwargs", ALL_RULES)
    def test_a_series_of_prices_is_refused(self, name, kwargs):
        """A plausible mistake -- rules take candles, indicators take a Series."""
        with pytest.raises(rules.RuleError, match="DataFrame"):
            rules.apply(name, pd.Series([1.0, 2.0, 3.0]), **kwargs)

    @pytest.mark.parametrize("name,kwargs", ALL_RULES)
    def test_a_frame_missing_a_column_it_needs_is_refused(self, name, kwargs):
        prices = random_walk(40, seed=1)
        needed = rules.get(name).columns
        for column in needed:
            with pytest.raises(rules.RuleError, match=column):
                rules.apply(name, prices.drop(columns=[column]), **kwargs)

    @pytest.mark.parametrize("name,kwargs", ALL_RULES)
    def test_a_hole_in_the_prices_is_refused(self, name, kwargs):
        """`prices.load` will not hand out a frame with a hole in it, so this
        can only arrive from a caller that built one itself. Averaging across a
        missing bar produces a value that spans a discontinuity and looks
        entirely ordinary, so the refusal is worth keeping at both layers.
        """
        prices = random_walk(40, seed=1)
        # Every price column, because the rules between them read close, high
        # and low, and a hole punched in only some of those is not a hole.
        prices.loc[prices.index[20], ["close", "high", "low"]] = np.nan
        with pytest.raises(ind.IndicatorError):
            rules.apply(name, prices, **kwargs)

    @pytest.mark.parametrize("name,kwargs", ALL_RULES)
    @pytest.mark.parametrize("bad", [0, -1, 2.5, "3", None, True])
    def test_a_period_that_is_not_a_positive_whole_number_is_refused(
        self, name, kwargs, bad
    ):
        """Periods are validated by the indicators, so these raise
        IndicatorError rather than RuleError. Both are ValueError, and the
        message names the indicator that objected, which is enough to find it.
        """
        for parameter in kwargs:
            # level and volume_mult are the two fractional dials in the project;
            # 2.5 is a perfectly good value for either, so they are validated by
            # their own rules' tests rather than by this whole-number check.
            if parameter in ("level", "volume_mult"):
                continue
            broken = dict(kwargs, **{parameter: bad})
            with pytest.raises(ValueError):
                rules.apply(name, candles([1, 2, 3, 4, 5, 6, 7, 8]), **broken)
