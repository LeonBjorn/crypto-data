"""The risk layer: volatility, the weights it implies, and the limits.

Everything here is checked against numbers worked out by hand, in the style the
trade tests established. That is possible because none of this predicts
anything -- an EWMA of a known series has one right answer, and so does an
inverse-volatility weight over two known volatilities.

The tests that matter most are the causality ones. A volatility estimate that
folds in the bar it is about to size a trade on is the same lookahead the guard
in signals/ exists to catch, one layer further down, and it would be invisible:
the numbers would simply be a little too good.
"""

import math

import pytest

from paper.risk import (
    DEFAULT_LAMBDA,
    DrawdownGuard,
    EwmaVolatility,
    RiskError,
    RiskModel,
    expected_shortfall,
    lambda_for,
    max_drawdown,
)


class TestEwmaVolatility:
    def test_it_is_not_ready_before_it_has_seen_enough(self):
        vol = EwmaVolatility(min_observations=5)
        for _ in range(4):
            vol.observe(0.01)
        assert not vol.ready
        assert vol.sigma is None

    def test_and_is_ready_once_it_has(self):
        vol = EwmaVolatility(min_observations=5)
        for _ in range(5):
            vol.observe(0.01)
        assert vol.ready
        assert vol.sigma == pytest.approx(0.01)

    def test_a_constant_absolute_return_gives_that_volatility(self):
        """Seeded with the first squared return and fed the same magnitude
        forever, the recursion is a fixed point: sigma stays exactly there.
        """
        vol = EwmaVolatility(0.94, min_observations=3)
        for _ in range(50):
            vol.observe(-0.02)
        assert vol.sigma == pytest.approx(0.02)

    def test_it_is_computed_by_hand(self):
        """lam=0.5, returns 0.10 then 0.00.
        seed  var = 0.10^2 = 0.01
        next  var = 0.5*0.01 + 0.5*0 = 0.005 -> sigma = sqrt(0.005)
        """
        vol = EwmaVolatility(0.5, min_observations=1)
        vol.observe(0.10)
        assert vol.variance == pytest.approx(0.01)
        vol.observe(0.0)
        assert vol.sigma == pytest.approx(math.sqrt(0.005))

    def test_a_bigger_move_raises_it(self):
        calm = EwmaVolatility(0.94, min_observations=2)
        wild = EwmaVolatility(0.94, min_observations=2)
        for _ in range(30):
            calm.observe(0.005)
            wild.observe(0.05)
        assert wild.sigma > calm.sigma

    def test_it_ignores_a_missing_return_rather_than_poisoning_itself(self):
        vol = EwmaVolatility(0.94, min_observations=1)
        vol.observe(0.02)
        before = vol.variance
        vol.observe(float("nan"))
        vol.observe(None)
        assert vol.variance == before

    def test_annualising_is_square_root_of_time(self):
        vol = EwmaVolatility(0.94, min_observations=1)
        vol.observe(0.01)
        assert vol.annualised(8760) == pytest.approx(0.01 * math.sqrt(8760))

    def test_the_decay_is_rescaled_for_the_bar_size(self):
        """0.94 means "about a month" on daily bars. Carried across to hourly
        data unchanged it would mean about a day and a half -- the same constant
        describing a different model.
        """
        assert lambda_for(1) == pytest.approx(DEFAULT_LAMBDA)
        hourly = lambda_for(24)
        assert hourly > DEFAULT_LAMBDA
        assert hourly ** 24 == pytest.approx(DEFAULT_LAMBDA)


class TestInverseVolatilityWeights:
    def build(self, sigmas, *, min_obs=1):
        model = RiskModel(list(sigmas), bars_per_year=8760, min_observations=min_obs)
        for symbol, sigma in sigmas.items():
            for _ in range(60):
                model.observe(symbol, sigma)
        return model

    def test_a_quieter_asset_gets_more_weight(self):
        model = self.build({"CALM": 0.01, "WILD": 0.04})
        weights = model.weights()
        assert weights["CALM"] > weights["WILD"]

    def test_the_weights_are_exactly_inverse_to_volatility(self):
        """sigma 0.01 and 0.03 -> 1/0.01 : 1/0.03 = 3 : 1 -> 0.75 and 0.25."""
        model = self.build({"A": 0.01, "B": 0.03})
        weights = model.weights()
        assert weights["A"] == pytest.approx(0.75)
        assert weights["B"] == pytest.approx(0.25)

    def test_they_sum_to_one(self):
        model = self.build({"A": 0.01, "B": 0.02, "C": 0.05})
        assert sum(model.weights().values()) == pytest.approx(1.0)

    def test_equal_volatility_gives_equal_weight(self):
        model = self.build({"A": 0.02, "B": 0.02, "C": 0.02})
        for weight in model.weights().values():
            assert weight == pytest.approx(1 / 3)

    def test_it_falls_back_to_equal_weight_while_warming_up(self):
        """Not a placeholder. Equal weighting is the benchmark this literature
        struggles to beat, so it is a defensible answer rather than a stand-in.
        """
        model = RiskModel(["A", "B"], min_observations=50)
        model.observe("A", 0.01)
        assert model.weights() == {"A": 0.5, "B": 0.5}
        assert not model.ready


class TestVolatilityTargeting:
    def warmed(self, sigma_per_bar, **kwargs):
        model = RiskModel(["A"], bars_per_year=8760, min_observations=1, **kwargs)
        for _ in range(60):
            model.observe("A", sigma_per_bar)
        return model

    def test_with_no_target_the_scale_is_just_the_cap(self):
        model = self.warmed(0.01)
        assert model.scale() == pytest.approx(1.0)

    def test_a_forecast_above_target_scales_the_book_down(self):
        # per-bar 0.01 over 8760 bars -> annualised 0.936
        model = self.warmed(0.01, target_vol=0.40)
        assert model.portfolio_vol() == pytest.approx(0.01 * math.sqrt(8760))
        assert model.scale() == pytest.approx(0.40 / (0.01 * math.sqrt(8760)))
        assert model.scale() < 1.0

    def test_a_forecast_below_target_is_capped_by_leverage(self):
        """The cap is the real risk decision, not the target."""
        model = self.warmed(0.0005, target_vol=0.80, max_leverage=1.0)
        assert model.scale() == pytest.approx(1.0)
        levered = self.warmed(0.0005, target_vol=0.80, max_leverage=3.0)
        assert 1.0 < levered.scale() <= 3.0

    def test_an_unwarmed_model_never_sizes_up(self):
        """The dangerous direction. With no usable forecast the scale must not
        reach for the leverage cap, or the very first trades -- taken with the
        least information -- would be the largest.
        """
        model = RiskModel(["A"], target_vol=0.40, max_leverage=3.0, min_observations=50)
        assert model.scale() == pytest.approx(1.0)

    def test_portfolio_volatility_assumes_everything_moves_together(self):
        """The weighted sum, not the quadratic form. Stated as a test because it
        is a deliberate pessimism rather than a missing feature: the correlation
        estimate the smaller number needs is unstable and rises toward one in
        exactly the stress it was meant to soften.
        """
        model = RiskModel(["A", "B"], bars_per_year=8760, min_observations=1)
        for _ in range(60):
            model.observe("A", 0.01)
            model.observe("B", 0.01)
        annual = 0.01 * math.sqrt(8760)
        assert model.portfolio_vol() == pytest.approx(annual)  # 0.5*a + 0.5*a

    def test_notional_is_weight_times_scale_times_capital(self):
        model = RiskModel(["A", "B"], bars_per_year=8760, min_observations=1)
        for _ in range(60):
            model.observe("A", 0.01)
            model.observe("B", 0.03)
        expected = 10_000 * model.weights()["A"] * model.scale()
        assert model.notional_for("A", 10_000) == pytest.approx(expected)


class TestExpectedShortfall:
    def test_it_is_the_mean_of_the_worst_tail(self):
        """Twenty returns, the worst being -0.10. At beta=0.95 the tail is the
        worst 1 of 20, so ES is 0.10 reported as a positive loss.
        """
        returns = [-0.10] + [0.01] * 19
        assert expected_shortfall(returns, 0.95) == pytest.approx(0.10)

    def test_it_averages_across_the_whole_tail(self):
        """Ten returns at beta=0.90 -> tail is the worst 1. At 0.80 -> worst 2,
        so the answer becomes the mean of -0.10 and -0.06.
        """
        returns = [-0.10, -0.06] + [0.02] * 8
        assert expected_shortfall(returns, 0.90) == pytest.approx(0.10)
        assert expected_shortfall(returns, 0.80) == pytest.approx(0.08)

    def test_a_portfolio_that_never_lost_reports_zero_not_a_gain(self):
        assert expected_shortfall([0.01] * 50, 0.95) == 0.0

    def test_it_is_worse_than_the_quantile_it_sits_beyond(self):
        """The whole reason to prefer it to VaR: VaR is where the tail starts,
        ES is how bad it is once you are in it.
        """
        returns = [-0.50, -0.20] + [0.01] * 18
        assert expected_shortfall(returns, 0.95) > 0.20

    def test_the_tail_size_does_not_move_with_float_representation(self):
        """1.0 - 0.80 is 0.19999999999999996. Without care the tail of a
        ten-observation sample floors to one instead of two, and the reported
        shortfall changes because of how a confidence level is stored.
        """
        returns = [-0.10, -0.06] + [0.02] * 8
        assert expected_shortfall(returns, 0.8) == pytest.approx(0.08)

    def test_no_target_volatility_is_a_valid_setting(self):
        """None means "do not target", which is the default, not a bad input."""
        assert RiskModel(["A"], target_vol=None).scale() == pytest.approx(1.0)

    def test_nothing_to_measure_is_not_an_error(self):
        assert expected_shortfall([], 0.95) is None


class TestDrawdownGuard:
    def test_it_allows_trading_while_above_the_limit(self):
        guard = DrawdownGuard(limit=0.25)
        assert guard.observe(10_000)
        assert guard.observe(9_000)
        assert not guard.tripped

    def test_it_trips_at_the_limit(self):
        guard = DrawdownGuard(limit=0.25)
        guard.observe(10_000)
        assert not guard.observe(7_500)
        assert guard.tripped

    def test_the_peak_is_what_it_measures_from_not_the_start(self):
        guard = DrawdownGuard(limit=0.25)
        guard.observe(10_000)
        guard.observe(20_000)
        assert guard.observe(16_000)      # -20% from peak
        assert not guard.observe(14_000)  # -30% from peak
        assert guard.tripped

    def test_it_stays_tripped_after_a_recovery(self):
        """Deliberate. A limit that un-trips as soon as the market bounces is a
        limit that does nothing in the case it exists for, and resuming should
        be a decision rather than a side effect of a good afternoon.
        """
        guard = DrawdownGuard(limit=0.25)
        guard.observe(10_000)
        guard.observe(7_000)
        assert not guard.observe(10_000)
        assert guard.tripped

    def test_resetting_is_explicit(self):
        guard = DrawdownGuard(limit=0.25)
        guard.observe(10_000)
        guard.observe(7_000)
        assert guard.reset().observe(9_500)

    def test_it_reports_the_current_fall(self):
        guard = DrawdownGuard(limit=0.5)
        guard.observe(10_000)
        guard.observe(8_000)
        assert guard.drawdown == pytest.approx(-0.20)


class TestMaxDrawdown:
    def test_a_rising_series_never_draws_down(self):
        assert max_drawdown([1, 2, 3, 4]) == 0.0

    def test_it_finds_the_worst_peak_to_trough(self):
        assert max_drawdown([100, 120, 60, 90]) == pytest.approx(-0.5)

    def test_it_is_peak_to_trough_not_first_to_last(self):
        assert max_drawdown([100, 200, 100, 150]) == pytest.approx(-0.5)


class TestItRefusesToBeConfiguredBadly:
    @pytest.mark.parametrize("bad", [0, -1, "x", True])
    def test_a_bad_target_volatility(self, bad):
        with pytest.raises(RiskError):
            RiskModel(["A"], target_vol=bad)

    @pytest.mark.parametrize("bad", [0, -1, "x", True])
    def test_a_bad_leverage_cap(self, bad):
        with pytest.raises(RiskError):
            RiskModel(["A"], max_leverage=bad)

    def test_a_lambda_above_one(self):
        with pytest.raises(RiskError):
            EwmaVolatility(1.5)

    def test_no_symbols(self):
        with pytest.raises(RiskError):
            RiskModel([])
