"""Corrections for having gone looking, checked against known values.

The functions here only ever lower a number, so the tests are mostly about
whether they lower it by the right amount and in the right direction. The one
that matters most is that more trials make the bar higher: that monotonicity is
the entire reason the correction exists.
"""

import math
import random

import pytest

from signals.validation import (
    EULER_MASCHERONI,
    ValidationError,
    deflated_sharpe,
    expected_max_sharpe,
    probability_of_backtest_overfitting,
    sharpe,
)


def normal(n, mean=0.0, sd=1.0, seed=0):
    rng = random.Random(seed)
    return [rng.gauss(mean, sd) for _ in range(n)]


class TestSharpe:
    def test_it_is_mean_over_standard_deviation(self):
        assert sharpe([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5 / (5 / 3) ** 0.5, rel=1e-6)

    def test_annualising_is_square_root_of_time(self):
        r = normal(500, 0.001, 0.01)
        assert sharpe(r, periods_per_year=8760) == pytest.approx(sharpe(r) * math.sqrt(8760))

    def test_a_flat_series_has_no_sharpe(self):
        with pytest.raises(ValidationError, match="no variance"):
            sharpe([0.01] * 20)

    def test_too_few_points_to_judge(self):
        with pytest.raises(ValidationError, match="at least 4"):
            sharpe([0.01, 0.02])


class TestTheBarRisesWithTheNumberOfTrials:
    """The reason any of this exists."""

    def test_one_trial_needs_to_clear_nothing(self):
        assert expected_max_sharpe(1, sharpe_variance=1.0) == 0.0

    def test_more_trials_means_a_higher_bar(self):
        bars = [expected_max_sharpe(n, sharpe_variance=1.0) for n in (2, 10, 40, 1000)]
        assert bars == sorted(bars)
        assert all(b > 0 for b in bars)

    def test_forty_trials_costs_about_two_standard_errors(self):
        """The count this project has actually reached. Expressed in units of
        the spread of the trial Sharpes, the best of forty worthless strategies
        lands roughly two standard errors up on luck alone.
        """
        assert 1.5 < expected_max_sharpe(40, sharpe_variance=1.0) < 2.5

    def test_a_wider_spread_of_trial_results_raises_it_further(self):
        assert (expected_max_sharpe(40, sharpe_variance=4.0)
                > expected_max_sharpe(40, sharpe_variance=1.0))

    def test_the_variance_has_to_be_given_because_it_sets_the_units(self):
        """A Sharpe per observation and a Sharpe per year differ by a factor of
        eighty. A default would silently be right for one and meaningless for
        the other, which is how this function was wrong when first written.
        """
        with pytest.raises(TypeError):
            expected_max_sharpe(40)

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "10", True])
    def test_a_nonsense_trial_count_is_refused(self, bad):
        with pytest.raises(ValidationError):
            expected_max_sharpe(bad, sharpe_variance=1.0)


class TestDeflatedSharpe:
    def test_a_strong_result_found_in_one_try_survives(self):
        strong = normal(2000, 0.10, 1.0, seed=1)
        assert deflated_sharpe(strong, trials=1) > 0.95

    def test_the_scale_is_the_estimator_not_an_assumed_one(self):
        """The bar is built from the same variance the test divides by, so the
        two cannot disagree about units. Without this every input saturated at
        zero and the function looked merely harsh rather than broken.
        """
        returns = normal(1000, 0.05, 1.0, seed=21)
        assert 0.0 < deflated_sharpe(returns, trials=1) < 1.0
        assert 0.0 < deflated_sharpe(returns, trials=40) < 1.0

    def test_the_same_result_found_after_many_tries_does_not(self):
        """Identical returns, identical Sharpe, different conclusion -- because
        the number of things tried is part of what the number means.
        """
        returns = normal(2000, 0.06, 1.0, seed=2)
        alone = deflated_sharpe(returns, trials=1)
        searched = deflated_sharpe(returns, trials=500)
        assert alone > searched
        assert searched < 0.5

    def test_it_falls_monotonically_as_trials_rise(self):
        returns = normal(2000, 0.08, 1.0, seed=3)
        values = [deflated_sharpe(returns, trials=n) for n in (1, 5, 25, 100, 1000)]
        assert values == sorted(values, reverse=True)

    def test_a_worthless_strategy_does_not_survive_even_one_trial(self):
        assert deflated_sharpe(normal(2000, 0.0, 1.0, seed=4), trials=1) < 0.95

    def test_negative_skew_and_fat_tails_are_penalised(self):
        """A strategy that sells tail risk shows a fine Sharpe until it does
        not, and the plain ratio cannot see that at all.
        """
        clean = normal(4000, 0.06, 1.0, seed=5)
        nasty = list(clean)
        for index in (10, 500, 1200, 2600):
            nasty[index] = -9.0     # a few catastrophes, similar mean
        assert deflated_sharpe(nasty, trials=10) < deflated_sharpe(clean, trials=10)

    def test_a_longer_record_earns_more_confidence(self):
        short = normal(300, 0.10, 1.0, seed=6)
        long = normal(6000, 0.10, 1.0, seed=6)
        assert deflated_sharpe(long, trials=20) > deflated_sharpe(short, trials=20)

    def test_it_returns_a_probability(self):
        value = deflated_sharpe(normal(500, 0.05, 1.0, seed=7), trials=10)
        assert 0.0 <= value <= 1.0


class TestProbabilityOfBacktestOverfitting:
    def test_a_consistent_winner_looks_like_an_edge(self):
        """Configuration 0 is best in every split, so it never ranks low
        elsewhere and the estimate sits near zero.
        """
        splits = [[9.0, 1.0, 2.0, 3.0]] * 4
        assert probability_of_backtest_overfitting(splits) == pytest.approx(0.0)

    def test_a_winner_that_changes_every_split_looks_like_noise(self):
        """Each split crowns a different configuration and the previous winner
        falls to the bottom, which is what pure selection looks like.
        """
        splits = [
            [9.0, 1.0, 1.0, 1.0],
            [1.0, 9.0, 1.0, 1.0],
            [1.0, 1.0, 9.0, 1.0],
            [1.0, 1.0, 1.0, 9.0],
        ]
        assert probability_of_backtest_overfitting(splits) > 0.5

    def test_it_needs_something_to_compare(self):
        with pytest.raises(ValidationError, match="at least 2 splits"):
            probability_of_backtest_overfitting([[1.0, 2.0]])

    def test_ragged_splits_are_refused(self):
        with pytest.raises(ValidationError, match="same"):
            probability_of_backtest_overfitting([[1.0, 2.0], [1.0]])


class TestAgainstThisProjectsOwnNumbers:
    """Applied to what has actually been measured here."""

    def test_forty_trials_is_the_relevant_bar(self):
        """docs/going-live-criteria.md names ~40 configurations. This is what
        that costs, expressed in standard errors of the trial Sharpes.
        """
        bar = expected_max_sharpe(40, sharpe_variance=1.0)
        assert bar > expected_max_sharpe(1, sharpe_variance=1.0)
        assert bar > 1.0

    def test_a_mediocre_edge_does_not_survive_this_projects_trial_count(self):
        """The measured mean is about +0.23% per trade against wide dispersion.
        A Sharpe that modest, found after forty tries, should not clear the bar
        -- and if the function said otherwise it would be useless here.
        """
        returns = normal(400, 0.03, 1.0, seed=11)
        assert deflated_sharpe(returns, trials=40) < 0.95
