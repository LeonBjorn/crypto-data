"""Evaluate: is this rule better than having no opinion at all?

The statistics in this file are arithmetic, and arithmetic is easy to test. The
part worth testing carefully is the comparison, because a benchmark can be
wrong in ways that never raise and never look wrong.

THE THREE WAYS A BENCHMARK QUIETLY LIES
---------------------------------------
*It is priced differently from the rule.* Charge the rule fees and let the
random entries trade free, or give one a stop and not the other, and the
comparison measures the settings rather than the signals. The defence here is
structural rather than diligent: `collect` takes the trade settings once and
uses the same ones for both, so the two cannot drift apart.

*It draws from bars the rule could never have used.* Trades near the end of the
file cannot finish, so they are dropped. If the random draws come from all bars
while the rule's come only from the ones with room, the random book quietly gets
fewer trades than the rule's, and two samples of different sizes are compared as
though they were the same. The population here is exactly the eligible bars.

*It pools symbols and loses track of which is which.* If a rule fired 300 times
on BTC and 50 on SOL, a random draw of 350 from a combined pool can land almost
entirely on SOL, and then the comparison is about which coins were picked rather
than when. So draws are stratified: each symbol contributes exactly as many
random trades as the rule took there.

WHAT A PERCENTILE IS AND IS NOT
-------------------------------
It says where the rule's mean sits among the means of random selections of the
same size, from the same bars, priced the same way. It is not a p-value and it
is not evidence of an edge -- one two-year sample, several rules and several
holding periods will produce a high percentile somewhere by chance alone. It is
the smallest honest thing that can be said, which is why it is what gets said.
"""

import numpy as np
import pandas as pd
import pytest

from signals import evaluate, trades

FIRST_MS = 1_700_000_000_000
HOUR_MS = 3_600_000


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


def ramp(count):
    """Prices 1, 2, 3, ... so every bar's forward return is known and distinct.

    A trade entered at bar `e` and left at bar `x` returns (x+1)/(e+1) - 1, and
    because the series never repeats a price, no two entry bars share a return.
    That is what makes "the rule picked the best bars" a statement a test can
    check rather than approximate.
    """
    return frame([(i + 1, i + 1, i + 1, i + 1) for i in range(count)])


def dipper(count=30):
    """A ramp with a hole punched in it, so a two percent stop has work to do."""
    rows = [(i + 100, i + 100, i + 100, i + 100) for i in range(count)]
    rows[8] = (108, 108, 90, 108)
    return frame(rows)


def signal_at(candles, *bars):
    fired = np.zeros(len(candles), dtype=bool)
    for bar in bars:
        fired[bar] = True
    return pd.Series(fired, index=candles.index)


def every_signal(candles):
    return pd.Series(np.ones(len(candles), dtype=bool), index=candles.index)


def returns(*values):
    return np.asarray(values, dtype="float64")


def part(taken, population, symbol="TEST"):
    return evaluate.Part(symbol=symbol, taken=returns(*taken), population=returns(*population))


class TestTheStatistics:
    def test_counts_the_trades(self):
        assert evaluate.stats_of(returns(0.1, -0.2, 0.3)).trades == 3

    def test_the_hit_rate_is_the_fraction_that_made_money(self):
        assert evaluate.stats_of(returns(0.1, -0.2, 0.3, -0.4)).hit_rate == pytest.approx(0.5)

    def test_breaking_exactly_even_is_not_a_win(self):
        # After costs a flat trade is a loss, and before costs it is a trade
        # that was not worth taking. Neither is a hit.
        assert evaluate.stats_of(returns(0.0, 0.0, 0.1)).hit_rate == pytest.approx(1 / 3)

    def test_the_mean_is_the_mean(self):
        assert evaluate.stats_of(returns(0.1, 0.2, 0.6)).mean == pytest.approx(0.3)

    def test_the_median_is_not_the_mean(self):
        # One huge winner is the classic way a mean flatters a rule, so both
        # numbers are reported and this test insists they can disagree.
        stats = evaluate.stats_of(returns(-0.01, -0.01, -0.01, -0.01, 1.0))
        assert stats.mean == pytest.approx(0.192)
        assert stats.median == pytest.approx(-0.01)

    def test_the_median_of_an_even_count_splits_the_middle(self):
        assert evaluate.stats_of(returns(0.0, 0.1, 0.3, 0.4)).median == pytest.approx(0.2)

    def test_the_best_and_the_worst(self):
        stats = evaluate.stats_of(returns(0.1, -0.5, 0.9, -0.2))
        assert stats.best == pytest.approx(0.9)
        assert stats.worst == pytest.approx(-0.5)

    def test_nothing_at_all_is_reported_rather_than_crashed_on(self):
        stats = evaluate.stats_of(returns())
        assert stats.trades == 0
        for value in (stats.hit_rate, stats.mean, stats.median, stats.best, stats.worst):
            assert np.isnan(value)

    def test_and_says_so_in_words(self):
        assert "no trades" in evaluate.stats_of(returns()).describe()

    def test_the_description_carries_every_number(self):
        described = evaluate.stats_of(returns(0.1, -0.2, 0.3, -0.4)).describe()
        assert "4 trade(s)" in described
        assert "50.0% hit" in described


class TestSummarisingABook:
    def test_it_reads_the_net_return_not_the_gross(self):
        # The whole point of charging costs is that the net number is the one
        # that gets reported. Reading gross here would make every rule look
        # about 0.3% better than it was, silently.
        candles = ramp(10)
        book = trades.round_trips(candles, signal_at(candles, 0), hold=2)
        stats = evaluate.summarise(book)
        assert stats.mean == pytest.approx(book.trades["net_return"].iloc[0])
        assert stats.mean != pytest.approx(book.trades["gross_return"].iloc[0])

    def test_a_book_with_no_trades_summarises_to_nothing(self):
        candles = ramp(10)
        book = trades.round_trips(candles, signal_at(candles, 8), hold=5)
        assert len(book.trades) == 0
        assert evaluate.summarise(book).trades == 0


class TestATradeAtEveryBar:
    def test_there_is_one_wherever_there_was_room(self):
        # 20 bars, 2-bar hold: a signal at bar b enters at b+1 and leaves at
        # b+3, so the last usable signal bar is 16. That is 17 trades.
        book = evaluate.every_bar(ramp(20), hold=2)
        assert len(book.trades) == 17
        assert list(book.trades["entry_bar"]) == list(range(1, 18))

    def test_and_none_where_there_was_not(self):
        book = evaluate.every_bar(ramp(20), hold=2)
        assert book.dropped == 3

    def test_a_longer_hold_leaves_less_room(self):
        assert len(evaluate.every_bar(ramp(20), hold=10).trades) == 9

    def test_it_is_priced_with_the_costs_it_was_given(self):
        free = evaluate.every_bar(ramp(20), hold=2, costs=trades.FREE)
        paid = evaluate.every_bar(ramp(20), hold=2)
        assert free.trades["net_return"].mean() > paid.trades["net_return"].mean()


class TestCollecting:
    def test_the_population_is_every_bar_that_had_room(self):
        candles = ramp(20)
        sample = evaluate.collect(candles, signal_at(candles, 3), hold=2)
        assert len(sample.population) == 17

    def test_the_rule_took_what_the_rule_took(self):
        candles = ramp(20)
        sample = evaluate.collect(candles, signal_at(candles, 3, 7), hold=2)
        assert len(sample.taken) == 2

    def test_the_taken_returns_are_the_population_at_those_bars(self):
        # The strongest statement this module makes: a rule adds nothing to a
        # trade except the decision to take it. If these ever disagree, either
        # the rule's trades are being priced differently from the benchmark's
        # or the population is not the thing the rule was choosing from --
        # and in both cases every percentile below is meaningless.
        candles = ramp(30)
        sample = evaluate.collect(candles, signal_at(candles, 0, 5, 11), hold=4)
        assert sample.taken == pytest.approx(sample.population[[0, 5, 11]])

    def test_the_rule_and_the_benchmark_are_charged_the_same_costs(self):
        candles = ramp(30)
        sample = evaluate.collect(candles, signal_at(candles, 5), hold=4, costs=trades.FREE)
        assert sample.taken == pytest.approx(sample.population[[5]])

    def test_the_rule_and_the_benchmark_get_the_same_stop(self):
        # A stop reaching only the rule's trades and not the benchmark's would
        # be invisible on a frame that never dips, so this one dips.
        candles = dipper()
        plain = evaluate.collect(candles, signal_at(candles, 0, 4), hold=10, costs=trades.FREE)
        stopped = evaluate.collect(
            candles, signal_at(candles, 0, 4), hold=10, stop=0.02, costs=trades.FREE
        )
        assert stopped.population != pytest.approx(plain.population)
        assert stopped.taken == pytest.approx(stopped.population[[0, 4]])

    def test_a_hold_too_long_for_the_file_is_refused(self):
        # Rather than returning an empty population, which would make every
        # comparison below vacuously pass.
        with pytest.raises(evaluate.EvalError, match="no room"):
            evaluate.collect(ramp(10), signal_at(ramp(10), 0), hold=50)

    def test_the_symbol_is_carried_along(self):
        candles = ramp(20)
        sample = evaluate.collect(candles, signal_at(candles, 3), hold=2, symbol="BTC/USDT")
        assert sample.parts[0].symbol == "BTC/USDT"

    def test_the_hold_is_carried_along(self):
        candles = ramp(20)
        assert evaluate.collect(candles, signal_at(candles, 3), hold=2).hold == 2


class TestPooling:
    def test_the_taken_trades_are_every_parts_taken_trades(self):
        pooled = evaluate.pool(
            [
                evaluate.Sample(parts=(part([0.1], [0.1, 0.2]),), hold=2),
                evaluate.Sample(parts=(part([0.3], [0.3, 0.4]),), hold=2),
            ]
        )
        assert pooled.taken == pytest.approx(returns(0.1, 0.3))

    def test_and_the_populations_likewise(self):
        pooled = evaluate.pool(
            [
                evaluate.Sample(parts=(part([0.1], [0.1, 0.2]),), hold=2),
                evaluate.Sample(parts=(part([0.3], [0.3, 0.4]),), hold=2),
            ]
        )
        assert pooled.population == pytest.approx(returns(0.1, 0.2, 0.3, 0.4))

    def test_the_parts_stay_separate(self):
        # Not cosmetic: the parts are what makes the random draws stratified.
        pooled = evaluate.pool(
            [
                evaluate.Sample(parts=(part([0.1], [0.1, 0.2], "BTC/USDT"),), hold=2),
                evaluate.Sample(parts=(part([0.3], [0.3, 0.4], "ETH/USDT"),), hold=2),
            ]
        )
        assert [p.symbol for p in pooled.parts] == ["BTC/USDT", "ETH/USDT"]

    def test_pooling_different_holding_periods_is_refused(self):
        with pytest.raises(evaluate.EvalError, match="hold"):
            evaluate.pool(
                [
                    evaluate.Sample(parts=(part([0.1], [0.1, 0.2]),), hold=2),
                    evaluate.Sample(parts=(part([0.3], [0.3, 0.4]),), hold=24),
                ]
            )

    def test_the_pool_keeps_the_holding_period(self):
        pooled = evaluate.pool(
            [
                evaluate.Sample(parts=(part([0.1], [0.1, 0.2]),), hold=24),
                evaluate.Sample(parts=(part([0.3], [0.3, 0.4]),), hold=24),
            ]
        )
        assert pooled.hold == 24

    def test_pooling_nothing_is_refused(self):
        with pytest.raises(evaluate.EvalError, match="nothing"):
            evaluate.pool([])


class TestTheRandomDraws:
    def sample(self, take, size):
        population = returns(*[i / 100 for i in range(size)])
        return evaluate.Sample(parts=(part(population[:take], population),), hold=2)

    def test_the_same_seed_gives_the_same_answer(self):
        first = evaluate.judge(self.sample(3, 20), draws=50, seed=7)
        second = evaluate.judge(self.sample(3, 20), draws=50, seed=7)
        assert first.means == pytest.approx(second.means)

    def test_a_different_seed_gives_a_different_answer(self):
        first = evaluate.judge(self.sample(3, 20), draws=50, seed=7)
        second = evaluate.judge(self.sample(3, 20), draws=50, seed=8)
        assert first.means != pytest.approx(second.means)

    def test_there_is_one_mean_per_draw(self):
        assert len(evaluate.judge(self.sample(3, 20), draws=50, seed=0).means) == 50

    def test_every_draw_is_the_size_the_rule_was(self):
        # Checked through the arithmetic: with a population of 0.00..0.19 and
        # a take of 4, the smallest possible mean is that of the four smallest
        # values and the largest that of the four largest. A draw of the wrong
        # size would land outside those bounds.
        verdict = evaluate.judge(self.sample(4, 20), draws=500, seed=0)
        assert verdict.means.min() >= (0.00 + 0.01 + 0.02 + 0.03) / 4
        assert verdict.means.max() <= (0.16 + 0.17 + 0.18 + 0.19) / 4

    def test_a_draw_never_takes_the_same_trade_twice(self):
        # A population of three distinct values, taking two. Without
        # replacement there are exactly three possible means; with
        # replacement there would be six. Five hundred draws will find them
        # all, so counting the distinct means settles it.
        population = returns(0.0, 0.3, 0.9)
        sample = evaluate.Sample(parts=(part([0.0, 0.3], population),), hold=2)
        means = evaluate.judge(sample, draws=500, seed=0).means
        assert sorted(set(np.round(means, 10))) == pytest.approx([0.15, 0.45, 0.6])

    def test_a_thousand_draws_unless_told_otherwise(self):
        # Pinned because the number is a claim about precision, not a taste.
        # Ten draws resolve a percentile only to the nearest ten points, and
        # the summary prints it to one decimal place -- four significant
        # figures for a number that has one.
        assert evaluate.DEFAULT_DRAWS == 1_000
        assert len(evaluate.judge(self.sample(3, 20)).means) == 1_000

    def test_the_draws_are_stratified_across_symbols(self):
        # One symbol whose every trade returned zero, another whose every trade
        # returned one. The rule took one from the first and three from the
        # second, so a stratified draw must always average exactly 0.75.
        # Drawing four from the combined pool instead would vary from 0 to 1.
        sample = evaluate.Sample(
            parts=(
                part([0.0], [0.0] * 40, "FLAT"),
                part([1.0, 1.0, 1.0], [1.0] * 40, "RICH"),
            ),
            hold=2,
        )
        means = evaluate.judge(sample, draws=200, seed=0).means
        assert means == pytest.approx(np.full(200, 0.75))


class TestThePercentile:
    def ramped(self, *bars, hold=2, count=40):
        candles = ramp(count)
        return evaluate.collect(candles, signal_at(candles, *bars), hold=hold)

    def test_a_rule_that_picked_the_best_bars_scores_high(self):
        # On a ramp the earliest bars have the largest forward returns, so
        # firing on bars 0, 1, 2 is a rule with perfect hindsight. No draw can
        # beat it. A draw can *tie* it, by picking those same three bars out of
        # the 7,770 possible triples, so the percentile lands just under 100
        # rather than exactly on it -- which is the honest answer, not an
        # off-by-one. Hence the two assertions rather than one.
        verdict = evaluate.judge(self.ramped(0, 1, 2), draws=1000, seed=0)
        assert (verdict.means <= verdict.rule.mean).all()
        assert verdict.percentile > 99.9

    def test_a_rule_that_picked_the_worst_bars_scores_low(self):
        verdict = evaluate.judge(self.ramped(34, 35, 36), draws=1000, seed=0)
        assert (verdict.means >= verdict.rule.mean).all()
        assert verdict.percentile < 0.1

    def test_a_rule_that_picked_every_bar_sits_exactly_in_the_middle(self):
        # If the rule fired everywhere then every random selection of the same
        # size is the whole population too, so all the draws tie with the rule.
        # A tie is not a win and it is not a loss, so it must land at 50.
        candles = ramp(40)
        sample = evaluate.collect(candles, every_signal(candles), hold=2)
        assert evaluate.judge(sample, draws=100, seed=0).percentile == pytest.approx(50.0)

    def test_a_middling_rule_lands_in_between(self):
        verdict = evaluate.judge(self.ramped(15, 17, 19), draws=1000, seed=0)
        assert 5.0 < verdict.percentile < 95.0


class TestTheVerdict:
    def verdict(self):
        candles = ramp(40)
        sample = evaluate.collect(candles, signal_at(candles, 2, 8, 20), hold=4, symbol="BTC/USDT")
        return evaluate.judge(sample, draws=100, seed=3)

    def test_it_reports_the_rules_own_statistics(self):
        assert self.verdict().rule.trades == 3

    def test_it_reports_the_whole_population_as_the_benchmark(self):
        assert self.verdict().baseline.trades == 35

    def test_it_remembers_how_it_was_asked(self):
        verdict = self.verdict()
        assert verdict.draws == 100
        assert verdict.seed == 3

    def test_the_summary_names_the_symbol_the_hold_and_the_percentile(self):
        summary = self.verdict().summary()
        assert "BTC/USDT" in summary
        assert "4-bar hold" in summary
        assert "percentile" in summary

    def test_the_summary_shows_both_sides(self):
        summary = self.verdict().summary()
        assert "rule" in summary
        assert "every bar" in summary

    def test_and_the_two_sides_carry_their_own_numbers(self):
        # The labels being present is not enough. Printing the benchmark on
        # both lines would satisfy the test above while showing a rule its own
        # result never appeared in.
        summary = self.verdict().summary()
        assert "rule       3 trade(s)" in summary
        assert "every bar  35 trade(s)" in summary

    def test_a_pooled_summary_names_all_of_its_symbols(self):
        candles = ramp(40)
        pooled = evaluate.pool(
            [
                evaluate.collect(candles, signal_at(candles, 2), hold=4, symbol="BTC/USDT"),
                evaluate.collect(candles, signal_at(candles, 8), hold=4, symbol="ETH/USDT"),
            ]
        )
        summary = evaluate.judge(pooled, draws=50, seed=0).summary()
        assert "BTC/USDT" in summary
        assert "ETH/USDT" in summary


class TestItRefusesRatherThanGuesses:
    def sample(self):
        candles = ramp(40)
        return evaluate.collect(candles, signal_at(candles, 2, 8), hold=4)

    def test_a_rule_that_never_fired_has_nothing_to_judge(self):
        candles = ramp(40)
        sample = evaluate.collect(candles, signal_at(candles, 38, 39), hold=4)
        assert len(sample.taken) == 0
        with pytest.raises(evaluate.EvalError, match="no trades"):
            evaluate.judge(sample)

    def test_no_draws_at_all(self):
        with pytest.raises(evaluate.EvalError, match="draws"):
            evaluate.judge(self.sample(), draws=0)

    def test_a_fractional_number_of_draws(self):
        with pytest.raises(evaluate.EvalError, match="draws"):
            evaluate.judge(self.sample(), draws=10.5)

    def test_draws_given_as_a_boolean(self):
        with pytest.raises(evaluate.EvalError, match="draws"):
            evaluate.judge(self.sample(), draws=True)

    def test_a_negative_seed(self):
        with pytest.raises(evaluate.EvalError, match="seed"):
            evaluate.judge(self.sample(), seed=-1)

    def test_a_fractional_seed(self):
        with pytest.raises(evaluate.EvalError, match="seed"):
            evaluate.judge(self.sample(), seed=0.5)

    def test_a_sample_that_took_more_trades_than_ever_existed(self):
        # Unreachable through `collect`, which builds both sides from the same
        # bars. It earns its place on the message: without it numpy says
        # "cannot take a larger sample than population", which in a pooled run
        # of five coins does not say which one.
        sample = evaluate.Sample(parts=(part([0.1, 0.2, 0.3], [0.1, 0.2], "BTC/USDT"),), hold=2)
        with pytest.raises(evaluate.EvalError, match="BTC/USDT"):
            evaluate.judge(sample)

    def test_something_that_is_not_a_sample(self):
        with pytest.raises(evaluate.EvalError, match="Sample"):
            evaluate.judge([0.1, 0.2])

    def test_something_that_is_not_a_book(self):
        with pytest.raises(evaluate.EvalError, match="Book"):
            evaluate.summarise([0.1, 0.2])

    def test_a_bad_hold_is_still_the_trade_modules_business(self):
        # collect does not re-validate what round_trips already validates, so
        # the error the user sees is the specific one, not a vaguer echo.
        candles = ramp(20)
        with pytest.raises(trades.TradeError):
            evaluate.collect(candles, signal_at(candles, 1), hold=0)
