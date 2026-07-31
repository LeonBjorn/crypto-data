"""Tests for the lookahead guard.

Everything else in this project is tested by asking whether it gives the right
answer. This module is tested by asking whether it *notices a wrong one*, which
means most of the file is rules that cheat on purpose. If the guard cannot catch
the cheats below, it is worse than having no guard, because it produces a line
of output saying the rule is honest.

The cheats are the real ones, not strawmen. Every one of them is a mistake that
gets made: a shifted series pointed the wrong way, a centred rolling window
(pandas will happily give you one), a threshold taken from the maximum or mean
of the whole series, a comparison against the last price in the file. All four
produce backtests that look excellent and cannot be traded, and none of them
looks wrong when you read the line.

The last class in the file is the one I care about most. It measures what the
guard actually covers rather than what it claims, by hiding a lookahead at a
single bar and checking which settings find it.
"""

import numpy as np
import pandas as pd
import pytest

from signals import lookahead, rules

HOUR = 3_600_000
T0 = 1_722_470_400_000


def walk(count, seed=0, start=100.0):
    """A candle frame following a random walk, with sane OHLC."""
    rng = np.random.default_rng(seed)
    close = start + rng.standard_normal(count).cumsum()
    spread = abs(rng.standard_normal(count)) + 0.1
    return pd.DataFrame(
        {
            "timestamp": [T0 + i * HOUR for i in range(count)],
            "open": close,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": 10.0 + abs(rng.standard_normal(count)),
        }
    )


def booleans(values, index):
    return pd.Series(np.asarray(values, dtype=bool), index=index)


def fired_at(result):
    return [int(i) for i in np.flatnonzero(result.to_numpy())]


# Small budgets so the suite stays quick. The library defaults are larger and
# are exercised once, in TestTheDefaultBudget.
QUICK = {"cuts": 10, "dense_through": 60, "perturbations": 10}


# --- rules that cheat ------------------------------------------------------


def peeks_at_the_next_bar(candles):
    """The perfect strategy: buy whenever the next bar closes higher.

    Nobody writes this deliberately. People write `.shift(-1)` when they meant
    `.shift(1)`, which is the same thing.
    """
    return booleans(
        (candles["close"].shift(-1) > candles["close"]).to_numpy(), candles.index
    )


def uses_a_centred_window(candles, window=11):
    """Compares each close against an average centred on it.

    Half that average is made of bars that had not happened yet. pandas offers
    `center=True` as a plain keyword and it reads perfectly naturally.
    """
    middle = candles["close"].rolling(window, center=True, min_periods=window).mean()
    return booleans((candles["close"] > middle).to_numpy(), candles.index)


def compares_against_the_all_time_high(candles):
    """Fires near the highest price in the file -- a high that, at the time,
    nobody could have known was the highest.
    """
    return booleans(
        (candles["close"] >= candles["close"].max() * 0.995).to_numpy(), candles.index
    )


def compares_against_the_whole_series_mean(candles):
    """Normalising against a statistic of the entire series. This is the one
    that hides best, because the number it uses looks like a property of the
    market rather than a property of the file.
    """
    return booleans(
        (candles["close"] < candles["close"].mean() * 0.98).to_numpy(), candles.index
    )


def compares_against_the_final_close(candles):
    return booleans(
        (candles["close"] < candles["close"].iloc[-1]).to_numpy(), candles.index
    )


CHEATS = [
    pytest.param(peeks_at_the_next_bar, id="next-bar"),
    pytest.param(uses_a_centred_window, id="centred-window"),
    pytest.param(compares_against_the_all_time_high, id="all-time-high"),
    pytest.param(compares_against_the_whole_series_mean, id="series-mean"),
    pytest.param(compares_against_the_final_close, id="final-close"),
]


# --- rules that are honest but badly behaved -------------------------------


def never_fires(candles):
    return booleans(np.zeros(len(candles), dtype=bool), candles.index)


def rolls_a_die(candles):
    """Not a lookahead, but it makes every result unreproducible, and a guard
    that only checked causality would pass it.
    """
    return booleans(
        np.random.default_rng().random(len(candles)) > 0.9, candles.index
    )


def scribbles_on_the_candles(candles):
    """Writes a working column into the frame it was given. The caller's next
    rule then sees a frame it did not build.
    """
    candles["working"] = candles["close"].rolling(5, min_periods=5).mean()
    return booleans(
        (candles["close"] > candles["working"]).to_numpy(), candles.index
    )


def honest_ma_cross(candles):
    return rules.ma_cross(candles, fast=5, slow=20)


class TestItCatchesCheating:
    @pytest.mark.parametrize("cheat", CHEATS)
    def test_a_rule_that_reads_the_future_is_reported(self, cheat):
        report = lookahead.check(walk(400, seed=1), cheat, **QUICK)
        assert not report.ok
        assert report.findings

    @pytest.mark.parametrize("cheat", CHEATS)
    def test_the_finding_says_which_bar_gave_it_away(self, cheat):
        """"This rule looks ahead" is not enough to act on. The bar number is
        what turns the report into somewhere to put a breakpoint.
        """
        report = lookahead.check(walk(400, seed=1), cheat, **QUICK)
        assert any(finding.bar is not None for finding in report.findings)

    @pytest.mark.parametrize("cheat", CHEATS)
    def test_the_summary_names_the_rule_and_says_it_failed(self, cheat):
        report = lookahead.check(walk(400, seed=1), cheat, **QUICK)
        summary = report.summary()
        assert cheat.__name__ in summary
        assert "looked ahead" in summary

    @pytest.mark.parametrize("cheat", CHEATS)
    def test_assert_causal_raises_for_it(self, cheat):
        with pytest.raises(lookahead.LookaheadError):
            lookahead.assert_causal(walk(400, seed=1), cheat, **QUICK)

    def test_truncation_alone_would_have_caught_these(self):
        """Recorded so the two halves of the guard can be told apart. Every
        cheat above changes its answer when the history is cut short.
        """
        for cheat in [p.values[0] for p in CHEATS]:
            report = lookahead.check(
                walk(400, seed=1), cheat, perturbations=0, **{k: v for k, v in QUICK.items() if k != "perturbations"}
            )
            assert not report.ok, cheat.__name__

    def test_perturbation_alone_would_also_have_caught_most(self):
        """Perturbation is the cheaper half and the one that generalises: it
        asks whether the future has any influence at all, rather than whether
        one particular split agrees.
        """
        caught = 0
        for cheat in [p.values[0] for p in CHEATS]:
            report = lookahead.check(
                walk(400, seed=1), cheat, cuts=0, dense_through=0, perturbations=10
            )
            caught += not report.ok
        assert caught >= 4


class TestItPassesHonestRules:
    @pytest.mark.parametrize("name", sorted(rules.RULES))
    def test_every_registered_rule_is_causal(self, name):
        """The point of the whole module: a rule added to the registry later is
        guarded by this test without anyone remembering to write one.
        """
        report = lookahead.check(walk(600, seed=2), name, **QUICK)
        assert report.ok, report.summary()

    @pytest.mark.parametrize("name", sorted(rules.RULES))
    def test_and_stays_causal_with_unusual_parameters(self, name):
        overrides = {"ma-cross": {"fast": 3, "slow": 7}, "rsi-oversold": {"period": 3}, "breakout": {"window": 4}}
        report = lookahead.check(walk(600, seed=3), name, **QUICK, **overrides[name])
        assert report.ok, report.summary()

    def test_a_bare_function_works_as_well_as_a_registered_name(self):
        """Guarding a rule should not require registering it first, or nobody
        will run the guard while they are still writing the rule.
        """
        assert lookahead.check(walk(400, seed=4), honest_ma_cross, **QUICK).ok

    def test_check_all_covers_every_registered_rule(self):
        reports = lookahead.check_all(walk(600, seed=5), **QUICK)
        assert set(reports) == set(rules.names())
        assert all(report.ok for report in reports.values())


class TestItRefusesToProveNothing:
    """The failure mode this project keeps running into, in a new place.

    Comparing two prefixes that are entirely False is an equality that always
    holds. A guard built only from those comparisons would report every rule as
    causal, including the cheats, and would do it in green.
    """

    def test_a_rule_that_never_fires_is_reported_as_inconclusive(self):
        report = lookahead.check(walk(400, seed=6), never_fires, **QUICK)
        assert not report.ok
        assert report.informative_cuts == 0
        assert any(finding.kind == "inconclusive" for finding in report.findings)

    def test_the_summary_says_so_rather_than_claiming_success(self):
        report = lookahead.check(walk(400, seed=6), never_fires, **QUICK)
        assert "nothing to compare" in report.summary()

    def test_too_little_history_to_warm_up_is_also_inconclusive(self):
        """Fifty bars cannot exercise a fifty-bar crossover. Saying "causal"
        here would be true and useless; the guard says it could not tell.
        """
        report = lookahead.check(walk(50, seed=7), "ma-cross", **QUICK)
        assert not report.ok
        assert any(finding.kind == "inconclusive" for finding in report.findings)

    def test_an_inconclusive_result_is_not_reported_as_a_lookahead(self):
        report = lookahead.check(walk(400, seed=6), never_fires, **QUICK)
        assert not any(finding.kind == "lookahead" for finding in report.findings)
        assert "looked ahead" not in report.summary()

    def test_a_real_rule_on_real_length_history_is_conclusive(self):
        report = lookahead.check(walk(600, seed=8), "rsi-oversold", **QUICK)
        assert report.informative_cuts > 0
        assert report.signals > 0

    def test_either_half_of_the_guard_on_its_own_still_counts(self):
        """Requiring both would make `cuts=0` report "I could not tell" about a
        rule it had in fact just tested ten times.
        """
        report = lookahead.check(
            walk(600, seed=8), "breakout", cuts=0, dense_through=0, perturbations=10
        )
        assert report.ok
        assert report.informative_cuts == 0
        assert report.informative_perturbations > 0

    def test_and_the_same_the_other_way_round(self):
        report = lookahead.check(
            walk(600, seed=8), "breakout", cuts=10, dense_through=60, perturbations=0
        )
        assert report.ok
        assert report.perturbations_checked == 0
        assert report.informative_cuts > 0


class TestOtherWaysARuleCanBeUntrustworthy:
    def test_a_rule_that_is_not_reproducible_is_reported(self):
        report = lookahead.check(walk(400, seed=9), rolls_a_die, **QUICK)
        assert not report.ok
        assert any(finding.kind == "unstable" for finding in report.findings)

    def test_a_rule_that_writes_to_the_candles_is_reported(self):
        """Guarded here rather than in rules.py because the damage is done to
        the *caller*: the next rule down the loop gets a frame with an extra
        column, or a changed one, and produces different signals for reasons
        nothing in its own file explains.
        """
        report = lookahead.check(walk(400, seed=10), scribbles_on_the_candles, **QUICK)
        assert not report.ok
        assert any(finding.kind == "mutates-input" for finding in report.findings)

    def test_the_candles_are_intact_after_the_guard_runs(self):
        """Including after a rule that scribbles: the guard hands out copies, so
        running it cannot be what breaks the next thing you do.
        """
        candles = walk(400, seed=10)
        before = candles.copy()
        lookahead.check(candles, scribbles_on_the_candles, **QUICK)
        pd.testing.assert_frame_equal(candles, before)

    def test_the_candles_are_intact_after_guarding_an_honest_rule_too(self):
        """The perturbations rewrite prices wholesale. Doing that to the frame
        the caller handed over -- rather than to a copy of it -- would leave
        every price in memory multiplied by a random number, and the next thing
        anyone measured would be wrong with no trace of why.
        """
        candles = walk(400, seed=10)
        before = candles.copy()
        lookahead.check(candles, "breakout", **QUICK)
        pd.testing.assert_frame_equal(candles, before)

    def test_a_rule_returning_the_wrong_length_is_reported(self):
        report = lookahead.check(
            walk(400, seed=11), lambda candles: pd.Series([True, False]), **QUICK
        )
        assert not report.ok
        assert any(finding.kind == "shape" for finding in report.findings)

    def test_and_is_reported_as_a_length_rather_than_as_an_index(self):
        """The index check catches this too, so the length check is redundant
        as a *check*. It is not redundant as a sentence: "returned 2 values for
        400 candles" is something to go and fix, and "the index does not match
        the candles" is something to go and read the source about.
        """
        report = lookahead.check(
            walk(400, seed=11), lambda candles: pd.Series([True, False]), **QUICK
        )
        assert any(
            "2 value(s) for 400 candle(s)" in finding.detail
            for finding in report.findings
        )

    def test_a_rule_returning_a_bare_array_is_reported(self):
        report = lookahead.check(
            walk(400, seed=11),
            lambda candles: np.zeros(len(candles), dtype=bool),
            **QUICK,
        )
        assert any(finding.kind == "shape" for finding in report.findings)

    def test_a_rule_returning_numbers_instead_of_booleans_is_reported(self):
        """0.0 and 1.0 compare and sum exactly like False and True, so this one
        would survive every other check in the file and only show up much later
        as a trade count that is somehow a float.
        """
        report = lookahead.check(
            walk(400, seed=11),
            lambda candles: pd.Series(
                (np.arange(len(candles)) % 10 == 0).astype("float64"),
                index=candles.index,
            ),
            **QUICK,
        )
        assert any(finding.kind == "shape" for finding in report.findings)

    def test_a_rule_returning_a_misaligned_index_is_reported(self):
        """Right length, right dtype, wrong labels. Positionally it looks
        perfect, which is exactly how it would get as far as being joined
        against the candles and silently producing NaN.
        """

        def misaligned(candles):
            return pd.Series(
                np.arange(len(candles)) % 10 == 0, index=candles.index + 5
            )

        report = lookahead.check(walk(400, seed=11), misaligned, **QUICK)
        assert any(finding.kind == "shape" for finding in report.findings)

    def test_a_rule_that_raises_is_reported_rather_than_crashing_the_guard(self):
        def explodes(candles):
            raise RuntimeError("kaboom")

        report = lookahead.check(walk(400, seed=12), explodes, **QUICK)
        assert not report.ok
        assert any("kaboom" in finding.detail for finding in report.findings)


class TestMistakesInTheCallingCodeAreNotFindings:
    """A rule that cannot be found and a frame that is not a frame are bugs in
    whoever called the guard. Reporting them as findings would put "this rule
    looks untrustworthy" next to a typo.
    """

    def test_an_unknown_rule_name_is_raised_not_reported(self):
        with pytest.raises(rules.RuleError):
            lookahead.check(walk(100), "ma_crossover", **QUICK)

    def test_something_that_is_not_a_candle_frame_is_raised_not_reported(self):
        with pytest.raises(ValueError):
            lookahead.check([1, 2, 3], "breakout", **QUICK)


class TestTheCutPoints:
    def test_they_are_sorted_and_unique_and_inside_the_series(self):
        points = lookahead.cut_points(1000, cuts=20, dense_through=60)
        assert points == sorted(set(points))
        assert points[0] >= 1
        assert points[-1] <= 1000

    def test_the_start_is_covered_bar_by_bar(self):
        """Warm-up boundaries are where off-by-ones live, and truncating near
        the start is cheap, so there is no reason to sample there.
        """
        points = lookahead.cut_points(1000, cuts=20, dense_through=60)
        assert set(range(1, 61)) <= set(points)

    def test_the_rest_is_spread_across_the_whole_series(self):
        points = [p for p in lookahead.cut_points(10_000, cuts=20, dense_through=60) if p > 60]
        assert len(points) >= 15
        assert max(points) > 9_000

    def test_a_short_series_is_checked_at_every_bar(self):
        assert lookahead.cut_points(40, cuts=20, dense_through=60) == list(range(1, 41))

    def test_thorough_means_every_bar(self):
        assert lookahead.cut_points(500, cuts=20, dense_through=60, thorough=True) == list(
            range(1, 501)
        )

    def test_asking_for_no_cuts_gives_none(self):
        assert lookahead.cut_points(500, cuts=0, dense_through=0) == []


class TestThePerturbation:
    """Why the future is rewritten the way it is.

    Named private functions are avoided everywhere else in this project, but
    the shape of this one is a claim about how much the guard proves, and it is
    not visible from any result the guard produces.
    """

    def test_it_reshapes_the_future_rather_than_merely_rescaling_it(self):
        """Every rule in the registry is scale-invariant. Multiplying the tail
        by one constant would therefore ask a much smaller question than it
        looks like -- only whether the future's price *level* leaks backwards,
        never whether its shape does.
        """
        candles = walk(200, seed=17)
        changed = lookahead._perturbed(candles, 100, np.random.default_rng(1))
        ratio = changed["close"].to_numpy()[100:] / candles["close"].to_numpy()[100:]
        assert ratio.std() > 0.1

    def test_it_leaves_everything_before_the_point_alone(self):
        """Otherwise the comparison would be against data the guard itself had
        changed, and every rule would look like it was reading the future.
        """
        candles = walk(200, seed=17)
        changed = lookahead._perturbed(candles, 100, np.random.default_rng(1))
        pd.testing.assert_frame_equal(changed.iloc[:100], candles.iloc[:100])

    def test_a_bar_still_looks_like_a_bar_afterwards(self):
        candles = walk(200, seed=17)
        changed = lookahead._perturbed(candles, 100, np.random.default_rng(1))
        assert (changed["high"] >= changed["close"]).all()
        assert (changed["low"] <= changed["close"]).all()


class TestTheReport:
    def test_it_records_how_much_work_was_actually_done(self):
        """A report that says "ok" without saying what it checked is a report
        that cannot be argued with. These are the numbers that make it possible
        to notice the guard was run on too little data.
        """
        report = lookahead.check(walk(600, seed=13), "breakout", **QUICK)
        assert report.bars == 600
        assert report.cuts_checked > 0
        assert report.perturbations_checked > 0
        assert report.informative_cuts > 0

    def test_the_summary_of_a_clean_run_says_what_was_compared(self):
        report = lookahead.check(walk(600, seed=13), "breakout", **QUICK)
        summary = report.summary()
        assert "breakout" in summary
        assert str(report.cuts_checked) in summary
        assert "no sign of lookahead" in summary

    def test_a_report_with_no_findings_is_ok(self):
        report = lookahead.check(walk(600, seed=13), "breakout", **QUICK)
        assert report.ok
        assert report.findings == []

    def test_findings_are_capped_so_a_broken_rule_does_not_print_forever(self):
        """A rule that looks ahead usually fails at nearly every cut. Printing
        four hundred identical lines buries the one useful number in it.
        """
        report = lookahead.check(walk(400, seed=14), peeks_at_the_next_bar, **QUICK)
        assert len(report.findings) <= lookahead.MAX_FINDINGS
        assert len(report.summary().splitlines()) < 30

    def test_but_it_says_how_many_it_dropped(self):
        """Capping silently would turn "this rule fails everywhere" into "this
        rule fails eight times", which reads like a much smaller problem.
        """
        report = lookahead.check(walk(400, seed=14), peeks_at_the_next_bar, **QUICK)
        assert report.findings_dropped > 0
        assert str(report.findings_dropped) in report.summary()


class TestWhatTheGuardActuallyCovers:
    """How well does sampling work when the cheat is rare?

    The honest answer is "not perfectly", and these tests pin down where the
    line is rather than letting the module imply it catches everything.
    """

    # Chosen by checking that the planted cheat actually changes the answer
    # there (see the first test below) and that no cut point or perturbation in
    # QUICK happens to land on it. Both parts matter and neither is obvious
    # from reading the number.
    HIDDEN_AT = 234

    def cheats_at_one_bar(self, at):
        def rule(candles):
            honest = rules.ma_cross(candles, fast=5, slow=20).to_numpy().copy()
            if at + 1 < len(candles):
                closes = candles["close"].to_numpy()
                honest[at] = bool(closes[at + 1] < closes[at])
            return booleans(honest, candles.index)

        rule.__name__ = f"cheats_at_bar_{at}"
        return rule

    @pytest.mark.parametrize("at", [40, HIDDEN_AT])
    def test_the_planted_cheat_really_does_change_the_answer(self, at):
        """Without this, the "can be missed" test below proves nothing.

        A cheat that happens to agree with the honest answer at that bar is not
        a cheat, and no guard could or should catch it -- so a passing report
        would say nothing about sampling. The first bar I picked, 237, was
        exactly that case: ma-cross is False there and so is the peeked-at
        comparison, and the whole class was green for the wrong reason.
        """
        candles = walk(400, seed=15)
        honest = rules.ma_cross(candles, fast=5, slow=20)
        planted = self.cheats_at_one_bar(at)(candles)
        assert not honest.equals(planted)
        assert fired_at(honest ^ planted) == [at]

    def test_a_single_cheating_bar_early_on_is_caught_by_the_dense_block(self):
        report = lookahead.check(walk(400, seed=15), self.cheats_at_one_bar(40), **QUICK)
        assert not report.ok

    def test_a_single_cheating_bar_in_the_middle_can_be_missed_by_sampling(self):
        """Not a bug -- a stated limit. Ten cut points over four hundred bars
        cannot notice a mistake at one of them, and pretending otherwise is
        what would make this module dangerous.
        """
        report = lookahead.check(
            walk(400, seed=15), self.cheats_at_one_bar(self.HIDDEN_AT), **QUICK
        )
        assert report.ok

    def test_but_thorough_catches_it(self):
        report = lookahead.check(
            walk(400, seed=15), self.cheats_at_one_bar(self.HIDDEN_AT), thorough=True
        )
        assert not report.ok

    def test_thorough_reports_the_exact_bar(self):
        report = lookahead.check(
            walk(400, seed=15), self.cheats_at_one_bar(self.HIDDEN_AT), thorough=True
        )
        assert any(finding.bar == self.HIDDEN_AT for finding in report.findings)

    def test_thorough_still_passes_an_honest_rule(self):
        """The check that stops the test above from being satisfied by a guard
        that simply fails everything in thorough mode.
        """
        assert lookahead.check(walk(400, seed=15), "breakout", thorough=True).ok


class TestTheDefaultBudget:
    """Run once with the shipped settings, so the defaults are known to work
    rather than merely known to exist.
    """

    def test_the_defaults_are_conclusive_on_a_realistic_series(self):
        report = lookahead.check(walk(1_200, seed=16), "ma-cross")
        assert report.ok
        assert report.informative_cuts > 10
        assert report.cuts_checked > 100
        assert report.perturbations_checked > 10

    def test_the_defaults_catch_the_obvious_cheat(self):
        assert not lookahead.check(walk(1_200, seed=16), peeks_at_the_next_bar).ok
