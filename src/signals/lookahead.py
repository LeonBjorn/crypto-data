"""The lookahead guard: does a rule read bars it could not have seen?

A backtest that reads the future is not a slightly optimistic backtest. It is a
number with no relationship to anything, produced by code that runs cleanly,
looks reasonable line by line, and gets better the worse the mistake is. Nothing
about it feels wrong from the inside, which is why it needs to be caught by a
machine rather than by reading.

Two questions, asked mechanically of any rule.

TRUNCATION
    Cut the history at bar `c`, run the rule on the first `c` bars alone, and
    compare against the first `c` answers from the full run. They must be
    identical. This is the strongest form of the question, because a prefix is
    exactly what a live system would have had.

PERTURBATION
    Take the whole history, replace every bar from `k` onward with wildly
    different prices, and compare the answers before `k`. They must be
    untouched. This asks something slightly different -- not "does the split
    agree" but "does the future have any influence at all" -- and it catches the
    rule that normalises against a statistic of the entire file, which is the
    cheat that hides best because the number it uses looks like a property of
    the market rather than a property of the file.

Both are needed. Truncation misses nothing in principle but is sampled in
practice; perturbation is cheaper per bar and generalises better.

WHAT THIS CANNOT DO
-------------------
It samples. Ten cut points over four hundred bars cannot find a mistake at one
particular bar, and the tests say so out loud rather than letting this module
imply otherwise. `thorough=True` checks every bar and is the thing to reach for
when a result looks too good.

It also cannot prove a rule causal by comparing two answers that are both empty.
A rule that never fires passes every comparison here trivially, so the guard
counts how many comparisons were *informative* and reports "I could not tell"
rather than "clean" when that count is zero. That distinction is the entire
reason this module returns a report instead of a boolean.

BEYOND LOOKAHEAD
----------------
Three other properties get checked while the rule is running anyway, because
each one breaks a downstream result in a way that looks like something else:
a rule that is not reproducible, a rule that writes to the frame it was handed,
and a rule that returns the wrong shape.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from . import rules as rules_module

__all__ = [
    "DEFAULT_CUTS",
    "DEFAULT_DENSE_THROUGH",
    "DEFAULT_PERTURBATIONS",
    "MAX_FINDINGS",
    "Finding",
    "LookaheadError",
    "Report",
    "assert_causal",
    "check",
    "check_all",
    "cut_points",
]


# Every bar up to here is truncated at, one at a time. Warm-up boundaries are
# where off-by-ones live and a short prefix is cheap to recompute, so there is
# no reason to sample near the start.
DEFAULT_DENSE_THROUGH = 150

# ...and this many cut points spread over everything after it.
DEFAULT_CUTS = 40

DEFAULT_PERTURBATIONS = 40

# A rule that looks ahead usually fails at nearly every cut. Four hundred
# identical lines bury the one useful number in them.
MAX_FINDINGS = 8

# Fixed, so that two runs of the guard on the same data give the same report.
# A guard whose verdict wobbles is one people learn to re-run until it is green.
PERTURBATION_SEED = 20_240_801

# The columns a perturbation rewrites. `timestamp` is left alone because a rule
# is allowed to read it and a scrambled clock would be a different complaint;
# `volume` is left alone so that a rule reading it is still tested against
# prices that moved underneath it.
PRICE_COLUMNS = ("open", "high", "low", "close")


class LookaheadError(AssertionError):
    """Raised by `assert_causal` when a rule does not pass the guard.

    AssertionError rather than ValueError, which is what the rest of the
    project raises, because nothing was wrong with the arguments. The claim
    being made is about the rule itself.
    """


@dataclass(frozen=True)
class Finding:
    """One thing the guard objected to, and where.

    `bar` is the part that matters. "This rule looks ahead" is not something
    anyone can act on; a bar number is somewhere to put a breakpoint.
    """

    kind: str
    detail: str
    bar: Optional[int] = None

    def __str__(self):
        where = "" if self.bar is None else f" at bar {self.bar}"
        return f"{self.kind}{where}: {self.detail}"


@dataclass(frozen=True)
class Report:
    """What the guard did and what it found.

    The counts are not decoration. A report that says "ok" without saying what
    it compared cannot be argued with, and the most likely way this module
    fails in practice is by being run on too little history and saying nothing
    about it.
    """

    rule: str
    bars: int
    signals: int
    cuts_checked: int
    informative_cuts: int
    perturbations_checked: int
    informative_perturbations: int
    findings: list = field(default_factory=list)
    findings_dropped: int = 0

    @property
    def ok(self):
        return not self.findings

    def _verdict(self):
        if any(finding.kind == "lookahead" for finding in self.findings):
            return "looked ahead"
        return "did not pass the guard"

    def summary(self):
        """One line if the rule passed, one line per objection if it did not."""
        did = (
            f"{self.cuts_checked} truncation(s), {self.informative_cuts} of them "
            f"informative, and {self.perturbations_checked} perturbation(s), over "
            f"{self.bars} bars carrying {self.signals} signal(s)"
        )
        if self.ok:
            return f"{self.rule}: no sign of lookahead -- {did}."
        lines = [f"{self.rule}: {self._verdict()} -- {did}."]
        lines.extend(f"    {finding}" for finding in self.findings)
        if self.findings_dropped:
            lines.append(
                f"    ...and {self.findings_dropped} more of the same, not shown."
            )
        return "\n".join(lines)


class _Findings:
    """A capped list that remembers how much it threw away."""

    def __init__(self):
        self.kept = []
        self.total = 0

    def add(self, kind, detail, bar=None):
        self.total += 1
        if len(self.kept) < MAX_FINDINGS:
            self.kept.append(Finding(kind, detail, bar))

    @property
    def dropped(self):
        return self.total - len(self.kept)


def cut_points(bars, cuts=DEFAULT_CUTS, dense_through=DEFAULT_DENSE_THROUGH, thorough=False):
    """The prefix lengths to re-run the rule on.

    Dense at the start and sampled afterwards. The shape of this list is the
    honest statement of what the guard covers: everything near the warm-up
    boundary, and a scattering of the rest.
    """
    bars = int(bars)
    if bars < 1:
        return []
    if thorough:
        return list(range(1, bars + 1))

    dense_end = min(int(dense_through), bars)
    points = set(range(1, dense_end + 1))
    if int(cuts) > 0 and dense_end < bars:
        spread = np.linspace(dense_end + 1, bars, int(cuts))
        points.update(int(round(float(value))) for value in spread)
    return sorted(points)


def _perturbation_points(bars, count):
    """The positions to rewrite the history from.

    The last bar is included deliberately: a rule comparing against the final
    close in the file is caught by that one and by no other.
    """
    if int(count) < 1 or bars < 2:
        return []
    spread = np.linspace(1, bars - 1, int(count))
    return sorted({int(round(float(value))) for value in spread})


def _perturbed(candles, at, rng):
    """A copy of `candles` with every price from `at` onward rewritten.

    A different multiplier for every bar, not one multiplier for all of them.
    Every rule in the registry is scale-invariant -- multiply every price by
    seven and a crossover, an RSI and a breakout all fire on exactly the same
    bars -- so rescaling the tail would leave its *shape* untouched and would
    only ask whether the future's price level leaks backwards. Per-bar factors
    ask whether anything about the future leaks backwards, which is the
    question. Mutation testing found this: replacing the whole thing with a
    single constant broke nothing.

    One factor per bar shared across all four prices, so open, high, low and
    close keep their order and the result still looks like candles rather than
    like noise a rule might reasonably refuse.
    """
    changed = candles.copy()
    factors = np.exp(rng.normal(0.0, 0.75, size=len(candles) - at))
    for column in PRICE_COLUMNS:
        if column in changed.columns:
            values = changed[column].to_numpy(dtype="float64", copy=True)
            values[at:] = values[at:] * factors
            changed[column] = values
    return changed


def _name_of(rule):
    if isinstance(rule, str):
        return rule
    return getattr(rule, "__name__", repr(rule))


def _call(rule, candles, params):
    if isinstance(rule, str):
        return rules_module.apply(rule, candles, **params)
    return rule(candles, **params)


def _shape_problem(result, candles):
    """Why `result` is not a usable answer for `candles`, or None."""
    if not isinstance(result, pd.Series):
        return f"returned {type(result).__name__}, not a pandas Series"
    if len(result) != len(candles):
        return f"returned {len(result)} value(s) for {len(candles)} candle(s)"
    if not result.index.equals(candles.index):
        return "returned a Series whose index does not match the candles"
    if result.dtype != bool:
        return f"returned dtype {result.dtype}, not bool"
    return None


def _empty_report(name, bars, findings):
    """A report for a rule that never got as far as being compared."""
    return Report(
        rule=name,
        bars=bars,
        signals=0,
        cuts_checked=0,
        informative_cuts=0,
        perturbations_checked=0,
        informative_perturbations=0,
        findings=findings.kept,
        findings_dropped=findings.dropped,
    )


def check(
    candles,
    rule,
    *,
    cuts=DEFAULT_CUTS,
    dense_through=DEFAULT_DENSE_THROUGH,
    perturbations=DEFAULT_PERTURBATIONS,
    thorough=False,
    **params,
):
    """Run `rule` over `candles` many times and report anything untrustworthy.

    `rule` is either a name in the registry or any callable taking a candle
    frame -- the second form on purpose, so that guarding a rule does not
    require registering it first. Nobody runs a guard that makes them do
    paperwork before they can use it.

    Every run gets its own copy of the frame, so this cannot be what breaks the
    next thing the caller does, even when the rule under test writes to it.
    """
    if not isinstance(candles, pd.DataFrame):
        raise ValueError(
            f"check expects a candle DataFrame, got {type(candles).__name__}"
        )
    name = _name_of(rule)
    if isinstance(rule, str):
        # Fails now, loudly, rather than becoming a finding. An unknown rule
        # name is a mistake in the calling code, not a property of a rule.
        rules_module.get(rule)

    bars = len(candles)
    findings = _Findings()
    pristine = candles.copy()

    handed = candles.copy()
    try:
        full = _call(rule, handed, params)
    except Exception as error:  # noqa: BLE001 -- reporting it is the job
        findings.add(
            "error",
            f"raised {type(error).__name__} on the full history: {error}",
        )
        return _empty_report(name, bars, findings)

    problem = _shape_problem(full, handed)
    if problem is not None:
        findings.add("shape", problem)
        return _empty_report(name, bars, findings)

    if not handed.equals(pristine):
        # Checked here rather than in rules.py because the damage is done to
        # the caller: the next rule down the loop gets a frame with an extra
        # column and produces different signals for reasons nothing in its own
        # file explains.
        findings.add(
            "mutates-input",
            "changed the candle frame it was given. Every later rule in the "
            "same loop then runs on data this one edited.",
        )
        return _empty_report(name, bars, findings)

    # A fresh copy rather than `handed`, which the check above has just shown to
    # be unchanged. Defensive: `equals` compares values, and a rule could in
    # principle leave something behind that it does not see.
    again = _call(rule, candles.copy(), params)
    if _shape_problem(again, candles) is not None or not full.equals(again):
        findings.add(
            "unstable",
            "gave two different answers for the same candles. Nothing below "
            "can be measured, because there is no single answer to measure.",
        )
        return _empty_report(name, bars, findings)

    full_array = full.to_numpy()
    informative_cuts = 0
    informative_perturbations = 0

    points = cut_points(bars, cuts=cuts, dense_through=dense_through, thorough=thorough)
    for point in points:
        prefix = candles.iloc[:point].copy()
        try:
            short = _call(rule, prefix, params)
        except Exception as error:  # noqa: BLE001
            findings.add(
                "error",
                f"raised {type(error).__name__} on the first {point} bar(s) "
                f"alone: {error}",
                bar=point - 1,
            )
            continue
        if _shape_problem(short, prefix) is not None:
            findings.add(
                "shape",
                f"on the first {point} bar(s) alone, "
                f"{_shape_problem(short, prefix)}",
                bar=point - 1,
            )
            continue

        expected = full_array[:point]
        got = short.to_numpy()
        if expected.any() or got.any():
            informative_cuts += 1
        differ = np.flatnonzero(expected != got)
        if len(differ):
            first = int(differ[0])
            findings.add(
                "lookahead",
                f"cutting the history at bar {point} changes the answer: "
                f"{bool(expected[first])} with the rest of the file present, "
                f"{bool(got[first])} without it.",
                bar=first,
            )

    rng = np.random.default_rng(PERTURBATION_SEED)
    places = _perturbation_points(bars, perturbations)
    for at in places:
        changed = _perturbed(candles, at, rng)
        try:
            moved = _call(rule, changed, params)
        except Exception as error:  # noqa: BLE001
            findings.add(
                "error",
                f"raised {type(error).__name__} when the bars from {at} onward "
                f"were changed: {error}",
                bar=at,
            )
            continue
        if _shape_problem(moved, changed) is not None:
            findings.add(
                "shape",
                f"with the bars from {at} onward changed, "
                f"{_shape_problem(moved, changed)}",
                bar=at,
            )
            continue

        expected = full_array[:at]
        got = moved.to_numpy()[:at]
        if expected.any() or got.any():
            informative_perturbations += 1
        differ = np.flatnonzero(expected != got)
        if len(differ):
            first = int(differ[0])
            findings.add(
                "lookahead",
                f"rewriting every bar from {at} onward changes the answer at "
                f"bar {first}, which came before any of them.",
                bar=first,
            )

    if informative_cuts == 0 and informative_perturbations == 0:
        findings.add(
            "inconclusive",
            "nothing to compare. The rule produced no signals anywhere the "
            "guard looked, so every comparison it made was between two empty "
            "answers, which is an equality that always holds. Run it on more "
            "history, or on parameters that fire.",
        )

    return Report(
        rule=name,
        bars=bars,
        signals=int(full_array.sum()),
        cuts_checked=len(points),
        informative_cuts=informative_cuts,
        perturbations_checked=len(places),
        informative_perturbations=informative_perturbations,
        findings=findings.kept,
        findings_dropped=findings.dropped,
    )


def assert_causal(candles, rule, **kwargs):
    """`check`, but raise if it did not come back clean.

    For use in tests and at the top of anything that is about to spend real
    effort on a rule's results. Returns the report when it passes, so the
    counts are still available to print.
    """
    report = check(candles, rule, **kwargs)
    if not report.ok:
        raise LookaheadError(report.summary())
    return report


def check_all(candles, **kwargs):
    """Guard every rule in the registry, keyed by name.

    The point of the module in one function: a rule added to the registry later
    is covered by this without anyone remembering to write a test for it.
    """
    return {
        name: check(candles, name, **kwargs) for name in rules_module.names()
    }
