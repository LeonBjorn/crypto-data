"""Evaluate: deciding whether a rule was better than having no opinion.

Increment 4 can tell you that breakout signals held for a week returned +0.36%
on average. That sentence is not yet a finding, because over the same two years
Bitcoin roughly doubled -- so a coin flip held for a week would also have made
money. A number with no benchmark is not a small result, it is no result.

This module supplies the benchmark.

THE TWO COMPARISONS
-------------------
*Every bar.* Take the trade at every candle the rule could have used, with the
same hold, the same stop and target, and the same costs. That is what not being
selective returns. It has no sampling noise at all: it is the whole population,
not an estimate of it.

*A thousand random selections.* Pick as many entries as the rule picked, at
random, from those same bars, and take the mean. Do it a thousand times and you
have the distribution of results that selection-by-luck produces at exactly this
sample size. The rule's percentile within that distribution is how unusual its
result was.

Both are needed and neither replaces the other. "Every bar" answers *was there
anything here at all*; the draws answer *and is this rule's version of it
bigger than chance*. A rule can beat the first and lose to the second, which is
the ordinary case for a rule that fired only two hundred times.

WHY THE DRAWS ARE SAMPLED RATHER THAN RE-TRADED
-----------------------------------------------
A thousand draws does not mean a thousand backtests. The trades in this project
are independent -- no shared wallet, no position sizing, and overlapping trades
are all kept -- so the return of a trade entered at bar 4,201 is the same number
whether or not anything else was open at the time. That means the every-bar book
already contains the outcome of every entry the random draws could possibly
make, and a random selection of K entries is a random selection of K rows from
it. Sampling is not an approximation of re-running; it is the same arithmetic,
in milliseconds instead of minutes.

This shortcut is exactly as valid as the independence assumption it rests on. If
this project ever grows a single account with finite cash, trades stop being
independent and this stops being correct -- so it is written down here rather
than left to be rediscovered.

WHERE THE DRAWS COME FROM
-------------------------
Two details, both places where a benchmark goes wrong quietly rather than
loudly.

The population is the bars that had *room* -- a signal too near the end of the
file cannot finish its hold and never becomes a trade. Drawing from all bars
instead would produce random books slightly smaller than the rule's, and
comparing a mean of 611 trades against a mean of 590 is a comparison of two
different questions.

The draws are *stratified by symbol*. If a rule fired 300 times on BTC and 50 on
SOL, an unstratified draw of 350 from the combined pool could land almost
entirely on SOL. The comparison would then be partly about which coins the rule
chose, which is a different claim from the one being tested. Each symbol
contributes exactly as many random trades as the rule took there, so what is
left being measured is the timing.

WHAT A PERCENTILE IS NOT
------------------------
It is not a p-value, and a high one is not evidence of an edge. Three rules at
four holding periods across five symbols is sixty comparisons against a single
two-year sample, and sixty comparisons produce a 98th percentile somewhere
whether or not anything is there. Ties count as neither wins nor losses, which
is why a rule that fired on every bar lands at exactly 50 rather than 0 or 100.

The percentile is the smallest honest thing that can be said about these
numbers. Saying anything larger would require data this project does not have.

WHAT IS NOT HERE
----------------
No total profit and no equity curve. Trades are allowed to overlap, so there is
no single account to compound and any "total return" would count the same money
several times over. The mean and median per trade are the honest figures.

No opinion about what to do next, either. This module measures; it does not
recommend.
"""

import numbers
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from . import trades as trades_module

__all__ = [
    "DEFAULT_DRAWS",
    "DEFAULT_SEED",
    "EvalError",
    "Part",
    "Sample",
    "Stats",
    "Verdict",
    "collect",
    "every_bar",
    "judge",
    "pool",
    "stats_of",
    "summarise",
]


# A thousand resolves a percentile to about a tenth of a point, which is finer
# than anything here deserves to be read to, and still runs in milliseconds.
DEFAULT_DRAWS = 1_000

# Fixed rather than random by default. A benchmark that changes when you rerun
# it is a benchmark you can rerun until you like it.
DEFAULT_SEED = 0

RETURN_COLUMN = "net_return"


class EvalError(ValueError):
    """Raised when a comparison cannot be made honestly.

    A ValueError subclass, matching PriceError, RuleError and TradeError, so the
    CLI can catch one broad type and print a message rather than a traceback.
    """


@dataclass(frozen=True)
class Stats:
    """What a set of trades did, in six numbers.

    All returns are net of costs. `hit_rate` counts trades that finished above
    zero: breaking exactly even is not a win, because a trade that returns
    nothing still consumed the attention of taking it.
    """

    trades: int
    hit_rate: float
    mean: float
    median: float
    best: float
    worst: float

    def describe(self) -> str:
        if not self.trades:
            return "no trades"
        return (
            f"{self.trades} trade(s), {self.hit_rate:.1%} hit, "
            f"mean {self.mean:+.3%}, median {self.median:+.3%}, "
            f"best {self.best:+.2%}, worst {self.worst:+.2%}"
        )


@dataclass(frozen=True)
class Part:
    """One symbol's trades, and the pool of trades they were chosen from.

    Kept separate rather than merged so that random draws can be stratified --
    see the module docstring. `taken` is always a subset of `population` in
    value as well as in spirit: the same bar priced the same way.
    """

    symbol: str
    taken: np.ndarray
    population: np.ndarray


@dataclass(frozen=True)
class Sample:
    """Everything a verdict is computed from: one part per symbol, one hold."""

    parts: Tuple[Part, ...]
    hold: int

    @property
    def taken(self) -> np.ndarray:
        return _joined(part.taken for part in self.parts)

    @property
    def population(self) -> np.ndarray:
        return _joined(part.population for part in self.parts)

    def label(self) -> str:
        return ", ".join(part.symbol for part in self.parts if part.symbol)


@dataclass(frozen=True)
class Verdict:
    """The rule, the benchmark, and where the rule sat among random selections."""

    sample: Sample
    rule: Stats
    baseline: Stats
    means: np.ndarray
    draws: int
    seed: int

    @property
    def percentile(self) -> float:
        """Where the rule's mean sits among the draws, from 0 to 100.

        Ties count as half, which is what makes a rule that fired on every bar
        land at 50: every draw is then the whole population, so every draw ties,
        and a result that is neither better nor worse than chance should not be
        reported as either.
        """
        below = float(np.count_nonzero(self.means < self.rule.mean))
        tied = float(np.count_nonzero(self.means == self.rule.mean))
        return 100.0 * (below + tied / 2) / len(self.means)

    def summary(self) -> str:
        label = self.sample.label()
        heading = f"{self.sample.hold}-bar hold" + (f", {label}" if label else "")
        return "\n".join(
            [
                heading,
                f"  rule       {self.rule.describe()}",
                f"  every bar  {self.baseline.describe()}",
                f"  the rule's mean sits at percentile {self.percentile:.1f} "
                f"of {self.draws} random selections of {self.rule.trades} trade(s)",
            ]
        )


def _joined(arrays):
    arrays = [np.asarray(array, dtype="float64") for array in arrays]
    if not arrays:
        return np.empty(0, dtype="float64")
    return np.concatenate(arrays)


def _whole_number(value, name, minimum):
    # bool is an Integral in Python, and `draws=True` is a typo rather than a
    # request for one draw.
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise EvalError(f"{name} must be a whole number, got {value!r}.")
    if int(value) < minimum:
        raise EvalError(f"{name} must be at least {minimum}, got {value!r}.")
    return int(value)


def stats_of(returns) -> Stats:
    """Six numbers from an array of returns. An empty array is not an error."""
    returns = np.asarray(returns, dtype="float64")
    if not len(returns):
        nothing = float("nan")
        return Stats(0, nothing, nothing, nothing, nothing, nothing)
    return Stats(
        trades=int(len(returns)),
        hit_rate=float(np.count_nonzero(returns > 0) / len(returns)),
        mean=float(np.mean(returns)),
        median=float(np.median(returns)),
        best=float(np.max(returns)),
        worst=float(np.min(returns)),
    )


def summarise(book) -> Stats:
    """The statistics of a Book, read from its net returns."""
    if not isinstance(book, trades_module.Book):
        raise EvalError(f"expected a Book, got {type(book).__name__}.")
    return stats_of(book.trades[RETURN_COLUMN].to_numpy(dtype="float64"))


def every_bar(candles, **kwargs):
    """The trade taken at every candle, as a Book.

    The benchmark of not being selective. Signals too near the end of the file
    are dropped by round_trips exactly as a rule's would be, so what comes back
    is the set of trades that were actually available.
    """
    signals = pd.Series(np.ones(len(candles), dtype=bool), index=candles.index)
    return trades_module.round_trips(candles, signals, **kwargs)


def collect(candles, signals, *, symbol="", **kwargs) -> Sample:
    """One symbol's rule trades alongside the pool they were chosen from.

    The trade settings -- hold, stop, target, costs -- are given once and used
    for both sides. That is deliberate: it makes the commonest way to rig a
    comparison impossible to express rather than merely discouraged.
    """
    taken = trades_module.round_trips(candles, signals, **kwargs)
    population = every_bar(candles, **kwargs)
    if not len(population.trades):
        raise EvalError(
            f"a {population.hold}-bar hold leaves no room for a single trade in "
            f"{len(candles)} candle(s), so there is nothing to compare against."
        )
    return Sample(
        parts=(
            Part(
                symbol=symbol,
                taken=taken.trades[RETURN_COLUMN].to_numpy(dtype="float64"),
                population=population.trades[RETURN_COLUMN].to_numpy(dtype="float64"),
            ),
        ),
        hold=population.hold,
    )


def pool(samples) -> Sample:
    """Several symbols as one sample, keeping the parts separate.

    Separate parts are not bookkeeping: they are what lets the draws in `judge`
    stay stratified by symbol.
    """
    samples = list(samples)
    if not samples:
        raise EvalError("there is nothing to pool.")
    holds = {sample.hold for sample in samples}
    if len(holds) != 1:
        raise EvalError(
            "cannot pool samples with different hold lengths: "
            f"{', '.join(str(hold) for hold in sorted(holds))}."
        )
    parts = tuple(part for sample in samples for part in sample.parts)
    return Sample(parts=parts, hold=samples[0].hold)


def _draw_means(parts, draws, seed):
    """One mean per draw, each drawn without replacement, stratified by part.

    Two details here exist to make a tie an actual tie, which the percentile
    depends on. Adding the same returns in a different order gives a different
    last bit, so the drawn indices are sorted: a draw's mean then depends on
    which trades were chosen and not on the order they came out of the
    generator. And the mean is taken the same way `stats_of` takes the rule's,
    over a single concatenated array, so that a draw which happens to select
    exactly what the rule selected produces exactly the rule's number rather
    than one that differs somewhere past the fifteenth decimal place.

    Without both, a rule that fired on nearly every bar would score at some
    arbitrary percentile decided by rounding noise instead of at 50.
    """
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype="float64")

    for index in range(draws):
        picks = []
        for part in parts:
            take = len(part.taken)
            if not take:
                continue
            chosen = rng.choice(len(part.population), size=take, replace=False)
            chosen.sort()
            picks.append(part.population[chosen])
        means[index] = float(np.mean(np.concatenate(picks)))
    return means


def judge(sample, *, draws=DEFAULT_DRAWS, seed=DEFAULT_SEED) -> Verdict:
    """Compare a sample's rule trades against the bars it chose them from."""
    if not isinstance(sample, Sample):
        raise EvalError(f"expected a Sample, got {type(sample).__name__}.")
    draws = _whole_number(draws, "draws", minimum=1)
    seed = _whole_number(seed, "seed", minimum=0)

    rule = stats_of(sample.taken)
    if not rule.trades:
        raise EvalError("the rule made no trades, so there is nothing to judge.")
    for part in sample.parts:
        if len(part.taken) > len(part.population):
            raise EvalError(
                f"{part.symbol or 'this symbol'} has {len(part.taken)} trade(s) drawn "
                f"from a pool of {len(part.population)}, which cannot be."
            )

    return Verdict(
        sample=sample,
        rule=rule,
        baseline=stats_of(sample.population),
        means=_draw_means(sample.parts, draws, seed),
        draws=draws,
        seed=seed,
    )
