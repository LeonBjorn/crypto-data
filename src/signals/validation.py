"""Correcting a Sharpe ratio for the number of times you went looking.

A Sharpe ratio computed on the best of forty strategies is not the Sharpe ratio
of a strategy. It is the maximum of forty draws, and the maximum of forty draws
from a distribution centred on zero is comfortably positive. Reporting it
without saying how many were tried is the single most common way a backtest
overstates itself, and this project has already tried about forty.

Two corrections, from Bailey and Lopez de Prado:

*The Deflated Sharpe Ratio* asks what the best of `trials` random strategies
would have scored, and then asks how confident we can be that the observed
Sharpe beats that. It also corrects for the two things a plain Sharpe ignores --
skew and kurtosis -- which matter enormously here, because a strategy that sells
tail risk shows a beautiful Sharpe right up until it does not.

*The probability of backtest overfitting* asks a different question: across
splits of the data, how often does the configuration that looked best in one
half rank below median in the other. A strategy selected by searching will do
this often, and the frequency is an estimate of how much of the result was
selection.

WHY THESE ARE IN THE REPOSITORY AND NOT IN A NOTEBOOK
-----------------------------------------------------
Because criterion 3 of `docs/going-live-criteria.md` requires a Deflated Sharpe
that survives the trial count, and a criterion nobody can evaluate is a
criterion that gets waived at exactly the moment it matters. It needs to be one
function call at the point of decision, not an afternoon's work that can be
deferred.

WHAT THEY CANNOT DO
-------------------
Neither rescues a strategy. They lower a number; they never raise one. If a
Deflated Sharpe comes out unimpressive, the honest reading is that the trials
explain the result -- not that the correction is too harsh.
"""

import math
import numbers

__all__ = [
    "EULER_MASCHERONI",
    "ValidationError",
    "deflated_sharpe",
    "expected_max_sharpe",
    "probability_of_backtest_overfitting",
    "sharpe",
]

EULER_MASCHERONI = 0.5772156649015329


class ValidationError(ValueError):
    """Raised when a correction cannot be computed honestly."""


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_ppf(p):
    """Inverse normal CDF, Acklam's rational approximation.

    Written out rather than pulled from scipy because scipy is not a dependency
    of this project and adding one for a single function that is accurate to
    about 1e-9 would be a poor trade.
    """
    if not 0.0 < p < 1.0:
        raise ValidationError(f"probability must be strictly between 0 and 1, got {p}")

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    low, high = 0.02425, 1 - 0.02425
    if p < low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _moments(returns):
    values = [float(r) for r in returns if r is not None and math.isfinite(r)]
    n = len(values)
    if n < 4:
        raise ValidationError(
            f"need at least 4 returns to estimate skew and kurtosis, got {n}"
        )
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    if var <= 0:
        raise ValidationError("returns have no variance, so a Sharpe ratio is undefined")
    sd = math.sqrt(var)
    skew = sum(((v - mean) / sd) ** 3 for v in values) / n
    kurt = sum(((v - mean) / sd) ** 4 for v in values) / n
    return values, n, mean, sd, skew, kurt


def sharpe(returns, *, periods_per_year=None):
    """Sharpe ratio of a return series. Annualised only if asked.

    No risk-free rate. At crypto's volatility it changes the third decimal of a
    number whose first decimal is in doubt, and pretending otherwise implies a
    precision this does not have.
    """
    _, _, mean, sd, _, _ = _moments(returns)
    ratio = mean / sd
    return ratio * math.sqrt(periods_per_year) if periods_per_year else ratio


def expected_max_sharpe(trials, *, sharpe_variance):
    """What the best of `trials` genuinely worthless strategies would score.

    The benchmark any searched result has to clear. It grows with the number of
    trials, which is the whole point: the more configurations tried, the higher
    the best one scores for no reason at all.

    `sharpe_variance` is the variance of the Sharpe ratios *across trials*, and
    it has to be supplied because it sets the units. There is no sensible
    default: a Sharpe measured per observation and a Sharpe measured per year
    differ by a factor of eighty, and a benchmark in the wrong one is not
    conservative or lenient, it is meaningless. `deflated_sharpe` estimates it
    from the return series when the caller has nothing better.
    """
    if isinstance(trials, bool) or not isinstance(trials, numbers.Integral) or trials < 1:
        raise ValidationError(f"trials must be a positive whole number, got {trials!r}")
    if trials == 1:
        return 0.0

    sd = math.sqrt(max(sharpe_variance, 0.0))
    left = _normal_ppf(1.0 - 1.0 / trials)
    right = _normal_ppf(1.0 - 1.0 / (trials * math.e))
    return sd * ((1.0 - EULER_MASCHERONI) * left + EULER_MASCHERONI * right)


def deflated_sharpe(returns, *, trials, sharpe_variance=None):
    """Probability the strategy's Sharpe is genuinely above what search explains.

    Returns a number in (0, 1). High means the result survives the trial count;
    low means the trials explain it. Around 0.95 is the conventional bar, and it
    is a bar this project's strategies have not been asked to clear before.

    The denominator carries the skew and kurtosis corrections. They matter: a
    strategy with negative skew and fat tails -- which is what selling tail risk
    looks like -- has a *less* reliable Sharpe than its value suggests, and the
    plain ratio cannot see that at all.
    """
    values, n, mean, sd, skew, kurt = _moments(returns)
    observed = mean / sd

    # Variance of the Sharpe estimator under non-normal returns. This is both
    # the denominator of the test and, absent better information, the spread the
    # trials themselves would have shown -- so it sets the units for the
    # benchmark too, and the two cannot then disagree about scale.
    numerator = 1.0 - skew * observed + ((kurt - 1.0) / 4.0) * observed ** 2
    if numerator <= 0:
        raise ValidationError(
            "the higher moments make the Sharpe estimator's variance non-positive, "
            "so no honest confidence can be computed. This usually means the return "
            "series is dominated by a handful of extreme observations."
        )
    estimator_variance = numerator / (n - 1)

    threshold = expected_max_sharpe(
        trials,
        sharpe_variance=estimator_variance if sharpe_variance is None else sharpe_variance,
    )
    return _normal_cdf((observed - threshold) / math.sqrt(estimator_variance))


def probability_of_backtest_overfitting(performance):
    """Estimate how much of a selection was selection, not skill.

    `performance` is a sequence of splits, each a sequence of the same
    configurations' scores: `performance[split][config]`. For every split, the
    configuration that scored best is looked up in the *other* splits, and the
    question is how often it lands below median there.

    A number near 0.5 means the winner is essentially chosen at random each time,
    which is what pure overfitting looks like. Near 0 means the same
    configuration wins consistently, which is what an edge looks like.

    A simplification of Bailey and Lopez de Prado's combinatorially symmetric
    cross-validation: this uses the splits it is given rather than generating
    every combination of them. It is the same question asked less exhaustively,
    and it is honest about being an estimate.
    """
    splits = [list(row) for row in performance]
    if len(splits) < 2:
        raise ValidationError("need at least 2 splits to compare in and out of sample")
    width = len(splits[0])
    if width < 2 or any(len(row) != width for row in splits):
        raise ValidationError("every split must score the same 2 or more configurations")

    below_median = 0
    comparisons = 0
    for index, chosen_in in enumerate(splits):
        best = max(range(width), key=lambda c: chosen_in[c])
        for other, scores in enumerate(splits):
            if other == index:
                continue
            ranked = sorted(range(width), key=lambda c: scores[c])
            rank = ranked.index(best) / (width - 1)
            below_median += 1 if rank < 0.5 else 0
            comparisons += 1
    return below_median / comparisons if comparisons else 0.0
