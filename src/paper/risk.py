"""Sizing and risk limits: deciding how much, having given up on deciding when.

Everything measured in this project points the same way. The searches found no
reliable directional edge; the account that does run drew down 41% from its peak
to deliver 20%; and almost all of the profit came from one of five symbols. That
is not a signal problem to be solved with a better rule, it is a *risk* problem,
and the literature on it is unusually settled.

So this module contains no forecast of anything. Every quantity in it is either a
volatility -- the best-estimated input in finance -- or a hard limit chosen in
advance. Specifically it does not estimate expected returns, because that is the
input mean-variance optimisation is most sensitive to and the one nobody can
estimate: DeMiguel, Garlappi and Uppal found no model among fourteen consistently
beat equal weighting out of sample, and put the sample needed for the sample-based
optimiser to win at around three thousand months for twenty-five assets.

WHAT IS HERE
------------
*EWMA volatility.* One parameter, no fitting, nothing to fail to converge. The
RiskMetrics decay of 0.94 on daily data is the convention; it is rescaled here
for the bar size actually in use.

*Inverse-volatility weights.* Size each position by 1/sigma rather than equally.
It needs only variances, not correlations and not means, which is exactly why it
survives contact with real data.

*A volatility target.* Scale the whole book so its forecast volatility sits at a
chosen level, capped by a leverage limit. Moreira and Muir found taking less risk
when volatility is high produced higher Sharpe ratios across the market, value,
momentum, profitability, investment and carry -- and the later literature
disputes how much of that survives costs, which is why the cap matters more than
the target.

*Expected Shortfall.* The average loss in the worst tail, rather than the
threshold of it. Subadditive where Value-at-Risk is not, which is why Basel's
FRTB moved to it. Here it is a monitor, not an objective.

*A drawdown limit.* Pre-committed, because the entire point of a limit is that it
is chosen before the loss rather than during it.

THE CORRELATION ASSUMPTION, STATED OUTRIGHT
-------------------------------------------
Portfolio volatility is computed as the weighted *sum* of component volatilities,
which is the value it takes when everything is perfectly correlated. That is
deliberately the most pessimistic reading, and it is chosen because the
alternative requires estimating a correlation matrix that is badly conditioned,
unstable, and -- on the evidence -- rises toward one precisely during the stress
it was supposed to protect against. Assuming the diversification benefit away
costs some position size in calm markets and does not lie to you in bad ones.
"""

import math
import numbers
from dataclasses import dataclass, field

__all__ = [
    "DEFAULT_LAMBDA",
    "DrawdownGuard",
    "EwmaVolatility",
    "RiskError",
    "RiskModel",
    "expected_shortfall",
    "max_drawdown",
]

# RiskMetrics' decay for daily data. Rescaled for other bar sizes in
# `lambda_for`, because 0.94 means "remember about a month" only if a step is a
# day; applied unchanged to hourly bars it would mean about a day and a half.
DEFAULT_LAMBDA = 0.94
DAILY_BARS_ASSUMED = 1

# Below this many observations an EWMA estimate is mostly its own seed. Sizing on
# it would be sizing on nothing, so the model says so and callers fall back to
# equal weight rather than pretending.
MIN_OBSERVATIONS = 30


class RiskError(ValueError):
    """Raised for a risk model that cannot be configured as asked.

    A ValueError subclass, matching every other error type in this project.
    """


def _positive(value, name, *, upper=None):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise RiskError(f"{name} must be a number, got {value!r}")
    value = float(value)
    if value <= 0:
        raise RiskError(f"{name} must be greater than zero, got {value}")
    if upper is not None and value > upper:
        raise RiskError(f"{name} must be at most {upper}, got {value}")
    return value


def lambda_for(bars_per_day, daily_lambda=DEFAULT_LAMBDA):
    """The decay that gives `daily_lambda`'s memory at a different bar size.

    A decay factor is a half-life in disguise, and the half-life is in *bars*.
    Carrying 0.94 across from daily to hourly data would keep the number and
    silently shorten the memory from about a month to about a day, which is a
    different model wearing the same constant.
    """
    bars_per_day = _positive(bars_per_day, "bars_per_day")
    return daily_lambda ** (1.0 / bars_per_day)


class EwmaVolatility:
    """One symbol's exponentially weighted volatility.

        sigma^2_t = lam * sigma^2_{t-1} + (1 - lam) * r^2_{t-1}

    Updated one return at a time, because that is the only way a live process
    can do it and the only way that cannot accidentally see the future. There is
    no fitting step and nothing that can fail to converge, which is most of the
    reason to prefer this to GARCH for a sizing input.
    """

    def __init__(self, lam=DEFAULT_LAMBDA, *, min_observations=MIN_OBSERVATIONS):
        self.lam = _positive(lam, "lam", upper=1.0)
        self.min_observations = int(min_observations)
        self.variance = None
        self.observations = 0

    def observe(self, ret):
        """Fold in one bar's return."""
        if ret is None or not math.isfinite(ret):
            return self
        squared = float(ret) ** 2
        # Seeded with the first squared return rather than zero: starting at zero
        # would take dozens of bars to climb to a believable level and would
        # oversize every position on the way there.
        self.variance = squared if self.variance is None else (
            self.lam * self.variance + (1.0 - self.lam) * squared
        )
        self.observations += 1
        return self

    @property
    def ready(self) -> bool:
        """Whether there is enough history for the estimate to mean anything."""
        return self.variance is not None and self.observations >= self.min_observations

    @property
    def sigma(self):
        """Per-bar volatility, or None while still warming up."""
        if not self.ready:
            return None
        return math.sqrt(self.variance)

    def annualised(self, bars_per_year):
        """Volatility scaled to a year by the square root of time."""
        sigma = self.sigma
        return None if sigma is None else sigma * math.sqrt(bars_per_year)


def expected_shortfall(returns, beta=0.95):
    """Mean loss in the worst `1 - beta` of outcomes, as a positive fraction.

    Reported rather than optimised. Optimising it needs a scenario set and a
    linear program; monitoring it needs the trades that actually happened, and
    the second is the honest thing to put on a dashboard.

    Positive means "this much is lost", so a portfolio that never lost anything
    returns zero rather than a negative number that reads like a gain.
    """
    values = sorted(float(r) for r in returns if r is not None and math.isfinite(r))
    if not values:
        return None
    beta = _positive(beta, "beta", upper=0.999999)

    # The nudge is not cosmetic. 1.0 - 0.80 is 0.19999999999999996, so a
    # ten-trade tail at beta=0.80 floors to one observation instead of two and
    # the reported shortfall changes with the float representation of the
    # confidence level. A risk number that moves for that reason is worse than
    # no risk number.
    cutoff = max(1, int(math.floor(len(values) * (1.0 - beta) + 1e-9)))
    tail = values[:cutoff]
    worst_mean = sum(tail) / len(tail)
    return max(0.0, -worst_mean)


def max_drawdown(equity):
    """Largest peak-to-trough fall in an equity series, as a negative fraction."""
    peak = None
    worst = 0.0
    for value in equity:
        value = float(value)
        peak = value if peak is None else max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


@dataclass
class DrawdownGuard:
    """A pre-committed limit on how far the account may fall from its peak.

    Chosen in advance and checked mechanically, because the one thing certain
    about a drawdown limit decided during a drawdown is that it will be moved.

    Tripping it stops *new* positions. It deliberately does not liquidate what is
    open: forced selling at the bottom is how a limit meant to preserve capital
    ends up destroying it, and the existing positions already have exits.
    """

    limit: float = 0.25
    peak: float = 0.0
    equity: float = 0.0
    tripped: bool = False
    tripped_at: float = None

    def observe(self, equity) -> bool:
        """Update with the latest equity. Returns whether trading is allowed."""
        self.equity = float(equity)
        self.peak = max(self.peak, self.equity)
        if self.peak > 0:
            fall = self.equity / self.peak - 1.0
            if fall <= -abs(self.limit) and not self.tripped:
                self.tripped = True
                self.tripped_at = self.equity
        return not self.tripped

    @property
    def drawdown(self) -> float:
        """Current fall from peak, as a negative fraction."""
        if not self.peak:
            return 0.0
        return self.equity / self.peak - 1.0

    def reset(self):
        """Clear the trip. Deliberately manual -- resuming is a decision."""
        self.tripped = False
        self.tripped_at = None
        return self


class RiskModel:
    """Volatilities, the weights they imply, and the scale the book is run at.

    Fed one bar per symbol at a time and asked, when a position is about to be
    opened, how large it should be. It holds no view on direction and no view on
    which symbol is attractive -- only on how much each one moves.
    """

    def __init__(
        self,
        symbols,
        *,
        bars_per_year=24 * 365,
        bars_per_day=24,
        target_vol=None,
        max_leverage=1.0,
        lam=None,
        min_observations=MIN_OBSERVATIONS,
    ):
        if not symbols:
            raise RiskError("a risk model needs at least one symbol")
        self.symbols = list(symbols)
        self.bars_per_year = _positive(bars_per_year, "bars_per_year")
        self.lam = lam if lam is not None else lambda_for(bars_per_day)
        self.target_vol = None if target_vol is None else _positive(target_vol, "target_vol")
        self.max_leverage = _positive(max_leverage, "max_leverage")
        self.vol = {
            s: EwmaVolatility(self.lam, min_observations=min_observations) for s in self.symbols
        }
        self._last_close = {}

    # -- observation --------------------------------------------------------

    def observe_close(self, symbol, close):
        """Fold in a bar by its close, deriving the return from the last one."""
        previous = self._last_close.get(symbol)
        self._last_close[symbol] = float(close)
        if previous and previous > 0:
            self.observe(symbol, float(close) / previous - 1.0)
        return self

    def observe(self, symbol, ret):
        if symbol in self.vol:
            self.vol[symbol].observe(ret)
        return self

    # -- what it can say ----------------------------------------------------

    def sigma(self, symbol):
        estimator = self.vol.get(symbol)
        return None if estimator is None else estimator.sigma

    def annualised(self, symbol):
        estimator = self.vol.get(symbol)
        return None if estimator is None else estimator.annualised(self.bars_per_year)

    @property
    def ready(self) -> bool:
        """Whether every symbol has enough history to be sized on."""
        return all(v.ready for v in self.vol.values())

    def weights(self) -> dict:
        """Inverse-volatility weights over the symbols, summing to one.

        Falls back to equal weight until the estimates are warm. That is not a
        placeholder: equal weighting is the benchmark this whole literature
        struggles to beat, so it is a defensible answer rather than a stand-in
        for a better one.
        """
        sigmas = {s: self.sigma(s) for s in self.symbols}
        usable = {s: v for s, v in sigmas.items() if v and v > 0}
        if len(usable) < len(self.symbols):
            share = 1.0 / len(self.symbols)
            return {s: share for s in self.symbols}
        inverse = {s: 1.0 / v for s, v in usable.items()}
        total = sum(inverse.values())
        return {s: v / total for s, v in inverse.items()}

    def portfolio_vol(self, weights=None) -> float:
        """Forecast volatility of the weighted book, annualised.

        The weighted sum of component volatilities, which is what the true value
        would be if everything moved together. See the module docstring: this is
        the pessimistic reading and it is chosen on purpose, because the
        correlation estimate the optimistic reading needs is both unstable and
        known to rise toward one exactly when it matters.
        """
        weights = weights or self.weights()
        total = 0.0
        for symbol, weight in weights.items():
            annual = self.annualised(symbol)
            if annual is None:
                return None
            total += weight * annual
        return total

    def scale(self) -> float:
        """How much of the book to actually run, as a multiplier.

            k = min(k_max, target / forecast)

        With no target this is just the leverage cap, so the machinery is inert
        rather than absent when it is switched off.
        """
        if self.target_vol is None:
            return min(1.0, self.max_leverage)
        forecast = self.portfolio_vol()
        if not forecast or forecast <= 0:
            # No usable forecast yet. Take the smaller of the cap and one, rather
            # than the cap, so an unwarmed model can never size *up*.
            return min(1.0, self.max_leverage)
        return min(self.max_leverage, self.target_vol / forecast)

    def notional_for(self, symbol, capital) -> float:
        """What one position in `symbol` should cost, before cash limits.

        Inverse-volatility weight times the scale the whole book is run at. The
        caller still has to check it can afford it -- this says what the risk
        model wants, not what the wallet permits.
        """
        return capital * self.weights().get(symbol, 0.0) * self.scale()

    def describe(self) -> str:
        vol = self.portfolio_vol()
        parts = [f"scale {self.scale():.2f}"]
        if vol is not None:
            parts.append(f"forecast vol {vol:.1%}")
        if self.target_vol is not None:
            parts.append(f"target {self.target_vol:.1%}")
        parts.append(f"cap {self.max_leverage:.2f}x")
        return ", ".join(parts)
