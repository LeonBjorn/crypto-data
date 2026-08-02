"""One wallet with finite money, which is the whole difference from a backtest.

`signals.evaluate` scores every signal on its own and lets trades overlap
freely, and says plainly why: that measures the rule rather than the luck of
which signal arrived while there was cash free. It also says what breaks the
assumption -- "if this project ever grows a single account with finite cash,
trades stop being independent and this stops being correct". This module is that
growth, kept separate from the engine so the line between the two is a file
boundary rather than a paragraph someone has to remember.

Three constraints, and each one exists because a real wallet has it:

*Cash runs out.* A position costs money that is then not available for the next
one. This is the constraint that turns a rule's hundredth-best signal into a
signal it never took.

*Positions are capped.* Not because an exchange forbids the eleventh, but
because a strategy holding every symbol it has ever liked is not a strategy, and
the cap is the smallest honest way to say so.

*One position per symbol.* A second breakout in the same coin while the first is
still open is the same opinion twice, and doubling into it is a sizing decision
wearing a signal's clothes. Defaults to on, and can be turned off.

WHAT A REJECTION IS
-------------------
Not a non-event. A signal the account could not afford is recorded, counted and
reported, because the difference between "the rule fired forty times" and "the
rule fired forty times and we could take nine of them" is the entire distance
between a backtest and a wallet. Dropping them silently would let a full account
look like a quiet market.

SIZING
------
A fixed fraction of the *starting* capital, capped by cash actually free. Fixed
fraction of current equity is the more usual choice and needs every open
position marked to market before a single order can be sized, which makes the
size of trade number four depend on an unrealised number that will change again
before it is realised. Starting capital is a number that does not move, so a
position size can be checked by hand -- and this project has an unbroken
preference for the arithmetic you can check by hand.
"""

import numbers
from dataclasses import dataclass

__all__ = [
    "Account",
    "AccountError",
    "DEFAULT_MAX_POSITIONS",
    "DEFAULT_SIZE_FRACTION",
    "Rejection",
]

# A fifth of the wallet per position, five positions, so a fully invested
# account holds five names and no cash. Conventional, round, and chosen before
# any result was looked at.
DEFAULT_SIZE_FRACTION = 0.2
DEFAULT_MAX_POSITIONS = 5


class AccountError(ValueError):
    """Raised for an account that cannot be configured as asked.

    A ValueError subclass, matching every other error type in this project, so
    a caller that only cares that the input was bad can catch one thing.
    """


@dataclass(frozen=True)
class Rejection:
    """A signal the account could not act on, and why.

    Carries the bar so it can be lined up against the candle that produced it,
    which is what makes "we were full all through March" visible rather than
    merely true.
    """

    bar: int
    symbol: str
    reason: str


def _positive_number(value, name, *, upper=None):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise AccountError(f"{name} must be a number, got {value!r}")
    value = float(value)
    if value <= 0:
        raise AccountError(f"{name} must be greater than zero, got {value}")
    if upper is not None and value > upper:
        raise AccountError(f"{name} must be at most {upper}, got {value}")
    return value


class Account:
    """Cash, position count, and the rules about when a new one may be opened.

    Deliberately knows nothing about candles, rules or fills. It is asked
    whether an order is allowed and told what one cost, and that is all -- which
    is what lets the whole policy be tested against a handful of numbers.
    """

    def __init__(
        self,
        starting_capital,
        *,
        size_fraction=DEFAULT_SIZE_FRACTION,
        max_positions=DEFAULT_MAX_POSITIONS,
        one_per_symbol=True,
    ):
        self.starting_capital = _positive_number(starting_capital, "starting_capital")
        self.size_fraction = _positive_number(size_fraction, "size_fraction", upper=1.0)

        if max_positions is not None:
            if isinstance(max_positions, bool) or not isinstance(max_positions, numbers.Integral):
                raise AccountError(
                    f"max_positions must be a whole number or None, got {max_positions!r}"
                )
            if int(max_positions) < 1:
                raise AccountError(
                    f"max_positions must be at least 1, got {max_positions}. "
                    f"An account that may hold nothing can never trade."
                )
            max_positions = int(max_positions)

        self.max_positions = max_positions
        self.one_per_symbol = bool(one_per_symbol)
        self.cash = self.starting_capital
        self.is_unlimited = False
        self._held = {}
        self.rejections = []

    @classmethod
    def unlimited(cls):
        """An account that never refuses anything.

        This is what makes the backtest invariant testable. `round_trips` scores
        every signal with no wallet at all, so reproducing it exactly requires
        an account that imposes nothing -- no cash limit, no cap, and
        overlapping positions in one symbol allowed, which is precisely the
        behaviour the backtest documents. Any divergence between the two is then
        a difference in the *fill model*, which is what the test is looking for,
        rather than a difference in policy.
        """
        account = cls(1.0, size_fraction=1.0, max_positions=None, one_per_symbol=False)
        account.is_unlimited = True
        account.cash = float("inf")
        return account

    @property
    def open_positions(self) -> int:
        return sum(self._held.values())

    def holds(self, symbol) -> int:
        return self._held.get(symbol, 0)

    def refusal(self, symbol):
        """Why `symbol` cannot be opened right now, or None if it can.

        Returns the reason rather than a bare False so that the caller can
        record it. A count of rejections says the account was busy; a reason
        says whether it was out of money or out of slots, and those call for
        opposite changes.
        """
        if self.is_unlimited:
            return None
        if self.one_per_symbol and self.holds(symbol):
            return f"already holding {symbol}"
        if self.max_positions is not None and self.open_positions >= self.max_positions:
            return f"at the {self.max_positions}-position cap"
        if self.notional_for(symbol) <= 0:
            return "no free cash"
        return None

    def notional_for(self, symbol=None) -> float:
        """What one new position may cost, in quote currency.

        A fixed slice of the starting capital, never more than the cash actually
        available -- so a nearly-empty account takes a smaller last position
        rather than refusing outright or, worse, spending money it does not have.
        """
        if self.is_unlimited:
            # A nominal unit. Nothing about the invariant depends on size, and a
            # real number keeps the arithmetic finite where infinite cash would
            # not.
            return 1.0
        return min(self.starting_capital * self.size_fraction, max(self.cash, 0.0))

    def opened(self, symbol, cash_spent):
        """Record that a position was opened, and what it cost."""
        if not self.is_unlimited:
            self.cash -= cash_spent
        self._held[symbol] = self._held.get(symbol, 0) + 1

    def closed(self, symbol, cash_received):
        """Record that a position was closed, and what came back."""
        if not self.is_unlimited:
            self.cash += cash_received
        remaining = self._held.get(symbol, 0) - 1
        if remaining > 0:
            self._held[symbol] = remaining
        else:
            self._held.pop(symbol, None)

    def reject(self, bar, symbol, reason):
        """Record a signal that could not be taken."""
        self.rejections.append(Rejection(bar=bar, symbol=symbol, reason=reason))

    def equity(self, marks=None) -> float:
        """Cash plus open positions marked at `marks` (symbol -> value).

        For display only. A mark is the last close, which is a price nobody
        traded at and nobody is promised; treating it as realised is how a
        paper account talks itself into a number the market never offered.
        """
        return self.cash + sum((marks or {}).values())
