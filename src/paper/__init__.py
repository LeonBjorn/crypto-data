"""Paper trading: the same rules, run forward one candle at a time.

Milestone 3, and the bridge between research and anything real. `signals`
answers "would this rule have worked", with every bar of history available at
once. This package answers the harder question that comes next: "what does this
rule do when it can only see the bar that just closed", which is the only
question a live system is ever asked.

The two must agree. A backtest and a paper run over the same candles are the
same arithmetic approached from opposite ends of time, and if they disagree then
one of them is wrong -- so the invariant this package is built around is that
replaying stored candles through the engine one at a time reproduces
`signals.trades.round_trips` exactly. That test is the reason to trust anything
printed later.

WHAT IS DIFFERENT FROM THE BACKTEST
-----------------------------------
One thing, deliberately: money. The backtest scores every signal on its own and
lets trades overlap, because that measures the *rule*. A wallet cannot do that.
It has finite cash, it can only hold so many positions at once, and two signals
on the same afternoon compete for the same money. `evaluate.py` names this
exactly -- "if this project ever grows a single account with finite cash, trades
stop being independent and this stops being correct" -- and this package is that
growth. So the fill model is shared and the account is new, kept in a separate
module so the boundary between "what the market did" and "what we could afford"
stays visible.

WHAT THIS PACKAGE STILL CANNOT DO
---------------------------------
Place an order. There are no credentials here, no signing, and no network calls
of any kind -- this package reads the store like `signals` does. Orders go
through the Broker seam in `broker.py`, and the only implementation that exists
today fills them out of the candle it was given. A live broker is a later
milestone, gated behind explicit configuration, and the seam exists so that
adding one is a new class rather than a rewrite of the engine.
"""

__version__ = "0.1.0"
