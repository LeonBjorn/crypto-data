"""Signal research: turning stored candles into testable opinions about them.

Milestone 2. This package reads the store that `collector` maintains and never
writes to it. That direction is one-way on purpose -- `collector` records what
the exchange said, which is fact, and `signals` records what we think it means,
which is not. Keeping them apart means a mistake in here can waste time but
cannot damage two years of candles.

Plural because `signal` is a standard-library module, and shadowing it inside a
project that will eventually want Ctrl-C handling is a bad trade for one letter.

Nothing here places an order or authenticates against anything.
"""

__version__ = "0.1.0"
