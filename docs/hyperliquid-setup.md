# Running the Hyperliquid broker

The broker exists and is proven against a fake venue. It has never sent an order
to anything. This is how you would let it, and the order of the steps is the
point.

**Nothing here should be done until `docs/going-live-criteria.md` is satisfied.**
Testnet is free and safe at any time; mainnet is not, and the criteria are not
met today.

---

## What it refuses to do

| guard | behaviour |
|---|---|
| `dry_run=True` (default) | logs the order, sends nothing |
| `network="testnet"` (default) | mainnet needs the argument **and** `HYPERLIQUID_ALLOW_MAINNET=yes` |
| kill switch | if `state/STOP` exists, every order is refused before signing |
| allow-list | a symbol not on it is refused |
| per-order cap | an order above it is **refused, not truncated** |
| exposure cap | refused if it would breach; also refused if positions cannot be read |
| withdrawals | there is no method — absent, not disabled |

Credentials are read from the environment only. There is no constructor argument
for a key, so one cannot arrive in the repository by being pasted into a config.

---

## Checking it, before anything else

```
uv run python scripts/testnet_check.py          # stages 1-3, no credentials needed
uv run python scripts/testnet_check.py --place-order   # stage 5, sends ONE small testnet order
```

Stages 1-3 need no credentials and already pass: the endpoint resolves to
`api.hyperliquid-testnet.xyz`, markets load, a live price is read, and the
allow-list, per-order cap and kill switch each refuse an order with a real
client underneath. Stage 4 reads your balances and positions. Stage 5 is the
only one that sends anything, is opt-in, and is testnet-only.

## Testnet

1. Get testnet funds from Hyperliquid's faucet at
   <https://app.hyperliquid-testnet.xyz>.

2. Create an **API wallet** (agent wallet) in the interface. This is the part
   that matters: an API wallet can place orders and **cannot withdraw**. Do not
   put your master key in an environment variable, ever.

### Setting them without leaving a copy anywhere

`read -rs` keeps the key off the screen and out of shell history:

```
read -rs HYPERLIQUID_PRIVATE_KEY && export HYPERLIQUID_PRIVATE_KEY
export HYPERLIQUID_WALLET_ADDRESS=<paste your 0x… account address here>
```

The second line will not run as written -- that is deliberate. A placeholder
that looks like a working command gets pasted as one, and `0xYourAccountAddress`
is a plausible-looking twenty characters that the venue would have rejected with
an unhelpful 422 before the format check existed.

**Watch for trailing whitespace.** A newline or space on the address makes the
venue answer every read with `422 Unprocessable Entity: failed to deserialize
the JSON body`, which names neither the field nor the whitespace, and surfaces
six frames down a traceback ending in this project refusing to trade. Both
values are stripped and the address is format-checked before any call, so this
now fails immediately and says what to look for -- but it is worth knowing why
that check exists.

Note also that the address should be your **account** address, and the key an
**API/agent wallet** key for that account. They are not the same thing.

3. Export the credentials in the shell that will run it — not in a file in this
   repository, and not in your shell profile if the machine is shared:

   ```
   export HYPERLIQUID_WALLET_ADDRESS=0x...      # the account
   export HYPERLIQUID_PRIVATE_KEY=0x...         # the API wallet key
   ```

4. Start in dry run, which needs no credentials at all, and confirm the wiring:

   ```python
   from paper.hyperliquid import HyperliquidBroker
   b = HyperliquidBroker(allowed_symbols={"BTC/USDC:USDC"}, max_order_notional=50)
   b.describe()      # hyperliquid testnet (dry run), ...
   ```

5. Only then set `dry_run=False`, still on testnet, with caps small enough that
   a bug is boring.

---

## Before mainnet is even considered

- Both switches: `network="mainnet"` **and** `HYPERLIQUID_ALLOW_MAINNET=yes`.
- An API wallet with withdrawals impossible, never the master key.
- Caps set from what you can afford to lose, not from what you hope to make.
- `state/STOP` tested — create it, confirm an order is refused, delete it.
- Reconciliation: `positions()` reads the venue, and the venue is the authority.
  Whatever this process remembers is a cache and may be wrong after a partial
  fill, a rejection, or a liquidation that happened while nothing was running.

---

## The thing most likely to hurt you

Not a bug in this file. It is that the strategy has no demonstrated edge — 38
configurations at the fiftieth percentile, cross-sectional at t = −0.67, and a
walk-forward with the edge in one window of six. Every guard above limits the
damage from a mistake. None of them makes a losing strategy profitable, and a
broker that works perfectly will execute a bad strategy perfectly.
