# Criteria for trading real money

Written 2026-08-04, while nothing is at stake, because that is the only time
criteria can be written honestly. The purpose of putting them here is to make
them awkward to quietly renegotiate later.

**Status: not met. The algorithm should not trade real money today.**

---

## Why not, in one paragraph

The strategy has no demonstrated edge. Thirty-eight directional configurations
came out around the fiftieth percentile — no better than random timing. The
cross-sectional version returns −0.74% per rebalance with a t-statistic of
−0.67. The one-minute microstructure test came out at zero once a lookahead in
the measurement was corrected. The walk-forward found the edge in one window of
six and *below random* in another. The +20.7% on the paper account is one bull
window plus XRP alone (+$1,609); ETH lost $749 and BTC was flat.

And the paper record is a **replay** of stored history. Genuinely forward,
out-of-sample testing began 2026-08-04 and is measured in days.

---

## What must be true first

All five. Not most.

1. **At least three months of genuinely forward paper record**, measured from
   `forward_from` in the state file — not from the replay. Ideally spanning a
   regime change rather than three quiet months.

2. **Forward results consistent with the replay.** If forward is materially
   worse, the replay was fitted and the difference is the size of the
   self-deception. This is the test that matters most.

3. **A Deflated Sharpe Ratio that survives the trial count.** Roughly forty
   configurations have been evaluated against this data. A Sharpe that does not
   survive correction for that is not a finding. See Bailey & López de Prado.

   Now computable, so this is no longer a judgement call:

   ```python
   from signals.validation import deflated_sharpe
   deflated_sharpe([t["net_return"] for t in ledger], trials=40)
   ```

   **Measured on the current ledger: 0.039.** The bar is 0.95. It is not close,
   and it is not close even at a single trial (0.667) -- the trial count is not
   what is failing this, the strategy is.

4. **The broker proven on testnet**: trade-only API keys with withdrawals
   disabled, hard caps on order size and total exposure, an allowed-symbol list,
   a kill-switch file, and reconciliation against the venue's own positions on
   every run.

5. **First real size small enough that losing all of it changes nothing.** Not
   "an amount I am comfortable with" — an amount whose total loss is
   uninteresting. Expect to lose it.

If 1–3 fail, that is an answer and not a delay.

---

## What would make this *more* likely to fail, not less

- Re-running the search until something passes. Every additional configuration
  raises the bar that any survivor must clear, and the count is already ~40.
- Moving the criteria after seeing the forward results.
- Enabling leverage. The account has already drawn down 41%; at 2.44x that
  drawdown is a wiped account, and at 2.86x the worst single trade liquidates.
- Reading the replay as a track record.

---

## Practical notes for when the time comes

- USDC is Hyperliquid's collateral, so there is no conversion friction. Logistics
  are not the binding constraint here; evidence is.
- Costs on Hyperliquid are roughly 0.13% round trip at taker, plus funding of
  about 0.2% per week on a long perp. Together that is comparable to the entire
  measured edge, which is why cost realism has been kept in every measurement.
- The `Broker` seam already exists, reads fills back rather than assuming them,
  and can express shorts. Adding a venue is a new class, not a rewrite.

---

## The honest alternative

Nothing here says "do not trade". It says this *algorithm* has not earned it. A
person who wants exposure to crypto can hold it, and over the window measured
here holding an equal-weight basket lost 15.8% while the strategy made 20.7% —
mostly by being out of the market, and mostly in one coin.

The paper account keeps running, costs nothing, and will answer question 2 in
about three months. That is the cheapest information available.
