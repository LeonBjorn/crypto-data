# Risk evaluation and portfolio construction for systematic crypto

A survey of the algorithms and frameworks, what the primary literature actually
supports, and a practical blueprint. Every substantive claim is linked to a
primary source.

**This is not investment advice.** It contains no recommendation to buy or sell
any asset. It is a methodology review, written to be argued with.

---

## 0. A note on the Harvard framing, before anything else

You asked for Harvard/HBS research as the conceptual foundation. Where it
genuinely is the foundation, it is cited as such — and where it is not, saying
so is more useful than manufacturing a lineage.

**Where Harvard genuinely anchors this material:**

- **Campbell & Viceira, *Strategic Asset Allocation: Portfolio Choice for
  Long-Term Investors*** (Oxford, 2002). John Campbell is Otto Eckstein
  Professor of Applied Economics at Harvard; Luis Viceira is at HBS. The book's
  central move is directly relevant here: it **critiques mean–variance analysis
  for giving inadequate guidance on long-horizon problems** and rebuilds
  portfolio choice on Merton's intertemporal model.
  [Harvard Scholars](https://campbell.scholars.harvard.edu/publications/strategic-asset-allocation-portfolio-choice-long-term-investors) ·
  [Oxford Academic](https://academic.oup.com/book/6093)
- **Robert C. Merton**, whose intertemporal portfolio choice framework underlies
  the above, was a professor at Harvard Business School from 1988 to 1998.
- **HBS Working Paper 23-073, "Who Invests in Crypto? Wealth, Financial
  Constraints, and Risk Attitudes"** (May 2023) — on crypto portfolio choice,
  risk preferences and hedging motives.
  [HBS](https://www.hbs.edu/faculty/Pages/item.aspx?num=64044) ·
  [PDF](https://www.hbs.edu/ris/Publication%20Files/23-073_2c50c117-a0af-4517-bb0f-06f8212f3177.pdf)

**Where it is not:** the operational algorithmic canon in this document —
Markowitz, Rockafellar–Uryasev CVaR, Ledoit–Wolf shrinkage, Engle's DCC,
López de Prado's HRP, Bailey's overfitting work — comes from Chicago/RAND,
Florida, Zurich, NYU Stern and Cornell. Presenting it as Harvard-derived would
be a fabricated pedigree. The Harvard contribution is **conceptual** (long-horizon
portfolio choice, scepticism of naive mean–variance); the machinery is not.

---

## 1. Executive conclusion

> **The most robust practical framework is a constrained, shrinkage-based
> risk-only allocation with an explicit tail-risk objective and a volatility
> target — never an unconstrained return-forecasting optimiser.**

Concretely, and in priority order:

1. **Do not estimate expected returns for the optimiser.** This is the single
   highest-value decision. Mean–variance optimisation is dominated by errors in
   the mean vector, and DeMiguel, Garlappi & Uppal found that **of 14 models
   across seven datasets, none consistently beat naive 1/N** on Sharpe ratio,
   certainty equivalent or turnover — they estimate you would need roughly
   **3,000 months of data for 25 assets** for sample mean–variance to reliably
   win. ([RFS 2009](https://academic.oup.com/rfs/article-abstract/22/5/1915/1592901))
2. **Use a shrunk or hierarchical covariance estimate**, not the sample
   covariance. ([Ledoit & Wolf](http://www.ledoit.net/Honey_2004.pdf))
3. **Optimise Expected Shortfall (CVaR), not variance or VaR**, via the
   Rockafellar–Uryasev linear program. ES is coherent and subadditive; VaR is
   not, and can penalise diversification.
4. **Overlay a volatility target with a hard leverage cap.** This is the
   mechanism with the strongest cross-asset evidence for improving risk-adjusted
   outcomes.
5. **Constrain everything** — position caps, liquidity floors, turnover budgets.
   Constraints substitute for estimation precision you do not have.
6. **Validate out-of-sample with an explicit multiple-testing correction.**

The honest summary of the literature: **structure and constraints beat cleverness**.
Every step above reduces reliance on parameters you cannot estimate well.

---

## 2. Risk modelling

### 2.1 Volatility forecasting

| Method | Formula | Verdict |
|---|---|---|
| **EWMA** (RiskMetrics) | σ²ₜ = λσ²ₜ₋₁ + (1−λ)r²ₜ₋₁, λ≈0.94 daily | One parameter, no fitting, no convergence failures. Excellent default. |
| **GARCH(1,1)** | σ²ₜ = ω + αr²ₜ₋₁ + βσ²ₜ₋₁ | Adds mean reversion. Hansen & Lunde compared 330 models and found **nothing significantly beat GARCH(1,1)** for exchange rates. ([J. Applied Econometrics 2005](https://ideas.repec.org/a/jae/japmet/v20y2005i7p873-889.html)) |
| **EGARCH** | log σ²ₜ with asymmetric term | Captures leverage effect (down-moves raise vol more). Crypto shows this, sometimes inverted. |
| **Realised volatility** | RVₜ = Σ r²ᵢ over intraday returns | Andersen & Bollerslev, *Answering the Skeptics* (IER 1998, 39:885–905), showed standard models forecast well **when evaluated against realised volatility** rather than squared daily returns. The earlier "GARCH doesn't work" literature was measuring badly. |

**Practical position:** EWMA for the volatility target (robust, no fitting);
realised volatility from your 1-minute data if available — it is a far less noisy
estimate of the same quantity. GARCH only if you need term structure.

**Crypto caveat:** volatility is regime-driven and clusters violently. A single
unconditional estimate is not merely imprecise, it is the wrong object.

### 2.2 Correlation and dependence

- **Sample covariance is unusable** for optimisation when assets are many
  relative to observations. Ledoit & Wolf: extreme coefficients "take on extreme
  values not because this is reality but because they contain an extreme amount
  of error"; shrinkage pulls them toward a structured target,
  **Σ̂ = δF + (1−δ)S**, with an analytically optimal δ.
  ([Ledoit & Wolf, *Honey, I Shrunk the Sample Covariance Matrix*](http://www.ledoit.net/Honey_2004.pdf) ·
  [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=433840))
- **DCC-GARCH** (Engle, *JBES* 2002, 20:339–350) models time-varying
  correlations with univariate-GARCH flexibility and two-step estimability.
  ([Taylor & Francis](https://www.tandfonline.com/doi/abs/10.1198/073500102288618487) ·
  [Engle's PDF](https://pages.stern.nyu.edu/~rengle/dccfinal.pdf))
- **Regime change is the practical problem.** Correlations in crypto are not
  stable parameters; they rise sharply in stress (§4.2).

### 2.3 Tail risk

**VaR is not a coherent risk measure.** Artzner, Delbaen, Eber & Heath showed
VaR can **violate subadditivity** — the VaR of a combined portfolio can exceed
the sum of its parts, so VaR can *punish diversification*. Expected Shortfall is
subadditive and does not.

This is not academic hair-splitting: the **Basel Committee's FRTB replaced VaR
(99%) with Expected Shortfall (97.5%)** for market risk capital, precisely
because ES captures losses beyond the quantile and is coherent. The 97.5% level
was chosen so ES is comparable to 99% VaR under approximate normality.
([BPI explainer](https://bpi.com/why-is-the-frtb-expected-shortfall-calculation-designed-as-it-is/))

| Method | Strength | Weakness |
|---|---|---|
| Historical VaR/ES | No distributional assumption | Cannot see a loss not in the sample; crypto history is short |
| Monte Carlo VaR/ES | Flexible, handles nonlinearity | Only as good as the assumed process — garbage in, confident garbage out |
| **Normal/parametric VaR** | Trivial | **Wrong for crypto.** Assumes away the fat tails that are the entire risk |
| **EVT (POT)** | Models the tail explicitly | Threshold choice is a judgement call; needs care |

**Best practice for tails:** the **McNeil–Frey two-step** — filter returns with a
GARCH model to remove heteroskedasticity, then fit a Generalised Pareto
distribution to the standardised residuals via Peaks-Over-Threshold. Their
backtests beat both unconditional EVT and GARCH with normal or Student-t
innovations, and it is now the standard approach. Applied to Bitcoin, dynamic POT
shows favourable backtesting statistics.
([Forecasting tail risk for Bitcoin, *Finance Research Letters*](https://www.sciencedirect.com/science/article/abs/pii/S1544612322003129))

### 2.4 The risks that are not in the covariance matrix

For crypto these frequently dominate everything above, and none appear in a
return series:

| Risk | Why it matters |
|---|---|
| **Drawdown** | Path matters. A −41% drawdown with a +20% end return is a different asset from a smooth +20%. |
| **Liquidity** | Depth vanishes exactly when you need it. Backtests assume you can trade. |
| **Leverage / liquidation** | On perps, liquidation is a hard absorbing barrier, not a bad day. |
| **Counterparty & custody** | FTX was not a market risk event. No covariance matrix contained it. |
| **Operational** | Key management, API failure, bad deploys, fat fingers. |
| **Regulatory** | Venue access can change by jurisdiction overnight. |

The BIS 2023 report on the crypto ecosystem documents structural flaws,
fragmentation, and **substantial de-facto centralisation despite decentralisation
claims**. ([BIS report coverage](https://www.regulationtomorrow.com/2023/07/bis-report-on-the-crypto-ecosystem-key-elements-and-risks/) ·
[BIS stablecoin bulletin](https://www.bis.org/publ/bisbull108.pdf))

---

## 3. Portfolio construction

### 3.1 Markowitz mean–variance, and why it fails in practice

Mean–variance is correct as theory and fragile as an estimator. Its failures are
well characterised:

- **Error maximisation.** The optimiser systematically overweights assets whose
  returns are *overestimated* and whose variances are *underestimated*. It is an
  error-seeking device.
- **Instability.** Small input changes produce large weight changes.
- **Concentration.** Solutions pile into few assets.
- **Requires an invertible covariance matrix**, which is ill-conditioned when
  assets are many relative to observations.

DeMiguel, Garlappi & Uppal is the decisive empirical result: **no model among 14
consistently beat 1/N out of sample.**
([RFS 2009](https://academic.oup.com/rfs/article-abstract/22/5/1915/1592901) ·
[SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1376199))

Campbell & Viceira's critique is complementary and deeper: single-period
mean–variance is simply the wrong *problem* for a long-horizon investor.

### 3.2 The methods worth using

| Method | What it does | Notes |
|---|---|---|
| **Constrained minimum-variance** | Minimise wᵀΣw subject to weight caps | Drops the mean vector entirely — removes the worst-estimated input. Strong, boring, hard to beat. |
| **CVaR / ES optimisation** | Minimise expected loss beyond a quantile | Rockafellar & Uryasev, *Journal of Risk* 2(3):21–42 (2000). Reduces to an **LP** with scenario data. Directly targets the tail. ([Uryasev lecture PDF](https://www2.mathematik.hu-berlin.de/~romisch/SP01/Uryasev.pdf)) |
| **Inverse-volatility / risk parity** | Weight ∝ 1/σᵢ, or equalise risk contributions | Needs only variances (well estimated), not correlations or means (badly estimated). Very robust. |
| **Hierarchical Risk Parity** | Cluster by correlation, recursively bisect | López de Prado, *JPM* 42(4):59–69 (2016). **Does not require inverting the covariance matrix.** ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678) · [JPM](https://jpm.pm-research.com/content/42/4/59.short)) |
| **Black–Litterman** | Blend equilibrium priors with explicit views | Useful *because* it forces you to state views and confidence separately. Without views it returns the market portfolio — a feature. |
| **Robust optimisation** | Optimise against a set of plausible parameters | Explicitly prices in the fact that you don't know the inputs. |

**On HRP, an honest caveat.** López de Prado's Monte Carlo experiments show HRP
delivers lower out-of-sample variance than the critical line algorithm — even
though minimum variance is CLA's *objective*. But independent replications are
mixed: recent work on S&P 500 constituents (2005–2023) found **1/N outperformed
HRP across all experimental setups**, and other studies find HRP underperforming
minimum variance. HRP is a reasonable, robust choice — not a settled winner.
([Empirical Economics comparison](https://link.springer.com/article/10.1007/s00181-026-02900-x) ·
[ScienceDirect implementation study](https://www.sciencedirect.com/science/article/abs/pii/S0167739X25000391))

### 3.3 Volatility targeting

Moreira & Muir (*Journal of Finance* 2017, 72:1611–1644) found that portfolios
taking **less risk when volatility is high** produce significant alphas and higher
Sharpe ratios across the market, value, momentum, profitability, investment,
betting-against-beta factors and the currency carry trade. The mechanism:
**changes in volatility are not offset by proportional changes in expected
returns.**
([Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513) ·
[NBER w22208](https://www.nber.org/system/files/working_papers/w22208/w22208.pdf))

Note the honest counterweight: subsequent work questions out-of-sample robustness
and implementability, including a *JF* multifactor reassessment.
([DeMiguel et al., *JF* 2024](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13395) ·
[Cederburg et al., *JFE*](https://www.sciencedirect.com/science/article/abs/pii/S0304405X2030132X))

Volatility targeting remains the best-evidenced overlay available, particularly in
an asset class whose volatility varies by an order of magnitude.

### 3.4 Rebalancing

Threshold ("no-trade band") rebalancing dominates calendar rebalancing on a
cost-adjusted basis: rebalance when a weight drifts outside ±τ of target, not
because it is Tuesday. Rebalancing is also a genuine risk control — the CFA
Institute brief notes it prevents unintended concentration after strong runs.
([CFA Institute cryptoassets brief](https://rpc.cfainstitute.org/sites/default/files/-/media/documents/article/rf-brief/rfbr-cryptoassets.pdf))

---

## 4. Crypto-specific factors

### 4.1 What actually prices crypto

Liu, Tsyvinski & Wu, *Common Risk Factors in Cryptocurrency* (NBER w25882;
*Journal of Finance* 2022) — **three factors capture the cross-section of
expected crypto returns: market, size, and momentum**, and they account for the
returns of nine crypto trading strategies.
([NBER](https://www.nber.org/papers/w25882))

Liu & Tsyvinski, *Risks and Returns of Cryptocurrency* (NBER w24877) — crypto's
risk-return tradeoff is **distinct from stocks, currencies and precious metals**,
with **no exposure to most common equity or macro factors**; returns are
predicted by **time-series momentum** and **investor attention proxies**.
([NBER](https://www.nber.org/papers/w24877))

Practical reading: crypto is largely a **single-factor market with a dominant
common component**, plus size and momentum. That has a direct consequence — a
long-only basket of majors is close to one bet.

### 4.2 Why diversification fails in crypto stress

Correlations rise sharply in exactly the states where diversification is needed.
Empirical work finds **pure financial contagion** between Bitcoin and developed
equity markets during turmoil, with spillover magnitude increasing in stress; and
that cross-market hedging "may be effective in times of normal market stability
but is more likely to fail in times of financial or economic turmoil."
([*Cogent Economics & Finance* 2023](https://www.tandfonline.com/doi/full/10.1080/23322039.2023.2203432) ·
[IMF WP 2023/213](https://www.elibrary.imf.org/view/journals/001/2023/213/article-A001-en.xml))

**Design implication:** a diversification assumption estimated in calm periods is
not merely optimistic, it fails *conditionally on the state you were protecting
against*. Stress-test with correlations forced toward 1.

### 4.3 Data quality — the one that invalidates backtests silently

Cong, Li, Tang & Yang, *Crypto Wash Trading* (*Management Science* 2023,
69(11):6427–6454; NBER w30783) found **wash trading averaged over 70% of reported
volume on unregulated exchanges**, detected via **Benford's-law first-digit
distributions, size-rounding patterns, and trade-size tail distributions**.
([Management Science](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2021.02709) ·
[NBER PDF](https://www.nber.org/system/files/working_papers/w30783/w30783.pdf) ·
[Cowles PDF](https://cowles.yale.edu/sites/default/files/2022-11/cryptowashtrading040521-crypto-wash-trading.pdf))

Consequences for any volume-based signal — including volume-confirmed breakouts —
are severe: **on unregulated venues, most of the volume is not real.**

Also material:

- **Survivorship bias.** Dead coins and failed exchanges vanish from datasets.
  Backtesting today's top 50 over five years measures having picked the survivors.
- **Exchange failure.** Mt. Gox, FTX. Custody risk is not diversifiable by
  holding more coins on the same venue.
- **Transaction costs.** Must be modelled at venue-realistic levels, both fees
  and slippage, plus **perpetual funding**, which for a week-long hold is
  comparable to a full round-trip fee.

---

## 5. Machine learning: when it helps and when it doesn't

**Where ML genuinely adds value**

- **Covariance/structure estimation** — HRP's clustering is ML used well: it
  imposes structure and *reduces* reliance on precise estimates.
- **Regime detection** (HMMs, clustering) — as an input to *risk scaling*, not
  return prediction.
- **Nonlinear feature extraction** where you have genuinely high-dimensional data
  (order book, on-chain) and lots of it.

**Where simpler methods dominate**

- **Return forecasting from price history.** Signal-to-noise is tiny; flexible
  models fit noise. This is where the losses come from.
- **Small samples.** Crypto has a decade of history, much of it one regime.

**Preventing self-deception — the non-negotiable part**

Bailey, Borwein, López de Prado & Zhu, *The Probability of Backtest Overfitting*,
show **the probability of selecting an overfit strategy grows rapidly with the
number of trials**, and propose **CSCV** (combinatorially symmetric
cross-validation) to estimate it.
([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253) ·
[PDF](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf))

Bailey & López de Prado's **Deflated Sharpe Ratio** corrects a reported Sharpe for
**number of trials, sample length, skewness and kurtosis**.
([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) ·
[PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf))

**If you try N strategies and report the best without deflating, you have
reported nothing.**

---

## 6. Comparison table

| Algorithm | Use case | Key weakness | Recommended implementation |
|---|---|---|---|
| **Sample mean–variance** | Teaching | Error-maximising; unstable; needs means | **Do not deploy unconstrained** |
| **Constrained min-variance** | Core allocator | Still needs Σ⁻¹ | Shrunk Σ + weight caps + long-only |
| **CVaR / ES optimisation** | Tail-aware core | Needs many scenarios; sensitive to sample | Rockafellar–Uryasev LP, β=0.95–0.975, historical + stressed scenarios |
| **Inverse volatility** | Robust baseline | Ignores correlation | EWMA vol, weight ∝ 1/σᵢ, capped |
| **Risk parity** | Robust core | Can lever low-vol assets | Equal risk contribution, **hard leverage cap** |
| **HRP** | Many correlated assets | Mixed independent replication | Clustering on shrunk correlation; benchmark against 1/N honestly |
| **Black–Litterman** | Blending views | Views are still your problem | Only with explicitly stated, confidence-weighted views |
| **Robust optimisation** | Parameter uncertainty | Conservative; more complex | Uncertainty sets around Σ |
| **EWMA / GARCH** | Vol forecasting | Assumes persistence | EWMA λ=0.94; realised vol if intraday data exists |
| **DCC-GARCH** | Time-varying correlation | Parameter-heavy | Only if correlation dynamics drive the decision |
| **EVT (GARCH+POT)** | Tail quantiles | Threshold choice | McNeil–Frey two-step |
| **Vol targeting** | Risk overlay | Turnover; some OOS doubt | k = min(k_max, σ_target/σ̂) |
| **1/N** | Benchmark you must beat | Ignores all information | **Always run it. It wins more often than people expect.** |

---

## 7. The recipe

**1 — Screen.** Liquidity floor (depth, spread, days-to-liquidate at target
size); venue quality (regulated/audited — recall the >70% wash trading finding);
minimum history; exclude anything you cannot custody or exit. Build the universe
**as it was at each point in time** to avoid survivorship bias.

**2 — Estimate.** EWMA or realised volatility per asset. Ledoit–Wolf shrunk
correlation. **No expected-return estimates entering the optimiser.**

**3 — Optimise.** CVaR minimisation (β = 0.95) or constrained minimum-variance,
with: long-only or bounded shorts, per-asset cap (e.g. 20–25%), sector/cluster
caps, turnover budget.

**4 — Scale.** Volatility target: k = min(k_max, σ_target/σ̂_portfolio). Cap
k_max well below the leverage at which historical drawdowns liquidate you.

**5 — Execute.** Model fees, slippage and funding at venue-realistic levels.
Prefer participation limits over speed. Assume you get the worse side.

**6 — Rebalance.** Threshold bands (±20–25% relative drift), not the calendar.
Compare expected risk reduction against round-trip cost before trading.

**7 — Stress test.** Historical replays (May 2021, Terra/Luna, FTX); correlations
forced to 1; liquidity halved; a venue set to zero; funding at historical
extremes.

**8 — Monitor.** Realised vs predicted volatility. ES exceedance counts vs
expectation. Drawdown against a pre-committed limit. Turnover vs budget. Weight
drift. **Pre-commit the de-risking rule before you need it.**

---

## 8. Formulas

### 8.1 Expected Shortfall / CVaR optimisation (Rockafellar–Uryasev)

Let `w` be weights, `r` a random return vector, and loss `L(w,r) = −wᵀr`.
For confidence level β ∈ (0,1), define the auxiliary function:

```
F_β(w, α) = α + (1 / (1 − β)) · E[ (L(w, r) − α)⁺ ]
```

where `(x)⁺ = max(x, 0)`.

**The key result:** minimising `F_β(w, α)` jointly over `(w, α)` yields
CVaR_β(w), and the minimising `α*` is VaR_β. You never need to compute VaR
separately — it falls out.

With `S` equally likely scenarios `r₁ … r_S`, this becomes a **linear program**:

```
minimise      α + (1 / ((1 − β) · S)) · Σₛ uₛ
over          w ∈ ℝⁿ, α ∈ ℝ, u ∈ ℝˢ

subject to    uₛ ≥ −wᵀrₛ − α      for all s = 1 … S
              uₛ ≥ 0               for all s
              Σᵢ wᵢ = 1
              0 ≤ wᵢ ≤ w_max       (or bounded shorts)
              (turnover / cluster constraints)
```

Solvable with any LP solver. Source: Rockafellar & Uryasev, *Optimization of
Conditional Value-at-Risk*, **Journal of Risk 2(3):21–42 (2000)**.
([Uryasev PDF](https://www2.mathematik.hu-berlin.de/~romisch/SP01/Uryasev.pdf))

**Scenario choice is the real modelling decision.** Historical scenarios embed
only what happened. Augment with stressed and simulated scenarios, or you are
optimising against a past that will not repeat exactly.

### 8.2 Volatility targeting

Forecast portfolio volatility (EWMA shown; RiskMetrics λ = 0.94 for daily):

```
σ̂²ₜ = λ · σ̂²ₜ₋₁ + (1 − λ) · r²ₜ₋₁
```

Annualise if needed: `σ̂_ann = σ̂ₜ · √P` (P = periods per year).

Scaling factor:

```
kₜ = min( k_max , σ_target / σ̂ₜ )
```

Applied weights: `w_applied = kₜ · w_optimised`; residual `(1 − kₜ)` in cash.

Three things that matter more than the formula:

- **`k_max` is the whole risk decision.** Set it from the drawdown you can
  survive, not from the return you want.
- **Cap turnover.** Recomputing `kₜ` every bar churns the book. Apply a band —
  only adjust when `|kₜ − k_applied| / k_applied > τ`.
- **σ̂ is a forecast.** It will be wrong when it matters most.

### 8.3 Ledoit–Wolf shrinkage

```
Σ̂ = δ · F + (1 − δ) · S
```

`S` = sample covariance, `F` = structured target (constant-correlation or
single-index), `δ* ∈ [0,1]` the analytically optimal intensity.
([Ledoit & Wolf](http://www.ledoit.net/Honey_2004.pdf))

---

## 9. What not to do

### 9.1 Do not run unconstrained Markowitz

It maximises estimation error, concentrates, and is unstable. DeMiguel et al.:
**no model of 14 consistently beat 1/N**; sample mean–variance would need ~3,000
months of data for 25 assets to reliably win.
([RFS 2009](https://academic.oup.com/rfs/article-abstract/22/5/1915/1592901))
If you use mean–variance at all: shrink, constrain, and drop the mean vector.

### 9.2 Do not use normal-distribution VaR

Two independent failures, either fatal for crypto:

1. **Normality is wrong.** Crypto returns are heavy-tailed and skewed. A Gaussian
   VaR assigns negligible probability to moves that occur repeatedly.
2. **VaR is not coherent.** It can violate subadditivity, so it can penalise
   diversification. Basel's FRTB moved to **ES at 97.5%** for exactly this reason.
   ([BPI](https://bpi.com/why-is-the-frtb-expected-shortfall-calculation-designed-as-it-is/))

VaR also says nothing about *how bad* the bad case is — which is the only question
that matters when it happens.

### 9.3 Do not optimise the Sharpe ratio as a single target

- It is **symmetric** — it penalises upside volatility identically to downside.
- It is **blind to skew and kurtosis** — a strategy selling tail risk shows a
  beautiful Sharpe until it doesn't.
- It says **nothing about path**. Two identical Sharpes can have very different
  maximum drawdowns, and drawdown is what forces capitulation.
- It is **trivially inflated by selection**. Hence the Deflated Sharpe Ratio.

Report Sharpe alongside max drawdown, ES, skew, kurtosis, turnover, and capacity.

### 9.4 Do not deploy unvalidated ML

- **Multiple testing without correction.** The probability of selecting an
  overfit strategy grows rapidly with trials. Report the trial count. Use CSCV /
  the Deflated Sharpe Ratio.
- **Look-ahead bias.** Feature scaling on full-sample statistics, survivorship-
  filtered universes, restated fundamentals, and centred windows all leak the
  future while looking fine line by line.
- **Backtest overfitting through iteration.** Re-running with tweaks until it
  works *is* fitting to the test set, however clean the code is.
- **Ignoring costs.** Many published ML edges are smaller than the spread.

### 9.5 Two more, specific to crypto

- **Do not trust reported volume.** >70% of it is fake on unregulated venues.
  Volume-based signals inherit that directly.
- **Do not assume diversification holds in stress.** It demonstrably does not.

---

## 10. Implementation blueprint

```
DATA          point-in-time universe (no survivorship)
              regulated/high-quality venues only
              OHLCV + funding + orderbook depth
              atomic, immutable storage; explicit gap handling
                    │
ESTIMATE      EWMA (λ=0.94) or realised vol per asset
              Ledoit-Wolf shrunk correlation → Σ̂
              NO expected-return model feeds the optimiser
                    │
OPTIMISE      CVaR LP (β=0.95)  [or constrained min-variance]
              long-only or bounded shorts
              per-asset cap ≤ 20-25%; cluster caps; turnover budget
                    │
SCALE         k = min(k_max, σ_target / σ̂_portfolio)
              k_max set from survivable drawdown, not desired return
                    │
EXECUTE       venue-realistic fees + slippage + funding
              participation limits; assume the worse fill
                    │
REBALANCE     threshold bands (±20-25% relative), not calendar
              trade only if expected risk reduction > round-trip cost
                    │
VALIDATE      walk-forward, out-of-sample
              Deflated Sharpe / CSCV with trial count reported
              benchmark vs 1/N and vs buy-and-hold, always
                    │
MONITOR       realised vs predicted vol; ES exceedances
              drawdown vs pre-committed limit; turnover
              pre-committed de-risking rule
```

**Minimum viable version**, if the above is too much at once — and this is a
defensible endpoint, not just a stepping stone:

> Inverse-volatility weights over a liquidity-screened universe, capped at 20%
> per asset, volatility-targeted with a hard leverage cap, rebalanced on ±25%
> drift bands, benchmarked against 1/N and buy-and-hold, with ES monitored and a
> pre-committed drawdown stop.

That uses only variances — the best-estimated inputs — and no correlations,
means, or optimiser. It is very hard to beat robustly, which is the point.

---

## 11. How this maps to the existing project

Honest read of what is already built here versus what this framework requires:

**Already aligned**
- Point-in-time discipline; no look-ahead (there is an explicit causality guard).
- Realistic costs, now with venue-specific fee presets and perpetual funding.
- Out-of-sample walk-forward validation, and trial counts reported.
- Benchmarking against buy-and-hold and against random entry.

**The gap**
The project is currently a **single-strategy signal engine**, not a portfolio
system. It selects *when* to trade one rule; it does not decide *how much* of
each asset to hold. Specifically missing: covariance estimation, any allocator,
volatility targeting, ES monitoring, and a drawdown limit.

**Where the research points, given what the measurements here have already shown**

The searches in this project found no reliable directional edge, a −41% drawdown,
and returns concentrated in one asset. That combination is exactly the profile
this literature says to respond to with **risk management rather than better
prediction**. The highest-value additions, in order:

1. **Volatility targeting** — best-evidenced single overlay; directly attacks the
   drawdown problem.
2. **Inverse-volatility or risk-parity sizing** across the five assets, replacing
   the current fixed 20% per position.
3. **ES monitoring and a pre-committed drawdown limit.**
4. **A 1/N benchmark on the dashboard** — the thing the literature says is
   hardest to beat, and currently absent.

Notably, **none of these require predicting anything.**

---

## Sources

**Harvard / HBS**
- [Campbell & Viceira, *Strategic Asset Allocation*](https://campbell.scholars.harvard.edu/publications/strategic-asset-allocation-portfolio-choice-long-term-investors) · [Oxford Academic](https://academic.oup.com/book/6093)
- [HBS WP 23-073, *Who Invests in Crypto?*](https://www.hbs.edu/faculty/Pages/item.aspx?num=64044) · [PDF](https://www.hbs.edu/ris/Publication%20Files/23-073_2c50c117-a0af-4517-bb0f-06f8212f3177.pdf)

**Portfolio construction**
- [DeMiguel, Garlappi & Uppal, *Optimal Versus Naive Diversification*, RFS 2009](https://academic.oup.com/rfs/article-abstract/22/5/1915/1592901) · [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1376199)
- [Rockafellar & Uryasev, *Optimization of CVaR*, J. Risk 2000](https://www2.mathematik.hu-berlin.de/~romisch/SP01/Uryasev.pdf) · [Semantic Scholar](https://www.semanticscholar.org/paper/Optimization-of-conditional-value-at-risk-Rockafellar-Uryasev/58444c142b6ea5c71a435cac7a0b4c66d6c68869)
- [Ledoit & Wolf, *Honey, I Shrunk the Sample Covariance Matrix*](http://www.ledoit.net/Honey_2004.pdf) · [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=433840)
- [López de Prado, *Building Diversified Portfolios that Outperform Out of Sample*, JPM 2016](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678) · [JPM](https://jpm.pm-research.com/content/42/4/59.short)
- [HRP independent comparison, *Empirical Economics*](https://link.springer.com/article/10.1007/s00181-026-02900-x) · [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167739X25000391)
- [Moreira & Muir, *Volatility-Managed Portfolios*, JF 2017](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513) · [NBER w22208](https://www.nber.org/system/files/working_papers/w22208/w22208.pdf)
- [DeMiguel et al., *A Multifactor Perspective on Volatility-Managed Portfolios*, JF 2024](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13395) · [Cederburg et al., JFE](https://www.sciencedirect.com/science/article/abs/pii/S0304405X2030132X)

**Risk modelling**
- [Engle, *Dynamic Conditional Correlation*, JBES 2002](https://www.tandfonline.com/doi/abs/10.1198/073500102288618487) · [PDF](https://pages.stern.nyu.edu/~rengle/dccfinal.pdf)
- [Hansen & Lunde, *Does anything beat a GARCH(1,1)?*](https://ideas.repec.org/a/jae/japmet/v20y2005i7p873-889.html)
- [McNeil & Frey EVT approach — Bitcoin dynamic POT application](https://www.sciencedirect.com/science/article/abs/pii/S1544612322003129) · [Autoregressive EVT for cryptocurrencies](https://pure.hud.ac.uk/ws/files/20412782/Autoregressive_EVT_to_Cryptocurrencies_accepted_version.pdf)
- [Why FRTB uses Expected Shortfall (Bank Policy Institute)](https://bpi.com/why-is-the-frtb-expected-shortfall-calculation-designed-as-it-is/)

**Crypto-specific**
- [Liu, Tsyvinski & Wu, *Common Risk Factors in Cryptocurrency*, NBER w25882](https://www.nber.org/papers/w25882)
- [Liu & Tsyvinski, *Risks and Returns of Cryptocurrency*, NBER w24877](https://www.nber.org/papers/w24877)
- [Cong, Li, Tang & Yang, *Crypto Wash Trading*, Management Science 2023](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2021.02709) · [NBER w30783](https://www.nber.org/system/files/working_papers/w30783/w30783.pdf) · [Cowles PDF](https://cowles.yale.edu/sites/default/files/2022-11/cryptowashtrading040521-crypto-wash-trading.pdf)
- [BIS crypto ecosystem report 2023](https://www.regulationtomorrow.com/2023/07/bis-report-on-the-crypto-ecosystem-key-elements-and-risks/) · [BIS stablecoin bulletin](https://www.bis.org/publ/bisbull108.pdf) · [BIS WP 1270](https://www.bis.org/publ/work1270.pdf)
- [Crypto–equity contagion, *Cogent Economics & Finance* 2023](https://www.tandfonline.com/doi/full/10.1080/23322039.2023.2203432) · [IMF WP 2023/213](https://www.elibrary.imf.org/view/journals/001/2023/213/article-A001-en.xml)
- [CFA Institute, *Cryptoassets* brief](https://rpc.cfainstitute.org/sites/default/files/-/media/documents/article/rf-brief/rfbr-cryptoassets.pdf) · [CFA Institute cryptoassets guide](https://rpc.cfainstitute.org/research/foundation/2021/cryptoassets) · [Valuation of Cryptoassets](https://rpc.cfainstitute.org/research/reports/2023/valuation-cryptoassets)

**Validation / overfitting**
- [Bailey, Borwein, López de Prado & Zhu, *The Probability of Backtest Overfitting*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253) · [PDF](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
- [Bailey & López de Prado, *The Deflated Sharpe Ratio*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) · [PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
