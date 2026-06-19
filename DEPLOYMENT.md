# UK ISA Deployment Spec — trend-gated leveraged-ETP rotation

The research is closed. This is the runbook for trading the strategy that
survived the audit: a 200-day trend filter with a volatility gate on leverage,
diversified across the Nasdaq-100 and S&P 500, run inside a Trading 212 Stocks
& Shares ISA. Generate the live book with `python scripts/uk_signal.py`.

## What was established (and how)

- **The edge is real and general.** Trend-gated, vol-stepped leverage beat
  buy-and-hold in *both* halves of 2000–2026 on *both* indices, after a 3.4%/yr
  real-product cost haircut (`scripts/uk_robustness.py`). The gate's job is
  crisis control: constant 3× lost −5.9%/yr through dot-com+GFC; the gated
  version made +6.7%/yr by stepping to 1× when volatility spiked.
- **The synthetic ETP model is trustworthy** — it matches the real TQQQ/QLD at
  correlation 1.000/0.999. The real LSE products tracked ~3.4%/yr *below* it
  (the haircut above), which is modelled, not ignored.
- **Leverage is regime-dependent.** Almost the entire leverage premium is
  post-2014; in a dot-com/GFC-style decade, full leverage adds only ~2%/yr over
  the unleveraged core while roughly doubling the drawdown. Size accordingly.

## Choose your point on the frontier (3× satellite, after costs)

| satellite | core | CAGR | worst drawdown | £10k → (27y) |
|----------:|-----:|-----:|---------------:|-------------:|
| 0%  | 100% | 8.6%  | −30% | £95k  |
| 25% | 75%  | 10.6% | −33% | £154k |
| **50%** | **50%** | **12.1%** | **−36%** | **£227k** |
| 75% | 25%  | 13.3% | −43% | £302k |
| 100%| 0%   | 14.1% | −52% | £364k |

**Recommended: 50% satellite / 50% core.** It captures most of the upside on
the efficient part of the curve; 75% is defensible only if you have genuinely
held a −43% loss without selling. 100% is the worst risk-adjusted point and the
most fragile to the regime not repeating — avoid. The decision is one honest
number: the deepest drawdown you will hold through. Pick the row for *that*.

## The instruments (all LSE-listed, UCITS, ISA-eligible)

| sleeve | role | ticker |
|---|---|---|
| Nasdaq core | 1× Nasdaq-100 | **EQQQ** (Invesco) |
| Nasdaq satellite | 3× Nasdaq-100 daily | **QQQ3** (WisdomTree) |
| S&P core | 1× S&P 500 | **CSPX** (iShares Core) |
| S&P satellite | 3× S&P 500 daily | **3USL** (WisdomTree) |
| risk-off (satellite) | physical gold | **SGLN** (iShares) |
| risk-off (core) | cash / money-market | cash or **CSH2** |

## The rules (what `uk_signal.py` computes)

Per index, independently:
1. **Trend filter.** In when the index closed above its 200-day average for 2
   straight days; out when it closed below 200d × 0.99 for 2 straight days
   (the band + confirmation kill whipsaws).
2. **Vol gate (only while trend-on).** "Calm" when 40-day EWMA annualised vol
   < 20%; "loud" when > 28%; hold the prior state in between (hysteresis).
3. **Position.** Trend-off → risk-off asset (gold for the satellite, cash for
   the core). Trend-on + calm → leveraged ETP. Trend-on + loud → 1× fund.

Book = `(1 − sat) ×` core + `sat ×` satellite, each split 50/50 Nasdaq/S&P.

## Monthly runbook

1. First trading day of the month, run `python scripts/uk_signal.py --fresh
   --sat 0.50`.
2. Compare the TARGET BOOK to current holdings; trade only the differences.
3. Heed the WATCH lines — if a trend or vol flip is near, you may prefer to act
   at month-end rather than mid-move. Do **not** trade intra-month on noise; the
   signal is slow by design (the lag study showed being a few days late helps).
4. Keep contributions within the £20k/yr ISA allowance.

## Risks — read before funding

- **Drawdown is the real cost.** −36% at the recommended weight means £100k →
  £64k, possibly for a year or more. Leverage only ever widens this.
- **Leveraged ETPs can approach zero** in a sustained, choppy decline; the
  trend filter mitigates but does not eliminate this.
- **Broker gate:** Trading 212 places leveraged ETPs behind an appropriateness
  questionnaire; confirm QQQ3/3USL are buyable in *your* ISA before relying on
  this. The 3× S&P UCITS is younger/thinner than QQQ3 — check liquidity.
- **Verify the haircut.** Pull the real QQQ3/3USL NAV history and confirm the
  tracking gap is ~3.4%/yr, not worse, before committing size.

## Before going live: forward-validate

Deploy a small fraction first and run the monthly process for 6–12 months.
Confirm the live trend/gate switches and the realised ETP drag match the model
before scaling up. The backtest earned a live trial — not a lump-sum leap.
