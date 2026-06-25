# Fixed Income Monte Carlo VaR Engine — SR 11-7 Model Risk Documentation

| Field | Value |
|-------|-------|
| **Document Version** | 3.0 |
| **Effective Date** | 2026-04-23 |
| **Model Name** | Fixed Income Monte Carlo VaR Engine |
| **Model Type** | Market Risk — Interest Rate & Spread |
| **Model Owner** | Quantitative Risk Management |
| **Validation Status** | Draft — Pending Independent MRM Review |
| **Active Rate Model** | Hull-White / Extended Vasicek (1990) |
| **Regulatory Framework** | Basel III IMA / SR 11-7 / OCC 2011-12 |

---

## 1. Model Overview

### 1.1 Purpose

This engine provides daily Value at Risk (VaR) and Conditional Value at Risk (CVaR / Expected Shortfall)
estimates for a diversified fixed-income portfolio comprising U.S. Treasuries, investment-grade and
high-yield corporate bonds, agency mortgage-backed securities (MBS), and asset-backed securities (ABS).
Results support:

- **Regulatory capital** computation under the Basel III Internal Models Approach (IMA).
- **Internal risk limits** monitoring and daily risk reporting to senior management.
- **Stress testing** under prescribed macroeconomic and rate-shock scenarios.
- **Model validation** through backtesting, benchmark comparison, and sensitivity analysis.

### 1.2 Computational Architecture

```
Yield Curve Inputs  →  Nelson-Siegel Fit  →  Smooth Zero-Rate Function
                                                        │
                              Short-Rate Model Calibration
                                    [Vasicek | Hull-White]
                                                        │
                     Monte Carlo Simulation  (10,000 paths × 252 steps)
                     + Antithetic Variates  (variance ÷ 2 at zero extra cost)
                                                        │
                              Full Portfolio Repricing
                         ├── Coupon Bonds: Analytical ZCB discounting
                         ├── MBS:          PSA prepayment model (rate-sensitive CPR)
                         └── ABS:          Effective duration (scheduled amortisation)
                                                        │
                     P&L Distribution  →  VaR (99%), CVaR (99%)
                                                        │
                         ├── Stress Testing  (6 scenarios, Nelson-Siegel shocked curves)
                         └── Backtesting & Benchmark Validation
```

### 1.3 Active Rate Model: Hull-White / Extended Vasicek (1990)

**SDE**: `dr_t = (θ(t) − a·r_t) dt + σ dW_t`

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Mean-reversion speed | a | 0.1000 | Controls speed of mean reversion |
| Volatility           | σ | 0.0120 | Diffusion coefficient |
| Initial short rate   | r₀ | 0.0450 | Current observed short rate |
| Drift function | θ(t) | Calibrated | Fit-to-market via Brigo-Mercurio cubic-spline |

**Discretisation**: Euler-Maruyama with time-varying drift θ(t) computed analytically from cubic-spline-fitted yield curve.

**ZCB Pricing**: Brigo-Mercurio market-consistent formula; exact fit to initial yield curve at t=0.

### 1.4 Yield Curve: Nelson-Siegel (1987)

```
y(τ) = β₀  +  β₁ · (1 − e^{-τ/λ}) / (τ/λ)  +  β₂ · [(1 − e^{-τ/λ}) / (τ/λ) − e^{-τ/λ}]
```

| Parameter | Interpretation |
|-----------|---------------|
| β₀ | Long-run level (limit as τ → ∞) |
| β₁ | Slope loading (short-end adjustment) |
| β₂ | Curvature / hump factor |
| λ  | Decay factor (location of hump) |

Fitted via Nelder-Mead to observed zero rates. Applied to both base and stressed yield curves.
Advantages over linear interpolation: smooth extrapolation, no kinks at knot tenors, economically
interpretable parameters.

### 1.5 MBS Prepayment: PSA Standard (1985) with Rate Sensitivity

```
CPR(m)   = min(m/30, 1) × 6% × (PSA_speed/100%)      [PSA ramp]
CPR_eff(r) = clip(CPR_base × (1 + β_refi × (r₀ − r)), 10%, 400%)
SMM      = 1 − (1 − CPR_eff)^(1/12)                   [Single Monthly Mortality]
```

`β_refi = 8.0`: a 100 bps rate decline accelerates CPR by 8 percentage points (refinancing wave).
MBS cash flows are projected month-by-month and discounted with model ZCB factors — replacing the
effective-duration approximation used in prior versions.

### 1.6 Variance Reduction: Antithetic Variates

For each seed, n/2 standard-normal draws Z are generated; paths are simulated for both Z and −Z.
This exploits the symmetry of the normal distribution to cancel first-order Monte Carlo error,
halving estimator variance without additional function evaluations.

---

## 2. Portfolio Composition

| Metric | Value |
|--------|-------|
| **Number of Positions** | 11 |
| **Total Notional** | $     97,000,000 |
| **Mark-to-Market Value** | $     90,661,195 |
| **Portfolio DV01** | $          5,828 |
| **Weighted Avg. Effective Duration** | 6.01 years |

**Allocation by Product Type**

product_type  Notional ($)  % of Portfolio
    abs_auto       5000000             5.2
    abs_card       4000000             4.1
     corp_hy       3000000             3.1
     corp_ig      15000000            15.5
         mbs      20000000            20.6
    treasury      50000000            51.5

---

## 3. Key Model Assumptions

| # | Assumption | Rationale |
|---|-----------|-----------|
| 1 | Short rate follows a single-factor mean-reverting Gaussian diffusion (no jumps) | Parsimony; analytical ZCB tractability; well-established in literature |
| 2 | 2. Hull-White θ(t) calibrated to initial yield curve via Brigo-Mercurio §3.3 formula, ensuring P(0,T)_model = P(0,T)_market for all T. | Ensures market-consistent pricing at t = 0 |
| 3 | Yield curve fitted via Nelson-Siegel (1987); shocked curves re-fitted to stressed tenors | Smooth, arbitrage-free interpolation and extrapolation beyond observed tenors |
| 4 | MBS repriced via PSA prepayment model with rate-sensitive CPR (β_refi = 8.0); ABS by effective duration | PSA captures negative convexity and extension/call risk; ABS has near-scheduled amortisation |
| 5 | Monte Carlo variance reduced via antithetic variates (n/2 independent draws, n/2 mirror draws) | Halves estimator variance; improves tail-quantile stability at fixed path count |
| 6 | Credit spreads held constant within the 1-day VaR horizon | Spread risk captured in dedicated stress scenarios (credit_spread_widen_50bps) |
| 7 | Static portfolio over the holding period; 1-day VaR scaled by √10 for 10-day regulatory capital | Conservative approximation; standard Basel III IMA provision |

---

## 4. Known Limitations

| # | Limitation | Potential Impact | Mitigant |
|---|-----------|-----------------|---------|
| 1 | Single-factor model: cannot independently capture curve twists | Underestimates non-parallel moves (e.g. bear flattener) | Steepener/flattener stress scenarios |
| 2 | Constant credit spreads in Monte Carlo simulation | Underestimates spread-jump and default risk | Credit spread widening stress scenario (+50 bps) |
| 3 | PSA CPR ignores burnout, WAC dispersion, and seasoning heterogeneity | Minor overshoot of prepayment in seasoned pools | Periodic PSA speed recalibration; OAS monitoring |
| 4 | Gaussian rate distribution permits negative short rates | Small probability mass below zero in low-rate environments | Monitor; consider log-normal or shifted log-normal extension |
| 5 | Euler-Maruyama for Hull-White is step-size dependent | Discretisation bias for large dt | 1-day dt (1/252) is very small; bias is negligible in practice |
| 6 | No rate–credit spread correlation in Monte Carlo | Flight-to-quality dynamics not fully captured in base VaR | Recession/flight-to-quality stress scenario |
| 7 | √10 scaling assumes i.i.d. daily returns | Overestimates 10-day VaR when mean reversion is strong | Conservative for regulatory capital purposes |

---

## 5. Scope of Application

### 5.1 Approved Use Cases

- Daily regulatory VaR and CVaR reporting under Basel III IMA
- Internal risk limits monitoring and breach reporting
- Stress testing and scenario analysis for ICAAP / ILAAP
- Model benchmarking and sensitivity-based risk limit calibration

### 5.2 Instruments Covered

| Instrument | Pricing Method | Coverage |
|-----------|---------------|---------|
| U.S. Treasuries | Analytical ZCB (Vasicek / HW) | ✅ Full |
| IG / HY Corporate Bonds | Analytical ZCB + constant OAS | ✅ Full |
| Agency MBS (FNMA / FHLMC) | PSA prepayment + ZCB discounting | ✅ Full |
| Auto / Card ABS | Effective duration | ✅ Approximate |
| CMBS, CLO, non-agency MBS | Not modelled | ❌ Out of scope |
| Equity, FX, Commodity | Not modelled | ❌ Out of scope |
| Derivatives (swaps, options) | Not modelled | ❌ Out of scope |

### 5.3 Not Approved For

- Real-time pricing or bid-offer quoting
- Individual trade P&L attribution or P&L explain
- Multi-currency portfolios (USD only in current version)
- Exotic instruments or non-linear payoffs

---

## 6. Risk Results (As of 2026-04-23)

### 6.1 VaR and CVaR

| Metric | Value |
|--------|-------|
| **Monte Carlo VaR (99%, 1-day)** | **$575,496** |
| **Monte Carlo CVaR (99%, 1-day)** | **$667,580** |
| VaR as % of Portfolio MtM | 0.635% |
| CVaR / VaR Ratio | 1.16× |
| **10-day Regulatory VaR (√10 scaled)** | **$1,819,879** |

### 6.2 Stress Test Results

| Scenario | Stressed P&L | % of Portfolio |
|----------|-------------|----------------|
| Parallel Up 200Bps | $-9,771,543 | -10.78% |
| Parallel Down 100Bps | $5,542,700 | 6.11% |
| Steepener 30S10S | $-7,811,588 | -8.62% |
| Flattener 30S10S | $7,352,275 | 8.11% |
| Credit Spread Widen 50Bps | $-21,862,300 | -24.11% |
| Recession Flight To Quality | $8,506,975 | 9.38% |

*Note: Stressed yield curves re-fitted via Nelson-Siegel. MBS repriced with rate-sensitive PSA CPR.*

---

## 7. Model Validation Results

### 7.1 VaR Backtesting (250-day window)

#### 7.1.1 Kupiec Unconditional Coverage (POF) Test

*H₀: Exception frequency = α = 1%*

| Exceptions Observed | Exceptions Expected | Exception Rate | LR Statistic (χ²₁) | p-value | Result |
|--------------------|--------------------|--------------|--------------------|---------|--------|
| 0 | 2 | 0.0000 | inf | 0.0000 | **FAIL** |

*Interpretation*: Zero exceptions indicate VaR is conservative (over-estimates risk). Statistically a FAIL on the POF test, but from a regulatory standpoint this is a conservative outcome.

#### 7.1.2 Christoffersen Conditional Coverage Test

*H₀: Exceptions are independently distributed (no clustering)*

| π₀₁ | π₁₁ | LR Independence | LR Combined (χ²₂) | p-value | Result |
|-----|-----|----------------|-------------------|---------|--------|
| 0.0000 | 0.0000 | -0.0000 | inf | 0.0000 | **FAIL** |

#### 7.1.3 Basel Traffic Light Assessment

| Exceptions (normalised to 250 days) | Zone | Signal | Regulatory Capital Multiplier |
|-------------------------------------|------|--------|------------------------------|
| 0 | **Green** | **PASS** | 3.0× |

### 7.2 Benchmark Comparison: Monte Carlo vs. Historical Simulation

| Method | VaR (99%, 1-day) | CVaR (99%, 1-day) |
|--------|------------------------|-------------------------|
| **Monte Carlo (HULL_WHITE)** | **$575,496** | **$667,580** |
| Historical Simulation | $484,497 | $530,746 |
| Absolute Difference | $+90,999 | $+136,834 |
| Relative Difference | +18.8% | +25.8% |

*Assessment*: Material divergence detected (18.8% > 15%). Investigate tail distributional assumptions — consider t-distribution extension.

### 7.3 Parameter Sensitivity Analysis

Perturbation VaR_99 ($) CVaR_99 ($) ΔVaR (%) ΔCVaR (%)
    Baseline    575,496     667,580        —         —
    σ × 1.25    729,576     841,265   +26.8%    +26.0%
    σ × 0.75    418,718     485,986   -27.2%    -27.2%
    a × 1.50    493,368     570,595   -14.3%    -14.5%
    a × 0.50    685,492     791,850   +19.1%    +18.6%

*Interpretation*: Volatility (σ) is the dominant driver of VaR. A ±25% perturbation in σ produces
an approximately proportional VaR response. Mean-reversion speed (a / κ) has a second-order effect
over a 1-day horizon but becomes material at longer horizons. See heatmap_alpha_sigma.png for the
full α×σ two-dimensional sensitivity surface.

---

## 8. SR 11-7 Compliance Statement

This document has been prepared in accordance with the Federal Reserve's
**Supervisory Guidance on Model Risk Management (SR 11-7, April 2011)**
and OCC Bulletin 2011-12.

### 8.1 Compliance Checklist

| SR 11-7 Requirement | Status | Reference |
|---------------------|--------|-----------|
| Model purpose and intended use documented | ✅ Complete | §1.1 |
| Computational architecture described | ✅ Complete | §1.2 |
| Conceptual soundness and theory reviewed | ✅ Complete | §1.3–1.6 |
| Key assumptions documented with rationale | ✅ Complete | §3 |
| Known limitations and compensating controls | ✅ Complete | §4 |
| Scope of application and out-of-scope instruments | ✅ Complete | §5 |
| Quantitative performance metrics reported | ✅ Complete | §6 |
| Backtesting — Kupiec unconditional coverage | ✅ Complete | §7.1.1 |
| Backtesting — Christoffersen conditional coverage | ✅ Complete | §7.1.2 |
| Basel Traffic Light assessment | ✅ Complete | §7.1.3 |
| Benchmark / challenger model comparison | ✅ Complete | §7.2 |
| Sensitivity and stability analysis | ✅ Complete | §7.3 |
| Ongoing monitoring plan | ⚠️ Recommended | Quarterly recalibration |
| Independent MRM validation | ⚠️ Pending | Pre-production requirement |
| Model approval and sign-off | ⚠️ Pending | Senior Risk Officer / CRO |
| Model inventory registration | ⚠️ Pending | Post-approval |

### 8.2 Attestation

> *This model documentation reflects the development state of the Fixed Income Monte Carlo VaR
> Engine v3.0 as of 2026-04-23. The following enhancements have been implemented relative to prior
> versions: (1) Nelson-Siegel yield curve fitting replacing linear interpolation; (2) PSA
> prepayment model for MBS replacing effective-duration approximation; (3) Antithetic Variates
> for Monte Carlo variance reduction; (4) α×σ two-dimensional VaR sensitivity heatmap.*
>
> *The model has not yet received independent Model Risk Management (MRM) validation.
> Deployment in a production risk system requires independent validation, formal approval by the
> Chief Risk Officer, and registration in the firm's Model Inventory in accordance with
> SR 11-7 §IV. Material changes to methodology, calibration, or scope require
> re-documentation and re-validation before deployment.*

---

*Fixed Income VaR Engine v3.0 | 2026-04-23 | SR 11-7 | OCC 2011-12 | Basel III IMA*
*Nelson-Siegel (1987) | PSA Standard (1985) | Vasicek (1977) | Hull & White (1990)*
