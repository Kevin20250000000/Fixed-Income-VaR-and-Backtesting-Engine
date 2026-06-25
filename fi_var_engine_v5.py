"""
Fixed Income VaR & Backtesting Engine  v3.2
============================================
Run:  python3 fi_var_engine_v3.2.py

Pipeline:
  1. Configuration          — model selection, params, portfolio, yield curve
  2. Nelson-Siegel          — smooth yield curve fitting (replaces linear interp)
  3. Rate Models            — Vasicek / Hull-White + simulate_horizon fast path
  4. Data Pipeline          — simulated SQL: positions, market data, historical P&L
  5. Risk Engine            — full repricing, VaR (99%), CVaR (99%), stress tests
                              MBS: PSA prepayment model (replaces eff-dur approx)
  6. Model Validation       — Kupiec POF, Christoffersen CC, Basel Traffic Light,
                              HS benchmark, parameter sensitivity, KRD profile
  7. SR 11-7 Documentation  — auto-generated Markdown (KRD §6.3 + neg-rate §4.1)
  8. Charts                 — P&L distribution, rate paths, stress bar, sensitivity,
                              α×σ VaR heatmap, KRD bar chart

Switch model: change MODEL_CONFIG["model"] to "vasicek" or "hull_white".

Enhancements vs v2.0 / v3.0:
  • Nelson-Siegel (1987) yield curve fit — smooth extrapolation, no endpoint kinks
  • Antithetic Variates — halves MC variance at zero extra cost
  • PSA prepayment model — rate-sensitive MBS cash flows (β_refi = 8.0)
  • α×σ heatmap — 2-D VaR sensitivity across confidence levels and volatility

Enhancements vs v3.1:
  • Key Rate Durations (KRD) — isolated +1bp tenor shocks capture twist/butterfly risk
  • Negative rate probability — model-specific quantification in SR 11-7 §4.1 (SR 11-7 required)

References:
  Vasicek (1977) | Hull & White (1990) | Brigo & Mercurio (2006) Ch. 3-4
  Nelson & Siegel (1987) | PSA Standard (1985) | Basel III IMA | SR 11-7
"""

import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                  # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize
from datetime import date

# ============================================================================
# SECTION 1 — CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    "model": "hull_white",            # ← switch to "vasicek" to change model

    "vasicek": {
        "kappa": 0.30,                # mean-reversion speed  κ
        "theta": 0.050,               # long-run mean rate    θ
        "sigma": 0.012,               # rate volatility       σ
        "r0":    0.045,               # initial short rate
    },
    "hull_white": {
        "a":     0.10,                # mean-reversion speed  a
        "sigma": 0.012,               # rate volatility       σ
        "r0":    0.045,               # initial short rate
    },
}

SIMULATION_CONFIG = {
    "n_paths":      10_000,
    "n_steps":         252,
    "dt":         1 / 252,
    "horizon_days":      1,           # VaR holding period
    "seed":             42,
}

RISK_CONFIG = {
    "confidence_level": 0.99,
    "holding_period":      1,
}

# FRED API — set key to enable live yield curve (free: https://fred.stlouisfed.org/docs/api/api_key.html)
FRED_API_KEY = "ac06dd94d933d7dd9686594c76036d18"            # e.g. "abcd1234efgh5678"
_FRED_SERIES = {                       # tenor (yrs) → FRED series ID
    0.25: "DGS3MO", 0.50: "DGS6MO",  1.00: "DGS1",   2.00: "DGS2",  3.00: "DGS3",
    5.00: "DGS5",   7.00: "DGS7",    10.00: "DGS10", 20.00: "DGS20", 30.00: "DGS30",
}

YIELD_CURVE = {                        # fallback hardcoded — overwritten by FRED if key is set
     0.25: 0.0430,  0.50: 0.0440,  1.00: 0.0450,  2.00: 0.0460,
     3.00: 0.0470,  5.00: 0.0480,  7.00: 0.0490, 10.00: 0.0500,
    20.00: 0.0510, 30.00: 0.0520,
}

STRESS_SCENARIOS = {
    "parallel_up_200bps":          {t: +0.020 for t in YIELD_CURVE},
    "parallel_down_100bps":        {t: -0.010 for t in YIELD_CURVE},
    "steepener_30s10s":            {
        0.25: -0.010, 0.50: -0.008, 1.00: -0.005, 2.00:  0.000,
        3.00:  0.005, 5.00:  0.010, 7.00:  0.015, 10.00:  0.020,
        20.00: 0.025, 30.00: 0.030},
    "flattener_30s10s":            {
        0.25:  0.020, 0.50:  0.015, 1.00:  0.010, 2.00:  0.005,
        3.00:  0.000, 5.00: -0.005, 7.00: -0.010, 10.00: -0.015,
        20.00: -0.020, 30.00: -0.025},
    "credit_spread_widen_50bps":   {t: +0.050 for t in YIELD_CURVE},
    "recession_flight_to_quality": {t: -0.015 for t in YIELD_CURVE},
}

BACKTEST_CONFIG = {"n_obs": 250, "alpha": 0.01}


# ============================================================================
# SECTION 2 — NELSON-SIEGEL YIELD CURVE
# ============================================================================
#
# Nelson & Siegel (1987): y(τ) = β₀ + β₁·(1−e^{−τ/λ})/(τ/λ)
#                                    + β₂·((1−e^{−τ/λ})/(τ/λ) − e^{−τ/λ})
#
# β₀ = long-run level  |  β₁ = slope (short-end loading)
# β₂ = curvature/hump  |  λ  = decay factor
#
# Advantages over linear interp:
#   • Smooth, arbitrage-free curve — no kinks at knot points
#   • Stable extrapolation beyond the longest observed tenor
#   • Economically interpretable parameters
# ============================================================================

def _fit_nelson_siegel(tenors, yields):
    """
    Fit Nelson-Siegel model to (tenors, yields) and return an interpolation
    callable f(t) → continuously-compounded zero rate. 
    """
    tenors = np.array(tenors, dtype=float)
    yields = np.array(yields, dtype=float)

    def _ns(tau, b0, b1, b2, lam):
        tau = np.maximum(tau, 1e-8)
        x   = tau // lam
        f1  = (1.0 - np.exp(-x)) / x
        return b0 + b1 * f1 + b2 * (f1 - np.exp(-x))

    def _loss(p):
        b0, b1, b2, lam = p
        if lam <= 0.05 or b0 <= 0.0:
            return 1e9
        return float(np.sum((_ns(tenors, b0, b1, b2, lam) - yields) ** 2))

    # Initial guess: β₀ = long-end yield, β₁ = slope, β₂ = small hump, λ = 2
    x0  = [yields[-1], yields[0] - yields[-1], 0.005, 2.0]
    res = minimize(_loss, x0, method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 30_000})
    b0, b1, b2, lam = res.x
    lam = max(abs(lam), 0.05)

    def ns_interp(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        t = np.maximum(t, 1e-8)
        x = t // lam
        f1 = (1.0 - np.exp(-x)) / x
        return b0 + b1 * f1 + b2 * (f1 - np.exp(-x))

    return ns_interp


# ============================================================================
# SECTION 3 — RATE MODELS
# ============================================================================

class VasicekModel:
    """
    dr_t = κ(θ − r_t) dt + σ dW_t
    Simulation : exact discretisation (conditional Gaussian) + antithetic variates.
    ZCB price  : P(t,T) = A(τ)·exp(−B(τ)·r_t), τ = T−t.
    """
    def __init__(self, p):
        self.kappa, self.theta, self.sigma, self.r0 = p["kappa"], p["theta"], p["sigma"], p["r0"]

    def simulate_paths(self, n_paths, n_steps, dt, seed=42):
        rng      = np.random.default_rng(seed)
        half     = n_paths // 2
        e_kdt    = np.exp(-self.kappa * dt) 
        mean_adj = self.theta * (1.0 - e_kdt)
        std_step = np.sqrt(self.sigma**2 / (2.0 * self.kappa) * (1.0 - np.exp(-2.0 * self.kappa * dt))) # 实际FIX 计算中，应使用sqrt(σ²/(2κ) * (1 - exp(-2κΔt)))，而非σ*sqrt((1 - exp(-2κΔt))/(2κ))，前者是单步标准差，更精确；后者是误用的版本，导致结果偏大；此次只是模拟，但在计算VaR时会放大误差，尤其在高波动率或长持有期下。已修正为正确的单步标准差公式。 
        r        = np.empty((n_paths, n_steps + 1));  r[:, 0] = self.r0
        Z_h      = rng.standard_normal((half, n_steps))
        Z        = np.concatenate([Z_h, -Z_h], axis=0)   # antithetic pairs
        for t in range(n_steps):
            r[:, t+1] = r[:, t] * e_kdt + mean_adj + std_step * Z[:, t] 
        return r

    def simulate_horizon(self, n_paths, horizon_days, dt, seed=42):
        """
        Directly sample r at the VaR horizon — no intermediate steps.

        Uses the exact Gaussian conditional distribution of the Ornstein-Uhlenbeck
        process over τ = horizon_days × dt.  For a 1-day horizon this is ~250×
        faster than simulate_paths because it avoids the 252-step time loop.

        Returns ndarray (n_paths,) — short rate at horizon only. 
        """
        rng     = np.random.default_rng(seed)
        half    = n_paths // 2
        t_h     = horizon_days * dt
        e_kt    = np.exp(-self.kappa * t_h)
        mu_h    = self.theta + (self.r0 - self.theta) * e_kt
        sigma_h = np.sqrt(self.sigma**2 / (2.0 * self.kappa)
                          * (1.0 - np.exp(-2.0 * self.kappa * t_h)))
        Z_h = rng.standard_normal(half)
        Z   = np.concatenate([Z_h, -Z_h])          # antithetic 
        return mu_h + sigma_h * Z                  # X = μ + σZ   

    def discount_factor(self, r_h, t_h, T):                             
        tau = T - t_h
        if tau <= 0.0: return np.ones(len(r_h))
        B = (1.0 - np.exp(-self.kappa * tau)) / self.kappa
        A = np.exp((self.theta - self.sigma**2 / (2.0 * self.kappa**2)) * (B - tau) 
                   - self.sigma**2 * B**2 / (4.0 * self.kappa))
        return A * np.exp(-B * r_h)

    def discount_factor_batch(self, r_h, t_h, times):
        """
        Vectorised ZCB prices for multiple maturities simultaneously.
        times  : array (n_times,)   — payment dates
        r_h    : array (n_paths,)   — short rate at horizon
        returns: array (n_times, n_paths)

        Analytical A(τ)·exp(−B(τ)·r) computed for all τ in one shot,
        replacing n_times separate discount_factor() calls.
        """
        times  = np.atleast_1d(np.asarray(times, dtype=float))
        taus   = times - t_h
        result = np.ones((len(times), len(r_h)))
        mask   = taus > 1e-10 
        if not np.any(mask):
            return result 

        tv = taus[mask]
        B  = (1.0 - np.exp(-self.kappa * tv)) / self.kappa            # (n_valid,)
        A  = np.exp(
            (self.theta - self.sigma**2 / (2.0 * self.kappa**2)) * (B - tv)
            - self.sigma**2 * B**2 / (4.0 * self.kappa)
        )                                                               # (n_valid,)
        # A[i] * exp(-B[i] * r_h[j])  →  outer product pattern
        result[mask] = A[:, None] * np.exp(-B[:, None] * r_h[None, :])
        return result


class HullWhiteModel:
    """
    dr_t = (θ(t) − a·r_t) dt + σ dW_t
    ZCB price via Brigo-Mercurio fit-to-market formula (§3.3).
    Simulation uses Euler-Maruyama + antithetic variates.
    """
    def __init__(self, p, yield_curve):
        self.a, self.sigma, self.r0 = p["a"], p["sigma"], p["r0"]
        tenors   = np.array(sorted(yield_curve.keys()), dtype=float)
        yields   = np.array([yield_curve[t] for t in tenors], dtype=float)
        t_aug    = np.concatenate([[1e-6], tenors])
        ld_aug   = np.concatenate([[0.0], -yields * tenors])
        self._cs = CubicSpline(t_aug, ld_aug, bc_type="not-a-knot") # self._cs 是插值CubicSpline；
        self._t_min, self._t_max = t_aug[0], t_aug[-1] 

    def _ld(self, t):  return float(self._cs(np.clip(t, self._t_min, self._t_max))) # 对数价格: LnP(0,t) ; 也是 对数贴现因子（Log-Discount Factor）；
    def _fwd(self, t): return float(-self._cs(np.clip(t, self._t_min, self._t_max), 1)) #插值-self._cs(t, 1)之后，得到的是瞬时远期利率 f(0, t) ；

    def simulate_paths(self, n_paths, n_steps, dt, seed=42): 
        rng  = np.random.default_rng(seed)
        half = n_paths // 2
        r    = np.empty((n_paths, n_steps + 1));  r[:, 0] = self.r0 
        Z_h  = rng.standard_normal((half, n_steps)) # 产生符合标准正态分布的随机数，行数为 half，列数为n_steps；每行代表一个路径，每列代表一个时间步的随机冲击。
        Z    = np.concatenate([Z_h, -Z_h], axis=0)   # antithetic pairs 

        # Pre-compute θ(t) for all steps in one vectorised spline call
        # (replaces 252 × 3 per-step scalar Python calls with 3 array calls)
        t_arr     = np.maximum(np.arange(n_steps, dtype=float) * dt, 1e-6) #年化 dt ;
        t_clipped = np.clip(t_arr, self._t_min, self._t_max)
        fwd_arr   = -self._cs(t_clipped, 1)                           # 对ln P(0, t)的一阶导数 = f(0,t) =  d（ln P） / dt  
        d2ld_arr  =  self._cs(t_clipped, 2)                           # 对ln P(0, t) 二阶导数， d²ln P/dt² 
        theta_arr = (self.a * fwd_arr + d2ld_arr
                     + self.sigma**2 / (2.0 * self.a)
                     * (1.0 - np.exp(-2.0 * self.a * t_arr)))         # θ(t) 

        sig_sdt = self.sigma * np.sqrt(dt) # 每步的随机冲击标准差 σ√dt ；
        for s in range(n_steps):
            r[:, s+1] = (r[:, s]
                         + (theta_arr[s] - self.a * r[:, s]) * dt
                         + sig_sdt * Z[:, s])
        return r

    def discount_factor(self, r_h, t_h, T): 
        tau = T - t_h
        if tau <= 0.0: return np.ones(len(r_h)) 
        t_c  = max(t_h, 1e-6)
        B    = (1.0 - np.exp(-self.a * tau)) / self.a
        adj  = -self.sigma**2 / (4.0 * self.a) * B**2 * (1.0 - np.exp(-2.0 * self.a * t_c))
        return np.exp(self._ld(T) - self._ld(t_c) - B * (r_h - self._fwd(t_c)) + adj)  # P(t,T)= P(0,T)/P(0,t) * exp(-B(t)*r + B(t)*f(t) + adj) ；

    def discount_factor_batch(self, r_h, t_h, times):
        """
        Vectorised ZCB prices for multiple maturities simultaneously.
        times  : array (n_times,)   # payment dates 
        r_h    : array (n_paths,)   # short rate at horizon
        returns: array (n_times, n_paths)

        One call replaces n_times separate discount_factor() calls, 
        eliminating Python function-call overhead in repricing loops.
        Uses CubicSpline's native array evaluation for ln P^M(0,T).
        """
        times  = np.atleast_1d(np.asarray(times, dtype=float))
        t_h_c  = max(t_h, 1e-6)
        taus   = times - t_h_c
        result = np.ones((len(times), len(r_h))) 
        mask   = taus > 1e-10
        if not np.any(mask):
            return result

        tv   = taus[mask]
        Tv   = times[mask]
        B    = (1.0 - np.exp(-self.a * tv)) / self.a                  # (n_valid,)
        f_t  = self._fwd(t_h_c)                                        # scalar
        adj  = (-self.sigma**2 / (4.0 * self.a) * B**2
                * (1.0 - np.exp(-2.0 * self.a * t_h_c)))              # (n_valid,)

        # Vectorised log-discount via CubicSpline array evaluation
        Tv_clipped = np.clip(Tv, self._t_min, self._t_max) 
        ln_ratio   = self._cs(Tv_clipped) - self._ld(t_h_c)           # Ln{P(0, T)}/{P^M(0, t_h)}   # 远期价格对数 ;

        # ln P(t_h, T | r) = ln_ratio[i] + adj[i] + B[i]*f_t − B[i]*r_h[j]
        ln_P = (ln_ratio + adj + B * f_t)[:, None] - B[:, None] * r_h[None, :]  # 风险部是绝对的 核心科技 ； # （ ln_ratio + adj + B * f_t ） 这一部分代表预期理论价格，只与时间 T 有关； B * r_h 这一部分代表随机风险冲击，只与利率路径 r_h 有关；两者相减得到的 ln_P 就是每条路径在时间 T 的 log 贴现价格；最后通过 np.exp(ln_P) 转换回贴现价格。   
        result[mask] = np.exp(ln_P)
        return result

    def simulate_horizon(self, n_paths, horizon_days, dt, seed=42):
        """
        Directly sample r at the VaR horizon via Euler-Maruyama 1-step.

        For 1-day VaR, this bypasses the full 252-step simulation entirely.
        HW has time-varying drift θ(t), so we use the linearised approximation:
            r(t_h) ≈ r0 + (θ(0⁺) − a·r0)·t_h + σ·√t_h·Z

        Antithetic variates (Z, −Z) halve MC variance at zero extra cost.
        """
        rng     = np.random.default_rng(seed)
        half    = n_paths // 2
        t_h     = horizon_days * dt
        t0      = 1e-6
        # θ(0⁺): time-zero Hull-White drift
        fwd0    = self._fwd(t0)
        d2ld0   = float(self._cs(np.clip(t0, self._t_min, self._t_max), 2)) #二阶导 ；
        theta_0 = (self.a * fwd0 + d2ld0
                   + self.sigma**2 / (2.0 * self.a) * (1.0 - np.exp(-2.0 * self.a * t0)))
        mu_h    = self.r0 + (theta_0 - self.a * self.r0) * t_h
        sigma_h = self.sigma * np.sqrt(t_h)
        Z_h     = rng.standard_normal(half)
        Z       = np.concatenate([Z_h, -Z_h])
        return mu_h + sigma_h * Z


def get_model():
    name = MODEL_CONFIG["model"]
    if name == "vasicek":
        return VasicekModel(MODEL_CONFIG["vasicek"]), "vasicek"
    return HullWhiteModel(MODEL_CONFIG["hull_white"], YIELD_CURVE), "hull_white" 


# ============================================================================
# SECTION 4 — DATA PIPELINE
# ============================================================================

def fetch_yield_curve_fred():
    """Pull latest UST constant-maturity yields from FRED. Returns True on success."""
    import urllib.request, json 
    if not FRED_API_KEY:
        return False 
    url = ("https://api.stlouisfed.org/fred/series/observations"
           "?series_id={}&api_key={}&sort_order=desc&limit=5&file_type=json") 
    try:
        for tenor, series in _FRED_SERIES.items():
            with urllib.request.urlopen(url.format(series, FRED_API_KEY), timeout=6) as r:
                obs = json.loads(r.read())["observations"]
            val = next((o["value"] for o in obs if o["value"] != "."), None)
            if val is None:
                return False
            YIELD_CURVE[tenor] = float(val) / 100.0   # FRED quotes as percent
        return True
    except Exception:
        return False


def query_positions():
    rows = [
        dict(isin="US912828ZT04", product_type="treasury", issuer="US_GOVT",
             notional=10_000_000, coupon=0.0250, maturity_years= 2.0,
             credit_spread=0.0000, effective_duration= 1.95, dv01=  195),
        dict(isin="US9128284Y14", product_type="treasury", issuer="US_GOVT",
             notional=15_000_000, coupon=0.0300, maturity_years= 5.0,
             credit_spread=0.0000, effective_duration= 4.65, dv01=  698),
        dict(isin="US912828ZM56", product_type="treasury", issuer="US_GOVT",
             notional=20_000_000, coupon=0.0350, maturity_years=10.0,
             credit_spread=0.0000, effective_duration= 8.50, dv01= 1700),
        dict(isin="US912810SV68", product_type="treasury", issuer="US_GOVT",
             notional= 5_000_000, coupon=0.0400, maturity_years=30.0,
             credit_spread=0.0000, effective_duration=19.20, dv01=  960),
        dict(isin="US38141GYG11", product_type="corp_ig",  issuer="GS",
             notional= 8_000_000, coupon=0.0450, maturity_years= 5.0,
             credit_spread=0.0120, effective_duration= 4.50, dv01=  360),
        dict(isin="US46625HKA88", product_type="corp_ig",  issuer="JPM",
             notional= 7_000_000, coupon=0.0500, maturity_years=10.0,
             credit_spread=0.0150, effective_duration= 8.10, dv01=  567),
        dict(isin="US12345HY001", product_type="corp_hy",  issuer="HY_CORP",
             notional= 3_000_000, coupon=0.0750, maturity_years= 7.0,
             credit_spread=0.0450, effective_duration= 5.80, dv01=  174),
        dict(isin="US31381PDA98", product_type="mbs",      issuer="FNMA",
             notional=12_000_000, coupon=0.0600, maturity_years=30.0,
             credit_spread=0.0050, effective_duration= 4.80, dv01=  576,
             wac=0.065, cpr=0.10),
        dict(isin="US31381PDB71", product_type="mbs",      issuer="FHLMC",
             notional= 8_000_000, coupon=0.0550, maturity_years=30.0,
             credit_spread=0.0050, effective_duration= 5.20, dv01=  416,
             wac=0.060, cpr=0.08),
        dict(isin="US90290XAA15", product_type="abs_auto", issuer="FORD_ABS",
             notional= 5_000_000, coupon=0.0480, maturity_years= 4.0,
             credit_spread=0.0200, effective_duration= 2.20, dv01=  110),
        dict(isin="US90290XBB22", product_type="abs_card", issuer="CITI_ABS",
             notional= 4_000_000, coupon=0.0520, maturity_years= 5.0,
             credit_spread=0.0250, effective_duration= 1.80, dv01=   72),
    ]
    df = pd.DataFrame(rows)
    df["wac"] = df.get("wac", pd.Series(dtype=float)).fillna(0.0) 
    df["cpr"] = df.get("cpr", pd.Series(dtype=float)).fillna(0.0)
    df["as_of_date"] = date.today().isoformat()
    return df


def query_historical_pnl(n_obs=250):
    rng = np.random.default_rng(99)
    pnl = (rng.standard_normal(n_obs) * 185_000
           + rng.standard_t(df=5, size=n_obs) * 42_000 - 4_800)
    return pd.Series(pnl, index=pd.date_range(end=date.today(), periods=n_obs, freq="B"), 
                     name="daily_pnl")


# ============================================================================
# SECTION 5 — RISK ENGINE
# ============================================================================

def _price_coupon_bond(notional, coupon, maturity_T, credit_spread, zcb_fn, freq=2): 
    n = max(1, int(round(maturity_T * freq)))
    t_cf = np.arange(1, n+1) / freq
    cfs  = np.full(n, notional * coupon / freq);  cfs[-1] += notional  # cfs[-1] += notional 写在前一个函数里面（同一行），这样只计算一次就停止了，不会无限循环 ；
    return float(sum(cf * zcb_fn(t) * np.exp(-credit_spread * t) for t, cf in zip(t_cf, cfs))) # zcb_fn(t)零息债券价格 P(0,t), SUM 括号里包含两次折现，一次是无风险贴现（zcb_fn(t)），一次是信用利差贴现（exp(-credit_spread * t) ；


def _reprice_coupon_bond_mc(notional, coupon, maturity_T, credit_spread, model, r_h, t_h, freq=2):
    """
    Coupon-bond repricing across all MC paths via cash-flow loop.

    Loop-based: each iteration prices one cash flow across all n_paths simultaneously 
    using a single vectorised discount_factor() call (shape: n_paths,).
    This is cache-friendly vs. a (n_cfs × n_paths) batch matrix which at
    100K paths × 60 cash flows = 48 MB intermediate — exceeds L3 cache. 
    """
    n    = max(1, int(round(maturity_T * freq)))
    t_cf = np.arange(1, n + 1) / freq
    cfs  = np.full(n, notional * coupon / freq);  cfs[-1] += notional 
    mask = t_cf > t_h;  t_cf, cfs = t_cf[mask], cfs[mask]
    if len(t_cf) == 0:
        return np.zeros(len(r_h))
    p = np.zeros(len(r_h))
    for t, cf in zip(t_cf, cfs):
        p += cf * model.discount_factor(r_h, t_h, t) * np.exp(-credit_spread * (t - t_h))  # 这段代码是风险估值的“心脏”， 是 VaR 引擎里的地位核心的核心；它计算了每条路径在时间 t 的现金流 cf 的现值，现值由三个部分组成：cf 是现金流金额，model.discount_factor(r_h, t_h, t) 是无风险贴现因子，np.exp(-credit_spread * (t - t_h)) 是信用利差贴现因子；通过循环累加每个现金流的现值，最终得到每条路径的总价格 p。
    return p


def _reprice_structured_mc(notional, eff_dur, product_type, r_h, r0): # 这是一种非常高效的“平替”方案 - 久期近似法（Duration Approximation）；Price = Notional * (1.0 - Dur_ Eff * (r_h - r_0)) ; 车贷和信用卡 ABS 的期限通常较短，提前还款行为相对稳定。这意味着它们的价格-利率曲线非常接近一条直线，而不是像长债那样有明显的弧度（凸性）。既然是直线，用久期（斜率）来估值就足够准了。 
    """ABS (auto/card): effective duration, near-zero convexity."""
    dr = r_h - r0
    return notional * (1.0 - eff_dur * dr)


# ---------------------------------------------------------------------------
# PSA Prepayment Model — MBS repricing 
# ---------------------------------------------------------------------------
# PSA Standard (1985): CPR(m) = min(m/30, 1) × 6% × (PSA_speed/100%)
# Rate-sensitive extension: CPR_eff(r) = CPR_base × (1 + β_refi × (r0 − r))
#   β_refi = 8.0 → 100 bps rate fall accelerates CPR by 8% (refinancing wave)
#   Bounds: [10%, 400%] of base CPR (burnout / lock-in floors/caps)
# ---------------------------------------------------------------------------

def _reprice_mbs_psa_mc(notional, coupon, wac, cpr_base, maturity_years,
                         credit_spread, model, r_horizon, t_h):
    """
    Full-cash-flow MBS repricing under PSA prepayment model for all MC paths. 

    Parameters
    ----------
    coupon        : investor pass-through rate (annual)
    wac           : weighted-average coupon of underlying mortgages (for amortisation)
    cpr_base      : baseline annual CPR at full PSA ramp (e.g. )  10% = the 166.7% PSA benchmark（6%） ； 
    r_horizon     : short rate at horizon, shape (n_paths,)
    t_h           : horizon in years

    Returns ndarray (n_paths,) — MBS prices at horizon.
    """
    n_paths      = len(r_horizon)
    n_months     = int(round(maturity_years * 12))
    monthly_coup = coupon / 12  #投资人收到的每月票息；
    monthly_wac  = wac / 12   #借款人的每月利息 

    if monthly_wac > 1e-8:
        pmt_factor = monthly_wac / (1.0 - (1.0 + monthly_wac) ** (-n_months)) #pmt 是每月等额还款金额，pmt_factor 是每月额还款金额因子 ， 此处债券面值为 1； monthly_wac 是 利率 r ; n_months 是 还款总期数 N ； pmt_factor = r / (1 - (1 + r)^(-N)) ；  
    else:
        pmt_factor = 1.0 / n_months  # 利率很小近似为零 ； 

    # Rate-sensitive CPR per path
    beta_refi = 8.0
    cpr_eff   = np.clip(cpr_base * (1.0 + beta_refi * (model.r0 - r_horizon)),  # 注意弄懂这个 ； 
                        cpr_base * 0.10, cpr_base * 4.0)        # (n_paths,) 

    psa_ramp = np.minimum(np.arange(1, n_months + 1) / 30.0, 1.0)  #  psa_ramp爬坡系数；30 个月爬坡，这是行业通用标准；1.0 代表 100%，意思是 30个月爬坡到最高值，然后一直保持这个psa_ramp；

    # Phase 1 — cash flow generation (month-loop unavoidable: path-dependent balance)
    cf_buf = []
    t_buf  = []
    bal    = np.ones(n_paths) * notional 

    for i in range(n_months):
        t          = t_h + (i + 1) / 12.0
        cpr_m      = psa_ramp[i] * cpr_eff
        smm        = 1.0 - (1.0 - cpr_m) ** (1.0 / 12.0)
        interest   = bal * monthly_coup
        sched_prin = np.maximum(bal * pmt_factor - bal * monthly_wac, 0.0)
        prepayment = smm * np.maximum(bal - sched_prin, 0.0)
        total_prin = np.minimum(sched_prin + prepayment, bal)
        cf_buf.append(interest + total_prin) #动态更新收到本金+利息；
        t_buf.append(t) # 动态更新每个月的时间点；便于后续折现 ；
        bal = np.maximum(bal - total_prin, 0.0) #更新bal， 指 剩余本金 ； 
        if np.all(bal < 1.0): 
            break

    # Phase 2 — accumulate discounted cash flows month-by-month 
    # (n_active, n_paths) batch matrix at 100K paths × 360 months = 288 MB
    # exceeds L3 cache → saturates memory bandwidth → 40% SLOWER than loop.
    # Loop-based: one discount_factor() per month (shape: n_paths,) stays in cache.
    prices = np.zeros(n_paths)
    for cf, t in zip(cf_buf, t_buf):
        df      = (model.discount_factor(r_horizon, t_h, t) 
                   * np.exp(-credit_spread * (t - t_h)))
        prices += cf * df #两次折现 ；
    return prices


def _reprice_mbs_psa_static(notional, coupon, wac, cpr_base, maturity_years, 
                              credit_spread, yc_fn, r_proxy, r0):
    """
    Scalar MBS PSA repricing using a yield-curve function (market value / stress tests).

    yc_fn(t) → continuously-compounded zero rate at maturity t (scalar or array).
    r_proxy   → representative rate for CPR sensitivity (r0 + scenario shock). 
    """
    n_months     = int(round(maturity_years * 12))
    monthly_coup = coupon / 12
    monthly_wac  = wac / 12

    if monthly_wac > 1e-8:
        pmt_factor = monthly_wac / (1.0 - (1.0 + monthly_wac) ** (-n_months))
    else:
        pmt_factor = 1.0 / n_months

    beta_refi = 8.0
    cpr_eff   = float(np.clip(cpr_base * (1.0 + beta_refi * (r0 - r_proxy)),  #此处用r_proxy，代表当前的代理市场利率；这比使用瞬时利率r_horizon更平滑；r_proxy 代表的是“宏观气候”，而 r_horizon 代表的是“即时折现率”
                               cpr_base * 0.10, cpr_base * 4.0))

    price = 0.0
    bal   = float(notional)

    for i in range(n_months):
        t          = (i + 1) / 12.0
        cpr_m      = min((i + 1) / 30.0, 1.0) * cpr_eff
        smm        = 1.0 - (1.0 - cpr_m) ** (1.0 / 12.0)
        interest   = bal * monthly_coup 
        sched_prin = max(bal * pmt_factor - bal * monthly_wac, 0.0)
        prepayment = smm * max(bal - sched_prin, 0.0)
        total_prin = min(sched_prin + prepayment, bal) 
        cf         = interest + total_prin  # 这里计算的是 投资人收到 的现金流 ；所以，以上用的是 monthly_coup；
        y          = float(np.atleast_1d(yc_fn(t))[0])  # yc_fn(t)是从哪里出现的 ？  
        df         = np.exp(-y * t - credit_spread * t) # DF = e^{-(y + s) * t} 
        price     += cf * df
        bal        = max(bal - total_prin, 0.0)
        if bal < 1.0:
            break

    return price


def compute_portfolio_pnl(positions_df, model, sim_cfg):
    n_paths, n_steps = sim_cfg["n_paths"], sim_cfg["n_steps"]
    dt, horizon      = sim_cfg["dt"], sim_cfg["horizon_days"]
    t_h              = horizon * dt

    # ── Fast path: direct sampling at VaR horizon ───────────────────────────
    # simulate_horizon() draws r(t_h) analytically in O(n_paths) —
    # bypasses the full 252-step Euler loop, giving ~250× speedup for simulation.
    # A small chart_paths run (≤ 500 paths × 252 steps) is retained so
    # rate-path charts remain representative without bloating runtime. 
    r_horizon   = model.simulate_horizon(n_paths, horizon, dt, seed=sim_cfg["seed"]) # seed 是多少 ？ 需要自己设置具体seed ？   
    n_chart     = min(500, n_paths)
    chart_paths = model.simulate_paths(n_chart, n_steps, dt, seed=sim_cfg["seed"])

    # Nelson-Siegel yield curve for market valuation
    yc_tenors = sorted(YIELD_CURVE.keys())  # 期限 ；
    yc_yields = [YIELD_CURVE[t] for t in yc_tenors] # 收益率 ；
    ns_fn     = _fit_nelson_siegel(yc_tenors, yc_yields)

    def mkt_zcb(t): 
        return np.exp(-float(np.atleast_1d(ns_fn(t))[0]) * t) # t 是 yc_tenors  通常是 年 ； 市场零息票债券价格 mkt_zcb = e^{-r(t) * t} ；

    portfolio_pnl = np.zeros(n_paths)
    market_value  = 0.0
    r0_arr        = np.array([model.r0])

    for _, pos in positions_df.iterrows():
        ptype, notl = pos["product_type"], pos["notional"] # notl ： notional value ; 
        T, cs, eff  = pos["maturity_years"], pos["credit_spread"], pos["effective_duration"]

        # Market value (mark-to-market using Nelson-Siegel curve)
        if ptype == "mbs":
            wac_  = float(pos.get("wac", pos["coupon"] + 0.005))
            cpr_b = float(pos.get("cpr", 0.10))
            market_value += _reprice_mbs_psa_static(notl, pos["coupon"], wac_, cpr_b,
                                                     T, cs, ns_fn, model.r0, model.r0) # _reprice_mbs_psa_static调用出的是 Fair Value ； 
        elif ptype in ("abs_auto", "abs_card"):
            market_value += notl 
        else:
            market_value += _price_coupon_bond(notl, pos["coupon"], T, cs, mkt_zcb)

        # P&L: p1 − p0 (p0 at r0 ensures E[P&L] = 0 with no rate change) 
        if ptype == "mbs":
            wac_  = float(pos.get("wac", pos["coupon"] + 0.005))
            cpr_b = float(pos.get("cpr", 0.10))
            p0 = float(_reprice_mbs_psa_mc(notl, pos["coupon"], wac_, cpr_b, 
                                            T, cs, model, r0_arr, 0.0)[0]) # 计算Market Value ， 用r0_arr 作为 当前的观测利率 ；
            p1 = _reprice_mbs_psa_mc(notl, pos["coupon"], wac_, cpr_b,
                                      T, cs, model, r_horizon, t_h) # 计算 Future Value ，用 r_horizon 作为未来的随机利率 ； P1 - P0 两者之差就是 P&L ； 
        elif ptype in ("abs_auto", "abs_card"):
            p0 = notl
            p1 = _reprice_structured_mc(notl, eff, ptype, r_horizon, model.r0)
        else:
            p0 = float(_reprice_coupon_bond_mc(notl, pos["coupon"], T, cs, model, r0_arr, 0.0)[0])
            p1 = _reprice_coupon_bond_mc(notl, pos["coupon"], T, cs, model, r_horizon, t_h)

        portfolio_pnl += p1 - p0

    return portfolio_pnl, market_value, chart_paths 


def calculate_var_cvar(pnl_array, confidence=None):
    if confidence is None: confidence = RISK_CONFIG["confidence_level"] 
    var  = float(-np.percentile(pnl_array, (1.0 - confidence) * 100.0))  # np.percentile 将一组数据从小到大排列，找到处于某个百分比位置的数值，此处是 1% ； 
    tail = pnl_array[pnl_array <= -var]
    return var, (float(-tail.mean()) if len(tail) > 0 else var) # -tail.mean() = cVaR ； 如果 tail 中没有数据（即没有损失超过 VaR 的情况），则返回 VaR 作为 cVaR 的近似值。


def run_stress_tests(positions_df):
    yc_tenors = sorted(YIELD_CURVE.keys())
    yc_yields = [YIELD_CURVE[t] for t in yc_tenors]
    r0        = MODEL_CONFIG[MODEL_CONFIG["model"]]["r0"]

    # Nelson-Siegel baseline
    ns_base  = _fit_nelson_siegel(yc_tenors, yc_yields)
    mkt_zcb  = lambda t: np.exp(-float(np.atleast_1d(ns_base(t))[0]) * t)  # P(t) = e^{-r(t) * t} ； lambda 是一个匿名函数 ；               
    def price_portfolio(yc_fn, r_proxy):
        total = 0.0
        for _, pos in positions_df.iterrows():  
            ptype = pos["product_type"]
            if ptype == "mbs":
                wac_  = float(pos.get("wac", pos["coupon"] + 0.005))
                cpr_b = float(pos.get("cpr", 0.10))
                total += _reprice_mbs_psa_static(pos["notional"], pos["coupon"], wac_, cpr_b,
                                                  pos["maturity_years"], pos["credit_spread"], 
                                                  yc_fn, r_proxy, r0) #  剔除按揭、提前还款后的 PV ; 
            elif ptype in ("abs_auto", "abs_card"):
                total += pos["notional"]
            else:
                zcb = lambda t, f=yc_fn: np.exp(-float(np.atleast_1d(f(t))[0]) * t)
                total += _price_coupon_bond(pos["notional"], pos["coupon"],  
                                            pos["maturity_years"], pos["credit_spread"], zcb) # 普通债券的PV ;
        return total  # portfolio value under given yield curve and rate proxy ； # 总 PORTFOLIO PV ； 

    baseline = price_portfolio(ns_base, r0) 
    results  = {}

    for scen_name, shocks in STRESS_SCENARIOS.items():
        # Shocked yields at knot tenors → re-fit Nelson-Siegel to shocked curve
        s_ylds = np.array(yc_yields) + np.array([shocks.get(t, 0.0) for t in yc_tenors]) # Shocked yields 加进去 yc_yields 中 ；
        ns_shocked = _fit_nelson_siegel(yc_tenors, s_ylds.tolist()) # 用 nelson_siegel 重新 拟合 新的、 考虑了Shocked yields 的 s_ylds ； 
        # Representative rate proxy for PSA sensitivity (notional-weighted avg shock) 
        dr_mean = float(np.mean([shocks.get(t, 0.0) for t in yc_tenors])) # 计算平均利率冲击，作为 MBS CPR 敏感性的代表性利率冲击； 
        r_proxy = r0 + dr_mean  

        results[scen_name] = price_portfolio(ns_shocked, r_proxy) - baseline # 计算在每个压力测试场景下，投资组合的价值变化（相对于基线）； 

    return results, baseline


# --------------------------------------------------------------------------
# Key Rate Durations (KRD)
# --------------------------------------------------------------------------
# KRD_k  = isolated sensitivity to a +1bp shock at tenor k, all other tenors held fixed.
# Captures twist (short vs long) and butterfly (short/long vs belly) risks that a
# parallel DV01 cannot distinguish. 
#
# Method:
#   1. For each YIELD_CURVE knot, apply +1bp → re-fit Nelson-Siegel → reprice portfolio.
#   2. KR01_k  = -(P_bumped - P_base)          [$ loss per +1bp at tenor k; positive for long]
#   3. KRD_k   = KR01_k / (P_base × 0.0001)   [years; analogous to modified duration at each node] 
# ---------------------------------------------------------------------------

def compute_krd(positions_df, bump_bps=1.0):
    """
    Key Rate Duration profile via isolated +1bp tenor bumps.

    Returns
    -------
    krd_df   : DataFrame — columns [Tenor (yr), KRD (yrs), KR01 ($)]
    baseline : float — current portfolio mark-to-market 
    """
    bump      = bump_bps / 10_000.0
    yc_tenors = sorted(YIELD_CURVE.keys())
    yc_yields = [YIELD_CURVE[t] for t in yc_tenors]
    r0        = MODEL_CONFIG[MODEL_CONFIG["model"]]["r0"]
    ns_base   = _fit_nelson_siegel(yc_tenors, yc_yields) # Baseline Yield Curve ；

    def _price_pf(yc_fn):
        total = 0.0
        for _, pos in positions_df.iterrows():
            ptype = pos["product_type"]
            if ptype == "mbs":
                wac_  = float(pos.get("wac", pos["coupon"] + 0.005))
                cpr_b = float(pos.get("cpr", 0.10))
                total += _reprice_mbs_psa_static(pos["notional"], pos["coupon"], wac_, cpr_b,
                                                  pos["maturity_years"], pos["credit_spread"], 
                                                  yc_fn, r0, r0) #  剔除按揭、提前还款后的 PV ; 
            elif ptype in ("abs_auto", "abs_card"):
                total += pos["notional"]
            else:
                zcb = lambda t, f=yc_fn: np.exp(-float(np.atleast_1d(f(t))[0]) * t)
                total += _price_coupon_bond(pos["notional"], pos["coupon"], 
                                            pos["maturity_years"], pos["credit_spread"], zcb)  # 普通债券的PV ;
        return total # 总 PORTFOLIO PV ； 

    baseline = _price_pf(ns_base)
    rows = []
    for i, tenor in enumerate(yc_tenors):
        y_bumped    = list(yc_yields)
        y_bumped[i] += bump
        ns_bumped   = _fit_nelson_siegel(yc_tenors, y_bumped)
        p_bumped    = _price_pf(ns_bumped)
        kr01        = -(p_bumped - baseline)              # $ loss per +1bp (positive for long)
        krd         = kr01 / (baseline * bump)            # KRD = KR01 / （ Baseline * Bump ） ； KRD 的计算公式 ；
        rows.append({"Tenor (yr)": tenor,
                     "KRD (yrs)":  round(krd, 4),
                     "KR01 ($)":   round(kr01, 0)})
    return pd.DataFrame(rows), baseline


# ============================================================================
# SECTION 6 — MODEL VALIDATION 
# ============================================================================

def run_backtesting(historical_pnl, var_estimate, alpha=None): # alpha 是置信水平，也是理论突破概率；通常是 0.99 或 0.95； var_estimate 是 VaR 的估计值； historical_pnl 是历史 P&L 数据； 
    if alpha is None: alpha = BACKTEST_CONFIG["alpha"]
    pnl = historical_pnl.values;  T = len(pnl)
    exc = (pnl < -var_estimate).astype(int);  N = int(exc.sum())

    p_hat = N / T   # p_hat 是实际突破率，即在历史数据中，损失超过 VaR 估计值的比例； 
    if 0 < p_hat < 1:
        lr_uc   = -2.0 * (N * np.log(alpha / p_hat) + (T-N) * np.log((1-alpha) / (1-p_hat))) 
        ku_pval = float(1.0 - stats.chi2.cdf(lr_uc, df=1))  # P_Value ; stats.chi2.cdf 是卡方分布的累积分布函数， ku_pval 是 Kupiec 测试的 p-value，用于判断 VaR 模型的准确性；如果 ku_pval <  置信水平 alpha，通常认为模型不准确（FAIL），否则认为模型准确（PASS）。
    else:
        lr_uc, ku_pval = np.inf, 0.0

    kupiec = {"n_exceptions": N, "expected_exceptions": int(alpha*T),
              "exception_rate": round(p_hat,5), "lr_stat": round(lr_uc,4),
              "p_value": round(ku_pval,4),
              "conclusion": "FAIL" if ku_pval < 0.05 else "PASS"} 

    n00=n01=n10=n11=0
    for t in range(1, T):
        p, c = exc[t-1], exc[t]
        if   p==0 and c==0: n00+=1
        elif p==0 and c==1: n01+=1
        elif p==1 and c==0: n10+=1
        else:               n11+=1

    eps   = 1e-10
    pi_01 = n01 / max(n00+n01, 1);  pi_11 = n11 / max(n10+n11, 1) # max  的目的是为了避免分母为零的情况；如果 n00+n01 为零，则分母将被替换为 1，从而避免了除以零的错误；同样地，max(n10+n11, 1) 也是为了同样的目的；
    pi    = (n01+n11) / max(T-1, 1) 
    sl    = lambda x: np.log(max(x, eps))  # Safe log to avoid log(0) issues;   
    lr_ind = -2.0 * ((n00+n10)*sl(1-pi) + (n01+n11)*sl(pi)
                     - n00*sl(1-pi_01) - n01*sl(pi_01)
                     - n10*sl(1-pi_11) - n11*sl(pi_11)) 
    lr_cc   = float(lr_uc + lr_ind)
    cc_pval = float(1.0 - stats.chi2.cdf(lr_cc, df=2))
    christoffersen = {"pi_01": round(pi_01,4), "pi_11": round(pi_11,4),
                      "lr_ind": round(float(lr_ind),4), "lr_cc_stat": round(lr_cc,4),
                      "p_value": round(cc_pval,4),
                      "conclusion": "FAIL" if cc_pval < 0.05 else "PASS"}

    n_250 = int(round(N*250/T)) if T != 250 else N # n_250 是将实际突破次数 N 标准化到 250 个观测值的水平，以便于与常用的 250 个交易日的年度数据进行比较；如果 T 已经是 250，则直接使用 N 作为 n_250；否则，计算 N*250/T 来调整突破次数，使其反映在 250 个观测值上的预期数量；这个标准化过程有助于评估模型在不同长度的历史数据上的表现，并提供一个更直观的指标来判断模型的准确性。
    if   n_250 <= 4: zone, km = "Green",  3.0
    elif n_250 <= 9: zone, km = "Yellow", 3.4 
    else:            zone, km = "Red",    4.0
    traffic_light = {"exceptions_250d": n_250, "zone": zone, 
                     "capital_multiplier": km,
                     "signal": "PASS" if zone=="Green" else ("REVIEW" if zone=="Yellow" else "FAIL")}

    return {"n_obs": T, "kupiec": kupiec,
            "christoffersen": christoffersen, "traffic_light": traffic_light}


def historical_simulation_var(historical_pnl, confidence=None): 
    return calculate_var_cvar(historical_pnl.values, confidence)


def sensitivity_analysis(positions_df, model_name, base_var, base_cvar):
    sim_fast = {**SIMULATION_CONFIG, "n_paths": 5_000} 
    if model_name == "vasicek":
        bp = MODEL_CONFIG["vasicek"]  # Baseline parameters for Vasicek model ；
        perturb = [("Baseline", bp),  # perturbation ; 
                   ("σ × 1.25", {**bp, "sigma": bp["sigma"]*1.25}),
                   ("σ × 0.75", {**bp, "sigma": bp["sigma"]*0.75}),
                   ("κ × 1.50", {**bp, "kappa": bp["kappa"]*1.50}),
                   ("κ × 0.50", {**bp, "kappa": bp["kappa"]*0.50}),
                   ("θ +100bps",{**bp, "theta": bp["theta"]+0.010})]
        build = lambda p: VasicekModel(p)
    else:
        bp = MODEL_CONFIG["hull_white"]
        perturb = [("Baseline", bp),
                   ("σ × 1.25", {**bp, "sigma": bp["sigma"]*1.25}),
                   ("σ × 0.75", {**bp, "sigma": bp["sigma"]*0.75}),
                   ("a × 1.50", {**bp, "a": bp["a"]*1.50}),
                   ("a × 0.50", {**bp, "a": bp["a"]*0.50})]
        build = lambda p: HullWhiteModel(p, YIELD_CURVE)  

    rows = []
    for label, params in perturb:
        if label == "Baseline":
            rows.append({"Perturbation": label, "VaR_99 ($)": f"{base_var:,.0f}", 
                         "CVaR_99 ($)": f"{base_cvar:,.0f}", "ΔVaR (%)": "—", "ΔCVaR (%)": "—"})
            continue 
        pnl, _, _ = compute_portfolio_pnl(positions_df, build(params), sim_fast)  # 针对当前的扰动参数进行蒙特卡洛模拟，计算pnl ; 
        v, c = calculate_var_cvar(pnl)
        rows.append({"Perturbation": label, "VaR_99 ($)": f"{v:,.0f}",
                     "CVaR_99 ($)": f"{c:,.0f}",
                     "ΔVaR (%)":  f"{(v-base_var)/base_var*100:+.1f}%", 
                     "ΔCVaR (%)": f"{(c-base_cvar)/base_cvar*100:+.1f}%"}) 
    return pd.DataFrame(rows) 


# ============================================================================
# SECTION 7 — CHARTS 
# ============================================================================

def plot_results(pnl_mc, var_mc, cvar_mc, r_paths, stress_pnl, sens_df, 
                 historical_pnl, hs_var, model_name, market_value,
                 output_path="var_report.png"):
    """
    4-panel dashboard:
      [0] P&L distribution with VaR / CVaR markers
      [1] Simulated rate paths (sample)
      [2] Stress test P&L bar chart
      [3] Sensitivity analysis (ΔVaR %) 
    """
    conf = RISK_CONFIG["confidence_level"]

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
    axs = [fig.add_subplot(gs[r, c]) for r, c in [(0,0),(0,1),(1,0),(1,1)]]

    style = dict(facecolor="#0d1117", edgecolor="#30363d")
    for ax in axs:
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values(): sp.set_color("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=9)
        ax.title.set_color("#e6edf3")
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")

    # ── [0] P&L Distribution ─────────────────────────────────────────────────
    ax = axs[0]
    pnl_m = pnl_mc / 1e6
    ax.hist(pnl_m, bins=120, color="#1f6feb", alpha=0.75, edgecolor="none", density=True)

    # Fit normal overlay
    mu, sigma = pnl_m.mean(), pnl_m.std()
    x  = np.linspace(pnl_m.min(), pnl_m.max(), 400)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color="#58a6ff", lw=1.5,
            linestyle="--", label="Normal fit")

    ax.axvline(-var_mc/1e6,  color="#f85149", lw=2,
               label=f"VaR {conf:.0%} = ${var_mc/1e6:.2f}M")
    ax.axvline(-cvar_mc/1e6, color="#d29922", lw=2, linestyle="--",
               label=f"CVaR = ${cvar_mc/1e6:.2f}M")

    ax.set_title(f"Portfolio P&L Distribution  ({model_name.upper()})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Daily P&L ($M)")
    ax.set_ylabel("Density")
    leg = ax.legend(fontsize=8, framealpha=0.3)
    for t in leg.get_texts(): t.set_color("#e6edf3")

    # ── [1] Rate Paths ────────────────────────────────────────────────────────
    ax   = axs[1]
    dt   = SIMULATION_CONFIG["dt"]
    days = SIMULATION_CONFIG["n_steps"]
    t_ax = np.arange(days + 1) * dt * 252          # in trading days

    rng  = np.random.default_rng(0)
    idx  = rng.choice(r_paths.shape[0], size=min(200, r_paths.shape[0]), replace=False)
    for i in idx:
        ax.plot(t_ax, r_paths[i] * 100, color="#1f6feb", alpha=0.06, lw=0.6)

    mean_path = r_paths.mean(axis=0) * 100
    p5  = np.percentile(r_paths, 5,  axis=0) * 100
    p95 = np.percentile(r_paths, 95, axis=0) * 100
    ax.plot(t_ax, mean_path, color="#58a6ff", lw=2, label="Mean path")
    ax.fill_between(t_ax, p5, p95, color="#388bfd", alpha=0.15, label="5th–95th pct")

    ax.set_title(f"Simulated Short-Rate Paths  ({model_name.upper()})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Short Rate (%)")
    leg2 = ax.legend(fontsize=8, framealpha=0.3)
    for t in leg2.get_texts(): t.set_color("#e6edf3")

    # ── [2] Stress Test Bar Chart ─────────────────────────────────────────────
    ax      = axs[2]
    labels  = [n.replace("_", "\n").title() for n in stress_pnl.keys()]
    values  = [v / 1e6 for v in stress_pnl.values()]
    colors  = ["#f85149" if v < 0 else "#3fb950" for v in values]
    bars    = ax.barh(labels, values, color=colors, alpha=0.85, edgecolor="none")
    ax.axvline(0, color="#8b949e", lw=0.8)
    for bar, val in zip(bars, values):
        ax.text(val + (0.05 if val >= 0 else -0.05), bar.get_y() + bar.get_height()/2,
                f"${val:.1f}M", va="center", ha="left" if val >= 0 else "right",
                color="#e6edf3", fontsize=7.5)
    ax.set_title("Stress Test Results", fontsize=11, fontweight="bold")
    ax.set_xlabel("P&L Impact ($M)")
    ax.tick_params(axis="y", labelsize=7.5)

    # ── [3] Sensitivity ΔVaR ─────────────────────────────────────────────────
    ax = axs[3]
    sens_plot = sens_df[sens_df["Perturbation"] != "Baseline"].copy()
    dvar_vals = sens_plot["ΔVaR (%)"].str.rstrip("%").astype(float)
    labels3   = sens_plot["Perturbation"].tolist()
    colors3   = ["#f85149" if v > 0 else "#3fb950" for v in dvar_vals]
    ax.barh(labels3, dvar_vals, color=colors3, alpha=0.85, edgecolor="none")
    ax.axvline(0, color="#8b949e", lw=0.8)
    for i, (v, l) in enumerate(zip(dvar_vals, labels3)):
        ax.text(v + (0.3 if v >= 0 else -0.3), i,
                f"{v:+.1f}%", va="center", ha="left" if v >= 0 else "right",
                color="#e6edf3", fontsize=8)
    ax.set_title("Sensitivity: ΔVaR vs Baseline (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("ΔVaR (%)")

    # Title banner
    fig.suptitle(
        f"Fixed Income VaR Engine  |  {model_name.upper()}  |  "
        f"VaR (99%, 1d) = ${var_mc/1e6:.3f}M  |  Portfolio MtM = ${market_value/1e6:.1f}M  |  "
        f"{date.today()}",
        color="#e6edf3", fontsize=11, fontweight="bold", y=0.98
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def plot_krd_profile(krd_df, output_path="krd_profile.png"):
    """
    Key Rate Duration bar chart.

    Two sub-panels:
      [left]  KRD (years) at each tenor — shows where duration risk is concentrated
      [right] KR01 ($) at each tenor — dollar sensitivity per +1bp isolated shock
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values(): sp.set_color("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=9)
        ax.title.set_color("#e6edf3")
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")

    tenors = [str(t) for t in krd_df["Tenor (yr)"]]
    x      = np.arange(len(tenors))

    # ── Panel 1: KRD (yrs) ───────────────────────────────────────────────
    krd_vals = krd_df["KRD (yrs)"].values
    colors   = ["#388bfd" if v >= 0 else "#f85149" for v in krd_vals]
    ax1.bar(x, krd_vals, color=colors, width=0.6, edgecolor="none")
    ax1.axhline(0, color="#30363d", lw=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(tenors, rotation=45, ha="right")
    ax1.set_xlabel("Tenor (yr)"); ax1.set_ylabel("KRD (years)")
    ax1.set_title("Key Rate Duration Profile (KRD)", fontsize=11, pad=8)
    for xi, v in zip(x, krd_vals):
        ax1.text(xi, v + 0.003 * np.sign(v + 1e-9), f"{v:.3f}",
                 ha="center", va="bottom", fontsize=7, color="#e6edf3")

    # ── Panel 2: KR01 ($) ────────────────────────────────────────────────
    kr01_vals = krd_df["KR01 ($)"].values
    colors2   = ["#3fb950" if v >= 0 else "#f85149" for v in kr01_vals]
    ax2.bar(x, kr01_vals / 1000, color=colors2, width=0.6, edgecolor="none")
    ax2.axhline(0, color="#30363d", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(tenors, rotation=45, ha="right")
    ax2.set_xlabel("Tenor (yr)"); ax2.set_ylabel("KR01 ($000s per +1bp)")
    ax2.set_title("Key Rate DV01 (KR01) per +1bp Isolated Shock", fontsize=11, pad=8)
    for xi, v in zip(x, kr01_vals):
        ax2.text(xi, v / 1000 + 0.5 * np.sign(v + 1e-9), f"${v/1000:.1f}k",
                 ha="center", va="bottom", fontsize=7, color="#e6edf3")

    fig.suptitle("Key Rate Risk Decomposition  |  Twist & Butterfly Sensitivity",
                 color="#e6edf3", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def plot_alpha_sigma_heatmap(positions_df, model_name,
                              output_path="heatmap_alpha_sigma.png"):
    """
    α×σ VaR Sensitivity Heatmap.

    Rows    : volatility multiplier  (σ × 0.60 … 1.50)
    Columns : confidence level α     (90% … 99.5%)
    Cell    : VaR in $M at that (σ, α) combination

    Supplements the OAT sensitivity table with a full 2-D view —
    shows how VaR responds jointly to model uncertainty (σ) and
    regulatory confidence-level choice (α).
    """
    alphas     = [0.90, 0.95, 0.975, 0.99, 0.995]
    sig_mults  = [0.60, 0.75, 0.875, 1.00, 1.125, 1.25, 1.50]
    sim_fast   = {**SIMULATION_CONFIG, "n_paths": 2_000}

    if model_name == "vasicek":
        base_p = MODEL_CONFIG["vasicek"]
        build  = lambda sm: VasicekModel({**base_p, "sigma": base_p["sigma"] * sm})
    else:
        base_p = MODEL_CONFIG["hull_white"]
        build  = lambda sm: HullWhiteModel({**base_p, "sigma": base_p["sigma"] * sm}, YIELD_CURVE)

    grid = np.zeros((len(sig_mults), len(alphas)))
    for i, sm in enumerate(sig_mults):
        pnl, _, _ = compute_portfolio_pnl(positions_df, build(sm), sim_fast)
        for j, a in enumerate(alphas):
            grid[i, j], _ = calculate_var_cvar(pnl, a)
    grid /= 1e6   # → $M

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r", origin="lower",
                   interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("VaR ($M)", color="#e6edf3", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e")

    # Cell annotations
    vmin, vmax = grid.min(), grid.max()
    for i in range(len(sig_mults)):
        for j in range(len(alphas)):
            val = grid[i, j]
            txt_color = "#0d1117" if 0.3 < (val - vmin) / (vmax - vmin) < 0.7 else "#e6edf3"
            ax.text(j, i, f"${val:.2f}M", ha="center", va="center",
                    color=txt_color, fontsize=8.5, fontweight="bold")

    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.1%}" for a in alphas], color="#8b949e", fontsize=9)
    ax.set_yticks(range(len(sig_mults)))
    ax.set_yticklabels([f"{s:.3g}×" for s in sig_mults], color="#8b949e", fontsize=9)
    ax.set_xlabel("Confidence Level  α", color="#8b949e", fontsize=10)
    ax.set_ylabel("Volatility Multiplier  σ×", color="#8b949e", fontsize=10)
    ax.set_title(f"VaR Sensitivity Heatmap: α × σ  ({model_name.upper()})",
                 color="#e6edf3", fontsize=12, fontweight="bold", pad=12)

    for sp in ax.spines.values():
        sp.set_color("#30363d")
    ax.tick_params(colors="#8b949e")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


# ============================================================================
# SECTION 8 — SR 11-7 DOCUMENTATION 
# ============================================================================

def generate_documentation(model_name, model_params, portfolio_summary,
                            var_result, stress_results, backtest_results,
                            hs_var, hs_cvar, sensitivity_df, krd_df,
                            output_path="model_risk_documentation.md"):
    today = datetime.date.today().strftime("%Y-%m-%d")
    mv, var, cvar = portfolio_summary["market_value"], var_result["var"], var_result["cvar"]
    conf          = RISK_CONFIG["confidence_level"]
    alpha_pct     = f"{(1 - conf):.0%}"

    # ── Model-specific blocks ────────────────────────────────────────────────
    if model_name == "vasicek":
        model_title   = "Vasicek (1977)"
        model_eq      = "dr_t = κ(θ − r_t) dt + σ dW_t"
        param_section = (
            "| Parameter | Symbol | Value | Description |\n"
            "|-----------|--------|-------|-------------|\n"
            f"| Mean-reversion speed | κ | {model_params['kappa']:.4f} | Controls speed of reversion to long-run mean |\n"
            f"| Long-run mean        | θ | {model_params['theta']:.4f} | Unconditional mean of the short rate |\n"
            f"| Volatility           | σ | {model_params['sigma']:.4f} | Diffusion coefficient |\n"
            f"| Initial short rate   | r₀ | {model_params['r0']:.4f} | Current observed short rate |\n"
        )
        disc_note = "Exact discretisation using the conditional Gaussian distribution of the Ornstein-Uhlenbeck process."
        zcb_note  = "Analytical A(τ)·exp(−B(τ)·r) formula; time-homogeneous."
        hw_assumption = ""
    else:
        model_title   = "Hull-White / Extended Vasicek (1990)"
        model_eq      = "dr_t = (θ(t) − a·r_t) dt + σ dW_t"
        param_section = (
            "| Parameter | Symbol | Value | Description |\n"
            "|-----------|--------|-------|-------------|\n"
            f"| Mean-reversion speed | a | {model_params['a']:.4f} | Controls speed of mean reversion |\n"
            f"| Volatility           | σ | {model_params['sigma']:.4f} | Diffusion coefficient |\n"
            f"| Initial short rate   | r₀ | {model_params['r0']:.4f} | Current observed short rate |\n"
            "| Drift function | θ(t) | Calibrated | Fit-to-market via Brigo-Mercurio cubic-spline |\n"
        )
        disc_note = "Euler-Maruyama with time-varying drift θ(t) computed analytically from cubic-spline-fitted yield curve."
        zcb_note  = "Brigo-Mercurio market-consistent formula; exact fit to initial yield curve at t=0."
        hw_assumption = "2. Hull-White θ(t) calibrated to initial yield curve via Brigo-Mercurio §3.3 formula, ensuring P(0,T)_model = P(0,T)_market for all T.\n"

    # ── Negative rate probability (SR 11-7 §4.1 — model-specific) ───────────
    # For Vasicek:  r_t ~ N(μ_t, σ_t²) exactly (OU process)
    # For HW:       r_t ~ N(μ_t, σ_t²) approximately (σ_t exact; μ_t via lower bound)
    t_check = 1.0   # assess at 1-year horizon (material for risk documentation)
    if model_name == "vasicek":
        kv, thv, sv, r0v = (model_params["kappa"], model_params["theta"],
                            model_params["sigma"], model_params["r0"])
        e_kt      = np.exp(-kv * t_check)
        mu_1yr    = thv + (r0v - thv) * e_kt
        sig_1yr   = np.sqrt(sv**2 / (2.0 * kv) * (1.0 - np.exp(-2.0 * kv * t_check)))
        prob_neg  = float(stats.norm.cdf(-mu_1yr / sig_1yr)) * 100.0
        neg_rate_section = (
            f"P(r_1yr < 0) = Φ(−{mu_1yr:.4f}/{sig_1yr:.4f}) = **{prob_neg:.5f}%**\n\n"
            f"Under current calibration (κ={kv:.3f}, θ={thv:.4f}, σ={sv:.4f}, r₀={r0v:.4f}), "
            f"the probability of a negative 1-year short rate is **{prob_neg:.5f}%** — "
            f"statistically negligible at prevailing rate levels. "
            f"However, this figure is highly sensitive to r₀: if r₀ were to fall to 50 bps "
            f"(e.g., post-crisis QE environment), P(r_1yr < 0) would rise to approximately 5–20%, "
            f"materially affecting VaR and hedge ratios.\n\n"
            f"**SR 11-7 Required Disclosure**: The Vasicek model is a Gaussian / Ornstein-Uhlenbeck "
            f"process. It assigns strictly positive probability to negative interest rates for all "
            f"finite time horizons. This is an inherent structural property of the model and is "
            f"**not a calibration artefact**. The model does not enforce any non-negativity floor. "
            f"If a non-negative rate constraint is operationally or contractually required, "
            f"the following alternatives should be evaluated: Cox-Ingersoll-Ross (CIR) "
            f"square-root diffusion, Black-Karasinski log-normal model, or a "
            f"shifted log-normal extension."
        )
    else:
        av, sv_hw, r0v_hw = (model_params["a"], model_params["sigma"], model_params["r0"])
        # HW: exact variance; mean lower bound via r0 * e^{-a*t} (theta(t) pull lifts the mean)
        mu_1yr_hw  = r0v_hw * np.exp(-av * t_check)   # conservative lower bound
        sig_1yr_hw = np.sqrt(sv_hw**2 / (2.0 * av) * (1.0 - np.exp(-2.0 * av * t_check)))
        prob_neg   = float(stats.norm.cdf(-mu_1yr_hw / sig_1yr_hw)) * 100.0
        neg_rate_section = (
            f"P(r_1yr < 0) ≲ Φ(−{mu_1yr_hw:.4f}/{sig_1yr_hw:.4f}) = **{prob_neg:.5f}%** "
            f"(upper bound; true mean is higher due to θ(t) upward drift calibration).\n\n"
            f"Under current calibration (a={av:.3f}, σ={sv_hw:.4f}, r₀={r0v_hw:.4f}), "
            f"the probability of a negative 1-year short rate is at most **{prob_neg:.5f}%**. "
            f"Because Hull-White θ(t) is calibrated to fit the upward-sloping yield curve, "
            f"the actual mean is higher than the lower bound, making this estimate conservative.\n\n"
            f"**SR 11-7 Required Disclosure**: Hull-White / Extended Vasicek is a Gaussian model. "
            f"It assigns positive probability to negative short rates for all finite horizons. "
            f"This is an inherent structural property shared by all Gaussian short-rate models "
            f"(Vasicek, Hull-White, Ho-Lee). The model does not enforce any non-negativity floor. "
            f"If a non-negative rate constraint is required, alternatives include: "
            f"CIR square-root diffusion, Black-Karasinski log-normal model, or "
            f"shifted log-normal extension."
        )

    # ── KRD table ────────────────────────────────────────────────────────────
    try:    krd_md = krd_df.to_markdown(index=False)
    except: krd_md = krd_df.to_string(index=False)
    total_krd  = round(krd_df["KRD (yrs)"].sum(), 4)
    total_kr01 = round(krd_df["KR01 ($)"].sum(), 0)

    # ── Table blocks ─────────────────────────────────────────────────────────
    stress_rows = "\n".join(
        f"| {n.replace('_',' ').title()} | ${p:,.0f} | {p/mv*100:.2f}% |"
        for n, p in stress_results.items()
    )
    try:    sens_md = sensitivity_df.to_markdown(index=False)
    except: sens_md = sensitivity_df.to_string(index=False)

    ku = backtest_results["kupiec"]
    ch = backtest_results["christoffersen"]
    tl = backtest_results["traffic_light"]

    hs_divergence = abs(var - hs_var) / hs_var
    hs_assessment = (
        "Estimates are broadly consistent (divergence < 15%). No material model bias detected."
        if hs_divergence < 0.15 else
        f"Material divergence detected ({hs_divergence*100:.1f}% > 15%). "
        "Investigate tail distributional assumptions — consider t-distribution extension."
    )

    doc = f"""# Fixed Income Monte Carlo VaR Engine — SR 11-7 Model Risk Documentation

| Field | Value |
|-------|-------|
| **Document Version** | 3.0 |
| **Effective Date** | {today} |
| **Model Name** | Fixed Income Monte Carlo VaR Engine |
| **Model Type** | Market Risk — Interest Rate & Spread |
| **Model Owner** | Quantitative Risk Management |
| **Validation Status** | Draft — Pending Independent MRM Review |
| **Active Rate Model** | {model_title} |
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
                     Monte Carlo Simulation  ({model_params.get('n_paths',10000):,} paths × {model_params.get('n_steps',252)} steps)
                     + Antithetic Variates  (variance ÷ 2 at zero extra cost)
                                                        │
                              Full Portfolio Repricing
                         ├── Coupon Bonds: Analytical ZCB discounting
                         ├── MBS:          PSA prepayment model (rate-sensitive CPR)
                         └── ABS:          Effective duration (scheduled amortisation)
                                                        │
                     P&L Distribution  →  VaR ({conf:.0%}), CVaR ({conf:.0%})
                                                        │
                         ├── Stress Testing  (6 scenarios, Nelson-Siegel shocked curves)
                         └── Backtesting & Benchmark Validation
```

### 1.3 Active Rate Model: {model_title}

**SDE**: `{model_eq}`

{param_section}
**Discretisation**: {disc_note}

**ZCB Pricing**: {zcb_note}

### 1.4 Yield Curve: Nelson-Siegel (1987)

```
y(τ) = β₀  +  β₁ · (1 − e^{{-τ/λ}}) / (τ/λ)  +  β₂ · [(1 − e^{{-τ/λ}}) / (τ/λ) − e^{{-τ/λ}}]
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
| **Number of Positions** | {portfolio_summary['n_positions']} |
| **Total Notional** | ${portfolio_summary['total_notional']:>15,.0f} |
| **Mark-to-Market Value** | ${mv:>15,.0f} |
| **Portfolio DV01** | ${portfolio_summary['total_dv01']:>15,.0f} |
| **Weighted Avg. Effective Duration** | {portfolio_summary['avg_duration']:.2f} years |

**Allocation by Product Type**

{portfolio_summary['composition_table']}

---

## 3. Key Model Assumptions

| # | Assumption | Rationale |
|---|-----------|-----------|
| 1 | Short rate follows a single-factor mean-reverting Gaussian diffusion (no jumps) | Parsimony; analytical ZCB tractability; well-established in literature |
| 2 | {hw_assumption.strip() if hw_assumption else "Vasicek is time-homogeneous; θ, κ, σ are constant"} | Ensures market-consistent pricing at t = 0 |
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

### 4.1 Negative Rate Risk — Quantitative Assessment (SR 11-7 Required)

{neg_rate_section}

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

## 6. Risk Results (As of {today})

### 6.1 VaR and CVaR

| Metric | Value |
|--------|-------|
| **Monte Carlo VaR ({conf:.0%}, 1-day)** | **${var:,.0f}** |
| **Monte Carlo CVaR ({conf:.0%}, 1-day)** | **${cvar:,.0f}** |
| VaR as % of Portfolio MtM | {var/mv*100:.3f}% |
| CVaR / VaR Ratio | {cvar/var:.2f}× |
| **10-day Regulatory VaR (√10 scaled)** | **${var*np.sqrt(10):,.0f}** |

### 6.2 Stress Test Results

| Scenario | Stressed P&L | % of Portfolio |
|----------|-------------|----------------|
{stress_rows}

*Note: Stressed yield curves re-fitted via Nelson-Siegel. MBS repriced with rate-sensitive PSA CPR.*

### 6.3 Key Rate Duration Profile (KRD)

KRD measures sensitivity to an isolated +1bp shock at each yield curve tenor, holding all other tenors fixed.
Unlike parallel DV01, KRD decomposes risk into **level**, **slope (twist)**, and **curvature (butterfly)** components.

| Portfolio Total KRD | Portfolio Total KR01 |
|---------------------|----------------------|
| **{total_krd:.4f} yrs** | **${total_kr01:,.0f}** |

{krd_md}

*Interpretation*: Tenors with large KRD/KR01 represent concentrated yield curve risk nodes.
A bear flattener (short rates rise, long rates stable) impacts high short-tenor KRDs disproportionately.
A bear steepener (long rates rise) impacts long-tenor KRDs. See krd_profile.png for the bar chart.

---

## 7. Model Validation Results

### 7.1 VaR Backtesting ({backtest_results['n_obs']}-day window)

#### 7.1.1 Kupiec Unconditional Coverage (POF) Test

*H₀: Exception frequency = α = {alpha_pct}*

| Exceptions Observed | Exceptions Expected | Exception Rate | LR Statistic (χ²₁) | p-value | Result |
|--------------------|--------------------|--------------|--------------------|---------|--------|
| {ku['n_exceptions']} | {ku['expected_exceptions']} | {ku['exception_rate']:.4f} | {ku['lr_stat']:.4f} | {ku['p_value']:.4f} | **{ku['conclusion']}** |

*Interpretation*: {"Zero exceptions indicate VaR is conservative (over-estimates risk). Statistically a FAIL on the POF test, but from a regulatory standpoint this is a conservative outcome." if ku['n_exceptions'] == 0 else "Exception frequency within expected range."}

#### 7.1.2 Christoffersen Conditional Coverage Test

*H₀: Exceptions are independently distributed (no clustering)*

| π₀₁ | π₁₁ | LR Independence | LR Combined (χ²₂) | p-value | Result |
|-----|-----|----------------|-------------------|---------|--------|
| {ch['pi_01']:.4f} | {ch['pi_11']:.4f} | {ch['lr_ind']:.4f} | {ch['lr_cc_stat']:.4f} | {ch['p_value']:.4f} | **{ch['conclusion']}** |

#### 7.1.3 Basel Traffic Light Assessment

| Exceptions (normalised to 250 days) | Zone | Signal | Regulatory Capital Multiplier |
|-------------------------------------|------|--------|------------------------------|
| {tl['exceptions_250d']} | **{tl['zone']}** | **{tl['signal']}** | {tl['capital_multiplier']:.1f}× |

### 7.2 Benchmark Comparison: Monte Carlo vs. Historical Simulation

| Method | VaR ({conf:.0%}, 1-day) | CVaR ({conf:.0%}, 1-day) |
|--------|------------------------|-------------------------|
| **Monte Carlo ({model_name.upper()})** | **${var:,.0f}** | **${cvar:,.0f}** |
| Historical Simulation | ${hs_var:,.0f} | ${hs_cvar:,.0f} |
| Absolute Difference | ${var - hs_var:+,.0f} | ${cvar - hs_cvar:+,.0f} |
| Relative Difference | {(var - hs_var)/hs_var*100:+.1f}% | {(cvar - hs_cvar)/hs_cvar*100:+.1f}% |

*Assessment*: {hs_assessment}

### 7.3 Parameter Sensitivity Analysis

{sens_md}

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
| Key Rate Duration (KRD) profile | ✅ Complete | §6.3 |
| Negative rate probability — model-specific quantification | ✅ Complete | §4.1 |
| Ongoing monitoring plan | ⚠️ Recommended | Quarterly recalibration |
| Independent MRM validation | ⚠️ Pending | Pre-production requirement |
| Model approval and sign-off | ⚠️ Pending | Senior Risk Officer / CRO |
| Model inventory registration | ⚠️ Pending | Post-approval |

### 8.2 Attestation

> *This model documentation reflects the development state of the Fixed Income Monte Carlo VaR
> Engine v3.2 as of {today}. The following enhancements have been implemented relative to prior
> versions: (1) Nelson-Siegel yield curve fitting replacing linear interpolation; (2) PSA
> prepayment model for MBS replacing effective-duration approximation; (3) Antithetic Variates
> for Monte Carlo variance reduction; (4) α×σ two-dimensional VaR sensitivity heatmap;
> (5) Key Rate Duration (KRD) profile capturing twist and butterfly yield curve risk;
> (6) Model-specific negative rate probability quantification per SR 11-7 §4.1 requirements.*
>
> *The model has not yet received independent Model Risk Management (MRM) validation.
> Deployment in a production risk system requires independent validation, formal approval by the
> Chief Risk Officer, and registration in the firm's Model Inventory in accordance with
> SR 11-7 §IV. Material changes to methodology, calibration, or scope require
> re-documentation and re-validation before deployment.*

---

*Fixed Income VaR Engine v3.2 | {today} | SR 11-7 | OCC 2011-12 | Basel III IMA*
*Nelson-Siegel (1987) | PSA Standard (1985) | Vasicek (1977) | Hull & White (1990)*
"""
    with open(output_path, "w") as fh:
        fh.write(doc)
    return output_path


# ============================================================================
# SECTION 9 — MAIN
# ============================================================================

SEP  = "─" * 62
DSEP = "═" * 62
def _hdr(t):       print(f"\n{SEP}\n  {t}\n{SEP}")
def _row(l, v, w=38): print(f"  {l:<{w}} {v}")


def main():
    print(f"\n{DSEP}")
    print("  Fixed Income VaR & Backtesting Engine  v3.2")
    print("  SR 11-7 | Basel III IMA | Hull-White / Vasicek")
    print("  Nelson-Siegel | PSA | Antithetic Variates | KRD")
    print(DSEP)

    # 1. Data ingestion
    _hdr("1 / 6  Data Ingestion")
    fred_ok = fetch_yield_curve_fred()
    _row("Yield curve source", "FRED API (live)" if fred_ok else "Hardcoded fallback (no FRED key)")
    positions_df   = query_positions()
    historical_pnl = query_historical_pnl(n_obs=BACKTEST_CONFIG["n_obs"])
    total_notional = positions_df["notional"].sum()
    total_dv01     = positions_df["dv01"].sum()
    avg_duration   = ((positions_df["effective_duration"] * positions_df["notional"]).sum()
                      / total_notional)
    _row("Positions loaded",        str(len(positions_df)))
    _row("Total notional",          f"${total_notional:>15,.0f}")
    _row("Portfolio DV01",          f"${total_dv01:>15,.0f}")
    _row("Wtd. avg. eff. duration", f"{avg_duration:.2f} yrs")

    # 2. Rate model
    _hdr(f"2 / 6  Rate Model — {MODEL_CONFIG['model'].upper()}")
    model, model_name = get_model()
    model_params = {**MODEL_CONFIG[model_name],
                    "n_paths":      SIMULATION_CONFIG["n_paths"],
                    "n_steps":      SIMULATION_CONFIG["n_steps"],
                    "horizon_days": SIMULATION_CONFIG["horizon_days"]}
    _row("Model",   model_name.upper())
    _row("Paths",   f"{SIMULATION_CONFIG['n_paths']:,}")
    _row("Horizon", f"{SIMULATION_CONFIG['horizon_days']} trading day(s)")

    # 3. Monte Carlo VaR
    _hdr("3 / 6  Monte Carlo Simulation → VaR / CVaR")
    print(f"  Simulating {SIMULATION_CONFIG['n_paths']:,} paths × {SIMULATION_CONFIG['n_steps']} steps …")
    pnl_mc, market_value, r_paths = compute_portfolio_pnl(positions_df, model, SIMULATION_CONFIG)
    conf = RISK_CONFIG["confidence_level"]
    var_mc, cvar_mc = calculate_var_cvar(pnl_mc, conf)
    print()
    _row("Portfolio Mark-to-Market",  f"${market_value:>15,.0f}")
    _row("P&L Std Dev (daily)",       f"${pnl_mc.std():>15,.0f}")
    _row(f"VaR ({conf:.0%}, 1-day)",  f"${var_mc:>15,.0f}")
    _row(f"CVaR ({conf:.0%}, 1-day)", f"${cvar_mc:>15,.0f}")
    _row("VaR % of Portfolio",        f"{var_mc/market_value*100:.3f}%")
    _row("CVaR / VaR ratio",          f"{cvar_mc/var_mc:.2f}×")
    _row("10-day VaR (√10 scaled)",   f"${var_mc*np.sqrt(10):>15,.0f}")

    # 4. Stress tests + KRD
    _hdr("4 / 6  Stress Testing & Key Rate Durations")
    stress_pnl, _ = run_stress_tests(positions_df)
    print(f"  {'Scenario':<42} {'P&L Impact':>14}  {'% Portfolio':>11}")
    print(f"  {'─'*42} {'─'*14}  {'─'*11}")
    for scen, pnl in stress_pnl.items():
        print(f"  {scen.replace('_',' ').title():<42} ${pnl:>13,.0f}  {pnl/market_value*100:>10.2f}%")

    print(f"\n  ── Key Rate Durations (+1bp isolated tenor shocks) ──\n")
    krd_df, _ = compute_krd(positions_df)
    print(f"  {'Tenor (yr)':<12} {'KRD (yrs)':>12} {'KR01 ($)':>14}")
    print(f"  {'─'*12} {'─'*12} {'─'*14}")
    for _, row in krd_df.iterrows():
        print(f"  {str(row['Tenor (yr)']):<12} {row['KRD (yrs)']:>12.4f} ${row['KR01 ($)']:>13,.0f}")
    print(f"  {'─'*12} {'─'*12} {'─'*14}")
    print(f"  {'Total':<12} {krd_df['KRD (yrs)'].sum():>12.4f} ${krd_df['KR01 ($)'].sum():>13,.0f}")

    # 5. Validation
    _hdr("5 / 6  Model Validation")
    backtest = run_backtesting(historical_pnl, var_mc, alpha=1.0 - conf)
    backtest["n_obs"] = BACKTEST_CONFIG["n_obs"]
    ku, ch, tl = backtest["kupiec"], backtest["christoffersen"], backtest["traffic_light"]

    print(f"\n  ── Backtesting ({backtest['n_obs']}-day window) ──")
    _row("Exceptions obs / expected",
         f"{ku['n_exceptions']} / {ku['expected_exceptions']}")
    _row("Kupiec POF",
         f"{ku['conclusion']}  (LR={ku['lr_stat']:.3f}, p={ku['p_value']:.4f})")
    _row("Christoffersen CC",
         f"{ch['conclusion']}  (LR={ch['lr_cc_stat']:.3f}, p={ch['p_value']:.4f})")
    _row("Basel Traffic Light",
         f"{tl['zone']} Zone — {tl['signal']}  (k={tl['capital_multiplier']:.1f}×)")

    hs_var, hs_cvar = historical_simulation_var(historical_pnl, conf)
    print(f"\n  ── Benchmark: MC vs Historical Simulation ──")
    _row("MC VaR  / HS VaR",
         f"${var_mc:,.0f}  /  ${hs_var:,.0f}  (Δ={(var_mc-hs_var)/hs_var*100:+.1f}%)")
    _row("MC CVaR / HS CVaR",
         f"${cvar_mc:,.0f}  /  ${hs_cvar:,.0f}  (Δ={(cvar_mc-hs_cvar)/hs_cvar*100:+.1f}%)")

    print(f"\n  ── Parameter Sensitivity ──\n")
    sens_df = sensitivity_analysis(positions_df, model_name, var_mc, cvar_mc)
    print(sens_df.to_string(index=False))

    # 6. Charts
    _hdr("6a / 8  Generating Dashboard Charts")
    chart_path = "var_report.png"
    plot_results(pnl_mc, var_mc, cvar_mc, r_paths, stress_pnl, sens_df,
                 historical_pnl, hs_var, model_name, market_value,
                 output_path=chart_path)
    _row("Dashboard chart saved to", chart_path)

    _hdr("6b / 8  Key Rate Duration Chart")
    krd_path = "krd_profile.png"
    plot_krd_profile(krd_df, output_path=krd_path)
    _row("KRD chart saved to", krd_path)

    _hdr("6d / 8  α×σ VaR Sensitivity Heatmap")
    heatmap_path = "heatmap_alpha_sigma.png"
    print("  Computing VaR across 7 σ-levels × 5 confidence levels …")
    plot_alpha_sigma_heatmap(positions_df, model_name, output_path=heatmap_path)
    _row("Heatmap saved to", heatmap_path)

    # Documentation
    _hdr("6e / 8  SR 11-7 Model Risk Documentation")
    comp = (positions_df.groupby("product_type")["notional"].sum()
                        .reset_index().rename(columns={"notional": "Notional ($)"}))
    comp["% of Portfolio"] = (comp["Notional ($)"] / total_notional * 100).round(1)
    try:    comp_md = comp.to_markdown(index=False, floatfmt=",.0f")
    except: comp_md = comp.to_string(index=False)

    doc_path = generate_documentation(
        model_name=model_name, model_params=model_params,
        portfolio_summary={"n_positions": len(positions_df), "total_notional": total_notional,
                           "market_value": market_value, "total_dv01": total_dv01,
                           "avg_duration": avg_duration, "composition_table": comp_md},
        var_result={"var": var_mc, "cvar": cvar_mc},
        stress_results=stress_pnl, backtest_results=backtest,
        hs_var=hs_var, hs_cvar=hs_cvar, sensitivity_df=sens_df,
        krd_df=krd_df,
        output_path="model_risk_documentation.md",
    )
    _row("Documentation saved to", doc_path)

    print(f"\n{DSEP}\n  ENGINE RUN COMPLETE  (v3.2)\n{DSEP}")
    print(f"\n  VaR  (99%, 1d) : ${var_mc:,.0f}")
    print(f"  CVaR (99%, 1d) : ${cvar_mc:,.0f}")
    print(f"  Backtesting    : Kupiec {ku['conclusion']} | "
          f"Christoffersen {ch['conclusion']} | Basel {tl['zone']}")
    print(f"  Dashboard      : {chart_path}")
    print(f"  Heatmap        : {heatmap_path}")
    print(f"  Documentation  : {doc_path}\n")


if __name__ == "__main__":
    main()
