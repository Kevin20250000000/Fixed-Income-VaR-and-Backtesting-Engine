"""Monte Carlo option pricing with Greeks and dynamic delta hedging."""

import numpy as np
from scipy.stats import norm
from .black_scholes import BlackScholesModel


def _geometric_brownian_motion_paths(S0, r, q, sigma, tau, n_paths, n_steps, seed=42):
    rng = np.random.default_rng(seed)
    dt = tau / n_steps
    S_paths = np.full((n_paths, n_steps + 1), S0, dtype=float)
    drift = (r - q - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    for t in range(n_steps):
        z = rng.standard_normal(n_paths)
        S_paths[:, t + 1] = S_paths[:, t] * np.exp(drift + vol * z)
    return S_paths


class MonteCarloModel:
    """Monte Carlo engine for pricing, Greeks, and hedging simulations."""

    def price(self, option_type, S, K, r, q, sigma, tau, n_paths, n_steps, seed=42):
        if tau <= 0.0:
            return float(max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0))
        S_paths = _geometric_brownian_motion_paths(S, r, q, sigma, tau, n_paths, n_steps, seed)
        payoff = np.maximum(S_paths[:, -1] - K, 0.0) if option_type == "call" else np.maximum(K - S_paths[:, -1], 0.0)
        return float(np.exp(-r * tau) * np.mean(payoff))

    def greeks(self, option_type, S, K, r, q, sigma, tau, n_paths=20_000, n_steps=252, seed=42):
        bs = BlackScholesModel()
        eps = max(1e-4, 1e-4 * S)
        vol_step = max(1e-4, 1e-4 * sigma)
        rate_step = max(1e-5, 1e-4 * abs(r) + 1e-5)
        time_step = max(1.0 / 252, tau / 252)

        def price_wrap(params):
            return self.price(option_type,
                              params["S"], params["K"], params["r"], params["q"], params["sigma"], params["tau"],
                              n_paths, n_steps, seed)

        base_params = {"S": S, "K": K, "r": r, "q": q, "sigma": sigma, "tau": tau}
        p0 = price_wrap(base_params)

        delta = (price_wrap({**base_params, "S": S + eps}) - price_wrap({**base_params, "S": S - eps})) / (2 * eps)
        gamma = (price_wrap({**base_params, "S": S + eps}) - 2 * p0 + price_wrap({**base_params, "S": S - eps})) / (eps**2)
        vega = (price_wrap({**base_params, "sigma": sigma + vol_step}) - price_wrap({**base_params, "sigma": sigma - vol_step})) / (2 * vol_step)
        theta = (price_wrap({**base_params, "tau": max(tau - time_step, 1e-8)}) - p0) / time_step
        rho = (price_wrap({**base_params, "r": r + rate_step}) - price_wrap({**base_params, "r": r - rate_step})) / (2 * rate_step)

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "theta": float(theta),
            "rho": float(rho),
        }

    def simulate_delta_hedge(self,
                             option_type,
                             S,
                             K,
                             r,
                             q,
                             sigma,
                             tau,
                             n_paths=20_000,
                             n_steps=252,
                             hedge_freq=21,
                             seed=42):
        if tau <= 0.0:
            return {"pnl": np.zeros(n_paths), "summary": {"mean": 0.0, "std": 0.0, "loss_pct": 0.0}}

        S_paths = _geometric_brownian_motion_paths(S, r, q, sigma, tau, n_paths, n_steps, seed)
        dt = tau / n_steps
        hedge_dates = np.arange(0, n_steps + 1, hedge_freq)
        if hedge_dates[-1] != n_steps:
            hedge_dates = np.append(hedge_dates, n_steps)

        bs = BlackScholesModel()
        cash = np.zeros(n_paths, dtype=float)
        delta = np.zeros(n_paths, dtype=float)
        shares = np.zeros(n_paths, dtype=float)

        def bs_delta(s_cur, tau_rem):
            if tau_rem <= 0.0:
                return np.zeros_like(s_cur)
            d1 = (np.log(s_cur / K) + (r - q + 0.5 * sigma**2) * tau_rem) / (sigma * np.sqrt(tau_rem))
            if option_type == "call":
                return np.exp(-q * tau_rem) * norm.cdf(d1)
            return np.exp(-q * tau_rem) * (norm.cdf(d1) - 1.0)

        delta[:] = bs_delta(S_paths[:, 0], tau)
        shares[:] = -delta
        cash[:] = delta * S - bs.price(option_type, S, K, r, q, sigma, tau)

        for i in range(1, len(hedge_dates)):
            t0 = hedge_dates[i - 1]
            t1 = hedge_dates[i]
            tau_next = tau - t1 * dt
            S_before = S_paths[:, t1]
            delta_new = bs_delta(S_before, tau_next)
            cash *= np.exp(r * (t1 - t0) * dt)
            cash -= (delta_new - shares) * S_before
            shares = -delta_new

        S_terminal = S_paths[:, -1]
        payoff = np.maximum(S_terminal - K, 0.0) if option_type == "call" else np.maximum(K - S_terminal, 0.0)
        cash += shares * S_terminal
        pnl = cash - payoff

        return {
            "pnl": pnl,
            "summary": {
                "mean": float(np.mean(pnl)),
                "std": float(np.std(pnl)),
                "loss_pct": float(np.mean(pnl < 0.0)),
                "min": float(np.min(pnl)),
                "max": float(np.max(pnl)),
            },
        }
