"""Black-Scholes pricing and Greeks."""

import numpy as np
from scipy.stats import norm


def _d1(S, K, r, q, sigma, tau):
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))


def _d2(d1, sigma, tau):
    return d1 - sigma * np.sqrt(tau)


class BlackScholesModel:
    """Black-Scholes closed-form option pricing and Greeks."""

    def price(self, option_type, S, K, r, q, sigma, tau):
        d1 = _d1(S, K, r, q, sigma, tau)
        d2 = _d2(d1, sigma, tau)
        if option_type == "call":
            return S * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
        return K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1)

    def greeks(self, option_type, S, K, r, q, sigma, tau):
        d1 = _d1(S, K, r, q, sigma, tau)
        d2 = _d2(d1, sigma, tau)
        pdf = norm.pdf(d1)
        delta = np.exp(-q * tau) * norm.cdf(d1 if option_type == "call" else d1 - sigma * np.sqrt(tau))
        gamma = np.exp(-q * tau) * pdf / (S * sigma * np.sqrt(tau))
        vega = S * np.exp(-q * tau) * pdf * np.sqrt(tau)
        theta = (-S * sigma * np.exp(-q * tau) * pdf / (2 * np.sqrt(tau))
                 - r * K * np.exp(-r * tau) * norm.cdf(d2 if option_type == "call" else -d2)
                 + q * S * np.exp(-q * tau) * norm.cdf(d1 if option_type == "call" else -d1))
        rho = K * tau * np.exp(-r * tau) * (norm.cdf(d2) if option_type == "call" else -norm.cdf(-d2))
        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "theta": float(theta),
            "rho": float(rho),
        }
