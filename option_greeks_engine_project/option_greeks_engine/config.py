"""Project configuration for option pricing and risk analytics."""

from dataclasses import dataclass
from typing import Dict, List

CONFIG = {
    "model": "black_scholes",  # options: black_scholes, heston, monte_carlo
    "market": {
        "spot": 100.0,
        "rate": 0.03,
        "dividend_yield": 0.01,
        "volatility": 0.20,
    },
    "simulation": {
        "n_paths": 50_000,
        "n_steps": 252,
        "dt": 1 / 252,
        "seed": 42,
    },
    "stress_scenarios": {
        "parallel_up_200bps": {"rate": +0.02, "volatility": +0.05},
        "parallel_down_100bps": {"rate": -0.01, "volatility": -0.03},
        "vol_spike": {"rate": 0.0, "volatility": +0.15},
    },
    "backtest": {"n_obs": 250, "alpha": 0.01},
}

OPTION_LEGS = [
    {
        "symbol": "CALL",
        "option_type": "call",
        "strike": 100.0,
        "expiry_days": 90,
        "notional": 1_000_000,
    },
    {
        "symbol": "PUT",
        "option_type": "put",
        "strike": 95.0,
        "expiry_days": 120,
        "notional": 1_000_000,
    },
]
