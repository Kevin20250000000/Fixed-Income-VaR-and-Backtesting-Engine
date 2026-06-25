"""Python-based data pipeline for Greeks report generation."""

import pandas as pd
from ..models.black_scholes import BlackScholesModel
from ..models.heston import HestonModel
from ..models.monte_carlo import MonteCarloModel


def build_greeks_report(trades: pd.DataFrame, market: pd.DataFrame, config: dict) -> pd.DataFrame:
    bs = BlackScholesModel()
    heston = HestonModel()
    mc = MonteCarloModel()
    rows = []

    for _, trade in trades.iterrows():
        S = trade["underlying_price"]
        K = trade["strike"]
        r = trade["interest_rate"]
        q = trade["dividend_yield"]
        sigma = trade["implied_vol"]
        tau = float((trade["expiry_date"] - trade["trade_date"]).days) / 252
        row = {"trade_id": trade["trade_id"], "symbol": trade["symbol"]}
        row["bs_price"] = bs.price(trade["option_type"], S, K, r, q, sigma, tau)
        row.update({f"bs_{k}": v for k, v in bs.greeks(trade["option_type"], S, K, r, q, sigma, tau).items()})
        row["heston_price"] = heston.price(trade["option_type"], S, K, r, q, tau)
        row["mc_price"] = mc.price(trade["option_type"], S, K, r, q, sigma, tau,
                                    config["simulation"]["n_paths"], config["simulation"]["n_steps"], config["simulation"]["seed"])
        rows.append(row)

    return pd.DataFrame(rows)
