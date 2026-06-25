"""Pricing bias benchmark utilities for Black-Scholes vs Heston."""

import pandas as pd


def compare_pricing_bias(report: pd.DataFrame) -> pd.DataFrame:
    if "bs_price" not in report or "heston_price" not in report:
        raise ValueError("report must contain bs_price and heston_price columns")
    report = report.copy()
    report["bias_bs_minus_heston"] = report["bs_price"] - report["heston_price"]
    report["bias_pct"] = report["bias_bs_minus_heston"] / report["heston_price"].replace(0.0, 1.0)
    return report
