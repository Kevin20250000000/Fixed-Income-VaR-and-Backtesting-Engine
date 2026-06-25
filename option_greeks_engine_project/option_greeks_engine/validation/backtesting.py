"""Backtesting and coverage tests for option risk models."""

import numpy as np


def kupiec_pof(actual_exceptions: np.ndarray, alpha: float) -> dict:
    n = len(actual_exceptions)
    x = actual_exceptions.sum()
    p = alpha
    prob = (p**x) * ((1 - p) ** (n - x))
    return {"n": n, "exceptions": int(x), "pof": float(prob)}


def christoffersen_cc(actual_exceptions: np.ndarray, alpha: float) -> dict:
    # Placeholder: implement independence and conditional coverage statistic
    return {"alpha": alpha, "exceptions": int(actual_exceptions.sum()), "status": "pending"}


def basel_traffic_light(actual_exceptions: np.ndarray, alpha: float) -> str:
    n = len(actual_exceptions)
    x = int(actual_exceptions.sum())
    if x <= 4:
        return "green"
    if x <= 9:
        return "yellow"
    return "red"
