"""Streamlit dashboard for real-time option pricing and Greeks visualization."""

import streamlit as st
import pandas as pd
from ..config import CONFIG
from ..pipelines.python_pipeline import build_greeks_report


def run_dashboard():
    st.set_page_config(page_title="Option Greeks Risk Dashboard", layout="wide")
    st.title("Option Pricing & Greeks Risk Dashboard")

    st.sidebar.header("Settings")
    model = st.sidebar.selectbox("Model", ["black_scholes", "heston", "monte_carlo"], index=0)
    st.sidebar.write("Stress scenarios")
    st.sidebar.write(CONFIG["stress_scenarios"].keys())

    st.markdown("## Sample Greeks Report")
    sample_trades = pd.DataFrame([
        {
            "trade_id": 1,
            "symbol": "CALL",
            "option_type": "call",
            "strike": 100.0,
            "expiry_date": pd.Timestamp("2025-01-01"),
            "underlying_price": 100.0,
            "implied_vol": 0.20,
            "interest_rate": 0.03,
            "dividend_yield": 0.01,
            "trade_date": pd.Timestamp("2024-10-01"),
        }
    ])
    report = build_greeks_report(sample_trades, sample_trades, CONFIG)
    st.dataframe(report)

    st.markdown("## Governance Notes")
    st.write("This dashboard is configured to support options pricing, Greeks analytics, hedging P&L, and stress-test output summaries.")


if __name__ == "__main__":
    run_dashboard()
