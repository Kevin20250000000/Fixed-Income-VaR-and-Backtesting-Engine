# Option Greeks Risk Engine

A Python project for option pricing, full-suite Greeks analytics, delta hedging, stress testing, backtesting, and SR 11-7 compliant documentation.

## Scope

- Black-Scholes, Heston, Monte Carlo pricing
- Greeks: Delta, Gamma, Vega, Theta, Rho
- Dynamic Delta hedging simulation
- SQL + Python data pipelines for trade and market data
- Model validation: Kupiec, Christoffersen, Basel Traffic Light
- Pricing bias and Greek sensitivity analysis
- Streamlit dashboard for real-time visualization
- Optional FastAPI pricing and Greeks API
- SR 11-7 compliant model governance documentation

## Project Layout

- `option_greeks_engine/`
  - `models/` – pricing and Greeks engines
  - `pipelines/` – SQL and Python ETL flows
  - `validation/` – backtesting, bias, sensitivity modules
  - `reports/` – SR 11-7 and model governance output
  - `dashboard/` – Streamlit visualization app
  - `api/` – FastAPI pricing API

## Getting Started

```bash
cd option_greeks_engine_project
python -m pip install -r requirements.txt
streamlit run option_greeks_engine/dashboard/streamlit_app.py
```

Optional API run:

```bash
uvicorn option_greeks_engine.api.fastapi_app:app --reload
```
