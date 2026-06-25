# Project Design: Option Greeks Risk Engine

## Objective

Design a modular project that extends the fixed-income risk engine style from `fi_var_engine_v5` to option pricing. The new engine should support:

- Black-Scholes, Heston, and Monte Carlo option pricing
- Full-suite Greeks (Delta, Gamma, Vega, Theta, Rho)
- Dynamic delta hedging simulation across multiple scenarios
- SQL + Python data pipelines for trade and market data ingestion
- Model validation and benchmarking
- Stress testing and sensitivity analysis
- SR 11-7 compliant documentation generation
- Streamlit dashboard visualization and optional FastAPI API

## Architecture

1. Configuration
   - Central config dictionary for model selection, market data, stress scenarios, backtest parameters.
2. Models
   - `black_scholes.py`: closed-form pricing and Greeks.
   - `heston.py`: semi-analytic Heston pricing and implied computational Greeks.
   - `monte_carlo.py`: pathwise simulation for pricing, Greeks, and hedging.
3. Data Pipeline
   - `sql_pipeline.py`: SQL templates and ingestion for trades, quotes, positions.
   - `python_pipeline.py`: DataFrame-based ETL, cleaning, and Greeks report generation.
4. Validation
   - `backtesting.py`: Kupiec, Christoffersen, Basel Traffic Light tests.
   - `bias.py`: pricing bias benchmark between BS and Heston.
   - `sensitivity.py`: parameter sensitivity analysis for Greeks.
5. Reporting
   - `documentation.py`: generate SR 11-7 model validation and governance documentation in Markdown.
6. Visualization
   - `streamlit_app.py`: dashboard for pricing surfaces, Greeks, hedging P&L, stress outputs.
7. API
   - `fastapi_app.py`: standardized pricing / Greeks REST API.

## Compliance Notes

- SR 11-7 documentation should cover:
  - model assumptions and limitations
  - data usage and governance
  - validation scope and results
  - stress-testing and scenario design
  - model change controls and ownership

## Deliverables

- Code skeleton with clearly separated modules
- End-to-end pipeline from ingestion to visualization
- Example streamlit dashboard and optional FastAPI endpoint
- Markdown documentation for validation and governance
