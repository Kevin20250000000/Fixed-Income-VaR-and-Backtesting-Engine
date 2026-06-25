"""FastAPI pricing API for option pricing and Greeks."""

from fastapi import FastAPI
from pydantic import BaseModel
from ..models.black_scholes import BlackScholesModel

app = FastAPI(title="Option Greeks Pricing API")

class PricingRequest(BaseModel):
    option_type: str
    strike: float
    spot: float
    rate: float
    dividend_yield: float
    volatility: float
    tau: float


@app.post("/price")
def price_option(request: PricingRequest):
    model = BlackScholesModel()
    price = model.price(request.option_type, request.spot, request.strike,
                        request.rate, request.dividend_yield, request.volatility, request.tau)
    greeks = model.greeks(request.option_type, request.spot, request.strike,
                          request.rate, request.dividend_yield, request.volatility, request.tau)
    return {"price": price, "greeks": greeks}
