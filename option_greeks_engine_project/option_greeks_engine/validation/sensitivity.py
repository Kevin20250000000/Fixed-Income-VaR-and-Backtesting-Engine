"""Greeks parameter sensitivity analysis."""

import numpy as np


def finite_difference_sensitivity(func, base_params, shift=1e-4):
    sensitivities = {}
    for name, value in base_params.items():
        bumped = dict(base_params)
        bumped[name] = value + shift
        base_price = func(base_params)
        bumped_price = func(bumped)
        sensitivities[name] = (bumped_price - base_price) / shift
    return sensitivities
