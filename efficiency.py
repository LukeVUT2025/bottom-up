# -*- coding: utf-8 -*-
"""Efficiency scenario: switching to the most efficient appliances available.

The appliance-level decomposition allows an efficiency class to be applied
*after the fact* -- the appliance power profile is scaled linearly by a
factor corresponding to a better class, without re-running the simulation.
Thermal appliances (kettle, hob, oven, toaster, microwave, boiler, heating)
are NOT changed: bringing a litre of water to the boil costs a fixed amount
of energy regardless of any "efficiency class".

Result from the paper: the efficiency upgrade cuts annual consumption by
about a quarter, but the evening peak only by about a sixth, because the
peak is dominated by thermal appliances with no efficiency headroom.
"""

# Appliance key -> power multiplier in the efficient scenario. Keys missing
# from the table are left unchanged (factor 1.0), typically thermal appliances.
EFFICIENCY_FACTORS = {
    'refrigerator':    0.55,
    'freezer':         0.55,
    'lighting':        0.50,
    'tv':              0.60,
    'pc':              0.60,
    'washing_machine': 0.75,
    'dishwasher':      0.75,
}


def apply(appliances: dict) -> dict:
    """Return a new dict {appliance: profile} with appliances scaled."""
    out = {}
    for key, arr in appliances.items():
        out[key] = arr * EFFICIENCY_FACTORS.get(key, 1.0)
    return out
