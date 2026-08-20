# -*- coding: utf-8 -*-
"""Seasonal localization by the national standard load profile (TDD4).

Key idea of the paper: the behavioural core of the model comes from a
foreign time-use survey, but the seasonal shape (how consumption is spread
over the year) is taken from the public regulatory standard of the target
country -- the TDD4 standard supply profile. The factors are applied
multiplicatively to the simulated daily profile by month; the annual energy
is unchanged, only its distribution over the year changes (the weighted
average of the factors equals 1).

The factors below are precomputed from the Czech TDD4 profile (OTE) so the
application does not have to redistribute the raw OTE input data.
"""

# Month (1-12) -> seasonal factor f_m; weighted annual average = 1.
# Source: TDD4, Czech Republic (OTE). January/July ~ 1.28; max/min ~ 1.30.
LOCALIZATION_FACTORS = {
    1: 1.1455,
    2: 1.1041,
    3: 1.0294,
    4: 0.9648,
    5: 0.9213,
    6: 0.9040,
    7: 0.8936,
    8: 0.9025,
    9: 0.9249,
    10: 0.9778,
    11: 1.0781,
    12: 1.1600,
}


def factor(month: int) -> float:
    """Return the seasonal factor for the given month (1-12)."""
    return LOCALIZATION_FACTORS[int(month)]
