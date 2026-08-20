# -*- coding: utf-8 -*-
"""Photovoltaic system model (pvlib + PVGIS TMY).

The computation chain matches the one in the paper: from the sun position
through plane-of-array irradiance and cell temperature (Faiman) to DC power
(PVWatts) and through the inverter model to AC power at the terminals.
The weather is downloaded as a typical meteorological year (TMY) from the
PVGIS database for the given location -- requires an internet connection.

Returns a representative daily generation profile (an average day of the
month) resampled to the requested time resolution. Positive values = generation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pvlib
    from pvlib import irradiance
    PVLIB_AVAILABLE = True
except Exception:  # pragma: no cover
    PVLIB_AVAILABLE = False

_TMY_CACHE: dict = {}


def _load_tmy(lat: float, lon: float, tz: str = "Europe/Prague") -> pd.DataFrame:
    """Download TMY from PVGIS and return a DataFrame with ghi/dni/dhi/temp_air/wind_speed."""
    key = (round(lat, 3), round(lon, 3))
    if key in _TMY_CACHE:
        return _TMY_CACHE[key]

    res = pvlib.iotools.get_pvgis_tmy(lat, lon, map_variables=True)
    # pvlib returns a tuple of varying length across versions; find the DataFrame.
    df = None
    for item in (res if isinstance(res, (tuple, list)) else (res,)):
        if isinstance(item, pd.DataFrame):
            df = item
            break
    if df is None:
        raise RuntimeError("PVGIS: could not obtain a TMY DataFrame.")

    # Normalise column names (map_variables=True usually yields these names).
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    for want in ("ghi", "dni", "dhi", "temp_air", "wind_speed"):
        if want in cols:
            rename[cols[want]] = want
    df = df.rename(columns=rename)
    for req in ("ghi", "dni", "dhi", "temp_air", "wind_speed"):
        if req not in df.columns:
            raise RuntimeError(f"PVGIS TMY: missing column {req!r}.")

    # Time axis in local time.
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert(tz)

    _TMY_CACHE[key] = df
    return df


def simulate_month(month: int, n_points: int, params: dict) -> np.ndarray:
    """Representative daily PV generation profile for the given month.

    params: latitude, longitude, kwp, tilt, azimuth (180 = south).
    Returns an array of length ``n_points`` in watts (positive = generation).
    """
    if not PVLIB_AVAILABLE:
        raise RuntimeError("pvlib is not installed. Run: pip install pvlib")

    lat = float(params["latitude"])
    lon = float(params["longitude"])
    tilt = float(params.get("tilt", 35.0))
    azimuth = float(params.get("azimuth", 180.0))
    pdc0 = 1000.0 * float(params.get("kwp", 1.0))  # W

    tmy = _load_tmy(lat, lon)
    site = pvlib.location.Location(lat, lon, tz="Europe/Prague")

    mdf = tmy[tmy.index.month == int(month)].copy()
    if mdf.empty:
        raise RuntimeError(f"PVGIS TMY: no data for month {month}.")

    solpos = site.get_solarposition(mdf.index)
    poa = irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=mdf["dni"],
        ghi=mdf["ghi"],
        dhi=mdf["dhi"],
        model="isotropic",
    )
    temp_cell = pvlib.temperature.faiman(
        poa_global=poa["poa_global"],
        temp_air=mdf["temp_air"],
        wind_speed=mdf["wind_speed"],
    )
    dc = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=poa["poa_global"],
        temp_cell=temp_cell,
        pdc0=pdc0,
        gamma_pdc=-0.004,
    )
    ac = pvlib.inverter.pvwatts(pdc=dc, pdc0=pdc0).clip(lower=0)

    # Random representative day of the month (1st-28th). Unlike a whole-month
    # average this gives run-to-run variability and a realistic peak (on a
    # clear day generation approaches the installed power; note that kWp is the
    # DC nameplate power, so the AC peak is naturally lower even on a clear day).
    days = mdf.index.normalize().unique()
    days = days[days.day <= 28]
    if len(days) == 0:
        days = mdf.index.normalize().unique()
    sel = days[np.random.randint(len(days))]
    day = ac[ac.index.normalize() == sel]
    daily = day.groupby(day.index.hour).mean().reindex(range(24), fill_value=0.0)
    hours = np.arange(24)
    x_dst = np.linspace(0, 24, n_points, endpoint=False)
    return np.interp(x_dst, hours, daily.to_numpy(), period=24)
