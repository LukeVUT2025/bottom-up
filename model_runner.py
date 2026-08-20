# -*- coding: utf-8 -*-
"""Orchestration of a single simulation: core + localization + efficiency
scenario + reactive power + photovoltaics.

This layer is called both by the graphical interface (``app.py``) and by the
example script (``examples/run_headless.py``). The GUI thus stays thin and the
whole computation is testable without a window.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Set

import numpy as np

import household_simulation as sd
import localization
import efficiency

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")

# Supported output resolutions (s) and horizon (always a single day here).
INTERVALS = {"1 s": 1, "1 min": 60, "10 min": 600, "15 min": 900}
DAY_TYPES = {"Weekday": 1, "Weekend": 2}

# Equipment category (consumption class) -> Low / Medium / High share.
# Controls appliance ownership probability (penetration_by_class.csv):
# a lower class typically lacks a dishwasher, dryer and boiler.
CLASS_MIX = {
    "Mix (1/3)": (1 / 3, 1 / 3, 1 / 3),
    "Low": (1.0, 0.0, 0.0),
    "Medium": (0.0, 1.0, 0.0),
    "High": (0.0, 0.0, 1.0),
}


@dataclass
class RunConfig:
    month: int = 2
    n_households: int = 50
    interval_seconds: int = 600
    period_type: int = 1              # 1 = weekday, 2 = weekend
    iterations: int = 5
    equip_class: str = "Mix (1/3)"           # equipment category (see CLASS_MIX)
    enabled_appliances: Optional[Set[str]] = None   # None = all
    reactive: bool = True
    localize: bool = True
    efficiency: bool = False
    dishwasher_hot_water: bool = False     # dishwasher: False = cold water (default), True = hot-water supply
    pv: bool = False
    pv_params: Dict = field(default_factory=lambda: {
        "latitude": 49.193, "longitude": 16.612,
        "kwp": 5.0, "tilt": 35.0, "azimuth": 180.0,
    })


def _groups_from_appliances(appl: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {}
    for g in sd.GROUP_KEYS:
        keys = [k for k in sd.APPLIANCE_GROUPS[g] if k in appl]
        if keys:
            out[g] = np.sum([appl[k] for k in keys], axis=0)
    return out


def run(cfg: RunConfig, progress: Optional[Callable[[int, str], None]] = None) -> dict:
    """Run the simulation according to the configuration and return results.

    Returned dict:
        t            time axis in hours (n points)
        total        aggregated active power (W)
        groups       {group: profile W}
        appliances   {appliance: profile W}
        reactive     total reactive power (var) or None
        reactive_appliances  {appliance: reactive-power profile (var)}
                     (only for those with a non-unity power factor)
        pv           PV generation (W, positive) or None
        net          total - pv (W) or None
        meta         run parameters
    """
    def _tick(pct, msg):
        if progress:
            progress(pct, msg)

    ag = cfg.interval_seconds
    n = 86400 // ag
    N = int(cfg.n_households)
    ea = cfg.enabled_appliances

    sim = sd.HouseholdSimulation(period_type=cfg.period_type, data_dir=DATA_DIR)
    # Equipment category: override the Low/Medium/High class shares.
    sim.class_mix = CLASS_MIX.get(cfg.equip_class, (1 / 3, 1 / 3, 1 / 3))
    sim.dishwasher_hot_water = bool(cfg.dishwasher_hot_water)

    appl_sum: Dict[str, np.ndarray] = {}
    iters = max(1, int(cfg.iterations))
    for i in range(iters):
        _tick(int(5 + 80 * i / iters), f"Simulation {i + 1}/{iters}…")
        sim.simulate(N, cfg.period_type, enabled_appliances=ea)
        appl = sim.get_appliance_aggregated_results(
            interval_seconds=ag, month=cfg.month, enabled_appliances=ea)
        for k, v in appl.items():
            appl_sum[k] = appl_sum.get(k, np.zeros(n)) + np.asarray(v, dtype=float)

    appliances = {k: v / iters for k, v in appl_sum.items()}

    # Efficiency scenario: scaling of individual appliances.
    if cfg.efficiency:
        appliances = efficiency.apply(appliances)

    # Seasonal localization: multiplicative envelope by month.
    f = localization.factor(cfg.month) if cfg.localize else 1.0
    if f != 1.0:
        appliances = {k: v * f for k, v in appliances.items()}

    total = np.sum(list(appliances.values()), axis=0) if appliances else np.zeros(n)
    groups = _groups_from_appliances(appliances)

    # Reactive power: per-appliance bottom-up sum Q(t) = sum_i P_i(t) * tan(phi_i)
    # with each appliance's measured cos(phi) and sign (leading = capacitive,
    # lagging = inductive). See sd.POWER_FACTORS / sd.APPLIANCE_TANPHI and the
    # citation to Hannagan et al. (2023, Sustainability 15(1):158) in
    # household_simulation.py.
    reactive = None
    reactive_by_appliance: Dict[str, np.ndarray] = {}
    if cfg.reactive:
        reactive = np.zeros(n, dtype=float)
        for key, tanphi in sd.APPLIANCE_TANPHI.items():
            if key in appliances:
                q_i = appliances[key] * tanphi
                reactive = reactive + q_i
                reactive_by_appliance[key] = q_i

    # Photovoltaics.
    pv_gen = net = None
    if cfg.pv:
        _tick(90, "PV generation (PVGIS/pvlib)…")
        import pv as pv_mod
        pv_gen = pv_mod.simulate_month(cfg.month, n, cfg.pv_params)
        net = total - pv_gen

    _tick(100, "Done")
    t = np.arange(n) * ag / 3600.0
    return {
        "t": t,
        "total": total,
        "groups": groups,
        "appliances": appliances,
        "reactive": reactive,
        "reactive_appliances": reactive_by_appliance,
        "pv": pv_gen,
        "net": net,
        "meta": {
            "month": cfg.month, "n_households": N, "interval_seconds": ag,
            "period_type": cfg.period_type, "iterations": iters, "equip_class": cfg.equip_class,
            "localize": cfg.localize, "localize_factor": f,
            "efficiency": cfg.efficiency, "reactive": cfg.reactive, "pv": cfg.pv,
        },
    }
