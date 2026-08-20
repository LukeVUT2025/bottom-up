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

# Supported output resolutions (s).
INTERVALS = {"1 s": 1, "1 min": 60, "10 min": 600, "15 min": 900}
DAY_TYPES = {"Weekday": 1, "Weekend": 2}

# Simulation horizons. `day` is a single day of the chosen type; the others
# concatenate multiple days, respecting the weekly (5 weekdays + 2 weekends)
# and monthly (localisation-factor) cycles. See `run()` below.
HORIZONS = {"Day": "day", "Week": "week", "Month": "month", "Year": "year"}
_HORIZON_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}

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
    period_type: int = 1              # 1 = weekday, 2 = weekend (only used when horizon = "day")
    horizon: str = "day"              # "day", "week", "month" or "year"; see HORIZONS
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


# Non-leap year: days in each month, starting January.
_MONTH_LENGTHS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def _day_schedule(cfg: "RunConfig"):
    """Return a list of (month, period_type) pairs, one per simulated day.

    Weekly cycle starts on Monday (day-of-week 0..4 -> weekday, 5..6 -> weekend).
    Year mode uses a non-leap calendar starting Jan 1 = Monday.
    """
    horizon = cfg.horizon
    if horizon == "day":
        return [(int(cfg.month), int(cfg.period_type))]
    if horizon == "week":
        # 5 weekdays then 2 weekends of the chosen month.
        return [(int(cfg.month), 1)] * 5 + [(int(cfg.month), 2)] * 2
    if horizon == "month":
        # Full length of the chosen month with a Mon-start weekly cycle.
        days = _MONTH_LENGTHS[int(cfg.month) - 1]
        return [(int(cfg.month), 2 if (i % 7) >= 5 else 1) for i in range(days)]
    if horizon == "year":
        sched = []
        doy = 0
        for m, ml in enumerate(_MONTH_LENGTHS, start=1):
            for _ in range(ml):
                sched.append((m, 2 if (doy % 7) >= 5 else 1))
                doy += 1
        return sched
    raise ValueError(f"Unknown horizon {horizon!r}; expected one of "
                     f"{list(_HORIZON_DAYS)}.")


def _simulate_one_day(sim, month, day_type, N, ea, ag, n_daily, iters):
    """Run `iters` stochastic sims for a single day and return the averages.

    Returns (day_appl, day_single_cp_appl, day_single_cp_total) where the
    dicts hold per-appliance profiles (length n_daily) for the aggregate and
    for the tracked single household. day_single_cp_total is the sum across
    single-CP appliances (length n_daily).
    """
    day_appl_sum: Dict[str, np.ndarray] = {}
    day_single_appl: Dict[str, np.ndarray] = {}
    for it in range(iters):
        sim.simulate(N, day_type, enabled_appliances=ea)
        appl = sim.get_appliance_aggregated_results(
            interval_seconds=ag, month=month, enabled_appliances=ea)
        for k, v in appl.items():
            day_appl_sum[k] = day_appl_sum.get(k, np.zeros(n_daily)) + np.asarray(v, dtype=float)
        if it == iters - 1:
            day_single_appl = sim.get_single_cp_appliance_results(
                interval_seconds=ag, month=month, enabled_appliances=ea)
    day_appl = {k: v / iters for k, v in day_appl_sum.items()}
    day_single_total = (np.sum(list(day_single_appl.values()), axis=0)
                        if day_single_appl else np.zeros(n_daily))
    return day_appl, day_single_appl, day_single_total


class SimulationCancelled(Exception):
    """Raised by run() when the caller-supplied stop_check returns True."""


def run(cfg: RunConfig,
        progress: Optional[Callable[[int, str], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None) -> dict:
    """Run the simulation according to the configuration and return results.

    Returned dict:
        t            time axis in hours (n points)
        total        aggregated active power (W)
        groups       {group: profile W}
        appliances   {appliance: profile W}
        single_cp    representative single-CP active-power profile (W),
                     one household from the last iteration (raw, pilovity)
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
    n_daily = 86400 // ag
    N = int(cfg.n_households)
    ea = cfg.enabled_appliances
    iters = max(1, int(cfg.iterations))

    schedule = _day_schedule(cfg)
    n_days = len(schedule)
    n = n_daily * n_days

    sim = sd.HouseholdSimulation(period_type=cfg.period_type, data_dir=DATA_DIR)
    # Equipment category: override the Low/Medium/High class shares.
    sim.class_mix = CLASS_MIX.get(cfg.equip_class, (1 / 3, 1 / 3, 1 / 3))
    sim.dishwasher_hot_water = bool(cfg.dishwasher_hot_water)

    # Per-appliance segments, one per calendar day; concatenated afterwards.
    appl_chunks: Dict[str, list] = {}
    single_appl_chunks: Dict[str, list] = {}
    single_cp_chunks: list = []

    total_sims = n_days * iters
    sim_no = 0
    for day_idx, (month, day_type) in enumerate(schedule):
        if stop_check is not None and stop_check():
            raise SimulationCancelled(
                f"Cancelled after day {day_idx}/{n_days}.")
        _tick(int(5 + 80 * sim_no / max(total_sims, 1)),
              f"Day {day_idx + 1}/{n_days} (m={month}, {'weekday' if day_type == 1 else 'weekend'})")
        day_appl, day_single_appl, day_single_total = _simulate_one_day(
            sim, month, day_type, N, ea, ag, n_daily, iters)
        sim_no += iters

        # Apply per-day seasonal localisation so each day carries its own
        # month's envelope (matters for horizon = year, harmless otherwise).
        f_day = localization.factor(month) if cfg.localize else 1.0
        if f_day != 1.0:
            day_appl = {k: v * f_day for k, v in day_appl.items()}
            if day_single_appl:
                day_single_appl = {k: v * f_day for k, v in day_single_appl.items()}
                day_single_total = day_single_total * f_day

        # Fill missing keys with zeros so np.concatenate later works even
        # when a given day never triggered a particular appliance.
        all_keys = set(appl_chunks) | set(day_appl)
        for k in all_keys:
            if k not in day_appl:
                day_appl[k] = np.zeros(n_daily)
            appl_chunks.setdefault(k, []).append(day_appl[k])
        all_single_keys = set(single_appl_chunks) | set(day_single_appl)
        for k in all_single_keys:
            if k not in day_single_appl:
                day_single_appl[k] = np.zeros(n_daily)
            single_appl_chunks.setdefault(k, []).append(day_single_appl[k])
        single_cp_chunks.append(day_single_total)

    # Concatenate to horizon-length arrays. Pad any chunk that was shorter
    # (appliance appeared later) with zeros for earlier days.
    def _concat(chunks_by_key):
        out = {}
        for k, chunks in chunks_by_key.items():
            padded = []
            for c in chunks:
                if c is None:
                    padded.append(np.zeros(n_daily))
                else:
                    padded.append(c)
            # Left-pad if chunks list is shorter than n_days (appliance seen late)
            while len(padded) < n_days:
                padded.insert(0, np.zeros(n_daily))
            out[k] = np.concatenate(padded)
        return out

    appliances = _concat(appl_chunks)
    single_cp_appliances = _concat(single_appl_chunks)
    single_cp = np.concatenate(single_cp_chunks) if single_cp_chunks else None

    # Efficiency scenario: applied after concatenation on the horizon.
    if cfg.efficiency:
        appliances = efficiency.apply(appliances)
        if single_cp_appliances:
            single_cp_appliances = efficiency.apply(single_cp_appliances)
            single_cp = np.sum(list(single_cp_appliances.values()), axis=0)

    # Legacy meta: report the seasonal factor of the chosen `month` for
    # display; per-day factors were already applied above.
    f = localization.factor(cfg.month) if cfg.localize else 1.0

    total = np.sum(list(appliances.values()), axis=0) if appliances else np.zeros(n)
    groups = _groups_from_appliances(appliances)

    # Reactive power: per-appliance bottom-up sum Q(t) = sum_i P_i(t) * tan(phi_i)
    # with each appliance's measured cos(phi) and sign (leading = capacitive,
    # lagging = inductive). See sd.POWER_FACTORS / sd.APPLIANCE_TANPHI and the
    # citation to Hannagan et al. (2023, Sustainability 15(1):158) in
    # household_simulation.py.
    reactive = None
    reactive_by_appliance: Dict[str, np.ndarray] = {}
    reactive_single_cp: Optional[np.ndarray] = None
    if cfg.reactive:
        reactive = np.zeros(n, dtype=float)
        for key, tanphi in sd.APPLIANCE_TANPHI.items():
            if key in appliances:
                q_i = appliances[key] * tanphi
                reactive = reactive + q_i
                reactive_by_appliance[key] = q_i
        # Same bottom-up formula on the single-CP appliance breakdown so
        # the top panel shows a proper spiky Q, not a rescaled aggregate.
        if single_cp_appliances:
            reactive_single_cp = np.zeros(n, dtype=float)
            for key, tanphi in sd.APPLIANCE_TANPHI.items():
                if key in single_cp_appliances:
                    reactive_single_cp = reactive_single_cp + single_cp_appliances[key] * tanphi

    # Photovoltaics: one representative day per month, tiled per calendar day.
    pv_gen = net = None
    if cfg.pv:
        _tick(90, "PV generation (PVGIS/pvlib)…")
        import pv as pv_mod
        pv_chunks = []
        month_cache: Dict[int, np.ndarray] = {}
        for month, _dt in schedule:
            if month not in month_cache:
                month_cache[month] = pv_mod.simulate_month(month, n_daily, cfg.pv_params)
            pv_chunks.append(month_cache[month])
        pv_gen = np.concatenate(pv_chunks) if pv_chunks else np.zeros(n)
        net = total - pv_gen

    _tick(100, "Done")
    t = np.arange(n) * ag / 3600.0
    return {
        "t": t,
        "total": total,
        "groups": groups,
        "appliances": appliances,
        "single_cp": single_cp,
        "single_cp_appliances": single_cp_appliances,
        "reactive": reactive,
        "reactive_single_cp": reactive_single_cp,
        "reactive_appliances": reactive_by_appliance,
        "pv": pv_gen,
        "net": net,
        "meta": {
            "month": cfg.month, "n_households": N, "interval_seconds": ag,
            "period_type": cfg.period_type, "horizon": cfg.horizon,
            "n_days": n_days, "iterations": iters, "equip_class": cfg.equip_class,
            "localize": cfg.localize, "localize_factor": f,
            "efficiency": cfg.efficiency, "reactive": cfg.reactive, "pv": cfg.pv,
        },
    }
