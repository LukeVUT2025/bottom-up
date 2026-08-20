"""Household load simulation engine.

Python port of the original MATLAB occupancy/appliance model. Simulates
per-appliance electricity demand for a set of households from a Markov
occupancy chain and measured appliance power traces.

Requires: numpy, scipy, pandas, openpyxl
Optional: h5py (only if kernely_2.mat is in MATLAB v7.3 / HDF5 format)

Data files are expected in the Data/ subfolder.
"""

import os
import numpy as np
import scipy.io
import scipy.signal
from typing import Optional, Set, Callable, Dict, Tuple, List
import pandas as pd

try:
    from pvlib import irradiance, solarposition
    _PVLIB_SOLAR_AVAILABLE = True
except Exception:
    _PVLIB_SOLAR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Supported appliance keys (used in enabled_appliances sets)
# ---------------------------------------------------------------------------
ALL_APPLIANCES: Set[str] = {
    'food_prep',        # Scheduler for kitchen appliances (required for kettle/toaster/oven)
    'kettle',           # Kettle / water heater
    'toaster',          # Toaster
    'oven',             # Oven
    'hob',              # Hob
    'microwave',        # Microwave oven
    'vacuum',           # Vacuum cleaner
    'washing_machine',  # Washing machine
    'dryer',            # Dryer (heat pump)
    'iron',             # Iron
    'tv',               # Television
    'pc',               # PC / printer
    'hair_dryer',       # Hair dryer
    'dishwasher',       # Dishwasher
    'refrigerator',     # Refrigerator
    'freezer',          # Freezer
    'router',           # Router / network
    'small_appliances', # Small appliances
    'boiler',           # Boiler / DHW heating
    'heating',          # Heating / climate control
    'heating_v2',       # Heating / climate control (v2 thermal model)
    'cooling_v2',       # Cooling / climate control (v2 thermal model)
    'lighting',         # Lighting
}

# ---------------------------------------------------------------------------
# Appliance → group mapping  (used for per-group breakdown output)
# Keys match CSV column prefixes: Group_Kitchen, Group_Cleaning, …
# ---------------------------------------------------------------------------
APPLIANCE_GROUPS = {
    'Kitchen':            {'kettle', 'toaster', 'oven', 'hob', 'microwave'},
    'Cleaning':           {'vacuum', 'washing_machine', 'dishwasher', 'dryer', 'iron'},
    'Entertainment_Work': {'tv', 'pc'},
    'Personal_Care':      {'hair_dryer'},
    'Always_On':          {'refrigerator', 'freezer', 'router', 'small_appliances'},
    'Climate_Control':    {'boiler', 'heating', 'heating_v2', 'cooling_v2'},
    'Lighting':           {'lighting'},
}
# Ordered list used to produce consistent column order
GROUP_KEYS = ['Kitchen', 'Cleaning', 'Entertainment_Work', 'Personal_Care',
              'Always_On', 'Climate_Control', 'Lighting']

# Ordered appliance keys — used for CSV column order.
# boiler / heating / cooling / lighting are computed in results_with_lighting_and_heating(), not in simulate().
APPLIANCE_KEYS = [
    'kettle', 'toaster', 'oven', 'hob', 'microwave',
    'vacuum', 'washing_machine', 'dishwasher', 'dryer', 'iron',
    'tv', 'pc',
    'hair_dryer',
    'refrigerator', 'freezer', 'router', 'small_appliances',
    'boiler', 'heating', 'heating_v2', 'cooling_v2', 'lighting',
]
# Keys tracked directly in simulate() (1D running sums)
_SIM_APPLIANCE_KEYS = [k for k in APPLIANCE_KEYS
                       if k not in ('boiler', 'heating', 'heating_v2', 'cooling_v2', 'lighting')]

# ---------------------------------------------------------------------------
# Monthly average temperatures (°F) - used for the monthly dependence of heating/boiler
# ---------------------------------------------------------------------------
_MONTHLY_TOUT_AVG_F = {
    1: 34, 2: 36, 3: 43, 4: 52, 5: 60, 6: 66,
    7: 69, 8: 68, 9: 62, 10: 52, 11: 42, 12: 36
}
_MONTHLY_TINLET_AVG_F = {
    1: 41, 2: 43, 3: 46, 4: 52, 5: 59, 6: 64,
    7: 68, 8: 68, 9: 63, 10: 55, 11: 47, 12: 43
}

# ---------------------------------------------------------------------------
# Heating v2 - apartment and thermal-envelope parameters (easily editable)
# ---------------------------------------------------------------------------
FLOOR_AREA_M2 = [46, 65, 84, 102, 121, 139, 158]  # legacy fallback by household type

# Living matrix bins (m2 per person):
# 1) 9.9, 2) 10-14.9, 3) 15-19.9, 4) 20-24.9,
# 5) 25-29.9, 6) 30-34.9, 7) 35-39.9, 8) >40
_LIVING_AREA_BINS_M2_PER_PERSON: List[Tuple[float, float]] = [
    (9.9, 9.9),
    (10.0, 14.9),
    (15.0, 19.9),
    (20.0, 24.9),
    (25.0, 29.9),
    (30.0, 34.9),
    (35.0, 39.9),
    (40.0, 60.0),
]

HEATING_V2_CONFIG: Dict[str, float] = {
    'ceiling_height_m': 2.5,
    'window_share_walls': 0.20,
    'r_wall_m2k_w': 3.4,
    'r_roof_m2k_w': 5.5,
    'r_floor_m2k_w': 3.2,
    'r_window_m2k_w': 0.9,
    'ach_1_h': 0.45,
    'solar_gain_window_g': 0.55,
    'solar_gain_opaque': 0.03,
    'heater_max_w_m2': 40.0,
    'min_wall_exposed_fraction': 0.25,
    'min_roof_exposed_fraction': 0.20,
    'min_floor_exposed_fraction': 0.20,
    'default_latitude': 49.193,
    'default_longitude': 16.612,
    'default_outdoor_c': 5.0,
}

# ---------------------------------------------------------------------------
# Cooling v2 - air-conditioning parameters (summer months)
# ---------------------------------------------------------------------------
COOLING_V2_CONFIG: Dict[str, float] = {
    'ceiling_height_m': 2.5,
    'window_share_walls': 0.20,
    'r_wall_m2k_w': 3.4,
    'r_roof_m2k_w': 5.5,
    'r_floor_m2k_w': 3.2,
    'r_window_m2k_w': 0.9,
    'ach_1_h': 0.45,
    'solar_gain_window_g': 0.55,
    'solar_gain_opaque': 0.03,
    'cooler_max_w_m2': 60.0,
    'min_wall_exposed_fraction': 0.25,
    'min_roof_exposed_fraction': 0.20,
    'min_floor_exposed_fraction': 0.20,
    'default_latitude': 49.193,
    'default_longitude': 16.612,
    'default_outdoor_c': 25.0,
}

# ---------------------------------------------------------------------------
# Helper: safe 1-based random integer (replaces MATLAB randi)
# ---------------------------------------------------------------------------
def _randi(lo: int, hi: int) -> int:
    """Uniform integer in [lo, hi] inclusive (MATLAB randi equivalent)."""
    if lo > hi:
        lo, hi = hi, lo
    return int(np.random.randint(lo, hi + 1))


def _randsample_weighted(n: int, k: int, replace: bool, weights: np.ndarray) -> np.ndarray:
    """
    Weighted random sample – replacement for MATLAB randsample(1:n, k, true/false, weights).
    Returns 1-based indices.
    weights must not contain NaN (caller's responsibility).
    """
    weights = np.asarray(weights, dtype=float)
    weights = np.nan_to_num(weights, nan=0.0)   # safety guard
    total = weights.sum()
    if total == 0:
        probs = np.ones(n) / n
    else:
        probs = weights / total
    return np.random.choice(np.arange(1, n + 1), size=k, replace=replace, p=probs)


# ===========================================================================
# HouseholdOccupancy
# ===========================================================================

class HouseholdOccupancy:
    """
    Python port of the original MATLAB household-occupancy model.

    Simulates household occupancy for an entire day using a Markov chain.
    States:
        1 = away from home
        2 = home and active
        3 = home and sleeping
    Time resolution: 144 x 10-minute slots (= 24 hours).
    """

    HOUSEHOLD_SIZE_DISTRIBUTION = [0.4433, 0.2977, 0.1392, 0.0941, 0.0185, 0.0062, 0.001]

    def __init__(self, n_households: int, kind: str, data_dir: str = "Data",
                 start_states: Optional[np.ndarray] = None):
        self.counts = np.zeros(7, dtype=int)
        self.size_distribution = np.zeros(7, dtype=int)
        self.final_states: Optional[np.ndarray] = None

        temp = n_households * np.array(self.HOUSEHOLD_SIZE_DISTRIBUTION)
        for i in range(7):
            self.size_distribution[i] = round(temp[i])
            self.counts[i] = round(temp[i]) * (i + 1)

        if int(np.sum(self.counts)) == 0:
            self.size_distribution[0] = 1
            self.counts[0] = 1

        self.n_people = int(np.sum(self.counts))

        csv_file = "markov_weekday.csv" if kind == "weekday" else "markov_weekend.csv"
        src = os.path.join(data_dir, csv_file)
        self.person_occupancy = self._simulate_occupancy(
            src, self.n_people, start_states=start_states
        )

    @staticmethod
    def _occupancy_person_day(transition_matrix: np.ndarray,
                                        initial_state: int) -> np.ndarray:
        s = np.zeros(145, dtype=int)
        s[0] = initial_state
        for k in range(144):
            U = np.random.random()
            cs = s[k] - 1
            p1  = transition_matrix[cs, 0, k]
            p12 = p1 + transition_matrix[cs, 1, k]
            if U < p1:
                s[k + 1] = 1
            elif U < p12:
                s[k + 1] = 2
            else:
                s[k + 1] = 3
        return s

    def _simulate_occupancy(self, src: str, N_lidi: int,
                           start_states: Optional[np.ndarray] = None) -> np.ndarray:
        x = np.loadtxt(src, delimiter=',')
        matrix = np.zeros((3, 3, 144))
        for k in range(144):
            matrix[:, :, k] = x[:3, k * 3 : (k + 1) * 3]

        weights = np.array([0.09, 0.2, 0.71])
        result = np.zeros((N_lidi, 145), dtype=int)
        final_states = np.zeros(N_lidi, dtype=int)
        if start_states is not None:
            start_states = np.asarray(start_states, dtype=int).reshape(-1)
            if len(start_states) != N_lidi:
                start_states = None
        for i in range(N_lidi):
            if start_states is not None and 1 <= int(start_states[i]) <= 3:
                state0_day1 = int(start_states[i])
            else:
                state0_day1 = np.random.choice([1, 2, 3], p=weights)
            day1 = self._occupancy_person_day(matrix, state0_day1)
            state0_day2 = int(day1[-1])
            result[i, :] = self._occupancy_person_day(matrix, state0_day2)
            final_states[i] = int(result[i, -1])
        self.final_states = final_states
        return result

    def occupancy_person(self, index_1based: int) -> np.ndarray:
        return self.person_occupancy[index_1based - 1, :]

    def household_type(self, index_1based: int) -> int:
        temp = 0
        for i in range(7):
            temp += (i + 1) * int(self.size_distribution[i])
            if temp >= index_1based:
                return i + 1
        return 7


# ===========================================================================
# HouseholdSimulation
# ===========================================================================

# Constant capacitive reactive offset per connection point. Calibrated on
# distribution-feeder measurements; the second-resolution flat measurement
# determines the shape, not the level.
Q0_VAR_PER_CP = -55.9


class HouseholdSimulation:
    """
    Python port of the original MATLAB household-simulation model.
    """

    def __init__(self, period_type: int = 1, data_dir: str = "Data"):
        self.period_type = period_type
        self.data_dir = data_dir
        self._living_area_probs = self._load_living_area_probabilities()
        self._load_all_data(period_type)
        self.model_total_profile: Optional[np.ndarray] = None
        self.lighting_profile: Optional[np.ndarray] = None
        self.model_boiler:     Optional[np.ndarray] = None
        self.model_heating:   Optional[np.ndarray] = None
        self.occupancy: Optional[HouseholdOccupancy] = None
        self.group_profile:    Optional[dict] = None   # derived lazily
        self.appliance_profile: Optional[dict] = None  # 1D sum arrays, set in simulate()
        self._tmy_loader: Optional[Callable] = None
        self._latitude: Optional[float] = None
        self._longitude: Optional[float] = None
        self._heating_v2_day_cache: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
        self._results_ag: Optional[int] = None
        self._num_households_requested: int = 0
        self._household_states_10min: Optional[np.ndarray] = None
        self._household_occupants: List[int] = []
        self._household_floor_area_m2: List[float] = []
        self._household_area_per_person_m2: List[float] = []
        self._household_area_bin_idx: List[int] = []
        self._person_floor_area_m2: Optional[np.ndarray] = None
        self.heating_v2_debug: Optional[dict] = None
        self._heating_schedule_mask_144: Optional[np.ndarray] = None
        self._heating_season_mask: Optional[np.ndarray] = None  # [365] bool array, True if heating active
        self._cooling_season_mask: Optional[np.ndarray] = None  # [365] bool array, True if cooling active

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _p(self, filename: str) -> str:
        return os.path.join(self.data_dir, filename)

    def _load_csv(self, filename: str) -> np.ndarray:
        return np.loadtxt(self._p(filename), delimiter=',')

    def _load_living_area_probabilities(self) -> np.ndarray:
        """
        Load Living.csv matrix [8 x 6] with probabilities of m2/person bins
        by household occupants (1,2,3,4,5,6+).
        """
        path = self._p("Living.csv")
        fallback = np.ones((8, 6), dtype=float) / 8.0
        try:
            df = pd.read_csv(path, sep=';', header=None, decimal=',', engine='python')
            arr = df.to_numpy(dtype=float)
            if arr.shape != (8, 6):
                raise ValueError(f"Expected Living.csv shape (8, 6), got {arr.shape}")

            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            col_sum = np.sum(arr, axis=0)
            for col in range(arr.shape[1]):
                if col_sum[col] > 0:
                    arr[:, col] = arr[:, col] / col_sum[col]
                else:
                    arr[:, col] = 1.0 / arr.shape[0]
            return arr
        except Exception as e:
            print(f"WARN: Failed to load Living.csv ({e}). Using uniform floor-area probabilities.")
            return fallback

    def _sample_household_floor_area(self, occupants: int) -> Tuple[float, float, int]:
        """
        Sample floor area from Living.csv probabilities.

        Returns
        -------
        tuple
            (area_floor_m2, area_per_person_m2, bin_idx_1based)
        """
        occ = max(1, int(occupants))
        col_idx = min(occ, 6) - 1  # 1..5 => 0..4, 6+ => 5
        probs = np.asarray(self._living_area_probs[:, col_idx], dtype=float)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        total = float(np.sum(probs))
        if total <= 0:
            probs = np.ones(8, dtype=float) / 8.0
        else:
            probs = probs / total

        bin_idx = int(np.random.choice(np.arange(8), p=probs))
        lo, hi = _LIVING_AREA_BINS_M2_PER_PERSON[bin_idx]
        area_per_person = float(np.random.uniform(lo, hi))
        area_floor = area_per_person * float(occ)
        return area_floor, area_per_person, bin_idx + 1

    @staticmethod
    def _f_to_c(temp_f: float) -> float:
        """Convert temperature from Fahrenheit to Celsius."""
        return (temp_f - 32.0) * 5.0 / 9.0

    def _calculate_heating_season(self) -> np.ndarray:
        """
        Calculate heating season mask for entire year (365 days).
        
        Rules for the Czech Republic:
        - Heating starts: when average daily outdoor temp falls below +13°C 
          for 2 consecutive days
        - Heating stops: when average daily outdoor temp rises above +13°C 
          for 2 consecutive days
        
        Returns
        -------
        np.ndarray
            Boolean array of shape (365,) where True indicates heating is active on that day
        """
        heating_threshold_c = 13.0
        consecutive_days_required = 2
        
        # Generate daily outdoor temperatures for the entire year
        # Using monthly averages with smooth interpolation and daily variation
        daily_temps_c = np.zeros(365)
        
        # Days per month (non-leap year)
        days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        
        day_idx = 0
        for month_idx, days_in_month in enumerate(days_per_month):
            month_num = month_idx + 1
            # Get this month and next month average temps
            month_temp_f = _MONTHLY_TOUT_AVG_F.get(month_num, 50.0)
            next_month_num = month_num + 1 if month_num < 12 else 1
            next_month_temp_f = _MONTHLY_TOUT_AVG_F.get(next_month_num, 50.0)
            
            month_temp_c = self._f_to_c(month_temp_f)
            next_month_temp_c = self._f_to_c(next_month_temp_f)
            
            # Generate daily temps for this month with smooth transition
            for day_in_month in range(days_in_month):
                # Linear interpolation within month
                t = day_in_month / days_in_month
                interpolated_temp = month_temp_c * (1.0 - t) + next_month_temp_c * t
                
                # Add daily variation (±2°C from average)
                daily_variation = np.random.uniform(-2.0, 2.0)
                daily_temps_c[day_idx] = interpolated_temp + daily_variation
                day_idx += 1
        
        # Apply heating season rules
        heating_mask = np.zeros(365, dtype=bool)
        heating_active = False
        consecutive_below = 0
        consecutive_above = 0
        
        for day_idx in range(365):
            temp = daily_temps_c[day_idx]
            
            if not heating_active:
                # Waiting for heating to start
                if temp < heating_threshold_c:
                    consecutive_below += 1
                    consecutive_above = 0
                else:
                    consecutive_below = 0
                    consecutive_above += 1
                
                if consecutive_below >= consecutive_days_required:
                    heating_active = True
                    heating_mask[day_idx] = True
            else:
                # Heating is active, waiting for it to stop
                if temp > heating_threshold_c:
                    consecutive_above += 1
                    consecutive_below = 0
                else:
                    consecutive_above = 0
                    consecutive_below += 1
                
                if consecutive_above >= consecutive_days_required:
                    heating_active = False
                else:
                    heating_mask[day_idx] = True
        
        return heating_mask

    def _is_heating_season(self, month: int) -> bool:
        """
        Check if a given month is in the heating season (simplified check).
        
        Returns True if ANY day in the month is in heating season.
        For more precise control, use _heating_season_mask directly.
        """
        if self._heating_season_mask is None:
            self._heating_season_mask = self._calculate_heating_season()
        
        # Get day of year for first day of this month
        days_per_month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
        month_start_doy = days_per_month[month - 1]
        month_end_doy = days_per_month[month] if month < 12 else 365
        
        # Check if any day in the month is in heating season
        return np.any(self._heating_season_mask[month_start_doy:month_end_doy])

    def _calculate_cooling_season(self) -> np.ndarray:
        """
        Calculate cooling season mask for entire year (365 days).
        
        Rules for the Czech Republic:
        - Cooling starts: when average daily outdoor temp rises above +21°C 
          for 2 consecutive days
        - Cooling stops: when average daily outdoor temp falls below +21°C 
          for 2 consecutive days
        - Approximately May-September in central Europe
        
        Returns
        -------
        np.ndarray
            Boolean array of shape (365,) where True indicates cooling is active on that day
        """
        cooling_threshold_c = 21.0
        consecutive_days_required = 2
        
        # Generate daily outdoor temperatures for the entire year
        daily_temps_c = np.zeros(365)
        
        # Days per month (non-leap year)
        days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        
        day_idx = 0
        for month_idx, days_in_month in enumerate(days_per_month):
            month_num = month_idx + 1
            # Get this month and next month average temps
            month_temp_f = _MONTHLY_TOUT_AVG_F.get(month_num, 50.0)
            next_month_num = month_num + 1 if month_num < 12 else 1
            next_month_temp_f = _MONTHLY_TOUT_AVG_F.get(next_month_num, 50.0)
            
            month_temp_c = self._f_to_c(month_temp_f)
            next_month_temp_c = self._f_to_c(next_month_temp_f)
            
            # Generate daily temps for this month with smooth transition
            for day_in_month in range(days_in_month):
                # Linear interpolation within month
                t = day_in_month / days_in_month
                interpolated_temp = month_temp_c * (1.0 - t) + next_month_temp_c * t
                
                # Add daily variation (±2°C from average)
                daily_variation = np.random.uniform(-2.0, 2.0)
                daily_temps_c[day_idx] = interpolated_temp + daily_variation
                day_idx += 1
        
        # Apply cooling season rules
        cooling_mask = np.zeros(365, dtype=bool)
        cooling_active = False
        consecutive_above = 0
        consecutive_below = 0
        
        for day_idx in range(365):
            temp = daily_temps_c[day_idx]
            
            if not cooling_active:
                # Waiting for cooling to start
                if temp > cooling_threshold_c:
                    consecutive_above += 1
                    consecutive_below = 0
                else:
                    consecutive_above = 0
                    consecutive_below += 1
                
                if consecutive_above >= consecutive_days_required:
                    cooling_active = True
                    cooling_mask[day_idx] = True
            else:
                # Cooling is active, waiting for it to stop
                if temp < cooling_threshold_c:
                    consecutive_below += 1
                    consecutive_above = 0
                else:
                    consecutive_below = 0
                    consecutive_above += 1
                
                if consecutive_below >= consecutive_days_required:
                    cooling_active = False
                else:
                    cooling_mask[day_idx] = True
        
        return cooling_mask

    def _is_cooling_season(self, month: int) -> bool:
        """
        Check if a given month is in the cooling season (simplified check).
        
        Returns True if ANY day in the month is in cooling season.
        For more precise control, use _cooling_season_mask directly.
        """
        if self._cooling_season_mask is None:
            self._cooling_season_mask = self._calculate_cooling_season()
        
        # Get day of year for first day of this month
        days_per_month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
        month_start_doy = days_per_month[month - 1]
        month_end_doy = days_per_month[month] if month < 12 else 365
        
        # Check if any day in the month is in cooling season
        return np.any(self._cooling_season_mask[month_start_doy:month_end_doy])

    def _load_all_data(self, period_type: int):

        npz_file = "weekday.npz" if period_type == 1 else "weekend.npz"
        try:
            X = np.load(self._p(npz_file))['data'].astype(float)
            X = np.nan_to_num(X, nan=0.0)

            if X.shape != (7, 9, 144):
                raise ValueError(f"Unexpected shape {X.shape}, expected (7, 9, 144).")

            n_app = X.shape[1]
            def _col(c1):
                return min(c1 - 1, n_app - 1)

            self.food_prep_data = X[:, _col(1), :]
            self.vacuum_data  = X[:, _col(2), :]
            self.washing_machine_data   = X[:, _col(3), :]
            self.ironing_data  = X[:, _col(4), :]
            self.tv_data       = X[:, _col(5), :]
            self.pc_data       = X[:, _col(7), :]
            self.hair_dryer_data      = X[:, _col(8), :]
            self.dishwasher_data    = X[:, _col(9), :]

            if period_type == 1:
                eve = self.tv_data[:, 96:120].mean()
                mid = self.tv_data[:, 54:90].mean()
                if np.isfinite(eve) and np.isfinite(mid) and eve < mid * 0.8:
                    print("  WARNING: X time axis appears transposed; applying np.flip.")
                    for attr in ["food_prep_data","vacuum_data","washing_machine_data",
                                 "ironing_data","tv_data","pc_data","hair_dryer_data","dishwasher_data"]:
                        setattr(self, attr, np.flip(getattr(self, attr), axis=1).copy())

        except Exception as e:
            raise RuntimeError(
                f"Cannot load {npz_file}: {e}\n"
                "Make sure the Data/ folder contains weekday.npz / weekend.npz."
            )

        try:
            import pandas as pd
            self.light_levels_by_month = pd.read_excel(
                self._p("illuminance.xlsx"), header=None
            ).values.astype(float)
        except Exception:
            self.light_levels_by_month = np.ones((96, 13)) * 300

        _fallback_mean = np.array([
            [3.5, 0.15, 0.12, 0.12, 1.8, 0, 1.5, 0.20, 0.12],
            [3.5, 0.15, 0.12, 0.12, 1.8, 0, 1.5, 0.20, 0.12],
            [4.0, 0.15, 0.18, 0.12, 2.2, 0, 1.8, 0.22, 0.18],
            [4.0, 0.15, 0.18, 0.12, 2.5, 0, 2.0, 0.25, 0.18],
            [4.5, 0.20, 0.20, 0.18, 2.8, 0, 2.2, 0.28, 0.20],
            [5.0, 0.20, 0.20, 0.18, 3.0, 0, 2.5, 0.30, 0.20],
            [5.5, 0.22, 0.28, 0.20, 3.2, 0, 2.8, 0.35, 0.28],
        ], dtype=float)
        _fallback_var = _fallback_mean * 0.3 + 0.1

        def _load_mean_switchons(fallback_7x9):
            try:
                arr = np.loadtxt(self._p("mean_switchons.csv"), delimiter=',').astype(float)
                n_rows, n_cols = arr.shape
                mu_col  = np.zeros(n_cols)
                std_col = np.zeros(n_cols)
                for c in range(n_cols):
                    # Average over ALL time-use-survey respondents, not only
                    # the non-zero ones. A zero record means that person did not
                    # use the appliance that day -- information that must be
                    # kept. Averaging only over users while assigning the
                    # appliance to everyone (penetration 1.0) would
                    # systematically overestimate consumption.
                    column = arr[:, c]
                    if np.any(column > 0):
                        mu_col[c]  = column.mean()
                        std_col[c] = max(column.std(), 0.1)
                    else:
                        mu_col[c]  = fallback_7x9[0, c] if c < fallback_7x9.shape[1] else 0.0
                        std_col[c] = 0.1
                # Restore the dependence on household size. A flat
                # `np.tile(..., (7, 1))` would make all seven rows identical,
                # so a single-person household would cook, wash and iron as
                # often as a five-person one; the `_fallback_mean` table above
                # varies with household size, confirming this is wrong.
                #
                # The tables are derived from the UK Time-Use Survey; values
                # are PER PERSON, matching the fact that simulate() iterates
                # over persons. Per person the number of switch-ons falls with
                # household size (in a larger household one person cooks for
                # all), while the household total grows sub-linearly.
                mu_7x9 = np.tile(mu_col, (7, 1))
                std_7x9 = np.tile(std_col, (7, 1))
                try:
                    _mu = np.loadtxt(self._p("mean_switchons_by_size.csv"),
                                     delimiter=',')
                    _sd = np.loadtxt(self._p("var_switchons_by_size.csv"),
                                     delimiter=',')
                    if _mu.shape == (7, n_cols) and _sd.shape == (7, n_cols):
                        mu_7x9, std_7x9 = _mu, np.maximum(_sd, 0.1)
                    else:
                        print("  WARN: mean_switchons_by_size.csv has an "
                              "unexpected shape; using a flat average.")
                except Exception:
                    print("  INFO: mean_switchons_by_size.csv not found; "
                          "switch-on count does not depend on household size.")
                return mu_7x9, std_7x9
            except Exception as err:
                print(f"  WARN: Cannot load mean_switchons.csv: {err}. Using fallback.")
                return fallback_7x9.copy(), (fallback_7x9 * 0.3 + 0.1)

        mean_9col, var_9col = _load_mean_switchons(_fallback_mean)
        # the original variance table was empty (all zeros) - variance is computed from mean_switchons.csv
        var_9col_file = var_9col.copy()

        def _pad_to_13(arr_7x9):
            out = np.ones((7, 13)) * 0.1
            r = min(arr_7x9.shape[0], 7)
            c = min(arr_7x9.shape[1], 13)
            out[:r, :c] = arr_7x9[:r, :c]
            return out

        # ---- single-phase connection points ----
        self.share_single_phase = 0.0      # share of 1-phase connections (0 = all 3-phase)
        self._single_phase_persons = None
        self._current_single_phase = False

        # ---- consumption classes (Low / Medium / High) ----
        # Appliance ownership is assigned PER HOUSEHOLD by class, not by an
        # independent draw for each appliance. This produces a realistic
        # correlation: a Low-class household has neither a dishwasher nor a
        # boiler.
        self.class_penetration = None      # DataFrame 3 x n_appliances
        self.class_mix = (1/3, 1/3, 1/3)  # Low / Medium / High shares
        self.dishwasher_hot_water = False    # False = cold water (default), True = hot-water supply
        self._class_persons = None
        try:
            _pt = pd.read_csv(self._p("penetration_by_class.csv"), index_col=0)
            if list(_pt.index) == ["low", "medium", "high"]:
                self.class_penetration = _pt
        except Exception:
            pass

        # ---- household-type axis ----
        # Besides household size we distinguish three types by member age
        # (pensioners 65+, family with a child <18, other), derived from the
        # UK Time-Use Survey. If the files are missing, the model behaves as
        # if this axis were absent.
        self.mean_switch_ons_by_type = None       # shape (3, 7, 13)
        self.p_type_by_size = None  # shape (7, 3)
        self._type_persons = None           # type assignment for individual persons
        self._current_type = None        # type of the household being processed
        try:
            _sp = np.loadtxt(self._p("mean_switchons_type_size.csv"),
                             delimiter=',').reshape(3, 7, 9)
            _pt = np.loadtxt(self._p("p_type_by_size.csv"), delimiter=',')
            if _pt.shape == (7, 3):
                self.mean_switch_ons_by_type = np.stack(
                    [_pad_to_13(_sp[i]) for i in range(3)])
                self.p_type_by_size = _pt / _pt.sum(axis=1, keepdims=True)
        except Exception:
            pass

        self.mean_switch_ons  = _pad_to_13(mean_9col)
        self.var_switch_ons = _pad_to_13(var_9col_file)

        _default_gamma = {
            0: (2.0, 1.0,  8),
            1: (2.5, 1.2, 15),
            2: (1.0, 1.0,  1),
            3: (2.0, 1.5, 18),
            4: (3.0, 2.0, 24),
            5: (1.0, 1.0,  1),
            6: (3.0, 2.5, 36),
            7: (2.0, 0.4,  3),
            8: (1.0, 1.0,  1),
        }
        self._gamma_k     = np.ones((7, 13))
        self._gamma_theta = np.ones((7, 13))
        self._gamma_max   = np.full((7, 13), 30.0)
        for j, (k, th, mx) in _default_gamma.items():
            self._gamma_k[:, j]     = k
            self._gamma_theta[:, j] = th
            self._gamma_max[:, j]   = mx

        _kernels_loaded = False

        try:
            import h5py
            with h5py.File(self._p("kernely_2.mat"), "r") as f:
                top_keys = list(f.keys())
                arr_key = "aaa" if "aaa" in top_keys else next(
                    (k for k in top_keys if not k.startswith("#")), None
                )
                if arr_key is None:
                    raise KeyError("No usable key in kernely_2.mat")

                aaa_ds = f[arr_key]
                rows, cols = aaa_ds.shape
                print(f"  INFO: kernely_2.mat (HDF5) '{arr_key}' shape={rows}x{cols}")

                for i in range(min(rows, 7)):
                    for j in range(min(cols, 13)):
                        try:
                            ref  = aaa_ds[i, j]
                            kern = f[ref]
                            bw   = None
                            for field in ["BandWidth","bandwidth","bw",
                                          "sigma","mu","scale","Sigma"]:
                                if field in kern.keys():
                                    val = float(np.array(kern[field]).flat[0])
                                    if val > 0:
                                        bw = val
                                        break
                            if bw is not None:
                                bw_slots = bw / 600.0
                                self._gamma_k[i, j]     = 2.0
                                self._gamma_theta[i, j] = bw_slots / 2.0
                                self._gamma_max[i, j]   = min(bw_slots * 5.0, 60.0)
                        except Exception:
                            pass

            _kernels_loaded = True
            print("  INFO: kernely_2.mat loaded via h5py.")

        except ImportError:
            print("  WARN: h5py not installed. "
                  "Run: pip install h5py --break-system-packages")
        except Exception as e:
            pass

        if not _kernels_loaded:
            try:
                km = scipy.io.loadmat(self._p("kernely_2.mat"),
                                      struct_as_record=False, squeeze_me=True)
                if "aaa" in km:
                    aaa = km["aaa"]
                    for i in range(min(aaa.shape[0], 7)):
                        for j in range(min(aaa.shape[1], 13)):
                            try:
                                obj = aaa[i, j]
                                for fname in ["BandWidth","sigma","mu","scale","bandwidth"]:
                                    if hasattr(obj, fname):
                                        val = float(getattr(obj, fname))
                                        if val > 0:
                                            bw_s = val / 600.0
                                            self._gamma_k[i, j]     = 2.0
                                            self._gamma_theta[i, j] = bw_s / 2.0
                                            self._gamma_max[i, j]   = min(bw_s * 5, 60)
                                            break
                            except Exception:
                                pass
                    _kernels_loaded = True
                    print("  INFO: kernely_2.mat loaded via scipy.io.")
            except Exception:
                pass

        try:
            kp_path = self._p("kernel_params.csv")
            if os.path.exists(kp_path):
                import pandas as pd
                kp = pd.read_csv(kp_path, index_col=0)
                for label, row in kp.iterrows():
                    parts = str(label).replace("dom","").replace("spot","").split("_")
                    if len(parts) == 2:
                        i_dom  = int(parts[0]) - 1
                        j_spot = int(parts[1]) - 1
                        if 0 <= i_dom < 7 and 0 <= j_spot < 13:
                            mu  = float(row.get("mean_10min", 0))
                            std = float(row.get("std_10min",  0))
                            p95 = float(row.get("p95", mu * 3))
                            if mu > 0.05 and std > 0:
                                k_g  = max(0.5, (mu / std) ** 2)
                                th_g = std ** 2 / mu
                                self._gamma_k[i_dom, j_spot]    = k_g
                                self._gamma_theta[i_dom, j_spot] = th_g
                                self._gamma_max[i_dom, j_spot]  = max(p95 * 1.5, 3.0)
                print("  INFO: kernel_params.csv applied.")
        except Exception:
            pass

        def lcsv(name):
            try:
                return self._load_csv(name).flatten()
            except Exception:
                return np.array([0.0])

        self.kettle_profile_1         = lcsv("WaterHeaterPower.csv")
        self.kettle_profile_2         = lcsv("WaterHeaterPower05Lt.csv")
        self.kettle_profile_3         = lcsv("WaterHeaterPower1.5Lt.csv")
        self.toaster_profile         = lcsv("ToastMachinePower.csv")
        self.oven_profile            = lcsv("Oven180C.csv")
        # Czech measured traces, used when available
        def _load_meas_csv(rel_path, fallback):
            try:
                a = np.loadtxt(self._p(rel_path))
                return a if a.size > 10 else fallback
            except Exception:
                return fallback
        self.oven_profile   = _load_meas_csv("measurements_cz/oven.csv",
                                        self.oven_profile)
        self.kettle_profile_3 = _load_meas_csv("measurements_cz/kettle.csv",
                                         self.kettle_profile_3)
        self.hob_profile_1zone = _load_meas_csv(
            "measurements_cz/hob_1zone.csv", np.zeros(1))
        self.hob_profile_max = _load_meas_csv(
            "measurements_cz/hob_max.csv", np.zeros(1))
        self.microwave_profile = _load_meas_csv(
            "measurements_cz/microwave.csv", np.zeros(1))
        self.tv_profile                = lcsv("TVPower.csv")
        self.pc_profile                = lcsv("PC_Power.csv")
        self.iron_profile          = lcsv("IronPower.csv")
        self.hair_dryer_profile               = lcsv("HairDryer.csv")
        # 40 °C = Czech measured cycle (majority); eco 30 °C is the minority
        _pr40 = _load_meas_csv("measurements_cz/washing_machine.csv",
                         lcsv("WashingMachinePower_40CMix.csv"))
        self.washing_machine_profile_1          = _pr40
        self.washing_machine_profile_2          = _pr40
        self.washing_machine_profile_3          = lcsv("WashingMachinePower_30C.csv")
        self.printer_profile          = lcsv("PrinterPower.csv")
        # Dishwasher in two variants (selected by self.dishwasher_hot_water):
        #  - COLD WATER (default, majority of households): the dishwasher heats
        #    its own water. Two Issi & Kaplan 2018 cycles (normal ~326 Wh,
        #    65 °C ~984 Wh), mixing to ~819 Wh/cycle, matching the usual
        #    700-1200 Wh and reproducing the paper's decomposition. CC-BY.
        #  - HOT WATER (option, minority): connected to a DHW supply, no
        #    electric heating -> authors' own measurement, 101 Wh/cycle (Table 2).
        self.dishwasher_profile_cold_1      = lcsv("DishWasherPower.csv")
        self.dishwasher_profile_cold_2      = lcsv("DishWasherPower_Power65C.csv")
        self.dishwasher_profile_hot         = _load_meas_csv("measurements_cz/dishwasher.csv",
                                                 self.dishwasher_profile_cold_1)
        # Heat-pump tumble dryer -- authors' own measurement
        # (872 Wh/cycle, ~700 W, see Table 2). No fallback: if missing, the
        # trace stays empty and the dryer contributes nothing.
        self.dryer_profile           = _load_meas_csv("measurements_cz/dryer_hp.csv",
                                                 np.zeros(1))
        self.fridge_profile           = lcsv("RefrigeratorPower.csv")
        self.vacuum_profile           = lcsv("vacuum.csv")
        # Router modelled as a constant always-on base load, merged with the
        # small-appliances constant (both are ~5 W, indistinguishable in a
        # whole-flat measurement). Previously loaded from a tracebase trace
        # (router.csv); replaced by a constant equal to that trace's mean so
        # model output is unchanged, and the tracebase dependency is removed.
        # See DATA_SOURCES.md / Data/README.md.
        self.router_profile            = 5.4 * np.ones(86400)
        self.small_appliances_profile = 5 * np.ones(86400)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _session_duration(self, size_idx0: int, appliance_idx0: int,
                   size: int) -> np.ndarray:
        """Gamma-distributed session duration with hard cap. Returns seconds."""
        j = min(appliance_idx0, self._gamma_k.shape[1] - 1)
        i = min(size_idx0,        self._gamma_k.shape[0] - 1)
        k_shape = float(self._gamma_k[i, j])
        theta   = float(self._gamma_theta[i, j])
        max_s   = float(self._gamma_max[i, j])
        if k_shape <= 0 or theta <= 0:
            k_shape, theta = 2.0, 1.0
        raw = np.random.gamma(shape=k_shape, scale=theta, size=size)
        return 600.0 * np.clip(raw, 0.1, max_s)

    def _switch_on_count(self, appliance_idx0: int, size_idx0: int) -> int:
        # If the household has an assigned type, use the rate for cell
        # (type, size); otherwise use the rate for size only.
        if (self.mean_switch_ons_by_type is not None
                and getattr(self, "_current_type", None) is not None):
            mu = float(self.mean_switch_ons_by_type[self._current_type,
                                           size_idx0, appliance_idx0])
            sigma = float(self.var_switch_ons[size_idx0, appliance_idx0])
            return max(0, round(np.random.normal(mu, max(sigma, 1e-6))))
        mu    = float(self.mean_switch_ons[size_idx0, appliance_idx0])
        sigma = float(self.var_switch_ons[size_idx0, appliance_idx0])
        return max(0, round(np.random.normal(mu, max(sigma, 1e-6))))

    def _extract_profile(self, duration: float, device_kind: str) -> np.ndarray:
        """Extract / tile a measured power profile to match requested duration."""
        dt = max(1, round(duration))

        if device_kind == "tv":
            src = self.tv_profile
            n = len(src)
            ends = [29339, 37039, 39399, 53999, 55499]
            sample_idx = (list(range(28299, 29340)) +
                     list(range(35679, 37040)) +
                     list(range(38799, 39400)) +
                     list(range(42099, 54000)) +
                     list(range(55139, 55500)))
            sample_idx = [idx for idx in sample_idx if idx < n]
            ends = [idx for idx in ends if idx < n]
            n_odber = len(sample_idx)
            if n_odber == 0:
                return np.zeros(dt)
            start = np.random.randint(0, n_odber)
            if start + dt > n_odber:
                segment = list(src[np.array(sample_idx[start:])])
                while len(segment) < dt:
                    need = dt - len(segment)
                    if need >= n_odber:
                        segment += list(src[np.array(sample_idx)])
                    else:
                        segment += list(src[np.array(sample_idx[:need])])
                profile = np.array(segment[:dt])
            else:
                profile = src[np.array(sample_idx[start : start + dt])]
            b1 = 900
            if ends:
                rk = ends[np.random.randint(0, len(ends))]
                t2 = src[rk : min(rk + b1 + 1, n)]
            else:
                t2 = np.array([])
            return np.concatenate([profile, t2])

        elif device_kind == "pc":
            src = self.pc_profile
            z, e = 799, min(6119, len(src) - 1)
            start = np.random.randint(z, e + 1)
            if start + dt > e:
                segment = list(src[start:e])
                while len(segment) < dt:
                    need = dt - len(segment)
                    chunk = e - z
                    segment += list(src[z : z + (need if need < chunk else chunk)])
                return np.array(segment[:dt])
            return src[start : start + dt]

        elif device_kind == "iron":
            src = self.iron_profile
            selection = list(range(109, min(2070, len(src)))) + \
                    list(range(2194, len(src)))
            n = len(selection)
            if n == 0:
                return np.zeros(dt)
            start = selection[np.random.randint(0, n)]
            end_s = min(start + dt, len(src))
            segment = list(src[start:end_s])
            while len(segment) < dt:
                need = dt - len(segment)
                segment += list(src[109 : 109 + need])
            return np.array(segment[:dt])

        elif device_kind == "hair_dryer":
            src = self.hair_dryer_profile
            z, e = 14, min(57, len(src) - 1)
            start = np.random.randint(z, e + 1)
            end_s = min(start + dt, len(src))
            segment = list(src[start:end_s])
            while len(segment) < dt:
                need = dt - len(segment)
                segment += list(src[z : z + need])
            return np.array(segment[:dt])

        elif device_kind == "vacuum":
            src = self.vacuum_profile
            z, e = 24, min(1729, len(src) - 1)
            start = np.random.randint(z, e + 1)
            end_s = min(start + dt, len(src))
            segment = list(src[start:end_s])
            while len(segment) < dt:
                need = dt - len(segment)
                segment += list(src[z : z + need])
            middle = np.array(segment[:dt])
            return np.concatenate([src[:z + 1], middle, src[e:]])

        elif device_kind == "oven":
            src = self.oven_profile
            if dt <= len(src):
                return src[:dt]
            chunks = [919, 1269, 1629, 1999, 2329, 2599, 2899, 3199, len(src) - 1]
            temp = list(src)
            while len(temp) < dt:
                idx  = np.random.randint(0, 8)
                temp += list(src[chunks[idx] : chunks[idx + 1]])
            return np.array(temp[:dt])

        else:
            return np.zeros(max(1, dt))

    # ------------------------------------------------------------------
    # Appliance simulation methods
    # ------------------------------------------------------------------

    def _food_prep(self, size_idx0: int, occupancy_person: np.ndarray):
        model_count = self._switch_on_count(0, size_idx0)
        prob = np.nan_to_num(
            occupancy_person * self.food_prep_data[size_idx0, :], nan=0.0)
        if prob.sum() == 0 or model_count == 0:
            return 0, np.array([], dtype=int)
        return model_count, _randsample_weighted(144, model_count, True, prob)

    def _kettle(self, model_meal_count, model_meal_starts, size_idx0):
        vol_probs = np.array([[0.55,0.55,0.33,0.33,0.25,0.12,0.12],
                          [0.35,0.35,0.40,0.40,0.35,0.30,0.30],
                          [0.10,0.10,0.27,0.27,0.40,0.58,0.58]])
        if model_meal_count == 0:
            return [], []
        count = np.random.randint(0, model_meal_count + 1)
        if count == 0:
            return [], []
        volumes = np.random.choice([1,2,3], size=count, p=vol_probs[:, size_idx0])
        starts, profiles = [], []
        for i in range(count):
            zp = int(model_meal_starts[i])
            z  = _randi((zp - 1) * 600, zp * 600)
            starts.append(z)
            profiles.append([self.kettle_profile_1,
                             self.kettle_profile_2,
                             self.kettle_profile_3][volumes[i]-1].copy())
        return starts, profiles

    def _toaster(self, model_meal_count, model_meal_starts, size_idx0):
        prob = [0.2,0.2,0.25,0.25,0.33,0.33,0.37][size_idx0]
        if model_meal_count == 0:
            return [], None
        mask = np.random.random(model_meal_count) < prob
        count = int(mask.sum())
        if count == 0:
            return [], None
        starts = []
        for idx in np.where(mask)[0]:
            zp = int(model_meal_starts[idx])
            starts.append(_randi((zp - 1) * 600, zp * 600))
        return starts, self.toaster_profile[:630].copy()

    def _oven(self, model_meal_count, model_meal_starts, size_idx0):
        tp = np.array([[0.45,0.45,0.55,0.55,0.65,0.65,0.75],
                       [0.10,0.10,0.20,0.20,0.25,0.25,0.30],
                       [0.02,0.02,0.07,0.07,0.12,0.12,0.18]])
        count = 0
        rnd = np.random.random(3)
        for i in range(min(model_meal_count, 3)):
            if rnd[i] <= tp[count, size_idx0]:
                count += 1
                if count == 3:
                    break
        if count == 0:
            return [], []
        durations = np.random.randint(1800, 7201, size=count)
        durations = durations[np.random.permutation(count)]
        starts, profiles = [], []
        for i in range(count):
            zp = int(model_meal_starts[i])
            z  = abs(_randi((zp-1)*600, zp*600) + np.random.randint(-3600, 3601))
            starts.append(z)
            profiles.append(self._extract_profile(durations[i], "oven"))
        return starts, profiles

    def _hob(self, model_meal_count, model_meal_starts,
                     size_idx0):
        """Hob.

        Unlike the oven, the hob is used in most meal preparations, so its
        probability of use is markedly higher than the oven's.

        The measured trace (a flat) shows pulse control: the zone switches
        between zero and rated power with a period on the order of seconds.
        Two variants are available -- one zone (1.7 kW) and a large zone (2.5 kW).
        """
        if model_meal_count <= 0 or self.hob_profile_1zone.size <= 10:
            return [], []
        # the hob is used in 60-85 % of preparations depending on household size
        p_use = 0.60 + 0.04 * min(size_idx0, 6)
        starts, profiles = [], []
        for i in range(min(model_meal_count, 4)):
            if np.random.random() > p_use:
                continue
            # A single-phase connection cannot carry the large zone: 2.49 kW
            # together with the 2.8 kW oven exceeds a 25 A breaker.
            large_possible = (not getattr(self, "_current_single_phase", False)
                           and self.hob_profile_max.size > 10)
            source = (self.hob_profile_max
                     if (large_possible and np.random.random() < 0.35)
                     else self.hob_profile_1zone)
            duration_s = int(np.random.randint(600, 3601))   # 10-60 min
            n = len(source)
            if duration_s <= n:
                start = np.random.randint(0, n - duration_s + 1)
                p = source[start:start + duration_s].copy()
            else:
                p = np.tile(source, duration_s // n + 1)[:duration_s].copy()
            zp = int(model_meal_starts[i])
            z = abs(_randi((zp - 1) * 600, zp * 600)
                    + np.random.randint(-1800, 1801))
            starts.append(z)
            profiles.append(p)
        return starts, profiles

    def _microwave(self, model_meal_count, model_meal_starts,
                    size_idx0):
        """Microwave oven.

        Short heat-ups, typically 1-8 minutes. The measured trace shows the
        magnetron cycling (about a 30 s period) at 1.45 kW.
        """
        if model_meal_count <= 0 or self.microwave_profile.size <= 10:
            return [], []
        starts, profiles = [], []
        for i in range(min(model_meal_count, 3)):
            if np.random.random() > 0.45:
                continue
            duration_s = int(np.random.randint(60, 481))     # 1-8 min
            n = len(self.microwave_profile)
            if duration_s <= n:
                p = self.microwave_profile[:duration_s].copy()
            else:
                p = np.tile(self.microwave_profile,
                            duration_s // n + 1)[:duration_s].copy()
            zp = int(model_meal_starts[i])
            z = abs(_randi((zp - 1) * 600, zp * 600)
                    + np.random.randint(-900, 901))
            starts.append(z)
            profiles.append(p)
        return starts, profiles

    def _vacuum(self, size_idx0, occupancy_person):
        model_count = min(self._switch_on_count(1, size_idx0), 1)
        if model_count <= 0:
            return [], []
        prob = np.nan_to_num(
            occupancy_person * self.vacuum_data[size_idx0, :], nan=0.0)
        if prob.sum() == 0:
            return [], []
        start_slots = _randsample_weighted(144, model_count, True, prob)
        durations = self._session_duration(size_idx0, 1, model_count)
        starts, profiles = [], []
        for i in range(model_count):
            zm = min(int(start_slots[i]), 143)
            starts.append(_randi(zm * 600, (zm + 1) * 600))
            profiles.append(self._extract_profile(durations[i], "vacuum"))
        return starts, profiles

    def _washing_machine(self, size_idx0, occupancy_person):
        wash_mode_probs = [0.40, 0.40, 0.20]   # 80 % at 40 °C, 20 % eco 30 °C
        prob = np.nan_to_num(
            occupancy_person * self.washing_machine_data[size_idx0, :], nan=0.0)
        if prob.sum() == 0:
            return 1, np.zeros(1)
        zm = int(_randsample_weighted(144, 1, True, prob)[0])
        z  = _randi((zm - 1) * 600, zm * 600)
        mod = np.random.choice([0,1,2], p=wash_mode_probs)
        return z, [self.washing_machine_profile_1, self.washing_machine_profile_2,
                   self.washing_machine_profile_3][mod].copy()

    def _ironing(self, size_idx0, occupancy_person):
        model_count = self._switch_on_count(3, size_idx0)
        if model_count <= 0:
            return [], []
        prob = np.nan_to_num(
            occupancy_person * self.ironing_data[size_idx0, :], nan=0.0)
        if prob.sum() == 0:
            return [], []
        start_slots = _randsample_weighted(144, model_count, True, prob)
        durations = self._session_duration(size_idx0, 3, model_count)
        starts, profiles = [], []
        for i in range(model_count):
            zp = int(start_slots[i])
            starts.append(_randi((zp - 1) * 600, zp * 600))
            profiles.append(self._extract_profile(durations[i], "iron"))
        return starts, profiles

    def _tv(self, size_idx0, occupancy_person):
        model_count = self._switch_on_count(4, size_idx0)
        if model_count <= 0:
            return [], []
        prob = np.nan_to_num(
            occupancy_person * self.tv_data[size_idx0, :], nan=0.0)
        if prob.sum() == 0:
            return [], []
        start_slots = _randsample_weighted(144, model_count, True, prob)
        durations = self._session_duration(size_idx0, 4, model_count)
        starts, profiles = [], []
        for i in range(model_count):
            zm = min(int(start_slots[i]), 143)
            starts.append(_randi(zm * 600, (zm + 1) * 600))
            profiles.append(self._extract_profile(durations[i], "tv"))
        return starts, profiles

    def _pc(self, size_idx0, occupancy_person):
        model_count = self._switch_on_count(6, size_idx0)
        if model_count <= 0:
            return [], []
        prob = np.nan_to_num(
            occupancy_person * self.pc_data[size_idx0, :], nan=0.0)
        if prob.sum() == 0:
            return [], []
        start_slots = _randsample_weighted(144, model_count, True, prob)
        durations = self._session_duration(size_idx0, 6, model_count)
        starts, profiles = [], []
        for i in range(model_count):
            zm = min(int(start_slots[i]), 143)
            profile = self._extract_profile(durations[i], "pc")
            if np.random.random() < 0.2:
                s2 = self.printer_profile
                temp = s2[:370] if np.random.random() < 0.5 else s2[1235:min(2075,len(s2))]
                if len(temp) > len(profile):
                    t = temp.copy(); t[:len(profile)] += profile; profile = t
                else:
                    profile = profile.copy(); profile[:len(temp)] += temp
            starts.append(_randi(zm * 600, (zm + 1) * 600))
            profiles.append(profile)
        return starts, profiles

    def _hair_dryer(self, size_idx0, occupancy_person):
        # The hair dryer is used at most once a day and only on some days;
        # the 'personal care' activity is not the same as running the dryer.
        P_HAIR_DRYER = 0.30   # estimate
        model_count = (1 if (self._switch_on_count(7, size_idx0) > 0
                             and np.random.random() < P_HAIR_DRYER) else 0)
        if model_count <= 0:
            return [], []
        prob = np.nan_to_num(
            occupancy_person * self.tv_data[size_idx0, :], nan=0.0)
        if prob.sum() == 0:
            return [], []
        start_slots = _randsample_weighted(144, model_count, True, prob)
        durations = self._session_duration(size_idx0, 7, model_count)
        starts, profiles = [], []
        for i in range(model_count):
            zm = min(int(start_slots[i]), 143)
            starts.append(_randi(zm * 600, (zm + 1) * 600))
            profiles.append(self._extract_profile(durations[i], "hair_dryer"))
        return starts, profiles

    def _dishwasher(self, size_idx0, occupancy_person):
        prob = np.nan_to_num(
            occupancy_person * self.dishwasher_data[size_idx0, :], nan=0.0)
        if prob.sum() == 0:
            return 1, np.zeros(1)
        zm = int(_randsample_weighted(144, 1, True, prob)[0])
        z  = _randi((zm - 1) * 600, zm * 600)
        if self.dishwasher_hot_water:
            # DHW connection: a single short cycle with no water heating.
            return z, self.dishwasher_profile_hot.copy()
        # Cold water (default): a mix of the normal and 65 °C cycles.
        mod = np.random.choice([0, 1], p=[0.25, 0.75])
        return z, (self.dishwasher_profile_cold_1 if mod == 0
                   else self.dishwasher_profile_cold_2).copy()


    def _appliance_24h(self, kind: str) -> np.ndarray:
        if kind == "fridge":
            temp  = self.fridge_profile
            n     = len(temp)
            mis   = max(0, 86400 - n)
            temp  = np.concatenate([temp, temp[:mis]])[:86400]
            start = np.random.randint(0, 86399)
            return np.concatenate([temp[start:], temp[:start]])
        elif kind == "router":
            # Constant base load (see __init__); no rotation needed.
            return self.router_profile.copy()
        elif kind == "tv_standby":
            # TV stand-by drawn from measured TVPower.csv low-power region.
            # Samples < 1 W in the CSV represent the stand-by state (~0.07 W).
            # Use the median of those samples as a realistic constant baseline.
            raw = self.tv_profile
            standby_samples = raw[raw < 1.0]
            standby_W = float(np.median(standby_samples)) if len(standby_samples) > 0 else 0.07
            return np.full(86400, standby_W)
        else:
            return self.small_appliances_profile.copy()

    def _freezer(self) -> np.ndarray:
        """Cycling freezer load.

        A freezer is common in Czech flats but is easy to omit; a missing
        freezer is one reason the consumption of cooling appliances looked
        underestimated.

        Parameters:
            rated power  40 / 60 / 80 / 100 W
            cycle half-period  18 / 20 / 24 / 30 / 36 / 40 min
        The duty cycle is 50 %, so the mean power works out to 20-50 W
        (175-438 kWh/year), consistent with energy labels.

        NOTE: this is a rectangular approximation, not a measured trace.
        Replace with a real measurement once available.
        """
        power_w = float(np.random.choice([40.0, 60.0, 80.0, 100.0]))
        half_period = int(np.random.choice([18, 20, 24, 30, 36, 40])) * 60
        period = 2 * half_period
        t = np.arange(86400)
        phase = np.random.randint(0, period)
        return np.where(((t + phase) % period) < half_period, power_w, 0.0)

    def _boiler(self, month: int = 1) -> np.ndarray:
        data = np.loadtxt(self._p('heating.csv'), delimiter=';')
        n = 144
        Ta    = np.random.uniform(66, 72, n)
        Tf    = np.random.uniform(130, 150, n)
        dTw   = np.random.uniform(5, 10, n)
        tinlet_base    = data[:, 1]
        offset         = _MONTHLY_TINLET_AVG_F.get(month, float(np.mean(tinlet_base))) - float(np.mean(tinlet_base))
        Tinlet         = tinlet_base + offset
        Vtank = np.random.uniform(20, 80, n)
        Rtank = np.random.uniform(12, 25, n)
        Pwh   = 2000.0;  nwh = 1.0;  Atank = 30.0  # Pwh in Watts (not kW)
        cwh   = data[:, 2]
        dt    = 10.0
        fr    = data[:, 3]
        Toutlet = np.zeros(n + 1);  Toutlet[0] = 80.0;  Toutlet[1] = 80.0
        wwh = np.zeros(n);  pwh = np.zeros(n)
        for i in range(1, n):
            if   Toutlet[i] > Tf[i]:               wwh[i] = 0.0
            elif Toutlet[i] < Tf[i] - dTw[i]:      wwh[i] = 1.0
            elif (Tf[i] - dTw[i]) <= Toutlet[i] <= Tf[i]: wwh[i] = wwh[i - 1]
            pwh[i] = wwh[i] * Pwh * nwh * cwh[i]
            Toutlet[i + 1] = (
                Toutlet[i] * (Vtank[i] - fr[i] * dt) / Vtank[i]
                + Tinlet[i] * fr[i] * dt / Vtank[i]
                + 1/8.34 * (pwh[i]*3.412 - (Atank*(Toutlet[i]-Ta[i]))/Rtank[i])
                * dt/60 * 1/Vtank[i]
            )
        return pwh/2

    def _heating(self, month: int = 1) -> np.ndarray:
        data = np.loadtxt(self._p('heating.csv'), delimiter=';')
        n = 144
        Tout_base = data[:, 0]
        offset    = _MONTHLY_TOUT_AVG_F.get(month, float(np.mean(Tout_base))) - float(np.mean(Tout_base))
        Tout      = Tout_base + offset
        Ts       = np.random.uniform(66, 72, n)
        dT       = 1.0
        Afloor   = np.random.normal(1700, 500, n)
        Awall    = 3 * Afloor;  Awall[n-1] = Afloor[n-2] - Afloor[0]
        Awindow  = 0.1 * Afloor;  Aceiling = Afloor
        Rwall    = np.random.uniform(13, 15, n)
        Rwindow  = np.random.uniform(0.8, 1.0, n)
        Rceiling = np.random.uniform(38, 60, n)
        Pac      = 2.0;  Chvac = 10000.0
        cac      = data[:, 5];  cw = data[:, 4]
        Vhouse   = 120 * 35.315;  Cair = 0.0195;  dc = Cair * Vhouse
        dt       = 10 / 60
        SHGC     = 0.6;  Hsolar = 1000.0;  Hp = 356.0
        T = np.zeros(n + 1);  T[0] = 68.0;  T[1] = 68.0
        wac = np.zeros(n);  G = np.zeros(n);  pac = np.zeros(n)
        for i in range(1, n):
            if   T[i] < (Ts[i] + cac[i]) - dT:   wac[i] = 0.0
            elif T[i] > (Ts[i] + cac[i]) + dT:   wac[i] = 1.0
            elif (Ts[i]-dT) <= (T[i]-cac[i]) <= (Ts[i]+dT): wac[i] = wac[i-1]
            G[i] = (
                (Awall[i]/Rwall[i] + Aceiling[i]/Rceiling[i]
                + Awindow[i]/Rwindow[i] + 11.77*(1/6)*Vhouse)
                * (Tout[i] - T[i])
                + SHGC * Awindow[i] * Hsolar * 3.412/10.76 + Hp
            )
            T[i+1] = T[i] + dt*G[i]/dc + dt*Chvac/dc*wac[i]
            pac[i] = Pac * wac[i] * cw[i]
        return pac

    def _envelope_geometry(self, size_idx0: int,
                           floor_area_m2: Optional[float] = None) -> Dict[str, float]:
        """Compute apartment geometry from floor area using a square base."""
        if floor_area_m2 is None or floor_area_m2 <= 0:
            idx = int(np.clip(size_idx0, 0, len(FLOOR_AREA_M2) - 1))
            area_floor = float(FLOOR_AREA_M2[idx])
        else:
            area_floor = float(floor_area_m2)
        h = float(HEATING_V2_CONFIG['ceiling_height_m'])
        side = np.sqrt(area_floor)
        area_walls = 4.0 * side * h
        window_share = float(HEATING_V2_CONFIG.get('window_share_walls', 0.20))
        area_win = window_share * area_walls
        area_wall_opaque = max(area_walls - area_win, 0.0)
        volume = area_floor * h
        return {
            'area_floor': area_floor,
            'area_roof': area_floor,
            'area_walls': area_walls,
            'area_window': area_win,
            'area_wall_opaque': area_wall_opaque,
            'volume': volume,
        }

    def _shared_envelope_exposure(self) -> Dict[str, float]:
        """
        Estimate exposed envelope fractions when multiple households are simulated.
        For 1 household: all surfaces exposed. For larger sets: assume apartments can
        be side-by-side and stacked, so only a fraction of walls/roof/floor is external.
        """
        n_hh = max(1, int(self._num_households_requested))
        if n_hh == 1:
            return {'wall': 1.0, 'roof': 1.0, 'floor': 1.0}

        floors = max(1, int(np.ceil(np.sqrt(n_hh))))
        units_per_floor = int(np.ceil(n_hh / floors))

        rows = max(1, int(np.floor(np.sqrt(units_per_floor))))
        cols = max(1, int(np.ceil(units_per_floor / rows)))
        units_on_plan = rows * cols

        if rows == 1 or cols == 1:
            perimeter_units = units_on_plan
        else:
            perimeter_units = 2 * rows + 2 * cols - 4

        wall_frac = perimeter_units / max(units_on_plan, 1)
        roof_frac = 1.0 / floors
        floor_frac = 1.0 / floors

        wall_frac = max(wall_frac, float(HEATING_V2_CONFIG.get('min_wall_exposed_fraction', 0.25)))
        roof_frac = max(roof_frac, float(HEATING_V2_CONFIG.get('min_roof_exposed_fraction', 0.20)))
        floor_frac = max(floor_frac, float(HEATING_V2_CONFIG.get('min_floor_exposed_fraction', 0.20)))

        return {'wall': min(1.0, wall_frac), 'roof': min(1.0, roof_frac), 'floor': min(1.0, floor_frac)}

    def _resample_hourly_to_ag(self, hourly: np.ndarray, ag: int) -> np.ndarray:
        """Resample 24 hourly values to target interval (1/60/600 s)."""
        n_pts = 86400 // ag
        xh = np.arange(24)
        x_dst = np.linspace(0, 23, n_pts)
        return np.interp(x_dst, xh, hourly.astype(float))

    def _load_tmy_day_for_ag(self, month: int, ag: int) -> Dict[str, np.ndarray]:
        """
        Load one random day (1-28) from PVGIS TMY and resample directly to ag.
        Returns weather and solar arrays at target interval.
        """
        cache_key = (int(month), int(ag))
        if cache_key in self._heating_v2_day_cache:
            return self._heating_v2_day_cache[cache_key]

        lat = float(self._latitude if self._latitude is not None else HEATING_V2_CONFIG['default_latitude'])
        lon = float(self._longitude if self._longitude is not None else HEATING_V2_CONFIG['default_longitude'])

        try:
            if self._tmy_loader is None:
                raise RuntimeError("TMY loader is not available.")

            tmy, _meta = self._tmy_loader(lat, lon, tz='Europe/Prague')
            month_df = tmy[tmy.index.month == int(month)].copy()
            if month_df.empty:
                raise RuntimeError("No TMY rows for selected month.")

            candidate_days = pd.DatetimeIndex(month_df.index.normalize().unique())
            candidate_days = candidate_days[candidate_days.day <= 28]
            if len(candidate_days) == 0:
                raise RuntimeError("No TMY days in range 1-28 for selected month.")

            selected_day = pd.Timestamp(np.random.choice(candidate_days))
            day_df = month_df.loc[month_df.index.normalize() == selected_day].copy()
            if day_df.empty:
                raise RuntimeError("Selected TMY day is empty.")

            if 'temp_air' in day_df.columns:
                temp_col = day_df['temp_air']
            elif 'T2m' in day_df.columns:
                temp_col = day_df['T2m']
            else:
                raise RuntimeError("TMY data has no outdoor temperature column (temp_air/T2m).")

            if 'ghi' in day_df.columns:
                ghi_col = day_df['ghi']
            elif 'G(h)' in day_df.columns:
                ghi_col = day_df['G(h)']
            else:
                ghi_col = pd.Series(0.0, index=day_df.index)

            if 'dni' in day_df.columns:
                dni_col = day_df['dni']
            elif 'Gb(n)' in day_df.columns:
                dni_col = day_df['Gb(n)']
            else:
                dni_col = pd.Series(0.0, index=day_df.index)

            if 'dhi' in day_df.columns:
                dhi_col = day_df['dhi']
            elif 'Gd(h)' in day_df.columns:
                dhi_col = day_df['Gd(h)']
            else:
                dhi_col = pd.Series(0.0, index=day_df.index)

            hour_temp = temp_col.groupby(day_df.index.hour).mean().reindex(range(24))
            hour_ghi = ghi_col.groupby(day_df.index.hour).mean().reindex(range(24), fill_value=0.0)
            hour_dni = dni_col.groupby(day_df.index.hour).mean().reindex(range(24), fill_value=0.0)
            hour_dhi = dhi_col.groupby(day_df.index.hour).mean().reindex(range(24), fill_value=0.0)

            # Robust fill if an hour is missing in source data
            hour_temp = hour_temp.interpolate(limit_direction='both').fillna(HEATING_V2_CONFIG['default_outdoor_c'])
            hour_ghi = hour_ghi.interpolate(limit_direction='both').fillna(0.0)
            hour_dni = hour_dni.interpolate(limit_direction='both').fillna(0.0)
            hour_dhi = hour_dhi.interpolate(limit_direction='both').fillna(0.0)

            tout = self._resample_hourly_to_ag(hour_temp.values, ag)
            ghi = np.clip(self._resample_hourly_to_ag(hour_ghi.values, ag), 0.0, None)
            dni = np.clip(self._resample_hourly_to_ag(hour_dni.values, ag), 0.0, None)
            dhi = np.clip(self._resample_hourly_to_ag(hour_dhi.values, ag), 0.0, None)

            if np.all(dni <= 1e-9) and np.any(ghi > 0):
                dni = 0.65 * ghi
            if np.all(dhi <= 1e-9) and np.any(ghi > 0):
                dhi = np.clip(ghi - 0.35 * dni, 0.0, None)

            start_ts = pd.Timestamp(selected_day).tz_convert(tmy.index.tz)
            idx_dst = pd.date_range(start=start_ts, periods=86400 // ag, freq=f'{ag}s', tz=tmy.index.tz)
            if _PVLIB_SOLAR_AVAILABLE:
                solpos = solarposition.get_solarposition(idx_dst, lat, lon)
                zenith = solpos['apparent_zenith'].values.astype(float)
                azimuth = solpos['azimuth'].values.astype(float)
            else:
                zenith = np.full(86400 // ag, 90.0)
                azimuth = np.full(86400 // ag, 180.0)

        except Exception as e:
            # Graceful fallback so consumption simulation still runs even without pvlib/TMY
            print(f"  WARN: Heating v2 TMY load failed ({e}). Using fallback outdoor profile.")
            n_pts = 86400 // ag
            x = np.linspace(0, 24, n_pts, endpoint=False)
            tout = HEATING_V2_CONFIG['default_outdoor_c'] + 4.0 * np.sin((x - 8.0) / 24.0 * 2.0 * np.pi)
            ghi = np.clip(700.0 * np.sin((x - 6.0) / 12.0 * np.pi), 0.0, None)
            dni = 0.65 * ghi
            dhi = 0.35 * ghi
            zenith = np.full(n_pts, 90.0)
            day = (x >= 6) & (x <= 18)
            zenith[day] = 80.0 - 70.0 * np.sin((x[day] - 6.0) / 12.0 * np.pi)
            azimuth = np.full(n_pts, 180.0)
            azimuth[day] = 90.0 + 180.0 * (x[day] - 6.0) / 12.0

        self._heating_v2_day_cache[cache_key] = {
            'tout': np.asarray(tout, dtype=float),
            'ghi': np.asarray(ghi, dtype=float),
            'dni': np.asarray(dni, dtype=float),
            'dhi': np.asarray(dhi, dtype=float),
            'zenith': np.asarray(zenith, dtype=float),
            'azimuth': np.asarray(azimuth, dtype=float),
        }
        return self._heating_v2_day_cache[cache_key]

    def _solar_gain_from_orientation(self, weather: Dict[str, np.ndarray],
                                     area_window: float, area_opaque_solar: float,
                                     enabled: bool = True) -> np.ndarray:
        """Compute solar gains using sun position and oriented surfaces."""
        if not enabled:
            return np.zeros(len(weather.get('ghi', [])), dtype=float)

        g_win = float(HEATING_V2_CONFIG['solar_gain_window_g'])
        g_opaque = float(HEATING_V2_CONFIG['solar_gain_opaque'])

        ghi = weather['ghi']
        dni = weather['dni']
        dhi = weather['dhi']
        zenith = weather['zenith']
        azimuth = weather['azimuth']

        n = len(ghi)
        if n == 0:
            return np.zeros(0)

        azimuths = [0.0, 90.0, 180.0, 270.0]  # N, E, S, W
        area_window_face = area_window / 4.0
        area_opaque_face = area_opaque_solar / 4.0
        q_solar = np.zeros(n, dtype=float)

        if _PVLIB_SOLAR_AVAILABLE:
            for face_az in azimuths:
                poa = irradiance.get_total_irradiance(
                    surface_tilt=90.0,
                    surface_azimuth=face_az,
                    solar_zenith=zenith,
                    solar_azimuth=azimuth,
                    dni=dni,
                    ghi=ghi,
                    dhi=dhi,
                )
                poa_global = np.asarray(poa['poa_global'], dtype=float)
                q_solar += poa_global * (area_window_face * g_win + area_opaque_face * g_opaque)

            poa_roof = irradiance.get_total_irradiance(
                surface_tilt=0.0,
                surface_azimuth=180.0,
                solar_zenith=zenith,
                solar_azimuth=azimuth,
                dni=dni,
                ghi=ghi,
                dhi=dhi,
            )
            poa_roof_global = np.asarray(poa_roof['poa_global'], dtype=float)
            q_solar += poa_roof_global * (area_opaque_solar * g_opaque)
            return np.clip(q_solar, 0.0, None)

        # Fallback without pvlib: split GHI approximately to two most sunlit facades
        sun_from_south = np.cos(np.deg2rad(np.clip(azimuth - 180.0, -180.0, 180.0)))
        south_weight = np.clip(sun_from_south, 0.0, 1.0)
        north_weight = np.clip(-sun_from_south, 0.0, 1.0)
        east_weight = np.clip(np.sin(np.deg2rad(azimuth - 180.0)), 0.0, 1.0)
        west_weight = np.clip(-np.sin(np.deg2rad(azimuth - 180.0)), 0.0, 1.0)
        weight_sum = np.maximum(south_weight + north_weight + east_weight + west_weight, 1e-6)
        wall_gain = ghi * (area_window * g_win + area_opaque_solar * g_opaque)
        q_solar += wall_gain * (south_weight + north_weight + east_weight + west_weight) / weight_sum
        q_solar += ghi * (area_opaque_solar * g_opaque)
        return np.clip(q_solar, 0.0, None)

    def _states_to_interval(self, states_10min: np.ndarray, ag: int) -> np.ndarray:
        """Expand occupancy states from 10-minute slots to target interval."""
        if ag == 600:
            return states_10min.astype(int)
        return np.repeat(states_10min.astype(int), 600 // ag)

    def _heating_v2(self, size_idx0: int, occupancy_144: np.ndarray,
                     month: int = 1, ag: int = 600,
                     return_components: bool = False,
                     include_solar_gains: bool = False,
                     floor_area_m2: Optional[float] = None,
                     heating_schedule_mask_144: Optional[np.ndarray] = None):
        """
        Heating demand model based on envelope losses and solar gains.
        Includes occupant sensible heat gains: 0 W (away), 80 W (active), 70 W (sleeping).
        Output is electric direct-heater power profile [W] in target interval ag.
        """
        weather = self._load_tmy_day_for_ag(month, ag)
        tout = weather['tout']
        geom = self._envelope_geometry(size_idx0, floor_area_m2=floor_area_m2)
        exposure = self._shared_envelope_exposure()

        r_wall = float(HEATING_V2_CONFIG.get('r_wall_m2k_w', HEATING_V2_CONFIG.get('r_opaque_m2k_w', 3.4)))
        r_roof = float(HEATING_V2_CONFIG.get('r_roof_m2k_w', r_wall))
        r_floor = float(HEATING_V2_CONFIG.get('r_floor_m2k_w', r_wall))
        r_window = float(HEATING_V2_CONFIG['r_window_m2k_w'])
        ach = float(HEATING_V2_CONFIG['ach_1_h'])
        pmax = float(HEATING_V2_CONFIG['heater_max_w_m2']) * geom['area_floor']

        area_window_ext = geom['area_window'] * exposure['wall']
        area_wall_opaque_ext = geom['area_wall_opaque'] * exposure['wall']
        area_roof_ext = geom['area_roof'] * exposure['roof']
        area_floor_ext = geom['area_floor'] * exposure['floor']

        ua_trans = (
            area_wall_opaque_ext / max(r_wall, 1e-6)
            + area_roof_ext / max(r_roof, 1e-6)
            + area_floor_ext / max(r_floor, 1e-6)
            + area_window_ext / max(r_window, 1e-6)
        )
        ua_vent = 0.33 * ach * geom['volume']

        state = self._states_to_interval(occupancy_144.astype(int), ag)
        tin = np.where(state == 1, 18.0, np.where(state == 2, 23.0, 21.0))
        dT = np.maximum(tin - tout, 0.0)

        q_loss = (ua_trans + ua_vent) * dT
        q_solar = self._solar_gain_from_orientation(
            weather,
            area_window=area_window_ext,
            area_opaque_solar=area_wall_opaque_ext + area_roof_ext,
            enabled=include_solar_gains,
        )
        # Occupant sensible heat gains (W)
        # state 1 (away): 0 W, state 2 (active): 80 W, state 3 (sleeping): 70 W
        q_occupants = np.where(state == 1, 0.0, np.where(state == 2, 80.0, 70.0))
        
        p_heat = np.clip(q_loss - q_solar - q_occupants, 0.0, pmax)
        schedule_mask_144 = self._heating_schedule_mask_144 if heating_schedule_mask_144 is None else np.asarray(
            heating_schedule_mask_144, dtype=bool
        ).reshape(-1)
        if schedule_mask_144 is not None:
            if schedule_mask_144.size != 144:
                raise ValueError("heating_schedule_mask_144 must contain exactly 144 values")
            schedule_mask = self._states_to_interval(schedule_mask_144.astype(int), ag).astype(bool)
            p_heat = np.where(schedule_mask, p_heat, 0.0)
        if return_components:
            return p_heat, tout, q_loss, q_solar, q_occupants
        return p_heat

    def _cooling_v2(self, size_idx0: int, occupancy_144: np.ndarray,
                     month: int = 1, ag: int = 600,
                     return_components: bool = False,
                     include_solar_gains: bool = False,
                     floor_area_m2: Optional[float] = None):
        """
        Cooling demand model based on envelope loads and solar gains.
        Output is electric cooling power profile [W] in target interval ag.
        
        Cooling is needed to remove excess heat:
        - When indoor temp would exceed setpoint due to outdoor temp and solar gains
        - Opposite logic to heating: p_cool = max(q_gain - q_loss, 0)
        """
        weather = self._load_tmy_day_for_ag(month, ag)
        tout = weather['tout']
        geom = self._envelope_geometry(size_idx0, floor_area_m2=floor_area_m2)
        exposure = self._shared_envelope_exposure()

        r_wall = float(COOLING_V2_CONFIG.get('r_wall_m2k_w', COOLING_V2_CONFIG.get('r_opaque_m2k_w', 3.4)))
        r_roof = float(COOLING_V2_CONFIG.get('r_roof_m2k_w', r_wall))
        r_floor = float(COOLING_V2_CONFIG.get('r_floor_m2k_w', r_wall))
        r_window = float(COOLING_V2_CONFIG['r_window_m2k_w'])
        ach = float(COOLING_V2_CONFIG['ach_1_h'])
        pmax = float(COOLING_V2_CONFIG['cooler_max_w_m2']) * geom['area_floor']

        area_window_ext = geom['area_window'] * exposure['wall']
        area_wall_opaque_ext = geom['area_wall_opaque'] * exposure['wall']
        area_roof_ext = geom['area_roof'] * exposure['roof']
        area_floor_ext = geom['area_floor'] * exposure['floor']

        ua_trans = (
            area_wall_opaque_ext / max(r_wall, 1e-6)
            + area_roof_ext / max(r_roof, 1e-6)
            + area_floor_ext / max(r_floor, 1e-6)
            + area_window_ext / max(r_window, 1e-6)
        )
        ua_vent = 0.33 * ach * geom['volume']

        state = self._states_to_interval(occupancy_144.astype(int), ag)
        # Cooling setpoints: 27°C away, 24°C active, 25°C sleeping
        tin = np.where(state == 1, 27.0, np.where(state == 2, 24.0, 25.0))
        dT = np.maximum(tout - tin, 0.0)

        # Heat from outdoor air
        q_gain_ext = (ua_trans + ua_vent) * dT
        
        # Solar gains - heat entering through windows and absorbed by opaque surfaces
        q_solar = self._solar_gain_from_orientation(
            weather,
            area_window=area_window_ext,
            area_opaque_solar=area_wall_opaque_ext + area_roof_ext,
            enabled=include_solar_gains,
        )
        
        # Occupant sensible heat gains (same as heating: 0 W away, 80 W active, 70 W sleeping)
        q_occupants = np.where(state == 1, 0.0, np.where(state == 2, 80.0, 70.0))
        
        # Total cooling demand is external heat + solar gains + occupant heat
        q_total = q_gain_ext + q_solar + q_occupants
        p_cool = np.clip(q_total, 0.0, pmax)
        
        if return_components:
            return p_cool, tout, q_gain_ext, q_solar, q_occupants
        return p_cool

    def _build_household_debug_data(self, target_households: int) -> None:
        """
        Build household-level occupancy states and metadata from person-level data.
        """
        occ = self.occupancy.person_occupancy[:, :144].astype(int)
        hh_sizes = [int(i + 1) for i, cnt in enumerate(self.occupancy.size_distribution)
                    for _ in range(int(cnt))]

        states: List[np.ndarray] = []
        occupants: List[int] = []
        hh_floor_area: List[float] = []
        hh_area_per_person: List[float] = []
        hh_area_bin_idx: List[int] = []
        person_floor_area = np.zeros(occ.shape[0], dtype=float)
        cursor = 0

        for size in hh_sizes:
            if cursor + size > occ.shape[0]:
                break
            start = cursor
            members = occ[cursor:cursor + size, :]
            cursor += size

            any_active = np.any(members == 2, axis=0)
            any_home = np.any(members != 1, axis=0)
            hh_state = np.where(any_active, 2, np.where(any_home, 3, 1)).astype(int)

            states.append(hh_state)
            occupants.append(size)
            area_floor, area_pp, area_bin = self._sample_household_floor_area(size)
            hh_floor_area.append(area_floor)
            hh_area_per_person.append(area_pp)
            hh_area_bin_idx.append(area_bin)
            person_floor_area[start:cursor] = area_floor

        if len(states) > target_households:
            states = states[:target_households]
            occupants = occupants[:target_households]
            hh_floor_area = hh_floor_area[:target_households]
            hh_area_per_person = hh_area_per_person[:target_households]
            hh_area_bin_idx = hh_area_bin_idx[:target_households]

        if len(states) < target_households and len(states) > 0:
            while len(states) < target_households:
                states.append(states[-1].copy())
                occupants.append(occupants[-1])
                hh_floor_area.append(hh_floor_area[-1])
                hh_area_per_person.append(hh_area_per_person[-1])
                hh_area_bin_idx.append(hh_area_bin_idx[-1])

        self._household_states_10min = (
            np.array(states, dtype=int) if states else np.zeros((0, 144), dtype=int)
        )
        self._household_occupants = occupants
        self._household_floor_area_m2 = [float(x) for x in hh_floor_area]
        self._household_area_per_person_m2 = [float(x) for x in hh_area_per_person]
        self._household_area_bin_idx = [int(x) for x in hh_area_bin_idx]
        self._person_floor_area_m2 = person_floor_area

    def _aggregate_people_to_households(self, people_matrix: np.ndarray,
                                        target_households: int) -> np.ndarray:
        """
        Aggregate per-person matrix to per-household matrix using occupancy distribution.
        """
        hh_sizes = [int(i + 1) for i, cnt in enumerate(self.occupancy.size_distribution)
                    for _ in range(int(cnt))]
        rows: List[np.ndarray] = []
        cursor = 0
        n_people, n_pts = people_matrix.shape

        for size in hh_sizes:
            if cursor + size > n_people:
                break
            rows.append(np.sum(people_matrix[cursor:cursor + size, :], axis=0))
            cursor += size

        if len(rows) > target_households:
            rows = rows[:target_households]

        if len(rows) < target_households and len(rows) > 0:
            while len(rows) < target_households:
                rows.append(rows[-1].copy())

        if rows:
            return np.array(rows)
        return np.zeros((0, n_pts))

    # ------------------------------------------------------------------
    # Lighting
    # ------------------------------------------------------------------
    def _light_levels(self, month: int) -> np.ndarray:
        light_lvls = self.light_levels_by_month
        col = min(month, light_lvls.shape[1] - 1)
        n   = light_lvls.shape[0]
        return np.interp(np.arange(1, 145) * 10.0,
                         np.arange(1, n + 1) * 15.0,
                         light_lvls[:, col])

    def _compute_lighting(self, P_absent, P_min, P_max, P_inactive,
                            L_lim, L, s, Q_adj, dP) -> np.ndarray:
        P = np.zeros(144); P_ideal = np.zeros(144)
        Pa = P_absent; Pi = P_inactive
        for k in range(1, 144):
            if np.random.random() > 0.3:
                Pa = 0.0; Pi = 0.0
            sk = int(s[k]); Lk = float(L[k])
            if sk == 1:
                P_ideal[k] = Pa
            elif sk == 2:
                P_ideal[k] = (P_min if Lk > L_lim
                               else P_min*(Lk/L_lim) + P_max*(1-Lk/L_lim))
            else:
                P_ideal[k] = Pi
        for k in range(1, 144):
            U = np.random.random()
            if U < Q_adj:
                if int(s[k]) in (1, 3):
                    P[k] = P_ideal[k]
            else:
                diff = P[k-1] - P_ideal[k]
                if diff > 0 and abs(P[k-1]-dP-P_ideal[k]) < abs(diff):
                    P[k] = P[k-1] - dP
                elif diff < 0 and abs(P[k-1]+dP-P_ideal[k]) < abs(diff):
                    P[k] = P[k-1] + dP
                else:
                    P[k] = P[k-1]
        return P

    # ------------------------------------------------------------------
    # Helper: insert profile
    # ------------------------------------------------------------------
    @staticmethod
    def _insert(total_arr: np.ndarray, start: int, profile: np.ndarray):
        z = max(0, min(int(start), 86399))
        e = min(z + len(profile), 86400)
        n = e - z
        if n > 0:
            total_arr[z:e] += profile[:n]
        overflow = (z + len(profile)) - 86400
        if overflow > 0:
            wrap_end = min(overflow, 86400)
            total_arr[:wrap_end] += profile[n : n + wrap_end]

    @staticmethod
    def _insert_no_overlap(total_arr: np.ndarray, occupied: np.ndarray,
                             start: int, profile: np.ndarray) -> bool:
        """Insert a profile only when the target interval is empty."""
        profile = np.asarray(profile, dtype=float).reshape(-1)
        if profile.size == 0:
            return False

        z = max(0, min(int(start), 86399))
        e = z + profile.size
        if e <= 86400:
            if np.any(occupied[z:e]):
                return False
            HouseholdSimulation._insert(total_arr, z, profile)
            occupied[z:e] = True
            return True

        wrap_end = e - 86400
        if np.any(occupied[z:]) or np.any(occupied[:wrap_end]):
            return False
        HouseholdSimulation._insert(total_arr, z, profile)
        occupied[z:] = True
        occupied[:wrap_end] = True
        return True

    # ------------------------------------------------------------------
    # Main simulation – now with enabled_appliances support
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Main simulation
    # ------------------------------------------------------------------
    def simulate(self, n_households: int, period_type: int,
                  enabled_appliances: Optional[Set[str]] = None,
                  latitude: Optional[float] = None,
                  longitude: Optional[float] = None,
                  tmy_loader: Optional[Callable] = None,
                  start_states: Optional[np.ndarray] = None,
                  heating_schedule_mask_144: Optional[np.ndarray] = None,
                  appliance_ownership: Optional[Dict[str, float]] = None):
        """
        Simulate consumption for all households.

        Tracks per-appliance 1D running-sum arrays (one per _SIM_APPLIANCE_KEYS
        entry).  Memory-efficient: only 86 400 floats per appliance regardless of
        household count.  Groups are computed on demand as sums of their appliances.

        boiler / heating / lighting are added later in
        ``results_with_lighting_and_heating``; they appear in
        ``appliance_profile`` after ``get_appliance_aggregated_results`` is called.
        """
        def _en(key: str) -> bool:
            return enabled_appliances is None or key in enabled_appliances

        kind = "weekday" if period_type == 1 else "weekend"
        self._latitude = latitude
        self._longitude = longitude
        self._tmy_loader = tmy_loader
        self._heating_v2_day_cache = {}
        self._results_ag = None
        self.heating_v2_debug = None
        self._heating_schedule_mask_144 = None if heating_schedule_mask_144 is None else np.asarray(
            heating_schedule_mask_144, dtype=bool
        ).reshape(-1)
        if self._heating_schedule_mask_144 is not None and self._heating_schedule_mask_144.size != 144:
            raise ValueError("heating_schedule_mask_144 must contain exactly 144 values")
        self._num_households_requested = int(n_households)
        self.occupancy  = HouseholdOccupancy(
            n_households, kind, self.data_dir, start_states=start_states
        )
        self._build_household_debug_data(self._num_households_requested)
        n_people     = self.occupancy.n_people
        self.model_total_profile = np.zeros((n_people, 86401))
        total_profile = np.zeros((n_people, 86400))

        # Reset fields computed in results_with_lighting_and_heating() so stale data is never reused
        self.lighting_profile = None
        self.model_boiler     = None
        self.model_heating   = None
        self.group_profile     = None

        # Per-appliance 1D running sums (86 400 s / day) ─────────────
        _ap: dict = {k: np.zeros(86400) for k in _SIM_APPLIANCE_KEYS}
        self.appliance_profile = _ap

        # Per-household ownership flags (at most one physical unit per appliance)
        # If `appliance_ownership` is provided it should map appliance key ->
        # ownership probability in [0,1]. By default every household has one unit.
        num_entities = n_people
        self._household_has_appliance: Dict[str, np.ndarray] = {}
        for k in _SIM_APPLIANCE_KEYS:
            if appliance_ownership and k in appliance_ownership:
                p = float(appliance_ownership[k])
                p = max(0.0, min(1.0, p))
                self._household_has_appliance[k] = (np.random.random(num_entities) < p)
            else:
                # default: everyone has the appliance (single unit)
                self._household_has_appliance[k] = np.ones(num_entities, dtype=bool)

        # ---- ownership by consumption class ----
        # Applied only when the caller did not pass appliance_ownership and
        # the class table is available.
        self._class_persons = None
        if self.class_penetration is not None and not appliance_ownership:
            _mix = np.asarray(self.class_mix, dtype=float)
            _mix = _mix / _mix.sum()
            _person_classes = np.zeros(num_entities, dtype=int)
            _p = 0
            for _t in range(7):
                _size = _t + 1
                for _ in range(int(self.occupancy.size_distribution[_t])):
                    _tr = int(np.random.choice(3, p=_mix))
                    _end = min(_p + _size, num_entities)
                    _person_classes[_p:_end] = _tr
                    _p += _size
            self._class_persons = _person_classes
            for k in _SIM_APPLIANCE_KEYS:
                if k not in self.class_penetration.columns:
                    continue
                _p_by_class = self.class_penetration[k].to_numpy(dtype=float)
                _p_persons = _p_by_class[_person_classes]
                self._household_has_appliance[k] = (
                    np.random.random(num_entities) < _p_persons)


        # ---- flat-level appliances: one per household ----
        # The loop below iterates over persons, but the refrigerator, router
        # and small appliances exist once per flat. The mask is therefore
        # narrowed to the first person of each household; persons are ordered
        # in blocks by household size.
        _first_in_hh = np.zeros(num_entities, dtype=bool)
        _pos = 0
        for _hh_kind in range(7):
            _n_hh = int(self.occupancy.size_distribution[_hh_kind])
            _size = _hh_kind + 1
            for _ in range(_n_hh):
                if _pos < num_entities:
                    _first_in_hh[_pos] = True
                _pos += _size
        for _key in ('refrigerator', 'freezer', 'router', 'small_appliances'):
            if _key in self._household_has_appliance:
                self._household_has_appliance[_key] = (
                    self._household_has_appliance[_key] & _first_in_hh)

        _need_food_sched = any(_en(k) for k in ('kettle', 'toaster', 'oven'))

        # ---- assign connection type (1/3-phase) to households ----
        self._single_phase_persons = None
        if self.share_single_phase > 0:
            _single_phase = np.zeros(n_people, dtype=bool)
            _p = 0
            for _t in range(7):
                _size = _t + 1
                for _ in range(int(self.occupancy.size_distribution[_t])):
                    is_single_phase = np.random.random() < self.share_single_phase
                    _end = min(_p + _size, n_people)
                    _single_phase[_p:_end] = is_single_phase
                    _p += _size
            self._single_phase_persons = _single_phase

        # ---- assign a type to individual households ----
        self._type_persons = None
        if self.p_type_by_size is not None:
            _types = np.zeros(n_people, dtype=int)
            _p = 0
            for _t in range(7):
                _size = _t + 1
                for _ in range(int(self.occupancy.size_distribution[_t])):
                    _hh_kind = int(np.random.choice(
                        3, p=self.p_type_by_size[_t]))
                    _end = min(_p + _size, n_people)
                    _types[_p:_end] = _hh_kind
                    _p += _size
            self._type_persons = _types

        person_idx1 = 1
        for size_class1 in range(1, 8):
            n_people = int(self.occupancy.counts[size_class1 - 1])
            size_idx0    = size_class1 - 1
            for _ in range(n_people):
                occ     = self.occupancy.occupancy_person(person_idx1)[:144].copy()
                occ_bin = (occ == 2).astype(float)
                idx     = person_idx1 - 1
                # household type of the person being processed
                self._current_type = (None if self._type_persons is None
                                      else int(self._type_persons[person_idx1 - 1]))
                # is this household single-phase?
                self._current_single_phase = (False if self._single_phase_persons is None
                                     else bool(self._single_phase_persons[person_idx1 - 1]))
                occupied = np.zeros(86400, dtype=bool)

                # Food preparation scheduler
                if _need_food_sched:   # only when a kitchen appliance is enabled
                    n_meals, meal_start_slots = self._food_prep(size_idx0, occ_bin)
                else:
                    n_meals, meal_start_slots = 0, np.array([], dtype=int)

                # Kitchen appliances
                if n_meals > 0:
                    s_kettle, p_kettle = (
                        self._kettle(n_meals, meal_start_slots, size_idx0)
                        if (_en('kettle') and self._household_has_appliance['kettle'][idx]) else ([], [])
                    )
                    s_toaster, p_toaster = (
                        self._toaster(n_meals, meal_start_slots, size_idx0)
                        if (_en('toaster') and self._household_has_appliance['toaster'][idx]) else ([], None)
                    )
                    s_oven, p_oven = (
                        self._oven(n_meals, meal_start_slots, size_idx0)
                        if (_en('oven') and self._household_has_appliance['oven'][idx]) else ([], [])
                    )
                    s_hob, p_hob = (
                        self._hob(n_meals, meal_start_slots, size_idx0)
                        if (_en('hob') and
                            self._household_has_appliance['hob'][idx])
                        else ([], [])
                    )
                    s_micro, p_micro = (
                        self._microwave(n_meals, meal_start_slots, size_idx0)
                        if (_en('microwave') and
                            self._household_has_appliance['microwave'][idx])
                        else ([], [])
                    )
                else:
                    s_kettle=[]; p_kettle=[]
                    s_toaster=[]; p_toaster=None
                    s_oven=[]; p_oven=[]
                    s_hob=[]; p_hob=[]
                    s_micro=[]; p_micro=[]

                # Other appliances
                s_vacuum,  p_vacuum  = self._vacuum(size_idx0, occ_bin)  if (_en('vacuum') and self._household_has_appliance['vacuum'][idx])          else ([], [])
                s_wash,   p_wash   = self._washing_machine(size_idx0, occ_bin)   if (_en('washing_machine') and self._household_has_appliance['washing_machine'][idx])  else (1, np.zeros(1))
                z_tv,       p_tv       = self._tv(size_idx0, occ_bin)       if (_en('tv') and self._household_has_appliance['tv'][idx])               else ([], [])
                z_pc,       p_pc       = self._pc(size_idx0, occ_bin)       if (_en('pc') and self._household_has_appliance['pc'][idx])               else ([], [])
                s_hairdryer,      p_hairdryer      = self._hair_dryer(size_idx0, occ_bin)      if (_en('hair_dryer') and self._household_has_appliance['hair_dryer'][idx])       else ([], [])
                s_dish,    p_dish    = self._dishwasher(size_idx0, occ_bin)    if (_en('dishwasher') and self._household_has_appliance['dishwasher'][idx])       else (1, np.zeros(1))
                s_iron, p_iron = self._ironing(size_idx0, occ_bin)  if (_en('iron') and self._household_has_appliance['iron'][idx])             else ([], [])

                p_fridge    = self._appliance_24h("fridge")    if (_en('refrigerator') and self._household_has_appliance['refrigerator'][idx])     else np.zeros(86400)
                p_router     = self._appliance_24h("router")     if (_en('router') and self._household_has_appliance['router'][idx])           else np.zeros(86400)
                p_small     = self._appliance_24h("small")     if (_en('small_appliances') and self._household_has_appliance['small_appliances'][idx]) else np.zeros(86400)
                p_freezer  = (self._freezer()
                                if (_en('freezer') and self._household_has_appliance.get('freezer', np.ones(1, bool))[idx if 'freezer' in self._household_has_appliance else 0])
                                else np.zeros(86400))
                p_tv_standby = self._appliance_24h("tv_standby") if (_en('tv') and self._household_has_appliance['tv'][idx])               else np.zeros(86400)

                row = total_profile[idx]

                # Insert into total row AND per-appliance 1D sums
                for z, p in zip(s_kettle, p_kettle):
                    if self._insert_no_overlap(row, occupied, z, p):
                        self._insert(_ap['kettle'], z, p)
                if p_toaster is not None:
                    for z in s_toaster:
                        if self._insert_no_overlap(row, occupied, z, p_toaster):
                            self._insert(_ap['toaster'], z, p_toaster)
                for z, p in zip(s_hob, p_hob):
                    if self._insert_no_overlap(row, occupied, z, p):
                        self._insert(_ap['hob'], z, p)
                for z, p in zip(s_micro, p_micro):
                    if self._insert_no_overlap(row, occupied, z, p):
                        self._insert(_ap['microwave'], z, p)
                for z, p in zip(s_oven, p_oven):
                    if self._insert_no_overlap(row, occupied, z, p):
                        self._insert(_ap['oven'], z, p)

                for z, p in zip(z_tv, p_tv):
                    if self._insert_no_overlap(row, occupied, z, p):
                        self._insert(_ap['tv'], z, p)
                for z, p in zip(z_pc, p_pc):
                    if self._insert_no_overlap(row, occupied, z, p):
                        self._insert(_ap['pc'], z, p)

                for z, p in zip(s_hairdryer, p_hairdryer):
                    if self._insert_no_overlap(row, occupied, z, p):
                        self._insert(_ap['hair_dryer'], z, p)

                for z, p in zip(s_vacuum, p_vacuum):
                    if self._insert_no_overlap(row, occupied, z, p):
                        self._insert(_ap['vacuum'], z, p)
                for z, p in zip(s_iron, p_iron):
                    if self._insert_no_overlap(row, occupied, z, p):
                        self._insert(_ap['iron'], z, p)

                if _en('washing_machine') and np.random.random() < 0.8:
                    if self._insert_no_overlap(row, occupied, s_wash, p_wash):
                        self._insert(_ap['washing_machine'], s_wash, p_wash)
                        # The dryer runs AFTER the wash (for owners), starting
                        # at the end of the washing-machine cycle. Via the
                        # overlap check it does not clash with the vacuum or
                        # other activity; ~70 % of washes are then dried.
                        if (_en('dryer') and self.dryer_profile.size > 10
                                and self._household_has_appliance.get('dryer', np.zeros(1, bool))[idx if 'dryer' in self._household_has_appliance else 0]
                                and np.random.random() < 0.7):
                            s_dry = int(s_wash) + len(p_wash)
                            if self._insert_no_overlap(row, occupied, s_dry, self.dryer_profile):
                                self._insert(_ap['dryer'], s_dry, self.dryer_profile)
                if _en('dishwasher') and np.random.random() < 0.7:
                    if self._insert_no_overlap(row, occupied, s_dish, p_dish):
                        self._insert(_ap['dishwasher'], s_dish, p_dish)

                # Always-On: direct array addition (constant 24-h profiles)
                row                      += (p_fridge + p_freezer + p_router
                                             + p_small + p_tv_standby)
                _ap['refrigerator']     += p_fridge
                _ap['freezer']          += p_freezer
                _ap['router']           += p_router
                _ap['small_appliances'] += p_small
                _ap['tv']              += p_tv_standby   # standby → TV appliance

                self.model_total_profile[idx, :86400] = row
                self.model_total_profile[idx, 86400]  = size_class1
                person_idx1 += 1

        return self

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def results_with_lighting_and_heating(self, month: int = 1, ag: int = 600,
                                          enabled_appliances: Optional[Set[str]] = None,
                                          include_solar_gains: bool = True) -> np.ndarray:
        """
        Aggregate results and optionally add lighting, boiler, heating.

        Parameters
        ----------
        month : int
            Month number (1–12).
        ag : int
            Aggregation interval in seconds (1, 60, or 600).
        enabled_appliances : set of str, optional
            Same set passed to simulate(). Controls whether lighting,
            boiler and heating are included.
        """
        def _en(key: str) -> bool:
            return enabled_appliances is None or key in enabled_appliances

        if ag not in (1, 60, 600):
            raise ValueError(f"Unsupported aggregation interval for heating model: {ag}")

        n_people = self.model_total_profile.shape[0]
        L = self._light_levels(month)
        n_pts = 86400 // ag

        self.lighting_profile = np.zeros((n_people, n_pts))
        self.model_boiler     = np.zeros((n_people, n_pts))
        self.model_heating   = np.zeros((n_people, n_pts))
        self.model_cooling   = np.zeros((n_people, n_pts))
        solar_gain_people = np.zeros((n_people, n_pts))

        debug_loss_total = np.zeros(n_pts)
        debug_solar_total = np.zeros(n_pts)
        debug_occupant_total = np.zeros(n_pts)
        debug_outdoor = None

        def _expand_144(arr144: np.ndarray) -> np.ndarray:
            if ag == 600:
                return arr144
            if ag == 60:
                return np.repeat(arr144, 10)
            return np.repeat(arr144, 600)

        for i in range(n_people):
            s = self.occupancy.person_occupancy[i, :144]
            if _en('lighting'):
                p_light = self._compute_lighting(
                    30, 30, 120, 30, 300, L, s, 0.1, 30
                )
                self.lighting_profile[i, :] = _expand_144(p_light)
            
            # Check if current month is in heating season
            is_in_heating_season = self._is_heating_season(month)
            
            # Boiler operates year-round, independent of heating season
            if _en('boiler'):
                self.model_boiler[i, :] = _expand_144(self._boiler(month))
            
            heating_enabled = _en('heating_v2')
            if heating_enabled and is_in_heating_season:
                size_idx0 = int(self.occupancy.household_type(i + 1) - 1)
                floor_area = None
                if self._person_floor_area_m2 is not None and i < len(self._person_floor_area_m2):
                    floor_area = float(self._person_floor_area_m2[i])
                p_heat, tout, q_loss, q_solar, q_occupants = self._heating_v2(
                    size_idx0, s, month=month, ag=ag,
                    return_components=True,
                    include_solar_gains=include_solar_gains,
                    floor_area_m2=floor_area,
                    heating_schedule_mask_144=self._heating_schedule_mask_144,
                )
                self.model_heating[i, :] = p_heat
                solar_gain_people[i, :] = q_solar
                debug_loss_total += q_loss
                debug_solar_total += q_solar
                debug_occupant_total += q_occupants
                if debug_outdoor is None:
                    debug_outdoor = tout
            
            # Check if current month is in cooling season
            is_in_cooling_season = self._is_cooling_season(month)
            
            if _en('cooling_v2') and is_in_cooling_season:
                size_idx0 = int(self.occupancy.household_type(i + 1) - 1)
                floor_area = None
                if self._person_floor_area_m2 is not None and i < len(self._person_floor_area_m2):
                    floor_area = float(self._person_floor_area_m2[i])
                p_cool, tout, q_gain, q_solar, q_occupants = self._cooling_v2(
                    size_idx0, s, month=month, ag=ag,
                    return_components=True,
                    include_solar_gains=include_solar_gains,
                    floor_area_m2=floor_area,
                )
                self.model_cooling[i, :] = p_cool

        raw  = self.model_total_profile[:, :86400]
        res2 = raw.copy() if ag == 1 else raw.reshape(n_people, 86400 // ag, ag).mean(axis=2)
        extra = self.model_boiler + self.model_heating + self.model_cooling

        if _en('heating_v2'):
            hh_states_10 = (self._household_states_10min
                            if self._household_states_10min is not None
                            else np.zeros((0, 144), dtype=int))
            hh_states = hh_states_10 if ag == 600 else np.repeat(hh_states_10, 600 // ag, axis=1)
            hh_occ = self._household_occupants
            hh_area = self._household_floor_area_m2
            hh_area_pp = self._household_area_per_person_m2
            hh_area_bin = self._household_area_bin_idx

            self.heating_v2_debug = {
                'interval': ag,
                'outdoor_temp': (debug_outdoor if debug_outdoor is not None else np.zeros(n_pts)),
                'heat_loss_total': debug_loss_total,
                'solar_gain_total': debug_solar_total,
                'occupant_heat_gain_total': debug_occupant_total,
                'household_states': hh_states,
                'household_occupants': np.array(hh_occ, dtype=int),
                'household_floor_area_m2': np.array(hh_area, dtype=float),
                'household_area_per_person_m2': np.array(hh_area_pp, dtype=float),
                'household_area_bin_idx': np.array(hh_area_bin, dtype=int),
            }
        else:
            self.heating_v2_debug = None

        self._results_ag = ag
        total_people = res2 + self.lighting_profile + extra

        if _en('heating_v2') and self.heating_v2_debug is not None:
            hh_consumption = self._aggregate_people_to_households(
                total_people,
                self._num_households_requested
            )
            hh_solar_gain = self._aggregate_people_to_households(
                solar_gain_people,
                self._num_households_requested
            )
            self.heating_v2_debug['household_consumption'] = hh_consumption
            self.heating_v2_debug['household_solar_gain'] = hh_solar_gain

        return total_people

    def get_aggregated_results(self, interval_seconds: int = 600,
                                month: int = 1,
                                enabled_appliances: Optional[Set[str]] = None,
                                include_solar_gains: bool = True) -> np.ndarray:
        return np.sum(
            self.results_with_lighting_and_heating(
                month=month, ag=interval_seconds,
                enabled_appliances=enabled_appliances,
                include_solar_gains=include_solar_gains,
            ),
            axis=0
        )

    def get_reactive_power(self, interval_seconds: int = 600,
                            month: int = 1,
                            enabled_appliances: Optional[Set[str]] = None,
                            include_solar_gains: bool = True) -> np.ndarray:
        # Reactive power is NOT proportional to active power. Measurements
        # show a capacitive draw practically independent of P, so each
        # connection point is assigned a constant offset Q0 (see the module
        # header). The power factor is no longer an input but an output:
        # cos(phi) = P / sqrt(P^2 + Q^2).
        active = self.get_aggregated_results(
            interval_seconds, month, enabled_appliances,
            include_solar_gains=include_solar_gains,
        )
        n_hh = int(np.sum(self.occupancy.size_distribution))
        return np.full(len(active), Q0_VAR_PER_CP * n_hh, dtype=float)

    def _ensure_results(self, interval_seconds: int, month: int,
                          enabled_appliances,
                          include_solar_gains: bool = True) -> None:
        """Compute lighting/boiler/heating once; skip if already populated."""
        if self.lighting_profile is None or self._results_ag != interval_seconds:
            self.results_with_lighting_and_heating(
                month=month, ag=interval_seconds,
                enabled_appliances=enabled_appliances,
                include_solar_gains=include_solar_gains,
            )

    def get_appliance_aggregated_results(self, interval_seconds: int = 600,
                                          month: int = 1,
                                          enabled_appliances: Optional[Set[str]] = None,
                                          include_solar_gains: bool = True
                                          ) -> dict:
        """
        Return aggregated load per individual appliance.

        Returns
        -------
        dict  {appliance_key: np.ndarray(n_points)}
            Keys match APPLIANCE_KEYS order.
        """
        if self.appliance_profile is None:
            raise RuntimeError("Call simulate() first.")

        # Populate lighting / boiler / heating if not yet done
        self._ensure_results(interval_seconds, month, enabled_appliances, include_solar_gains)

        ag = interval_seconds

        def _agg_1d(arr: np.ndarray) -> np.ndarray:
            """Aggregate (86400,) 1-s sum array to requested interval."""
            if ag == 1:
                return arr.copy()
            pts = 86400 // ag
            return arr.reshape(pts, ag).mean(axis=1)

        def _sum_matrix(matrix: np.ndarray) -> np.ndarray:
            """Sum matrix over people and align to requested length."""
            s = np.sum(matrix, axis=0)
            expected = 86400 // ag
            if len(s) == expected:
                return s
            if len(s) == 144:
                if ag == 600:
                    return s
                if ag == 60:
                    return np.repeat(s, 10)
                return np.repeat(s, 600)
            x_src = np.linspace(0, 1, len(s), endpoint=False)
            x_dst = np.linspace(0, 1, expected, endpoint=False)
            return np.interp(x_dst, x_src, s)

        out: dict = {}
        use_v2 = (enabled_appliances is None) or ('heating_v2' in enabled_appliances)
        use_legacy_heating = ((enabled_appliances is None) or ('heating' in enabled_appliances)) and (not use_v2)
        use_cooling = (enabled_appliances is None) or ('cooling_v2' in enabled_appliances)
        for key in APPLIANCE_KEYS:
            if key in self.appliance_profile:
                out[key] = _agg_1d(self.appliance_profile[key])
            elif key == 'boiler' and self.model_boiler is not None:
                out[key] = _sum_matrix(self.model_boiler)
            elif key == 'heating' and self.model_heating is not None and use_legacy_heating:
                out[key] = _sum_matrix(self.model_heating)
            elif key == 'heating_v2' and self.model_heating is not None and use_v2:
                out[key] = _sum_matrix(self.model_heating)
            elif key == 'cooling_v2' and self.model_cooling is not None and use_cooling:
                out[key] = _sum_matrix(self.model_cooling)
            elif key == 'lighting' and self.lighting_profile is not None:
                out[key] = _sum_matrix(self.lighting_profile)
        return out

    def get_heating_v2_debug(self, interval_seconds: int = 600,
                              month: int = 1,
                              enabled_appliances: Optional[Set[str]] = None,
                              include_solar_gains: bool = True) -> Optional[dict]:
        """Return heating v2 debug arrays for export."""
        self._ensure_results(interval_seconds, month, enabled_appliances, include_solar_gains)
        return self.heating_v2_debug

    def get_group_aggregated_results(self, interval_seconds: int = 600,
                                      month: int = 1,
                                      enabled_appliances: Optional[Set[str]] = None,
                                      include_solar_gains: bool = True
                                      ) -> dict:
        """
        Return aggregated load per appliance group.

        Groups are computed as sums of their constituent appliance arrays,
        so this calls get_appliance_aggregated_results() internally.

        Returns
        -------
        dict  {group_key: np.ndarray(n_points)}  — keys match GROUP_KEYS.
        """
        ap = self.get_appliance_aggregated_results(
            interval_seconds, month, enabled_appliances,
            include_solar_gains=include_solar_gains,
        )
        out: dict = {}
        for gkey in GROUP_KEYS:
            member_keys = APPLIANCE_GROUPS.get(gkey, set())
            arrays = [ap[k] for k in member_keys if k in ap]
            if arrays:
                out[gkey] = sum(arrays)
        return out


# ===========================================================================
# Wrapper – drop-in replacement for old HouseholdSimulator
# ===========================================================================

class HouseholdSimulatorReal:
    def __init__(self, period_type: int = 1, data_dir: str = "Data"):
        self.period_type = period_type
        self.data_dir    = data_dir
        self._sim: Optional[HouseholdSimulation] = None
        self._month = 1
        self._enabled_appliances: Optional[Set[str]] = None
        self._latitude: Optional[float] = None
        self._longitude: Optional[float] = None
        self._tmy_loader: Optional[Callable] = None
        self._heating_schedule_mask_144: Optional[np.ndarray] = None

    def simulate(self, num_households: int, period_type: int = 1,
                 month: int = 1,
                 enabled_appliances: Optional[Set[str]] = None,
                 latitude: Optional[float] = None,
                 longitude: Optional[float] = None,
                 tmy_loader: Optional[Callable] = None,
                 start_states: Optional[np.ndarray] = None,
                 heating_schedule_mask_144: Optional[np.ndarray] = None,
                 appliance_ownership: Optional[Dict[str, float]] = None) -> "HouseholdSimulatorReal":
        self.period_type         = period_type
        self._month              = month
        self._enabled_appliances = enabled_appliances
        self._latitude           = latitude
        self._longitude          = longitude
        self._tmy_loader         = tmy_loader
        self._heating_schedule_mask_144 = heating_schedule_mask_144
        self._sim = HouseholdSimulation(period_type=period_type, data_dir=self.data_dir)
        self._sim.simulate(
            num_households,
            period_type,
            enabled_appliances,
            latitude=latitude,
            longitude=longitude,
            tmy_loader=tmy_loader,
            start_states=start_states,
            heating_schedule_mask_144=heating_schedule_mask_144,
            appliance_ownership=appliance_ownership,
        )
        return self

    def get_final_states(self) -> Optional[np.ndarray]:
        if self._sim is None or self._sim.occupancy is None:
            return None
        states = getattr(self._sim.occupancy, 'final_states', None)
        if states is None:
            return None
        return np.asarray(states, dtype=int).copy()

    def get_aggregated_results(self, interval_seconds: int = 600) -> np.ndarray:
        if self._sim is None:
            raise RuntimeError("Call simulate() first.")
        return self._sim.get_aggregated_results(
            interval_seconds, month=self._month,
            enabled_appliances=self._enabled_appliances,
            include_solar_gains=True,
        )

    def get_reactive_power(self, interval_seconds: int = 600) -> np.ndarray:
        if self._sim is None:
            raise RuntimeError("Call simulate() first.")
        return self._sim.get_reactive_power(
            interval_seconds, month=self._month,
            enabled_appliances=self._enabled_appliances,
            include_solar_gains=True,
        )

    def get_appliance_aggregated_results(self, interval_seconds: int = 600) -> dict:
        """Return per-appliance aggregated power profiles."""
        if self._sim is None:
            raise RuntimeError("Call simulate() first.")
        return self._sim.get_appliance_aggregated_results(
            interval_seconds, month=self._month,
            enabled_appliances=self._enabled_appliances,
            include_solar_gains=True,
        )

    def get_group_aggregated_results(self, interval_seconds: int = 600) -> dict:
        """Return per-group aggregated power profiles."""
        if self._sim is None:
            raise RuntimeError("Call simulate() first.")
        return self._sim.get_group_aggregated_results(
            interval_seconds, month=self._month,
            enabled_appliances=self._enabled_appliances,
            include_solar_gains=True,
        )

    def get_heating_v2_debug(self, interval_seconds: int = 600) -> Optional[dict]:
        """Return heating v2 debug arrays for CSV export."""
        if self._sim is None:
            raise RuntimeError("Call simulate() first.")
        return self._sim.get_heating_v2_debug(
            interval_seconds, month=self._month,
            enabled_appliances=self._enabled_appliances,
            include_solar_gains=True,
        )