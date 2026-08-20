# Data sources and attribution

This folder contains the model inputs. Files fall into four groups by origin.

## Appliance power traces — Issi & Kaplan (2018), CC-BY 4.0

The following measured appliance load curves are taken from:

> Issi, F.; Kaplan, O. *The Determination of Load Profiles and Power
> Consumptions of Home Appliances.* Energies 2018, 11(3), 607.
> https://doi.org/10.3390/en11030607

Licensed under the Creative Commons Attribution 4.0 International licence
(CC-BY 4.0). Reused here with attribution.

`TVPower.csv`, `RefrigeratorPower.csv`, `WaterHeaterPower.csv`,
`WaterHeaterPower05Lt.csv`, `WaterHeaterPower1.5Lt.csv`,
`ToastMachinePower.csv`, `Oven180C.csv`, `PC_Power.csv`, `IronPower.csv`,
`HairDryer.csv`, `WashingMachinePower_30C.csv`, `WashingMachinePower_40C.csv`,
`WashingMachinePower_40CMix.csv`, `PrinterPower.csv`, `DishWasherPower.csv`,
`DishWasherPower_Power65C.csv`.

Note: for oven, kettle, hob, microwave and washing machine these act as
**fallbacks** — the model prefers the group's own measurement in
`measurements_cz/` when present. The **dishwasher** is different: the Issi
cycles (normal 326 Wh + 65 °C 984 Wh) are the **cold-water default** (the
machine heats its own water, ~819 Wh — the majority case), while the own
`measurements_cz/dishwasher.csv` (101 Wh) is the **hot-water-connected option**
selected via the `dishwasher_hot_water` toggle.

## Vacuum — authors' own measurement

`vacuum.csv` is re-derived from the authors' own KMB SMY-CA measurement.

## Router — no data file

The router is not shipped as a data file. It is a constant always-on base
load (~5 W), modelled in code alongside the small-appliances constant
(`household_simulation.py`, `router_profile`). A whole-flat measurement cannot
isolate a single ~5 W device, so rather than fabricate a device-level trace,
the router is represented by a constant equal to the previous trace's mean.

## Authors' own measurements — `measurements_cz/`

Second-resolution measurement of a Czech flat, KMB SMY-CA analyser,
December 2021 (the traces behind Table 2 of the paper). Author-owned.

`measurements_cz/oven.csv`, `kettle.csv`, `hob_1zone.csv`, `hob_max.csv`,
`microwave.csv`, `washing_machine.csv`, `dishwasher.csv` (dishwasher,
hot-water-connected option, 101 Wh/cycle — the trace behind Table 2),
`dryer_hp.csv` (heat-pump tumble dryer, 872 Wh/cycle — a full appliance in
the model, penetration 0/42/100 by class).

## Derived statistics and model parameters

- `markov_weekday.csv`, `markov_weekend.csv`, `weekday.npz`, `weekend.npz`,
  `mean_switchons*.csv`, `var_switchons*.csv`, `p_type_by_size.csv` —
  aggregates derived from the **UK Time-Use Survey 2014–2015** (Gershuny &
  Sullivan; UK Data Service SN 8128). The survey microdata are **not**
  redistributable and are not included; only these aggregates are.
- `Living.csv` — floor-area sampling table by household size, built from the
  **Czech Census of People, Houses and Dwellings (SLDB 2021)**. Aggregated
  public-census data.
- `illuminance.xlsx` — daylight illuminance (lux) by time-of-day and month,
  read from **PVGIS** (EU Joint Research Centre) solar-irradiation data for
  the Czech Republic and converted to illuminance (685 lux = 1 W/m²). PVGIS
  is free to use with attribution. The lighting model that consumes it
  follows Widén et al. (2009); the switching threshold is from ČSN EN 17037.
- `penetration_by_class.csv`, `heating_schedule.csv`, `heating.csv` — model
  parameters chosen by the authors.

The TDD4 seasonal factors are hard-coded constants in `../localization.py`,
derived from the public OTE standard load profile; the raw OTE dataset is not
redistributed here.
