# Localizable bottom-up household load model

A localizable, decomposable **bottom-up model of residential electricity
demand** with 1-second resolution and a simple PySide6 desktop app. The model
derives the load diagram from occupant behaviour (a Markov occupancy chain
driven by a time-use survey) and measured appliance power curves, so the total
diagram can be broken down to individual appliances. This is the reference
implementation for the paper *"A localisable bottom-up model of household
load profiles with appliance-level decomposition: validation against five
independent measurement sources"* (Applied Energy, in preparation).

## Key features

- **Appliance-level decomposition** — the aggregate diagram is always the sum
  of individual appliance contributions, so efficiency scenarios can be applied
  afterwards without re-simulating.
- **Seasonal localization by the national standard load profile (TDD4)** — the
  behavioural core comes from a foreign time-use survey; the seasonal shape is
  taken from the target country's public settlement profile. No local time-use
  survey required.
- **Reactive power (Q₀)** — measurements show residential reactive demand is
  persistently *capacitive* and practically independent of active power, so it
  is modelled as a constant capacitive baseline per connection point, with a
  time-varying inductive contribution added while occasional motor-driven
  appliances (washing machine, dryer, dishwasher, vacuum) are running (power
  factor becomes an output, not an input).
- **Efficiency scenario** — one click applies best-in-class appliance factors
  and shows why appliance renewal cuts annual energy by ~25 % but the evening
  peak only by ~20 % (the peak is thermal).
- **Photovoltaics (pvlib + PVGIS TMY)** — optional on-site generation and net
  load, for self-consumption/sizing studies.

## Install

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (source venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

Python 3.9+ recommended. `pvlib` is only needed for the photovoltaics feature
(and it downloads TMY data from PVGIS, so that feature needs internet access).

## Usage

**Desktop app:**

```bash
python app.py
```

Pick a month, number of dwellings, resolution, day type and iterations, toggle
the features you want (reactive power, TDD localization, efficiency scenario,
PV), run, and export the result to CSV.

**From a script (no GUI):**

```bash
python examples/run_headless.py
```

See that file for how to drive `model_runner.run()` for batch runs or
sensitivity studies.

## Repository layout

| File | Role |
|------|------|
| `app.py` | PySide6 desktop app (entry point) |
| `model_runner.py` | Orchestration: core + localization + efficiency + Q₀ + PV |
| `household_simulation.py` | Simulation engine (occupancy, appliances, thermal model) |
| `localization.py` | TDD4 seasonal factors |
| `efficiency.py` | Efficiency-scenario appliance factors |
| `pv.py` | Photovoltaic chain (pvlib + PVGIS TMY) |
| `Data/` | Model inputs (Markov matrices, appliance curves, …) |
| `examples/` | Headless usage example |

## Data and licensing

The `Data/` folder ships the inputs the engine needs to run. Full
per-file provenance is in [`Data/README.md`](Data/README.md); in short:

- **Appliance power traces** — most are from Issi, F. & Kaplan, O. (2018),
  *The Determination of Load Profiles and Power Consumptions of Home
  Appliances*, Energies 11(3):607, [doi:10.3390/en11030607](https://doi.org/10.3390/en11030607),
  reused under **CC-BY 4.0** (attribution required). The kitchen/thermal
  appliances and the vacuum in `Data/measurements_cz/` are the authors' own
  second-resolution measurements of a Czech flat (KMB SMY-CA analyser).
- **Occupancy matrices and appliance-usage statistics** (`markov_*.csv`,
  `*switchons*.csv`, `weekday.npz`, `weekend.npz`) are aggregates derived from
  the **UK Time-Use Survey 2014–2015** (UK Data Service SN 8128). The survey
  microdata are not redistributable and are not included; only these
  aggregates are.
- **`Living.csv`** (floor-area sampling by household size) is built from the
  **Czech Census 2021 (SLDB)**; **`illuminance.xlsx`** (daylight illuminance)
  is derived from **PVGIS** (EU Joint Research Centre). The TDD4 seasonal
  factors in `localization.py` are precomputed constants from the public OTE
  standard load profile.

Restricted validation data (distribution feeder and smart-meter measurements
from the distribution system operators) are **not** included and are not
required to run the model; they underlie the paper's validation only.

## How it maps to the paper

| Paper | Code |
|-------|------|
| Model, appliance decomposition | `household_simulation.py`, `model_runner.py` |
| Localization by TDD4 | `localization.py` |
| Efficiency scenario | `efficiency.py` |
| Reactive power (Q₀ = −55.9 var/CP baseline + motor inductive term) | `household_simulation.get_reactive_power` |
| Photovoltaics | `pv.py` |

## Citation

If you use this model, please cite both the paper and this repository. See
[`CITATION.cff`](CITATION.cff); a Zenodo DOI is minted from the tagged release.

## License

Released under the MIT License (see [`LICENSE`](LICENSE)). The MIT licence
covers the code; third-party input data keep their own licences, documented
in [`Data/README.md`](Data/README.md).
