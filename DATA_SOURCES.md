# Data sources and redistribution status

The repository is MIT-licensed, but the MIT licence covers **the code only**.
Third-party input data keep their own licences, summarised below. Per-file
provenance is in [`Data/README.md`](Data/README.md).

| Data | Source | Licence / redistribution |
|------|--------|--------------------------|
| Appliance power traces (`*Power*.csv`, `Oven180C.csv`, `HairDryer.csv`, …) | Issi, F. & Kaplan, O. (2018), *The Determination of Load Profiles and Power Consumptions of Home Appliances*, Energies 11(3):607, [doi:10.3390/en11030607](https://doi.org/10.3390/en11030607) | **CC-BY 4.0** — redistributed here with attribution |
| Kitchen/thermal appliances and vacuum (`measurements_cz/`, `vacuum.csv`) | Authors' own second-resolution measurement of a Czech flat (KMB SMY-CA analyser, December 2021) | Author-owned — redistributed by the authors' decision |
| Occupancy matrices and appliance-usage statistics (`markov_*.csv`, `mean_switchons*.csv`, `var_switchons*.csv`, `p_type_by_size.csv`, `weekday.npz`, `weekend.npz`) | Aggregates derived from the **UK Time-Use Survey 2014–2015** (UK Data Service SN 8128) | Only aggregates are shipped; the survey microdata are **not** redistributable and are **not** included |
| Floor-area sampling (`Living.csv`) | **Czech Census (SLDB 2021)** | Aggregated public-census data |
| Daylight illuminance (`illuminance.xlsx`) | **PVGIS** (EU Joint Research Centre) | Free to use with attribution |
| Seasonal factors (`localization.py`) | Precomputed constants from the public **OTE** TDD4 standard load profile | Only the twelve derived factors are shipped, not the raw dataset |
| Model parameters (`penetration_by_class.csv`, `heating.csv`, `heating_schedule.csv`) | Chosen by the authors | Author-owned |

## Restricted data — deliberately excluded

Distribution-feeder and smart-meter measurements used only for the paper's
validation (from the distribution system operators) are **not** included in
this repository and are **not** required to run the model. They are covered
by the paper's Code and data availability statement.
