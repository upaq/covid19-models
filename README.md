# Data and models for COVID-19

Data processing and models for COVID-19

## Usage example:

```python
from covid_models import CovidData
```
and

```python
cd = CovidData()

max_date = datetime.datetime(2020, 4, 24)
isos = cd.countries_with_cumulative_mortality_greater_than(100, max_date)

npis = ['npi_stringency_index']

for idx in range(len(isos)):
    iso = isos[idx]
    x, controls, future_controls = cd.get_country_data(
        iso,
        npis,
        max_date=max_date,
        weeks_back=8,
        global_bias=True,
        country_bias=True,
        country_index=idx,
        total_countries=len(isos),
        dynamic_bias=True,
        future_days=21)
    
```
