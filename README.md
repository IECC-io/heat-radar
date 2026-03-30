# Heat-Radar

Short-term extreme heat event prediction using ERA5 reanalysis and deep learning.

Heat-Radar combines physiological heat stress modeling (EHI-N*) with ERA5 climate data to forecast dangerous heat conditions 1–7 days ahead. The pipeline uses stochastic variability decomposition, a GRU classifier, and an autoencoder for anomaly detection.

## Repository Structure

```
heat-radar/
├── nowcast/
│   ├── src/
│   │   ├── heatradar_nowcast.py   # Full pipeline (data → features → GRU → evaluation)
│   │   └── download_ssrd.py       # Download ERA5 SSRD data from ARCO-ERA5
│   ├── data/                      # ERA5 NetCDF files (not tracked — see below)
│   ├── notebooks/
│   └── models/
│
├── shared/
│   ├── config.py                  # Body params, MET levels, zone definitions
│   └── ehi/
│       ├── generate_lookup_tables.py   # Generate EHI-N* lookup tables (requires heatindex_ek.py)
│       └── lookup_tables/              # Pre-computed .npz tables (one per MET level)
│
└── docs/
```

## Quick Start

```bash
git clone https://github.com/IECC-io/heat-radar.git
cd heat-radar
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the classifier pipeline (MET 4 = outdoor labor, default)
python nowcast/src/heatradar_nowcast.py \
    --data_dir nowcast/data \
    --lookup_table_dir shared/ehi/lookup_tables \
    --mode classifier

# Run for all MET levels
for m in 1 2 3 4 5 6; do
  python nowcast/src/heatradar_nowcast.py \
      --data_dir nowcast/data \
      --lookup_table_dir shared/ehi/lookup_tables \
      --met_level $m --mode classifier
done
```

## ERA5 Data

ERA5 hourly data for Kolkata (2005–2024, March–July) is included in `nowcast/data/`.
To download additional years or cities, use `nowcast/src/download_ssrd.py`, which pulls from the public
[ARCO-ERA5](https://cloud.google.com/storage/docs/public-datasets/era5) dataset on Google Cloud.

## EHI Lookup Tables

Pre-computed EHI-N* lookup tables are in `shared/ehi/lookup_tables/` — one `.npz` file per MET level (1–6).
These replace the `heatindex_ek.py` Numba dependency at runtime.
To regenerate them (requires `heatindex_ek.py`):

```bash
python shared/ehi/generate_lookup_tables.py
```

## License

See [LICENSE](LICENSE) for details.
