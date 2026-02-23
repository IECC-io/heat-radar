# Heat-Radar

Predicting future human heat stress using climate projections and machine learning.

Heat-Radar combines physiological heat stress modeling (EHI-N*) with climate data to forecast dangerous heat conditions across two timescales:

- **[Nowcast](nowcast/)** — Short-term extreme heat event prediction (1–7 days) using ERA5 reanalysis and deep learning (GRU + autoencoder)
- **[Longterm](longterm/)** — Future heat stress projections (2025–2100) using CMIP6 climate models and ML ensembles (RF, XGBoost)

Both pipelines share a common physiological engine in [`shared/`](shared/) based on the Extended Heat Index (EHI-N*), which accounts for metabolic workload, solar radiation, and population-specific body parameters.

## Repository Structure

```
heat-radar/
├── nowcast/          # Short-term forecasting (1-7 days)
│   ├── data/         # ERA5 hourly data
│   ├── src/          # Stochastic decomposition, GRU, autoencoder
│   ├── notebooks/
│   └── models/
│
├── longterm/         # Future projections (2025-2100)
│   ├── data/         # CMIP6 data
│   ├── src/          # RF, XGBoost, ensemble pipeline
│   ├── notebooks/
│   └── models/
│
├── shared/           # Shared across pipelines
│   ├── config.py     # Body params, MET levels, constants
│   ├── ehi/          # EHI-N* heat index engine
│   └── visualization/
│
└── docs/             # Landing page, figures
```

## Setup

```bash
git clone https://github.com/IECC-io/heat-radar.git
cd heat-radar
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## License

See [LICENSE](LICENSE) for details.
