# Nowcast — Short-Term Heat Stress Forecasting (1–7 days)

Short-term extreme heat event prediction using ERA5 reanalysis, stochastic variability decomposition, and deep learning.

## Pipeline

1. **Data**: ERA5 hourly T2m, Td2m, SSRD for Kolkata (2005–present)
2. **Stochastic Decomposition**: LOWESS smoothing → mean diurnal cycle + residuals
3. **EHI-N* Features**: Physiological heat stress zones via lookup tables (no Numba required)
4. **Anomaly Detection**: Autoencoder on residual patterns
5. **Classification**: GRU on residual sequences → extreme event probability (48h ahead)

## Usage

```bash
python nowcast/src/heatradar_nowcast.py \
    --data_dir nowcast/data \
    --lookup_table_dir shared/ehi/lookup_tables \
    --met_level 4 \
    --mode classifier
```

See `python nowcast/src/heatradar_nowcast.py --help` for all options.
