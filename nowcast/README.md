# Nowcast — Short-Term Heat Stress Forecasting (1–7 days)

Short-term extreme heat event prediction using ERA5 reanalysis, stochastic variability decomposition, and deep learning.

## Pipeline

1. **Data**: ERA5 hourly T2m, Td2m for target city (default: Kolkata)
2. **Stochastic Decomposition**: LOWESS smoothing → mean diurnal cycle + residuals
3. **Anomaly Detection**: Autoencoder on residual patterns
4. **Sequential Prediction**: GRU on residual sequences → extreme event probability
5. **Severity**: EHI-N* zone classification (from `shared/ehi/`)

## Status

Pipeline is under active development. See project slides for current progress.
