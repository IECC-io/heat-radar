# Longterm — Future Heat Stress Projections (2025–2100)

ML-based projections of future heat stress using CMIP6 climate model output and EHI-N* physiological modeling.

## Pipeline

1. **Data**: NASA NEX-GDDP-CMIP6 (bias-corrected, 0.25° resolution)
2. **Variables**: Near-surface temperature (tas), relative humidity (hurs)
3. **Models**: Random Forest → XGBoost → Stacked Ensemble
4. **Target**: EHI-N* values and zone classifications under SSP scenarios
5. **Output**: Gridded future heat stress projections through 2100

## Data Source

- **NASA NEX-GDDP-CMIP6** via AWS S3 (`s3://nex-gddp-cmip6`)
- License: CC0 (Public Domain)
- Resolution: 0.25° (~25km)

## Status

Baseline Random Forest model complete. XGBoost and ensemble in progress.
