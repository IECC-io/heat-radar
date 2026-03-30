"""
================================================================================
HeatRadar Nowcast — Weather GRU + EHI-N* Pipeline
================================================================================

7-day EHI-N* zone forecasting for Kolkata using:
  - Autoencoder (unsupervised, runs FIRST): learns "normal" weather patterns,
    outputs anomaly_score + latent_z that feed into downstream models
  - Weather GRU (primary): 240h lookback → 168h forecast of T, RH, SSRD
    Uses 12 input features INCLUDING autoencoder outputs
  - EHI-N* (physics, no training): converts forecasts to physiological zones 1–6
    Uses real solar heat load (Qs) from SSRD + solar altitude
  - GRU Classifier (legacy): binary extreme heat classification

Pipeline:
  ERA5 (T2m, D2m, SSRD) → PySolar + RH calc → LOWESS decomposition
  → Autoencoder (learn normal → anomaly_score, latent_z)
  → Weather GRU (forecast 7 days) → EHI-N* physics → Zone alerts

Authors: Elif Kilic, Simone Robson

Usage:
  python heatradar_nowcast.py --data_dir ./data --met_level 4
      → Runs autoencoder (features) + Weather GRU (primary forecast model)
  python heatradar_nowcast.py --mode full --data_dir ./data
      → Runs ALL three: autoencoder → classifier → Weather GRU
  python heatradar_nowcast.py --mode classifier --data_dir ./data
      → Runs GRU classifier (predicts 48h ahead by default)
  python heatradar_nowcast.py --mode classifier --lead_time 72
      → Classifier predicts 72h (3 days) ahead
  python heatradar_nowcast.py --mode classifier --lead_time 0
      → Classifier in nowcasting mode (is it extreme NOW?)
  python heatradar_nowcast.py --mode autoencoder --data_dir ./data
      → Runs only the autoencoder (anomaly detection)
  python heatradar_nowcast.py --mode inference
      → Live 7-day forecast using trained model + Open-Meteo API
  python heatradar_nowcast.py --help

References:
  - Lu & Romps (2022), "Extending the Heat Index", JAMC
  - heatindex_ek.py (canonical EHI-N* implementation with Qs=0 fix)
================================================================================
"""

# =============================================================================
# Section 0: Imports & Config
# =============================================================================
#
# Here is why we need each of these imports:
#   - argparse: allows command-line control of the pipeline (e.g. --met_level 5)
#     so you don't have to edit the code to change parameters
#   - xarray: reads NetCDF climate data files (ERA5 reanalysis)
#   - tensorflow/keras: builds and trains the GRU neural networks
#   - sklearn: MinMaxScaler (normalizes data to 0-1 range), evaluation metrics
#   - matplotlib: generates all plots (training curves, forecasts, etc.)
#
# WHY NOT PyTorch?
#   The original notebooks used Keras/TensorFlow, so we keep the same
#   framework for consistency. Both work fine for GRUs.
# =============================================================================

import argparse
import datetime as dt
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib

# "Agg" backend = render plots to PNG files without needing a display window.
# Required when running on servers or in scripts (no GUI).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
)

# Suppress TensorFlow's verbose startup messages (GPU info, etc.)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    GRU,         # Gated Recurrent Unit — a type of recurrent neural network
    Dense,       # Standard fully-connected layer
    Dropout,     # Randomly drops neurons during training to prevent overfitting
    Input,       # Defines the input shape
    RepeatVector,    # Used in autoencoder decoder to repeat the bottleneck
    TimeDistributed, # Applies a layer to each timestep independently
)
from tensorflow.keras.optimizers import Adam  # Adaptive learning rate optimizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

warnings.filterwarnings("ignore", category=FutureWarning)

# PySolar computes exact sun position (altitude angle above horizon) for any
# location and time. We use this to determine:
#   1. Whether the sun is up (sun_flag) — affects solar radiation on the body
#   2. The solar altitude angle — determines how much of your body is exposed
#      to direct sunlight (low sun = long shadow = more frontal exposure)
try:
    from pysolar.solar import get_altitude
    _HAS_PYSOLAR = True
except ImportError:
    _HAS_PYSOLAR = False
    warnings.warn(
        "pysolar not installed — solar angle will be estimated from SSRD. "
        "Install with: pip install pysolar",
        stacklevel=2,
    )

# MET level → metabolic heat flux (W/m²)
# MET = Metabolic Equivalent of Task. It represents how much heat the body
# generates internally from physical activity. Higher activity = more internal
# heat = harder to cool down = higher risk of heat stress.
#
# WHY THIS MATTERS:
#   A person sitting in shade (MET 1 = 65 W/m²) can tolerate much higher
#   air temperatures than a construction worker (MET 4 = 240 W/m²) because
#   the worker's body is producing 4x more internal heat that must be
#   dissipated. The EHI-N* model accounts for this.
#
# Default is MET 4 (outdoor labor) because HeatRadar targets outdoor workers
# in Kolkata who are most vulnerable to extreme heat events.
MET_LEVELS = {
    1: 65,    # Resting (sitting, reading)
    2: 130,   # Low activity (walking slowly, light office work)
    3: 200,   # Moderate activity (walking briskly, light manual labor)
    4: 240,   # High activity (outdoor labor, construction — DEFAULT)
    5: 290,   # Very high activity (heavy lifting, fast-paced manual work)
    6: 400,   # Extreme activity (sprinting, firefighting)
}


@dataclass
class Config:
    """
    All tunable parameters for the pipeline, collected in one place.

    WHY A DATACLASS?
      Instead of having magic numbers scattered throughout the code,
      every parameter lives here. You can change any setting from the
      command line (e.g. --met_level 5 --epochs 100) without editing code.
      The dataclass also serializes to JSON for reproducibility — when you
      load a saved model, you know exactly what settings produced it.
    """

    # -------------------------------------------------------------------------
    # Pipeline mode — which neural network(s) to train
    # -------------------------------------------------------------------------
    # "forecast"    = Weather GRU (PRIMARY) — predicts T, RH, SSRD 7 days ahead
    # "autoencoder" = unsupervised anomaly detection (learns "normal" weather)
    # "classifier"  = binary classifier (will it be extreme in X hours?)
    # "full"        = autoencoder → classifier (autoencoder features feed classifier)
    # "inference"   = live mode — fetches real-time data from Open-Meteo API,
    #                 runs the trained Weather GRU, and outputs zone alerts
    mode: str = "forecast"

    # -------------------------------------------------------------------------
    # EHI-N* physiology parameters
    # -------------------------------------------------------------------------
    # These define the "person" whose heat stress we're modeling.
    # Different people tolerate heat differently based on:
    #   - Activity level (MET): a resting person vs an outdoor worker
    #   - Body size: affects surface area for heat exchange
    met_level: int = 4            # MET 4 = outdoor labor (240 W/m²)
    body_mass_kg: float = 65.0    # Average body mass
    body_height_m: float = 1.65   # Average body height

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    # WHY SEEDS?
    #   Neural networks use random numbers at many stages:
    #     - Random weight initialization (how the network starts)
    #     - Random dropout (which neurons to temporarily disable)
    #     - Random shuffling of training batches
    #   Without a fixed seed, you get slightly different results each run.
    #   Setting seed=42 means we both get the same results — essential for
    #   scientific reproducibility. If I run this with seed=42, I should
    #   get the same model performance as you.
    seed: int = 42

    # -------------------------------------------------------------------------
    # Data paths
    # -------------------------------------------------------------------------
    data_dir: str = "./data"      # Where ERA5 NetCDF files live
    output_dir: str = "./output"  # Where models, plots, CSVs get saved
    lookup_table_dir: str = "./lookup_tables"  # Pre-computed EHI lookup tables

    # WHY MARCH–JULY?
    #   These are the months when Kolkata experiences extreme heat events:
    #   - March: pre-monsoon heating begins
    #   - April–May: peak heat season (40°C+ common)
    #   - June–July: monsoon onset, still dangerously hot with high humidity
    #   Training on only the "hot season" gives the model focused signal.
    months: List[int] = field(default_factory=lambda: [3, 4, 5, 6, 7])

    # Kolkata coordinates for solar geometry calculations
    kolkata_lat: float = 22.57
    kolkata_lon: float = 88.36

    # -------------------------------------------------------------------------
    # Temporal train/test split
    # -------------------------------------------------------------------------
    # WHY 2019?
    #   Train on 2005–2018 (14 years), test on 2019–2024 (6 years).
    #   This is a TEMPORAL split — we never let the model see future data
    #   during training. This prevents data leakage (a bug in the original
    #   notebooks where statistics were computed on the full dataset including
    #   test data, which artificially inflated performance).
    split_year: int = 2019

    # -------------------------------------------------------------------------
    # Weather GRU forecasting parameters
    # -------------------------------------------------------------------------
    # WHY 240 HOURS (10 DAYS) LOOKBACK?
    #   The model needs enough past weather to learn patterns. ERA5 reanalysis
    #   has a ~5-day lag from real-time, so 10 days of context gives the model
    #   the most recent available data plus 5 days of older context.
    #   Shorter lookback (e.g., 24h) would miss multi-day weather patterns.
    #   Longer lookback (e.g., 720h) would be too slow to train and risks
    #   the GRU "forgetting" recent patterns (vanishing gradient problem).
    lookback: int = 240          # hours (10 days)

    # WHY 168 HOURS (7 DAYS) FORECAST?
    #   This is the practical limit for useful weather forecasting. Beyond
    #   7 days, weather becomes essentially chaotic and unpredictable.
    #   For heat warnings, 7 days gives public health officials enough
    #   lead time to prepare cooling centers, issue advisories, etc.
    forecast_horizon: int = 168  # hours (7 days ahead)

    # WHAT WE PREDICT:
    #   - Ta_C: 2-meter air temperature in Celsius
    #   - RH: relative humidity (0–1 fraction)
    #   - SSRD: surface solar radiation downwards (W/m²)
    #   These three variables are everything the EHI-N* model needs to
    #   compute physiological heat stress zones.
    forecast_targets: List[str] = field(
        default_factory=lambda: ["Ta_C", "RH", "SSRD"]
    )

    # At these forecast lead times, we report the probability of reaching
    # Zone 5+ (Very Hot / Extreme) for alert purposes.
    alert_horizons: List[int] = field(
        default_factory=lambda: [24, 48, 72, 120, 168]  # 1, 2, 3, 5, 7 days
    )

    # -------------------------------------------------------------------------
    # Legacy classification parameters
    # -------------------------------------------------------------------------
    # The classifier labels an hour as "extreme" if T_resid (temperature
    # anomaly above the diurnal mean) exceeds the 95th percentile of the
    # TRAINING data. This means roughly 5% of training hours are "extreme".
    extreme_quantile: float = 0.95

    # WHY LEAD TIME?
    #   Without lead_time, the classifier answers "is it extreme RIGHT NOW?"
    #   That's nowcasting — useful only if you already have real-time data.
    #   By shifting the label forward by `lead_time` hours, the classifier
    #   instead answers "WILL it be extreme in X hours?" — true forecasting.
    #
    #   Example with lead_time=48 and lookback=240:
    #     Input:  weather from hours [i, i+240)   (past 10 days)
    #     Label:  extreme flag at hour i+240+48    (2 days into the future)
    #
    #   This gives public health officials 48 hours of advance warning
    #   to open cooling centers, issue advisories, and alert hospitals.
    #   Set to 0 for original nowcasting behavior.
    classifier_lead_time: int = 48  # hours ahead to predict (default 48h = 2 days)

    # -------------------------------------------------------------------------
    # Training hyperparameters
    # -------------------------------------------------------------------------
    # WHY 50 EPOCHS?
    #   An epoch = one full pass through all training data. 50 is enough for
    #   the GRU to converge (loss stops decreasing) without overfitting.
    #   EarlyStopping will halt training if validation loss hasn't improved
    #   for `patience` epochs, so 50 is an upper bound.
    epochs: int = 50

    # WHY BATCH SIZE 32?
    #   Each training step updates weights using a batch of 32 samples.
    #   Smaller batches = noisier gradients but better generalization.
    #   Larger batches = smoother training but can overfit. 32 is a
    #   well-established default that works across most problems.
    batch_size: int = 32

    # WHY LEARNING RATE 0.001?
    #   This controls how big each weight update step is. Too high (0.1)
    #   and the model overshoots and never converges (loss oscillates).
    #   Too low (0.00001) and training takes forever. 0.001 is Adam's
    #   default and works well for most GRU problems.
    learning_rate: float = 0.001

    # WHY PATIENCE 10?
    #   If validation loss doesn't improve for 10 consecutive epochs,
    #   stop training and restore the best weights. This prevents
    #   overfitting (where the model memorizes training data instead of
    #   learning generalizable patterns). 10 epochs gives enough time
    #   for the loss to recover from temporary plateaus.
    patience: int = 10

    # -------------------------------------------------------------------------
    # Weather GRU architecture
    # -------------------------------------------------------------------------
    # WHY 64 UNITS?
    #   Each GRU layer has 64 "neurons" (hidden units). More units = more
    #   capacity to learn complex patterns, but also more parameters to
    #   train and higher overfitting risk. 64 is a good balance for our
    #   12-feature input. 
    gru_units: int = 64

    # WHY 2 LAYERS?
    #   Stacking 2 GRU layers lets the network learn hierarchical patterns:
    #   - Layer 1 might learn hourly patterns (day/night cycles)
    #   - Layer 2 might learn multi-day patterns (heat waves building)
    #   More layers (3+) rarely help for weather data and slow training.
    gru_layers: int = 2

    # WHY 0.2 DROPOUT?
    #   During each training step, 20% of neurons are randomly "turned off".
    #   This forces the network to learn redundant representations — it can't
    #   rely on any single neuron. This prevents overfitting and improves
    #   generalization to unseen test data. 0.2 is a standard value;
    #   higher (0.5) can hurt performance by removing too much signal.
    dropout_rate: float = 0.2

    # -------------------------------------------------------------------------
    # Autoencoder architecture
    # -------------------------------------------------------------------------
    # WHY 24-HOUR WINDOW?
    #   The autoencoder reconstructs one full day of weather data. If it
    #   can't reconstruct a day well (high reconstruction error), that day
    #   is "anomalous" — possibly an extreme heat event. 24h captures one
    #   complete diurnal cycle (sunrise → peak heat → sunset → night cooling).
    ae_window: int = 24  # 1-day window

    # WHY 32 UNITS (SMALLER THAN WEATHER GRU)?
    #   The autoencoder's bottleneck must be SMALL — it compresses 24 hours
    #   of 5 features (120 values) into just 32 numbers. This forces it to
    #   learn the essential patterns of "normal" weather. If the bottleneck
    #   were too large, it could memorize everything including anomalies.
    ae_gru_units: int = 32

    @property
    def Qm(self) -> float:
        """Metabolic heat flux in W/m². Looked up from the MET_LEVELS table."""
        return float(MET_LEVELS[self.met_level])

    @property
    def n_forecast_targets(self) -> int:
        """How many variables we're predicting (3: Ta_C, RH, SSRD)."""
        return len(self.forecast_targets)

    @property
    def forecast_input_cols(self) -> List[str]:
        """
        Input features for the Weather GRU forecaster — 12 features.

        WHY THESE 12 FEATURES?
          The GRU sees a 240-hour window of these 12 values per hour.
          Each feature adds different information:

          1. Ta_C          — raw temperature: the main thing we're forecasting
          2. RH            — raw humidity: affects how dangerous heat is
          3. SSRD          — solar radiation: drives daytime heating
          4. T_resid       — temperature ANOMALY above the typical daily pattern.
                             If T_resid is +3°C, it's 3° hotter than usual for
                             that hour. This helps the GRU focus on unusual events
                             rather than the predictable day/night cycle.
          5. RH_resid      — same idea for humidity
          6. hour_sin      — sine encoding of hour (0–23). WHY SINE?
          7. hour_cos        Because hour 23 is close to hour 0, but numerically
                             they're far apart. sin/cos makes them adjacent on a
                             circle, which helps the GRU learn diurnal patterns.
          8. doy_sin       — sine encoding of day-of-year (1–365). Same trick
          9. doy_cos         for seasonal patterns (late December ≈ early January).
         10. sun_flag      — binary: is the sun above the horizon? (1=yes, 0=no)
                             Simple but powerful — separates day from night physics.
         11. anomaly_score — autoencoder reconstruction error. Higher values mean
                             current weather is UNUSUAL compared to what the
                             autoencoder learned as "normal". This tells the GRU
                             "something abnormal is happening — adjust forecast."
         12. latent_z      — autoencoder bottleneck representation. A compressed
                             "fingerprint" of the current weather state. Gives
                             the GRU additional context about what TYPE of pattern
                             is happening (e.g., pre-monsoon buildup vs heat wave).

        WHY ADD AUTOENCODER FEATURES?
          The autoencoder runs FIRST and learns what normal weather looks like.
          When it encounters unusual patterns (heat wave building, sudden humidity
          shift), its reconstruction error spikes. By feeding this anomaly signal
          into the Weather GRU, the forecaster gets an early warning that
          conditions are deviating from normal — which may improve forecasts
          during extreme events (exactly when accuracy matters most).
        """
        return [
            "Ta_C",           # raw 2m air temperature (°C)
            "RH",             # relative humidity (0–1)
            "SSRD",           # surface solar radiation downwards (W/m²)
            "T_resid",        # temperature anomaly (°C above diurnal mean)
            "RH_resid",       # humidity anomaly (above diurnal mean)
            "hour_sin",       # cyclical hour encoding (sin)
            "hour_cos",       # cyclical hour encoding (cos)
            "doy_sin",        # cyclical day-of-year encoding (sin)
            "doy_cos",        # cyclical day-of-year encoding (cos)
            "sun_flag",       # 1 if sun is above horizon, 0 if night
            "anomaly_score",  # autoencoder reconstruction error (higher = unusual)
            "latent_z",       # autoencoder compressed weather "fingerprint"
        ]

    @property
    def feature_cols(self) -> List[str]:
        """
        Features for the legacy GRU classifier (original model).

        This classifier uses 9 features per timestep, including outputs
        from the autoencoder (latent_z, anomaly_score). That's why the
        autoencoder must run FIRST — it produces the latent_z and
        anomaly_score columns that the classifier consumes.
        """
        return [
            "EHI_zone",       # EHI-N* physiological zone (1–6)
            "hour_sin",       # cyclical hour
            "hour_cos",
            "sun_flag",       # day/night flag
            "T_resid",        # temperature anomaly
            "RH_resid",       # humidity anomaly
            "latent_z",       # autoencoder bottleneck value (compressed representation)
            "SSRD",           # solar radiation
            "anomaly_score",  # autoencoder reconstruction error (higher = more unusual)
        ]

def parse_args() -> Config:
    """Parse CLI arguments into a Config object."""
    parser = argparse.ArgumentParser(
        description="HeatRadar Nowcast — Weather GRU → EHI-N* zone forecast"
    )
    parser.add_argument(
        "--mode",
        choices=["forecast", "classifier", "autoencoder", "full", "inference"],
        default="forecast",
        help="Pipeline mode (default: forecast)",
    )
    parser.add_argument(
        "--met_level",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=4,
        help="MET activity level 1-6 (default: 4 = 240 W/m²)",
    )
    parser.add_argument("--data_dir", default="./data", help="ERA5 data directory")
    parser.add_argument("--output_dir", default="./output", help="Output directory")
    parser.add_argument("--lookup_dir", default="./lookup_tables",
                        help="Directory with pre-computed EHI lookup tables")
    parser.add_argument(
        "--split_year", type=int, default=2019, help="Train/test split year"
    )
    parser.add_argument(
        "--lookback", type=int, default=240, help="Lookback window in hours (default: 240 = 10 days)"
    )
    parser.add_argument(
        "--horizon", type=int, default=168, help="Forecast horizon in hours (default: 168 = 7 days)"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--lead_time", type=int, default=48,
        help="Classifier lead time in hours (default: 48 = predict 2 days ahead). "
             "Set to 0 for nowcasting (is it extreme NOW?)."
    )

    args = parser.parse_args()

    return Config(
        mode=args.mode,
        met_level=args.met_level,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        lookup_table_dir=args.lookup_dir,
        split_year=args.split_year,
        lookback=args.lookback,
        forecast_horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        classifier_lead_time=args.lead_time,
    )


# =============================================================================
# Section 1: EHI Computation (Physics-Based Heat Stress Model)
# =============================================================================
#
# EHI-N* (Environmental Heat Index, New Star) is a physics-based model that
# computes the body's equilibrium temperature given:
#   - Air temperature (Ta)
#   - Relative humidity (RH)
#   - Metabolic heat production (Qm, from activity level)
#   - Solar radiation absorbed by the body (Qs)
#   - Body dimensions (height, mass → surface area)
#
# WHAT IT DOES:
#   Solves the human thermoregulatory heat balance equation. The body gains
#   heat from metabolism + solar radiation, and loses heat via sweating,
#   convection, and radiation. EHI-N* finds the steady-state core temperature.
#
# THE 6 ZONES:
#   Zone 1 ("Cool")     — body easily maintains 37°C, no stress
#   Zone 2 ("Comfort")  — comfortable, minimal sweating needed
#   Zone 3 ("Warm")     — noticeable sweating, still manageable
#   Zone 4 ("Hot")      — significant heat stress, take precautions
#   Zone 5 ("Very Hot") — dangerous, limit exposure time
#   Zone 6 ("Extreme")  — life-threatening, body cannot maintain 37°C
#
# Each zone corresponds to a different physiological cooling mechanism
# becoming saturated (skin vasodilation, sweating rate limits, etc.)
#
# Note: WHY NOT JUST USE TEMPERATURE?
#   40°C at 20% humidity (dry) is Zone 3 — uncomfortable but manageable.
#   40°C at 80% humidity (humid) is Zone 5+ — sweat can't evaporate,
#   body temperature rises uncontrollably. EHI-N* captures this correctly.
#
# Reference: Lu & Romps (2022), "Extending the Heat Index", JAMC
# Implementation: heatindex_ek.py (Elif Kilic, with Qs=0 fix for shade)
# =============================================================================

# EHI computation uses pre-computed lookup tables only.
# heatindex_ek.py is NOT required — Simone runs this with lookup tables.
modifiedheatindex = None  # Not used; kept as sentinel for any stray checks


# =============================================================================
# Lookup Table Support
# =============================================================================
# Pre-computed EHI lookup tables eliminate the Numba/NumbaMinpack dependency.
# Run generate_lookup_tables.py once to create them, then the nowcast can
# use fast scipy interpolation instead of solving thermoregulatory equations.

class EHILookupTable:
    """Loads pre-computed EHI tables and provides fast interpolation."""

    def __init__(self, filepath: str):
        from scipy.interpolate import RegularGridInterpolator

        data = np.load(filepath)
        self.Ta_grid = data["Ta_grid"]
        self.RH_grid = data["RH_grid"]
        self.Qs_grid = data["Qs_grid"]
        self.Qm = float(data["Qm"])
        self.H = float(data["H"])
        self.M = float(data["M"])

        self._ehi_shade = RegularGridInterpolator(
            (self.Ta_grid, self.RH_grid), data["ehi_shade"],
            method="linear", bounds_error=False, fill_value=None,
        )
        self._ehi_sun = RegularGridInterpolator(
            (self.Ta_grid, self.RH_grid), data["ehi_sun"],
            method="linear", bounds_error=False, fill_value=None,
        )
        self._zone = RegularGridInterpolator(
            (self.Ta_grid, self.RH_grid, self.Qs_grid),
            data["zone_table"].astype(np.float64),
            method="nearest", bounds_error=False, fill_value=None,
        )

    def _clip(self, Ta_K, RH, Qs=None):
        Ta_K = np.clip(Ta_K, self.Ta_grid[0], self.Ta_grid[-1])
        RH = np.clip(RH, self.RH_grid[0], self.RH_grid[-1])
        if Qs is not None:
            Qs = np.clip(Qs, self.Qs_grid[0], self.Qs_grid[-1])
            return Ta_K, RH, Qs
        return Ta_K, RH

    def lookup_ehi_value(self, Ta_K, RH, mrt):
        Ta_K, RH = self._clip(Ta_K, RH)
        interp = self._ehi_sun if mrt > 0 else self._ehi_shade
        return float(interp([[Ta_K, RH]])[0])

    def lookup_ehi_array(self, Ta_K, RH, mrt):
        Ta_K, RH = self._clip(Ta_K, RH)
        interp = self._ehi_sun if mrt > 0 else self._ehi_shade
        points = np.column_stack([Ta_K, RH])
        return interp(points)

    def lookup_zone(self, Ta_K, RH, Qs):
        Ta_K, RH, Qs = self._clip(Ta_K, RH, Qs)
        return int(np.round(self._zone([[Ta_K, RH, Qs]])[0]))

    def lookup_zones_array(self, Ta_K, RH, Qs):
        Ta_K, RH, Qs = self._clip(Ta_K, RH, Qs)
        points = np.column_stack([Ta_K, RH, Qs])
        return np.round(self._zone(points)).astype(np.int32)


# Module-level lookup table (loaded when pipeline starts)
_lookup_table: Optional[EHILookupTable] = None


def load_lookup_table(config) -> None:
    """Load pre-computed EHI lookup table for the configured MET level. Raises if not found."""
    global _lookup_table

    lookup_dir = config.lookup_table_dir
    search_paths = [
        lookup_dir,
        os.path.join(config.data_dir, "lookup_tables"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "lookup_tables"),
    ]

    filename = f"ehi_lookup_met{config.met_level}.npz"
    for path in search_paths:
        filepath = os.path.join(path, filename)
        if os.path.isfile(filepath):
            _lookup_table = EHILookupTable(filepath)
            print(f"  Loaded EHI lookup table: {filepath}")
            print(f"    Grid: Ta {len(_lookup_table.Ta_grid)} pts, "
                  f"RH {len(_lookup_table.RH_grid)} pts, "
                  f"Qs {len(_lookup_table.Qs_grid)} pts")
            return

    raise FileNotFoundError(
        f"EHI lookup table not found for MET level {config.met_level}.\n"
        f"Searched:\n" + "\n".join(f"  {os.path.join(p, filename)}" for p in search_paths) + "\n"
        f"Run: python generate_lookup_tables.py --met_levels {config.met_level}"
    )


def ssrd_to_Qs(ssrd_wm2, solar_altitude_deg, absorptivity: float = 0.7):
    """
    Convert SSRD (W/m²) + solar altitude to absorbed solar heat Qs (W/m²)
    for a standing person. Accepts scalars or numpy arrays.

    Physics (Fanger 1970, ISO 7726):
      f_proj = 0.308 × sin(altitude)
        — sin because altitude=0° (horizon) → no overhead projection → f_proj≈0,
          altitude=90° (overhead) → maximum overhead projection → f_proj=0.308.
      Qs = absorptivity × f_proj × SSRD

    Returns 0 at night (SSRD≤0 or altitude≤0).
    """
    ssrd = np.asarray(ssrd_wm2, dtype=np.float64)
    alt  = np.asarray(solar_altitude_deg, dtype=np.float64)
    alt_rad = np.deg2rad(np.clip(alt, 0.0, 90.0))
    f_proj = 0.308 * np.sin(alt_rad)
    Qs = np.where((ssrd > 0) & (alt > 0), absorptivity * f_proj * ssrd, 0.0)
    # Return scalar if scalar input, array otherwise
    return float(Qs) if Qs.ndim == 0 else np.maximum(Qs, 0.0)


def compute_ehi_value(Ta_K: float, RH: float, Qm: float, mrt: float,
                      H: float, M: float) -> float:
    """
    Compute a single EHI-N* value in Kelvin.

    This is the core heat index computation — it solves the human
    thermoregulatory equation to find what core temperature the body
    would reach under these environmental conditions.

    A healthy person's core temp is ~310 K (37°C). If EHI returns 312 K,
    the person's body would stabilize at 39°C — that's heat stress.
    If EHI returns > 315 K, the body CAN'T stabilize — that's Zone 6 (lethal).
    """
    return _lookup_table.lookup_ehi_value(Ta_K, RH, mrt)


def compute_ehi_array(Ta_K: np.ndarray, RH: np.ndarray, Qm: float,
                      mrt: float, H: float, M: float) -> np.ndarray:
    """
    Vectorized EHI computation for arrays. Returns EHI in Kelvin.

    WHY VECTORIZED?
      Computing EHI for 100,000+ hourly data points one-by-one would be
      very slow. Uses scipy RegularGridInterpolator on pre-computed lookup
      tables — ~0.1 seconds for the full dataset, no Numba required.
    """
    return _lookup_table.lookup_ehi_array(
        Ta_K.astype(np.float64), RH.astype(np.float64), mrt
    )


def compute_zone(Ta_K: float, RH: float, Qm: float, H: float,
                 M: float, Qs: float = 0.0) -> int:
    """
    Compute EHI-N* physiological zone (1–6) for a single data point.

    HOW ZONE ASSIGNMENT WORKS:
      The EHI model solves the heat balance equation. Depending on which
      cooling mechanism is the "limiting factor" (the bottleneck), you're
      in a different zone:

      Zone 1 (phi)   — skin vasodilation alone handles the heat
      Zone 2 (Rf)    — sweating has started but is well within limits
      Zone 3 (Rf)    — sweating is significant (Pa test distinguishes 2 vs 3)
      Zone 4 (Rs)    — sweating at maximum sustainable rate
      Zone 5 (Rs*)   — sweating can't keep up, core temp rising
      Zone 6 (dTcdt) — thermoregulation has FAILED, core temp rising fast

    Parameters
    ----------
    Ta_K : float — Air temperature in Kelvin (e.g., 313.15 = 40°C)
    RH : float — Relative humidity as fraction (0.8 = 80%)
    Qm : float — Metabolic heat production (W/m²), from MET level
    H : float — Body height (m)
    M : float — Body mass (kg)
    Qs : float — Solar heat load on the body (W/m²), computed by ssrd_to_Qs().
                  Default 0.0 = full shade (no solar radiation hitting the body).
                  Typical sunny value: 50-150 W/m² depending on sun angle.
    """
    return _lookup_table.lookup_zone(Ta_K, RH, Qs)


def compute_zones_array(Ta_K: np.ndarray, RH: np.ndarray, Qm: float,
                        H: float, M: float,
                        Qs: np.ndarray = None) -> np.ndarray:
    """
    Compute EHI-N* zones for an array of data points via lookup table.
    Uses vectorized scipy interpolation (~0.1 seconds for 100k points).
    """
    if Qs is None:
        Qs = np.zeros(len(Ta_K))
    return _lookup_table.lookup_zones_array(
        Ta_K.astype(np.float64), RH.astype(np.float64),
        Qs.astype(np.float64)
    )


# =============================================================================
# Section 2: Data Loading
# =============================================================================
#
# ERA5 REANALYSIS — WHAT IS IT?
#   ERA5 is a global weather dataset produced by ECMWF (European Centre for
#   Medium-Range Weather Forecasts). It's not a forecast — it's a "best estimate"
#   of what the weather actually was, computed by combining observations from
#   weather stations, satellites, radiosondes, etc. with a physics-based model.
#
#   It provides hourly data at 0.25° resolution (~25 km) globally from 1979
#   to present, with a ~5-day lag. We use:
#     - T2m: 2-meter air temperature (what you'd feel standing outside)
#     - D2m: 2-meter dewpoint temperature (indicator of moisture)
#     - SSRD: Surface Solar Radiation Downwards (W/m²) — how much sun hits the ground
#
# ARCO-ERA5 (ANALYSIS-READY CLOUD-OPTIMIZED):
#   Google hosts the full ERA5 dataset on Google Cloud Storage as Zarr arrays.
#   This is where our download_ssrd.py script pulls SSRD data from. It's free,
#   public, and doesn't require authentication. Much faster than ECMWF's CDS API.
#
# NETCDF FILES:
#   Our local data is stored as NetCDF (.nc) files — one per month per year.
#   NetCDF is the standard format for climate data. Each file is a small
#   ~100 KB file containing hourly T, dewpoint, RH, and optionally SSRD
#   for a single grid point (Kolkata: 22.5°N, 88.25°E).
# =============================================================================

def load_era5_data(config: Config) -> pd.DataFrame:
    """
    Load ERA5 data from NetCDF files.

    Expects files named: {data_dir}/era5_kolkata_{year}_{month:02d}.nc
    Each file should have variables: Ta (°C), Td (°C), RH (0-1), and optionally SSRD.

    Returns DataFrame with columns: [Ta_K, Ta_C, RH, SSRD]

    IMPORTANT DETAILS:
      - Uses xr.open_mfdataset to open all files at once (handles merging)
      - combine="nested" + concat_dim="time" because files from different
        months may not align perfectly in coordinates
      - sortby("time") + unique index dedup fixes non-monotonic timestamps
        that can occur when files overlap at month boundaries
    """
    data_dir = Path(config.data_dir)
    files = []

    # Try glob pattern for individual month files
    for year in range(2005, 2025):
        for month in config.months:
            pattern = f"era5_kolkata_{year}_{month:02d}.nc"
            matches = list(data_dir.rglob(pattern))
            files.extend(matches)

    if not files:
        # Try a single combined file
        combined = list(data_dir.rglob("era5_kolkata_*.nc"))
        if combined:
            files = sorted(combined)
        else:
            raise FileNotFoundError(
                f"No ERA5 NetCDF files found in {data_dir}. "
                f"Expected files like era5_kolkata_2020_05.nc"
            )

    files = sorted(set(files))
    print(f"  Found {len(files)} NetCDF files")

    ds = xr.open_mfdataset(
        [str(f) for f in files],
        combine="nested",
        concat_dim="time",
        engine="netcdf4",
        coords="minimal",
        compat="override",
    )
    # Sort by time to fix non-monotonic index from mixed files
    ds = ds.sortby("time")
    # Drop duplicate timestamps (from overlapping files)
    _, unique_idx = np.unique(ds.time.values, return_index=True)
    ds = ds.isel(time=unique_idx)

    # Select nearest grid point to Kolkata
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"

    if len(ds[lat_name]) > 1 or len(ds[lon_name]) > 1:
        ds = ds.sel(
            {lat_name: config.kolkata_lat, lon_name: config.kolkata_lon},
            method="nearest",
        )

    # Build DataFrame
    time = pd.to_datetime(ds.time.values)

    # Detect temperature variable
    ta_var = None
    for name in ["Ta", "t2m", "2m_temperature"]:
        if name in ds.data_vars:
            ta_var = name
            break
    if ta_var is None:
        raise KeyError(f"No temperature variable found. Available: {list(ds.data_vars)}")

    Ta_vals = ds[ta_var].values.flatten()

    # Check if temperature is in Kelvin (> 200) or Celsius
    if np.nanmean(Ta_vals) > 200:
        Ta_C = Ta_vals - 273.15
        Ta_K = Ta_vals
    else:
        Ta_C = Ta_vals
        Ta_K = Ta_vals + 273.15

    # RH (Relative Humidity): try direct variable, else compute from dewpoint.
    #
    # WHY COMPUTE RH FROM DEWPOINT?
    #   ERA5 provides dewpoint temperature (Td) rather than RH directly.
    #   Dewpoint = the temperature at which air becomes saturated (100% RH).
    #   If Td is close to T, the air is very humid. If Td is much lower, it's dry.
    #
    # MAGNUS FORMULA:
    #   RH = e_actual / e_saturated
    #   where e = 6.112 × exp(17.625 × T / (243.04 + T))
    #   This is an approximation of the Clausius-Clapeyron equation that
    #   relates temperature to the amount of water vapor air can hold.
    if "RH" in ds.data_vars:
        RH_vals = ds["RH"].values.flatten()
    else:
        td_var = None
        for name in ["Td", "d2m", "2m_dewpoint_temperature"]:
            if name in ds.data_vars:
                td_var = name
                break
        if td_var is None:
            raise KeyError("No RH or dewpoint variable found")
        Td_vals = ds[td_var].values.flatten()
        if np.nanmean(Td_vals) > 200:
            Td_C = Td_vals - 273.15
        else:
            Td_C = Td_vals
        # Magnus formula for saturation vapor pressure
        e_sat = 6.112 * np.exp(17.625 * Ta_C / (243.04 + Ta_C))
        e_act = 6.112 * np.exp(17.625 * Td_C / (243.04 + Td_C))
        RH_vals = np.clip(e_act / e_sat, 0.0, 1.0)

    # Ensure RH is 0-1 (not percentage)
    if np.nanmean(RH_vals) > 1.5:
        RH_vals = RH_vals / 100.0

    # SSRD: Surface Solar Radiation Downwards
    #
    # WHAT IS SSRD?
    #   The total amount of solar radiation (sunlight) hitting the ground surface,
    #   in Watts per square meter (W/m²). At peak noon in clear sky tropical
    #   conditions, this can reach ~1000 W/m². At night it's 0. On cloudy days
    #   it's reduced.
    #
    # WHY DO WE NEED IT?
    #   Solar radiation is a MAJOR driver of heat stress. A person standing in
    #   direct sun absorbs significant heat (50-150 W/m² on the body surface).
    #   This can push someone from Zone 3 (warm) to Zone 5 (dangerous).
    #   Without SSRD, we can only model shade conditions.
    #
    # UNIT CONVERSION:
    #   ERA5 stores SSRD as accumulated energy per hour in J/m² (Joules).
    #   We convert to instantaneous power W/m² by dividing by 3600 seconds.
    #   If the values are already in W/m² (< 2000), we skip the conversion.
    ssrd_var = None
    for name in ["SSRD", "ssrd", "surface_solar_radiation_downwards"]:
        if name in ds.data_vars:
            ssrd_var = name
            break

    if ssrd_var is not None:
        SSRD_vals = ds[ssrd_var].values.flatten()
        # ERA5 SSRD is accumulated J/m² per hour; convert to W/m² (÷ 3600)
        if np.nanmax(SSRD_vals) > 2000:
            SSRD_vals = SSRD_vals / 3600.0
        SSRD_vals = np.clip(SSRD_vals, 0.0, None)  # no negative radiation
        print(f"  SSRD loaded: {np.nanmin(SSRD_vals):.0f}–{np.nanmax(SSRD_vals):.0f} W/m²")
    else:
        # If no SSRD in the data files, we'll estimate it later from solar
        # geometry in compute_solar_features(). This is less accurate but
        # allows the pipeline to run without SSRD data.
        print("  WARNING: No SSRD variable found — will be estimated from solar geometry")
        SSRD_vals = np.full_like(Ta_C, np.nan)

    df = pd.DataFrame(
        {"Ta_K": Ta_K, "Ta_C": Ta_C, "RH": RH_vals, "SSRD": SSRD_vals},
        index=time,
    )
    df.index.name = "datetime"

    # Convert to IST (UTC+5:30) for Kolkata
    df.index = df.index + pd.Timedelta(hours=5, minutes=30)
    df.index.name = "datetime_IST"

    # Filter to configured months
    df = df[df.index.month.isin(config.months)]

    # Drop NaN rows (except SSRD which may be NaN)
    essential = ["Ta_K", "Ta_C", "RH"]
    n_before = len(df)
    df = df.dropna(subset=essential)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} NaN rows")

    print(f"  Loaded {len(df)} hourly records")
    print(f"  Time range: {df.index[0]} → {df.index[-1]}")
    print(f"  Temp range: {df['Ta_C'].min():.1f}°C – {df['Ta_C'].max():.1f}°C")

    ds.close()
    return df


# =============================================================================
# Section 3: Solar Geometry (PySolar)
# =============================================================================
#
# WHY SOLAR GEOMETRY?
#   The sun's position in the sky determines two critical things:
#   1. WHETHER the sun is up (sun_flag) — day vs night
#   2. The sun's ALTITUDE ANGLE — how high the sun is above the horizon
#
#   The altitude angle matters for heat stress because:
#   - Low sun (near horizon, e.g., 6 AM): your body casts a LONG shadow,
#     meaning more of your frontal area is exposed to direct sunlight
#   - High sun (noon): your body casts a SHORT shadow, less frontal exposure
#     BUT the total radiation (SSRD) is stronger
#
#   PySolar uses astronomical equations to compute exact sun position for
#   any latitude/longitude/time. For Kolkata (22.5°N), the sun reaches
#   ~85° altitude at summer solstice noon and ~45° in winter.
# =============================================================================

def compute_solar_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Compute solar altitude and sun flag using PySolar.

    If PySolar is not installed, estimates sun flag from SSRD (>20 W/m² = sun).
    Solar altitude is used by EHI-N* for Qsolar computation.

    FALLBACK HIERARCHY (if PySolar not installed):
      1. Use SSRD values (>20 W/m² = sun is up)
      2. If SSRD also missing, use hour-of-day heuristic (6-18 IST = daylight)
    """
    lat = config.kolkata_lat
    lon = config.kolkata_lon

    if _HAS_PYSOLAR:
        print("  Computing solar altitude via PySolar...")
        altitudes = np.zeros(len(df))
        for i, timestamp in enumerate(df.index):
            # PySolar needs timezone-aware datetime
            aware_dt = timestamp.to_pydatetime().replace(
                tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30))
            )
            altitudes[i] = get_altitude(lat, lon, aware_dt)
        df["solar_altitude"] = altitudes
        df["sun_flag"] = (altitudes > 0).astype(float)
        print(f"  Solar altitude range: {altitudes.min():.1f}° – {altitudes.max():.1f}°")
    else:
        # Fallback: estimate from SSRD
        if "SSRD" in df.columns and df["SSRD"].notna().any():
            print("  Estimating sun flag from SSRD (>20 W/m²)...")
            df["sun_flag"] = (df["SSRD"] > 20).astype(float)
            # Rough altitude estimate from SSRD (normalized 0–90°)
            ssrd_max = df["SSRD"].quantile(0.99)
            if ssrd_max > 0:
                df["solar_altitude"] = np.clip(df["SSRD"] / ssrd_max * 90, 0, 90)
            else:
                df["solar_altitude"] = 0.0
        else:
            # Last resort: use hour-of-day heuristic for tropical location
            print("  Estimating sun flag from hour of day (6-18 IST)...")
            hour = df.index.hour
            df["sun_flag"] = ((hour >= 6) & (hour <= 18)).astype(float)
            df["solar_altitude"] = np.where(
                df["sun_flag"] > 0,
                np.sin(np.pi * (hour - 6) / 12) * 75,  # rough tropical arc
                0.0,
            )

    # If SSRD is missing or mostly NaN, estimate from solar altitude
    ssrd_nan_rate = df["SSRD"].isna().mean()
    if ssrd_nan_rate > 0.1:  # more than 10% NaN → replace all with estimate
        print(f"  SSRD has {ssrd_nan_rate:.0%} NaN — estimating from solar altitude (clear-sky approx)...")
        alt_rad = np.deg2rad(np.clip(df["solar_altitude"].values, 0, 90))
        df["SSRD"] = np.where(alt_rad > 0, 1000.0 * np.sin(alt_rad), 0.0)

    sun_hours = df["sun_flag"].sum()
    print(f"  Sun hours: {sun_hours}/{len(df)} ({sun_hours/len(df)*100:.1f}%)")

    return df


# =============================================================================
# Section 4: Preprocessing & Feature Engineering
# =============================================================================
#
# The most important principle in temporal ML: NEVER use test data to compute
# anything that the model sees during training. This is called "data leakage"
# and it artificially inflates performance metrics.
#
# ALL statistics (means, thresholds, scaler parameters) are computed on
# TRAIN data only (years < split_year), then applied to the full dataset.
# This ensures the model never "sees" future data during training.
# =============================================================================

def compute_residuals(df: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
    """
    Compute diurnal residuals using TRAIN-only diurnal mean.

    WHAT ARE RESIDUALS?
      Temperature follows a predictable daily pattern: cool at night,
      warm in the afternoon. This is the "diurnal cycle". The RESIDUAL
      is how much a particular hour deviates from its typical value.

      Example: If 2 PM typically averages 35°C but today it's 38°C,
      the residual is +3°C. This +3°C anomaly is the signal that
      something unusual (potentially dangerous) is happening.

    WHY TRAIN-ONLY MEAN?
      Computing the diurnal mean from ALL years (including the test period)
      would leak future information into training — a form of data leakage.
      Instead, the mean is computed from training years only (2005-2018),
      ensuring the model never benefits from future observations.
    """
    train_df = df.loc[train_mask]

    # Mean diurnal cycle from TRAIN only
    diurnal_T_mean = train_df.groupby(train_df.index.hour)["Ta_C"].mean()
    diurnal_RH_mean = train_df.groupby(train_df.index.hour)["RH"].mean()

    # Apply to full dataset
    df["T_resid"] = df["Ta_C"] - df.index.hour.map(diurnal_T_mean)
    df["RH_resid"] = df["RH"] - df.index.hour.map(diurnal_RH_mean)

    return df


def compute_ehi_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Add EHI-N* shade, sun, and zone features.

    WHAT THIS COMPUTES:
      - EHI_shade_C: heat index in shade (Qs=0, mrt=0)
      - EHI_sun_C: heat index in direct sun (mrt=1)
      - EHI_active_C: shade or sun value per timestep based on sun_flag
      - EHI_zone: physiological zone (1–6) using real Qs from SSRD + solar altitude

    WHY REAL Qs FOR ZONES?
      The lookup tables have a full 3D zone_table indexed by (Ta, RH, Qs).
      Passing Qs=0 everywhere ignores the solar dimension and always reads
      from the Qs=0 slice — equivalent to shade-only conditions even at noon.
      Using real Qs from SSRD + solar altitude activates the full table and
      gives physically correct zones (e.g., Zone 5 at noon in the sun vs
      Zone 3 in shade at the same temperature and humidity).
    """
    Ta_K = df["Ta_K"].values
    RH = df["RH"].values
    Qm = config.Qm
    H = config.body_height_m
    M = config.body_mass_kg

    print("  Computing EHI-N* (shade)...")
    ehi_shade_K = compute_ehi_array(Ta_K, RH, Qm, mrt=0.0, H=H, M=M)
    df["EHI_shade_C"] = ehi_shade_K - 273.15

    print("  Computing EHI-N* (sun)...")
    ehi_sun_K = compute_ehi_array(Ta_K, RH, Qm, mrt=1.0, H=H, M=M)
    df["EHI_sun_C"] = ehi_sun_K - 273.15

    # Active EHI: use sun_flag to pick shade or sun value per timestep
    sun = df["sun_flag"].values.astype(bool)
    ehi_active = np.where(sun, ehi_sun_K, ehi_shade_K)
    df["EHI_active_C"] = ehi_active - 273.15

    # Qs: real solar heat load on body from SSRD + solar altitude.
    # This is what activates the Qs dimension of the lookup zone_table.
    # Night hours get Qs=0 automatically (SSRD=0 → ssrd_to_Qs returns 0).
    print("  Computing EHI-N* zones (with solar Qs from SSRD)...")
    ssrd_vals = df["SSRD"].fillna(0).values
    if "solar_altitude" in df.columns:
        Qs_arr = ssrd_to_Qs(ssrd_vals, df["solar_altitude"].values)
    else:
        # solar_altitude not yet available — use 45° as midday proxy for sun hours
        alt_proxy = np.where(sun, 45.0, 0.0)
        Qs_arr = ssrd_to_Qs(ssrd_vals, alt_proxy)

    df["EHI_zone"] = compute_zones_array(Ta_K, RH, Qm, H=H, M=M, Qs=Qs_arr)

    zone_dist = df["EHI_zone"].value_counts().sort_index()
    print(f"  Zone distribution: {dict(zone_dist)}")

    return df


def compute_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical hour and day-of-year features using sine/cosine encoding.

    WHY SINE/COSINE INSTEAD OF RAW NUMBERS?
      If we encode hour as 0-23, the model thinks hour 23 and hour 0 are
      very far apart (23 units away). But in reality, 11 PM and midnight
      are only 1 hour apart! Sine/cosine encoding wraps time into a circle:

        hour_sin = sin(2π × hour / 24)
        hour_cos = cos(2π × hour / 24)

      Now hour 23 and hour 0 are adjacent on the unit circle. Same logic
      applies to day-of-year: December 31 and January 1 should be close.

    WHY BOTH SIN AND COS?
      sin alone can't distinguish 6 AM from 6 PM (both have sin≈0).
      Adding cos resolves this ambiguity. Together, (sin, cos) uniquely
      identifies every hour/day on the circle.
    """
    hour = df.index.hour
    doy = df.index.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    return df


def create_extreme_labels(df: pd.DataFrame, train_mask: np.ndarray,
                          config: Config) -> pd.DataFrame:
    """
    Label extreme heat events — binary (0 = normal, 1 = extreme).

    An hour is labeled "extreme" if its temperature residual (T_resid)
    exceeds the 95th percentile of the TRAINING data's residuals.

    WHY 95TH PERCENTILE?
      This means ~5% of training hours are "extreme" — the hottest 5%.
      This creates a meaningful but rare event to detect. Using a higher
      threshold (99th pctl) would make events too rare to learn.

    WHY TRAIN-ONLY THRESHOLD?
      The threshold must be computed from training data only. If the full
      dataset (including test years) were used, the model's labels would
      be influenced by future temperature distributions — data leakage.

      A good sanity check: train extreme rate should be ≈ 5% (by
      definition of the 95th percentile), and test extreme rate should
      DIFFER (usually 5-8%, because recent years tend to be hotter
      due to climate change).
    """
    train_resid = df.loc[train_mask, "T_resid"]
    threshold = np.nanquantile(train_resid.values, config.extreme_quantile)

    df["extreme"] = (df["T_resid"] >= threshold).astype(int)

    train_rate = df.loc[train_mask, "extreme"].mean()
    test_rate = df.loc[~train_mask, "extreme"].mean()

    print(f"  Extreme threshold: T_resid >= {threshold:.3f}°C")
    print(f"  Train extreme rate: {train_rate:.4f} ({train_rate*100:.1f}%)")
    print(f"  Test extreme rate:  {test_rate:.4f} ({test_rate*100:.1f}%)")

    return df


def build_feature_matrix(df: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Full preprocessing pipeline. Returns (df_with_features, train_mask).

    Pipeline (matches architecture diagram):
      1. Temporal split mask
      2. Solar geometry (PySolar) → solar_altitude, sun_flag
      3. Diurnal residuals via LOWESS (train-only mean)
      4. EHI-N* features (shade/sun/zone) — physics, no training
      5. Cyclical time features
      6. Extreme event labels (train-only threshold)
    """
    print("\n--- Feature Engineering ---")

    # 1. Temporal split
    train_mask = df.index.year < config.split_year
    n_train = train_mask.sum()
    n_test = (~train_mask).sum()
    print(f"  Split year: {config.split_year}")
    print(f"  Train: {n_train} rows | Test: {n_test} rows")

    # 2. Solar geometry
    print("  Computing solar geometry...")
    df = compute_solar_features(df, config)

    # 3. Diurnal residuals
    print("  Computing diurnal residuals (train-only mean)...")
    df = compute_residuals(df, train_mask)

    # 4. EHI-N* features
    df = compute_ehi_features(df, config)

    # 5. Cyclical time features
    df = compute_cyclical_time_features(df)

    # 6. Extreme labels
    df = create_extreme_labels(df, train_mask, config)

    # Initialize autoencoder-derived columns as zeros (filled later)
    df["latent_z"] = 0.0
    df["anomaly_score"] = 0.0

    print(f"  Total columns: {list(df.columns)}")

    return df, train_mask


# =============================================================================
# Section 5: Scaling & Sequences
# =============================================================================
#
# WHY SCALE THE DATA?
#   Neural networks work best when all input features are on a similar scale.
#   Without scaling:
#     - Temperature: 20–45 (range of 25)
#     - RH: 0.2–1.0 (range of 0.8)
#     - SSRD: 0–1000 (range of 1000!)
#
#   The GRU would focus almost entirely on SSRD because its values are so
#   much larger, ignoring the subtle but important RH signal. MinMaxScaler
#   squishes everything to 0–1, giving all features equal "voice."
#
# SINGLE SCALER PRINCIPLE:
#   It is critical to use ONE scaler, fit on TRAIN data only, and apply it
#   to ALL data. If separate scalers are fit on different subsets, the same
#   temperature (e.g., 35°C) would map to different scaled values in train
#   vs test — the model would see inconsistent inputs.
# =============================================================================

def scale_features(df: pd.DataFrame, train_mask: np.ndarray,
                   feature_cols: List[str]) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Single MinMaxScaler fit on TRAIN only, transform all.

    WHY FIT ON TRAIN ONLY?
      If we fit the scaler on all data, the min/max would include test
      data values — another form of data leakage. By fitting on train
      only, some test values might exceed [0, 1] range, but that's
      correct behavior (the model sees genuinely new conditions).
    """
    scaler = MinMaxScaler(feature_range=(0, 1))

    train_data = df.loc[train_mask, feature_cols].values
    scaler.fit(train_data)

    all_scaled = scaler.transform(df[feature_cols].values)
    print(f"  Scaled {len(feature_cols)} features, shape: {all_scaled.shape}")

    return all_scaled, scaler


def create_sequences(data: np.ndarray, labels: np.ndarray,
                     lookback: int,
                     lead_time: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences: X(N, lookback, features), y(N,).

    HOW SEQUENCES WORK:
      GRUs process data as sequences (windows of consecutive timesteps).
      For each position i, we create:
        X[i] = data[i : i + lookback]              — the past 'lookback' hours
        y[i] = labels[i + lookback + lead_time]     — the label 'lead_time' hours ahead

    LEAD TIME — TURNING NOWCASTING INTO FORECASTING:
      With lead_time=0 (default), the classifier asks "is it extreme NOW?"
      — the label corresponds to the hour immediately after the input window.

      With lead_time=48, the classifier asks "WILL it be extreme in 48 hours?"
      — the label is shifted 48 hours into the future relative to the window end.

      Example with lookback=3, lead_time=2:
        Time:    [t0, t1, t2, t3, t4, t5, t6, ...]
        X[0] = [t0, t1, t2],  y[0] = label at t5  (t3 + lead_time=2)
        X[1] = [t1, t2, t3],  y[1] = label at t6  (t4 + lead_time=2)

      This costs lead_time fewer sequences (some at the end have no label),
      but gives the model a genuine prediction horizon — exactly what's
      needed for an early warning system.
    """
    X, y = [], []
    for i in range(len(data) - lookback - lead_time):
        X.append(data[i : i + lookback])
        y.append(labels[i + lookback + lead_time])
    return np.array(X), np.array(y)


def temporal_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    lookback: int,
    lead_time: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split sequences respecting temporal ordering.

    WHY NOT RANDOM SPLIT?
      In weather data, time matters. If we randomly split, a sequence
      from June 15 might be in the test set while June 14 and June 16
      are in training. The model would effectively "see" the test data
      through its neighbors — data leakage through temporal proximity.

      Instead, we split by year: train = before split_year, test = after.
      The model has NEVER seen any data from the test period during training.

    LOOKBACK + LEAD_TIME OFFSET:
      A sequence at position i contains data from [i, i+lookback).
      Its label is at position i+lookback+lead_time. So we assign the
      sequence to train/test based on the LABEL's timestamp, not the
      input window's start. This ensures that when lead_time > 0, we
      don't accidentally train on sequences whose prediction target
      falls in the test period.
    """
    # Label positions correspond to indices [lookback+lead_time, ...)
    offset = lookback + lead_time
    label_mask = train_mask[offset:]
    assert len(label_mask) == len(y), (
        f"Mask length {len(label_mask)} != y length {len(y)}"
    )

    X_train = X[label_mask]
    y_train = y[label_mask]
    X_test = X[~label_mask]
    y_test = y[~label_mask]

    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")
    print(f"  Train extreme rate: {y_train.mean():.4f}")
    print(f"  Test extreme rate:  {y_test.mean():.4f}")

    return X_train, y_train, X_test, y_test


def compute_class_weights(y_train: np.ndarray) -> dict:
    """
    Compute class weights for imbalanced binary classification.

    WHY CLASS WEIGHTS?
      Only ~5% of hours are "extreme" (label=1). Without class weights,
      the model could predict "normal" for EVERY hour and still get 95%
      accuracy! But it would miss ALL extreme events (0% recall) — useless
      for a heat warning system.

      Class weights tell the model: "each extreme hour counts as much as
      ~19 normal hours." This forces the model to pay attention to extreme
      events. The formula is: weight = N_total / (2 × N_class).

      With 5% extreme: weight_extreme ≈ 10, weight_normal ≈ 0.53.
      So misclassifying one extreme event is penalized 19x more than
      misclassifying one normal event.
    """
    n = len(y_train)
    n_pos = y_train.sum()
    n_neg = n - n_pos
    weights = {0: n / (2 * n_neg), 1: n / (2 * n_pos)}
    print(f"  Class weights: {weights}")
    return weights


# =============================================================================
# Section 6: GRU Classifier (Original Architecture — Legacy)
# =============================================================================
#
# WHAT IS A GRU?
#   GRU (Gated Recurrent Unit) is a type of Recurrent Neural Network (RNN)
#   designed to process SEQUENCES of data. Unlike a regular Dense layer
#   that sees one data point at a time, a GRU maintains a "hidden state"
#   that carries information forward through the sequence.
#
#   Think of it like reading a book: a Dense layer reads one word at a time
#   with no memory. A GRU reads word by word but REMEMBERS what it read
#   before, so it can understand "it was hot yesterday AND today" as a
#   pattern (heat wave building).
#
# WHY GRU INSTEAD OF LSTM?
#   LSTM (Long Short-Term Memory) is the other common RNN type. GRU is
#   simpler (2 gates vs 3) and often performs equally well while training
#   faster. For weather time series, there's no clear advantage to LSTM.
#
# THIS CLASSIFIER vs THE WEATHER GRU:
#   - Classifier (this): answers "is the current hour extreme? YES/NO"
#     Output = single probability (sigmoid). Binary Cross-Entropy loss.
#     This is the original approach.
#   - Weather GRU (Section 10): answers "what will T, RH, SSRD be for
#     each of the next 168 hours?" Output = (168, 3) matrix. MSE loss.
#     This is the new primary model — actually forecasts the future.
#
#   The classifier tells you "it's hot NOW." The Weather GRU tells you
#   "it WILL BE hot in 3 days." The Weather GRU is more useful for
#   early warning systems.
# =============================================================================

def build_classifier(n_timesteps: int, n_features: int,
                     config: Config) -> Sequential:
    """
    GRU classifier architecture (legacy):
      GRU(64) → Dropout(0.2) → GRU(64) → Dropout(0.2) → Dense(1, sigmoid)

    WHY SIGMOID ACTIVATION?
      The final Dense(1, sigmoid) outputs a probability between 0 and 1:
        - 0.0 = definitely normal
        - 1.0 = definitely extreme
        - 0.62 = 62% chance of extreme (above our optimal threshold → classify as extreme)

    WHY BINARY CROSSENTROPY LOSS?
      This is the standard loss function for binary classification.
      It measures how far the predicted probability is from the true label.
      It penalizes confident wrong predictions heavily (predicting 0.99
      when the answer is 0 costs much more than predicting 0.6 when the
      answer is 0).
    """
    model = Sequential([
        Input(shape=(n_timesteps, n_features)),
        GRU(config.gru_units, return_sequences=True),
        Dropout(config.dropout_rate),
        GRU(config.gru_units),
        Dropout(config.dropout_rate),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.Precision(name="precision")],
    )
    return model


def train_classifier(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Config,
) -> tf.keras.callbacks.History:
    """
    Train the GRU classifier with class weights and early stopping.

    KEY TRAINING CONCEPTS:
      - class_weight: upweights extreme events so the model pays attention
        to rare but important events (see compute_class_weights)
      - EarlyStopping: stops training when validation loss stops improving
        for 'patience' epochs. Prevents overfitting (memorizing training data).
        restore_best_weights=True goes back to the best epoch's weights.
      - ModelCheckpoint: saves the best model to disk so we never lose it,
        even if training continues past the best point before early stopping.

    WHY MONITOR VAL_LOSS (NOT VAL_RECALL)?
      We used to monitor val_recall, but recall can hit 1.0 very early
      (epoch 3) if the model just predicts "extreme" for everything.
      val_loss is a more reliable signal of genuine improvement because
      it balances both precision and recall through the class weights.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(config.output_dir, "best_classifier.keras")

    class_weights = compute_class_weights(y_train)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    return history


# =============================================================================
# Section 7: GRU Autoencoder (Unsupervised Anomaly Detection)
# =============================================================================
#
# WHAT IS AN AUTOENCODER?
#   An autoencoder learns to COMPRESS data and then RECONSTRUCT it.
#   It's like a photocopier — if the copy matches the original, the
#   autoencoder has learned the pattern well.
#
#   Structure: Input → Encoder (compress) → Bottleneck → Decoder (expand) → Output
#
#   The key insight: we train it ONLY on NORMAL weather windows.
#   So it learns what normal weather looks like. When we feed it an
#   EXTREME event, it can't reconstruct it well (high error), because
#   it never saw anything like it during training.
#
#   High reconstruction error → anomaly → possibly extreme heat event
#
# WHY UNSUPERVISED?
#   The autoencoder doesn't need labels (extreme/normal). It just learns
#   patterns from data. This is useful because:
#   1. We might not have reliable labels for all types of anomalies
#   2. It can detect novel patterns the classifier hasn't seen
#   3. Its outputs (latent_z, anomaly_score) feed into the classifier
#      and Weather GRU as extra features
#
# OUTPUTS THAT FEED DOWNSTREAM:
#   - latent_z: the 32-dimensional bottleneck vector — a compressed
#     "fingerprint" of the current 24-hour weather pattern
#   - anomaly_score: the reconstruction error (MSE between input and output)
#     Higher = more anomalous. These become features for the classifier
#     and Weather GRU.
# =============================================================================

def build_autoencoder(n_timesteps: int, n_features: int,
                      config: Config) -> Model:
    """
    GRU Autoencoder architecture:
      Encoder: Input(24h, 5 features) → GRU(32) → 32-dim bottleneck
      Decoder: RepeatVector(24) → GRU(32) → TimeDistributed(Dense(5))

    WHY GRU (NOT DENSE)?
      Weather data is sequential — the order of hours matters.
      A Dense autoencoder would treat each hour independently, losing
      temporal patterns like "temperature rising steadily for 6 hours."
      A GRU encoder captures these temporal dynamics.

    WHY RepeatVector?
      The bottleneck is a single 32-dim vector (no time dimension).
      RepeatVector copies it 24 times so the decoder GRU can process
      it as a sequence, reconstructing each hour one by one.
    """
    inputs = Input(shape=(n_timesteps, n_features))

    # Encoder
    encoded = GRU(config.ae_gru_units, activation="tanh")(inputs)

    # Decoder
    decoded = RepeatVector(n_timesteps)(encoded)
    decoded = GRU(config.ae_gru_units, activation="tanh",
                  return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(n_features))(decoded)

    model = Model(inputs, decoded)
    model.compile(optimizer=Adam(config.learning_rate), loss="mse")
    return model


def train_autoencoder(
    model: Model,
    X_train_normal: np.ndarray,
    X_test: np.ndarray,
    config: Config,
) -> tf.keras.callbacks.History:
    """
    Train autoencoder on NORMAL windows only.

    WHY NORMAL ONLY?
      If we trained on all data (including extreme events), the autoencoder
      would learn to reconstruct extreme patterns too — defeating the purpose.
      By training only on normal weather, extreme events become "out of
      distribution" and produce high reconstruction errors.

    NOTE: validation_data=(X_test, X_test)
      The target IS the input (reconstruct itself). X_test here includes
      both normal and extreme windows — we use it to monitor whether the
      autoencoder is learning general normal patterns (not just memorizing
      the training set).
    """
    os.makedirs(config.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(config.output_dir, "best_autoencoder.keras")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train_normal,
        X_train_normal,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_data=(X_test, X_test),
        callbacks=callbacks,
        shuffle=False,
        verbose=1,
    )

    return history


# =============================================================================
# Section 8: (Removed — GNN requires multiple spatial grid points)
# =============================================================================

    # GNN removed — single grid point has no spatial graph to model.
    # If expanded to multiple cities/grid points, add GNN back here.


# =============================================================================
# Section 9: Evaluation & Visualization
# =============================================================================
#
# These functions evaluate model performance and generate plots.
# Key concept: we compute metrics on the TEST set only (data the model
# never saw during training) to get an honest estimate of real-world
# performance.
# =============================================================================

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Sweep thresholds to maximize F1 score.

    WHY NOT JUST USE 0.5?
      The classifier outputs a probability (0–1). The default rule is:
      if P > 0.5, predict "extreme." But with imbalanced data (95% normal,
      5% extreme), 0.5 is often too high — the model is too conservative.

      F1 score = harmonic mean of precision and recall. It balances:
        - Precision: "of the hours I called extreme, how many really were?"
        - Recall: "of the actually extreme hours, how many did I catch?"

      By sweeping from 0.05 to 0.95, we find the threshold that best
      balances catching extreme events (recall) vs. not crying wolf too
      often (precision). Typically the optimal threshold is 0.4–0.7.
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_f1, best_t = 0.0, 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    print(f"  Optimal threshold: {best_t:.2f} (F1={best_f1:.4f})")
    return best_t


def evaluate_classifier(
    model: Sequential,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Config,
) -> dict:
    """Full evaluation: confusion matrix, classification report, ROC-AUC."""
    print("\n--- Classifier Evaluation ---")

    y_prob = model.predict(X_test, verbose=0).flatten()

    # Default threshold
    y_pred_05 = (y_prob >= 0.5).astype(int)
    print("\n  At threshold=0.5:")
    print(classification_report(y_test, y_pred_05, target_names=["Normal", "Extreme"]))

    # Optimal threshold
    opt_threshold = find_optimal_threshold(y_test, y_prob)
    y_pred_opt = (y_prob >= opt_threshold).astype(int)
    print(f"\n  At threshold={opt_threshold:.2f}:")
    report = classification_report(
        y_test, y_pred_opt, target_names=["Normal", "Extreme"], output_dict=True
    )
    print(classification_report(y_test, y_pred_opt, target_names=["Normal", "Extreme"]))

    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred_opt)
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    lt = config.classifier_lead_time
    metrics = {
        "roc_auc": float(auc),
        "optimal_threshold": float(opt_threshold),
        "recall_at_optimal": float(report["Extreme"]["recall"]),
        "precision_at_optimal": float(report["Extreme"]["precision"]),
        "f1_at_optimal": float(report["Extreme"]["f1-score"]),
        "accuracy": float(report["accuracy"]),
        "confusion_matrix": cm.tolist(),
        "lead_time_hours": lt,
        "prediction_type": "forecasting" if lt > 0 else "nowcasting",
    }

    return metrics


def evaluate_autoencoder(
    model: Model,
    X_train_normal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Config,
) -> dict:
    """Evaluate autoencoder anomaly detection."""
    print("\n--- Autoencoder Evaluation ---")

    # Reconstruction errors on train (normal)
    train_pred = model.predict(X_train_normal, verbose=0)
    train_errors = np.mean((X_train_normal - train_pred) ** 2, axis=(1, 2))

    # Threshold = 95th percentile of train errors
    threshold = np.quantile(train_errors, 0.95)
    print(f"  Anomaly threshold (95th pctl of train error): {threshold:.6f}")

    # Test reconstruction errors
    test_pred = model.predict(X_test, verbose=0)
    test_errors = np.mean((X_test - test_pred) ** 2, axis=(1, 2))

    # Anomaly labels
    anomaly_pred = (test_errors >= threshold).astype(int)

    if y_test is not None and len(y_test) == len(anomaly_pred):
        print("\n  Autoencoder vs extreme labels:")
        print(classification_report(
            y_test, anomaly_pred, target_names=["Normal", "Anomaly"]
        ))
        auc = roc_auc_score(y_test, test_errors)
        print(f"  ROC-AUC (error as score): {auc:.4f}")
    else:
        auc = None

    metrics = {
        "anomaly_threshold": float(threshold),
        "anomaly_rate": float(anomaly_pred.mean()),
        "roc_auc": float(auc) if auc is not None else None,
    }

    return metrics


def plot_training_history(history: tf.keras.callbacks.History, config: Config,
                          prefix: str = "classifier"):
    """Plot training/validation loss and metrics."""
    os.makedirs(config.output_dir, exist_ok=True)

    # Determine loss type for labeling
    if prefix == "weather_gru":
        loss_label = "Mean Squared Error (MSE)"
        loss_title = "Weather GRU — Training vs Validation MSE Loss"
    elif prefix == "autoencoder":
        loss_label = "Mean Squared Error (MSE)"
        loss_title = "Autoencoder — Reconstruction MSE Loss"
    else:
        loss_label = "Binary Cross-Entropy (BCE)"
        loss_title = "GRU Classifier — Training vs Validation BCE Loss"

    # Check if we have a second metric to plot
    has_second = ("mae" in history.history or "recall" in history.history
                  or "accuracy" in history.history)
    ncols = 2 if has_second else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    # Loss
    axes[0].plot(history.history["loss"], label="Train", linewidth=1.5)
    axes[0].plot(history.history["val_loss"], label="Validation", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(loss_label)
    axes[0].set_title(loss_title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Second metric
    if "mae" in history.history:
        axes[1].plot(history.history["mae"], label="Train MAE", linewidth=1.5)
        axes[1].plot(history.history["val_mae"], label="Validation MAE", linewidth=1.5)
        axes[1].set_ylabel("Mean Absolute Error (MAE)")
        axes[1].set_title(f"{prefix.replace('_', ' ').title()} — MAE per Epoch")
    elif "recall" in history.history:
        axes[1].plot(history.history["recall"], label="Train Recall", linewidth=1.5)
        axes[1].plot(history.history["val_recall"], label="Validation Recall", linewidth=1.5)
        axes[1].set_ylabel("Recall")
        axes[1].set_title("GRU Classifier — Recall per Epoch")
    elif "accuracy" in history.history:
        axes[1].plot(history.history["accuracy"], label="Train Accuracy", linewidth=1.5)
        axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=1.5)
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title(f"{prefix.replace('_', ' ').title()} — Accuracy per Epoch")
    if has_second:
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(config.output_dir, f"{prefix}_training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          config: Config):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Extreme"])
    ax.set_yticklabels(["Normal", "Extreme"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)

    plt.colorbar(im)
    plt.tight_layout()
    path = os.path.join(config.output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_pr_curves(y_true: np.ndarray, y_prob: np.ndarray,
                       config: Config):
    """Plot ROC and Precision-Recall curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax1.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ax2.plot(rec, prec, label="Precision-Recall")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(config.output_dir, "roc_pr_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_autoencoder_errors(
    model, X_train_normal: np.ndarray, X_test: np.ndarray,
    y_test: np.ndarray, config: Config,
):
    """Plot histogram of reconstruction errors and reconstructed time series."""
    os.makedirs(config.output_dir, exist_ok=True)

    # Compute reconstruction errors
    train_pred = model.predict(X_train_normal, verbose=0)
    train_errors = np.mean((X_train_normal - train_pred) ** 2, axis=(1, 2))
    test_pred = model.predict(X_test, verbose=0)
    test_errors = np.mean((X_test - test_pred) ** 2, axis=(1, 2))
    threshold = np.quantile(train_errors, 0.95)

    # --- Plot 1: Histogram of reconstruction errors ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(train_errors, bins=80, alpha=0.6, label="Train (normal)", color="steelblue", density=True)
    if y_test is not None and len(y_test) == len(test_errors):
        normal_mask = y_test == 0
        ax.hist(test_errors[normal_mask], bins=80, alpha=0.5, label="Test normal", color="green", density=True)
        ax.hist(test_errors[~normal_mask], bins=80, alpha=0.5, label="Test extreme", color="red", density=True)
    else:
        ax.hist(test_errors, bins=80, alpha=0.5, label="Test", color="orange", density=True)
    ax.axvline(threshold, color="k", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold:.4f})")
    ax.set_xlabel("Mean Squared Reconstruction Error")
    ax.set_ylabel("Density")
    ax.set_title("Autoencoder Reconstruction Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(config.output_dir, "autoencoder_error_histogram.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # --- Plot 2: Reconstructed vs original time series (first test window) ---
    n_features = X_test.shape[2]
    feat_names = ["T_resid", "RH_resid", "SSRD", "hour_sin", "hour_cos"][:n_features]
    n_show = min(3, n_features)
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 3.5 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    # Pick one normal and one anomalous window if available
    idx_normal, idx_extreme = 0, None
    if y_test is not None and len(y_test) == len(test_errors):
        normals = np.where(y_test == 0)[0]
        extremes = np.where(y_test == 1)[0]
        if len(normals) > 0:
            idx_normal = normals[len(normals) // 2]
        if len(extremes) > 0:
            idx_extreme = extremes[len(extremes) // 2]

    for fi, ax in enumerate(axes):
        fname = feat_names[fi] if fi < len(feat_names) else f"Feature {fi}"
        ax.plot(X_test[idx_normal, :, fi], label=f"Original (normal)", color="steelblue", linewidth=1.2)
        ax.plot(test_pred[idx_normal, :, fi], label=f"Reconstructed (normal)", color="steelblue", linestyle="--", linewidth=1.2)
        if idx_extreme is not None:
            ax.plot(X_test[idx_extreme, :, fi], label=f"Original (extreme)", color="red", linewidth=1.2)
            ax.plot(test_pred[idx_extreme, :, fi], label=f"Reconstructed (extreme)", color="red", linestyle="--", linewidth=1.2)
        ax.set_ylabel(fname)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Timestep (hours)")
    axes[0].set_title("Autoencoder: Original vs Reconstructed Time Series")
    plt.tight_layout()
    path = os.path.join(config.output_dir, "autoencoder_reconstruction.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # --- Plot 3: ROC + PR curves for autoencoder (if labels available) ---
    if y_test is not None and len(y_test) == len(test_errors) and y_test.sum() > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fpr, tpr, _ = roc_curve(y_test, test_errors)
        auc = roc_auc_score(y_test, test_errors)
        ax1.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title("Autoencoder ROC Curve")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        prec, rec, _ = precision_recall_curve(y_test, test_errors)
        ax2.plot(rec, prec, label="Precision-Recall")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Autoencoder Precision-Recall Curve")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(config.output_dir, "autoencoder_roc_pr_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")

        # Confusion matrix for autoencoder
        anomaly_pred = (test_errors >= threshold).astype(int)
        cm = confusion_matrix(y_test, anomaly_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Oranges")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Anomaly"])
        ax.set_yticklabels(["Normal", "Anomaly"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Autoencoder Confusion Matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
        plt.colorbar(im)
        plt.tight_layout()
        path = os.path.join(config.output_dir, "autoencoder_confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


def save_model_artifacts(model, scaler: MinMaxScaler, config: Config,
                         metrics: dict, prefix: str = "classifier"):
    """Save model, scaler, config, and metrics."""
    os.makedirs(config.output_dir, exist_ok=True)

    # Model
    model_path = os.path.join(config.output_dir, f"{prefix}_model.keras")
    model.save(model_path)
    print(f"  Saved model: {model_path}")

    # Scaler
    import joblib
    scaler_path = os.path.join(config.output_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Saved scaler: {scaler_path}")

    # Config
    config_path = os.path.join(config.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2, default=str)
    print(f"  Saved config: {config_path}")

    # Metrics
    metrics_path = os.path.join(config.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics: {metrics_path}")


def load_model_artifacts(output_dir: str, prefix: str = "classifier"):
    """Load model, scaler, and config for inference."""
    import joblib

    model = tf.keras.models.load_model(
        os.path.join(output_dir, f"{prefix}_model.keras")
    )
    scaler = joblib.load(os.path.join(output_dir, "scaler.pkl"))
    with open(os.path.join(output_dir, "config.json")) as f:
        config_dict = json.load(f)

    return model, scaler, config_dict


# =============================================================================
# Section 10: Weather GRU Forecaster (Multi-Step Regression — PRIMARY MODEL)
# =============================================================================
#
# THIS WILL BE THE MAIN MODEL.
#
# The Weather GRU forecasts what the weather will be for each of the next 168 hours
# (7 days). Then the EHI-N* physics model converts those forecasts to
# physiological zones (1–6) for each hour.
#
# HOW IT WORKS:
#   1. Look at the last 240 hours (10 days) of weather features
#   2. Predict [temperature, humidity, solar radiation] for hours +1 to +168
#   3. Apply EHI-N* physics to each predicted hour → get zone
#   4. Report alerts: "Day 3: Zone 5 (Very Hot) with 60% probability"
#
# WHY REGRESSION (NOT CLASSIFICATION)?
#   Classification loses information. Saying "extreme=yes" doesn't tell
#   you HOW extreme, or WHEN in the next week. Regression predicts actual
#   values (35.2°C, 82% RH, 450 W/m²) that can be processed by physics.
#
# LOSS FUNCTION: MSE (Mean Squared Error)
#   MSE = average of (predicted - actual)² across all hours and targets.
#   Squaring penalizes large errors more — a 5°C error costs 25x more
#   than a 1°C error. This is desirable: we care more about getting
#   extreme temperatures right (which are far from the mean).
# =============================================================================

def build_weather_forecaster(
    n_timesteps: int,
    n_input_features: int,
    n_targets: int,
    forecast_horizon: int,
    config: Config,
) -> Model:
    """
    Weather GRU: predicts T, RH, SSRD at each hour for +1h to +168h.

    Architecture:
      Input(240, 12)                     ← 10 days × 12 features per hour
      → GRU(64, return_sequences=True)   ← 1st GRU: processes sequence, outputs hidden state at EACH timestep
      → Dropout(0.2)                     ← randomly drops 20% of neurons (regularization)
      → GRU(64)                          ← 2nd GRU: processes sequence, outputs SINGLE hidden state (summary)
      → Dropout(0.2)
      → Dense(168 × 3 = 504)            ← fully connected: maps 64-dim summary to all predictions
      → Reshape(168, 3)                  ← reshape flat vector to (hours, targets)

    Output shape: (batch, 168, 3) — [Ta_C, RH, SSRD] for each of the next 168 hours

    WHY return_sequences=True ON FIRST GRU?
      The first GRU outputs its hidden state at EVERY timestep (240 outputs).
      The second GRU reads all 240 of these and produces a SINGLE summary
      vector (64 numbers) that captures the entire 10-day weather history.
      This hierarchical processing lets the network learn patterns at
      different time scales.
    """
    inputs = Input(shape=(n_timesteps, n_input_features), name="lookback_window")

    x = GRU(config.gru_units, return_sequences=True)(inputs)
    x = Dropout(config.dropout_rate)(x)
    x = GRU(config.gru_units)(x)
    x = Dropout(config.dropout_rate)(x)

    # Dense to full forecast: horizon * targets
    x = Dense(forecast_horizon * n_targets, activation="linear")(x)

    # Reshape to (horizon, n_targets)
    outputs = tf.keras.layers.Reshape(
        (forecast_horizon, n_targets), name="forecast"
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name="WeatherGRU")
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def create_forecast_sequences(
    data: np.ndarray,
    targets: np.ndarray,
    lookback: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create (input, target) pairs for multi-step forecasting.

    HOW THIS DIFFERS FROM CLASSIFIER SEQUENCES:
      The classifier predicts ONE label for the next hour.
      The Weather GRU predicts 168 TARGET VALUES (7 days × 3 variables).

      For each position i:
        X[i] = data[i : i+lookback]                      shape: (240, 12) — past weather
        y[i] = targets[i+lookback : i+lookback+horizon]   shape: (168, 3)  — future weather

      So each training example says:
        "Given these 240 hours of weather history, predict the next 168 hours
        of temperature, humidity, and solar radiation."

    NOTE: We need lookback + horizon consecutive hours (240 + 168 = 408).
    Data gaps (missing months between March-July) create invalid sequences
    at year boundaries, but these are a small fraction of the total.
    """
    n = len(data) - lookback - horizon
    if n <= 0:
        raise ValueError(
            f"Not enough data: {len(data)} points for lookback={lookback} + horizon={horizon}"
        )
    X, y = [], []
    for i in range(n):
        X.append(data[i : i + lookback])
        y.append(targets[i + lookback : i + lookback + horizon])
    return np.array(X), np.array(y)


def train_weather_forecaster(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Config,
) -> tf.keras.callbacks.History:
    """Train the Weather GRU with early stopping."""
    os.makedirs(config.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(config.output_dir, "best_weather_gru.keras")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )
    return history


def forecast_to_ehi_zones(
    predicted: np.ndarray,
    config: Config,
    start_datetime: pd.Timestamp = None,
) -> pd.DataFrame:
    """
    Apply EHI-N* physics to Weather GRU predictions.

    THIS IS WHERE ML MEETS PHYSICS.

    The Weather GRU outputs raw numbers: temperature, humidity, and solar
    radiation for each of the next 168 hours. These are just numbers —
    they don't tell you whether a person is in danger.

    This function takes those predictions and runs them through the EHI-N*
    thermoregulatory physics model to compute:
      - What zone (1–6) a person would be in at each predicted hour
      - The actual body heat load from solar radiation (Qs)
      - Whether the sun is up (sun_flag)

    CONTINUOUS SOLAR HEAT LOAD (Qs)
      Solar heat load on the body depends on three physical quantities:
        1. How much sunlight is hitting the ground (SSRD, from the forecast)
        2. The sun's angle (determines how much of the body is exposed)
        3. How much radiation skin/clothes absorb (absorptivity ≈ 0.7)

      The function ssrd_to_Qs() computes the actual Qs from these inputs,
      accounting for the sun's position and body geometry. For example,
      "SSRD = 800 W/m² at solar altitude 60°" produces a specific Qs value
      rather than a simple binary sun/shade flag.

    Parameters:
      predicted: shape (168, 3) — [Ta_C, RH, SSRD] for each future hour
      config: pipeline configuration (MET level, body params)
      start_datetime: if provided, uses PySolar for exact sun position;
                      if None, estimates solar altitude from SSRD magnitude

    Returns DataFrame with columns:
      [hour_ahead, Ta_C, RH, SSRD, sun_flag, EHI_C, zone]
    """
    horizon = predicted.shape[0]
    Ta_C = predicted[:, 0]
    RH = predicted[:, 1]
    SSRD = predicted[:, 2]

    # Ensure physical bounds
    Ta_C = np.clip(Ta_C, -10, 60)
    RH = np.clip(RH, 0.0, 1.0)
    SSRD = np.clip(SSRD, 0.0, 1500.0)

    Ta_K = Ta_C + 273.15
    Qm = config.Qm
    H = config.body_height_m
    M = config.body_mass_kg

    # Compute solar altitude for each forecast hour
    # This determines both sun_flag and the projected area for Qs
    solar_alts = np.zeros(horizon)
    if start_datetime is not None and _HAS_PYSOLAR:
        for h in range(horizon):
            future_dt = start_datetime + pd.Timedelta(hours=h + 1)
            aware_dt = future_dt.to_pydatetime().replace(
                tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30))
            )
            solar_alts[h] = get_altitude(config.kolkata_lat, config.kolkata_lon, aware_dt)
    else:
        # Estimate solar altitude from SSRD: higher SSRD → higher sun
        # Rough inverse: at peak SSRD ~1000 W/m², altitude ~75°
        solar_alts = np.where(SSRD > 20, np.arcsin(np.clip(SSRD / 1000, 0, 1)) * 180 / np.pi, 0.0)

    sun_flags = (solar_alts > 0).astype(float)

    # Compute Qs (solar heat load on body) from SSRD + solar altitude
    # This replaces the binary mrt=0/1 with actual physics
    Qs_body = np.array([
        ssrd_to_Qs(float(SSRD[h]), float(solar_alts[h]))
        for h in range(horizon)
    ])

    # Compute EHI-N* per hour using real solar load
    ehi_vals = np.zeros(horizon)
    zones = np.zeros(horizon, dtype=int)
    for h in range(horizon):
        mrt = 1.0 if sun_flags[h] > 0 else 0.0
        ehi_vals[h] = compute_ehi_value(
            float(Ta_K[h]), float(RH[h]), Qm, mrt, H, M
        ) - 273.15  # to Celsius
        # Zone uses actual Qs from SSRD (not binary shade/sun)
        zones[h] = compute_zone(
            float(Ta_K[h]), float(RH[h]), Qm, H, M,
            Qs=float(Qs_body[h])
        )

    result = pd.DataFrame({
        "hour_ahead": np.arange(1, horizon + 1),
        "Ta_C": Ta_C,
        "RH": RH,
        "SSRD": SSRD,
        "sun_flag": sun_flags,
        "EHI_C": ehi_vals,
        "zone": zones,
    })

    return result


def compute_alert_levels(
    zone_forecast: pd.DataFrame,
    config: Config,
) -> dict:
    """
    Compute alert levels for each forecast horizon.

    HOW ALERTS WORK:
      For each horizon (e.g., "next 48 hours"), we count what fraction
      of hours reach Zone 5+ (Very Hot or Extreme). This gives a
      probability-like measure of dangerous heat.

      Alert colors (modeled after weather warning systems):
        RED    — ≥70% of hours at Zone 5+ → "high confidence dangerous heat"
        ORANGE — ≥40% of hours at Zone 5+ → "likely dangerous heat"
        YELLOW — ≥20% of hours at Zone 5+ → "possible dangerous heat"
        GREEN  — <20% → "no significant heat stress expected"

    WHY P(Zone ≥ 5) AND NOT JUST MAX ZONE?
      A single hour at Zone 5 in 168 hours (0.6%) is not alarming — it
      might be a brief afternoon spike. But 48 hours at Zone 5 (29%) means
      sustained dangerous heat with no nighttime relief. The probability
      captures the DURATION of dangerous conditions, not just the peak.

    Returns dict like:
      {24: {"prob_zone_gte_5": 0.0, "alert": "GREEN", "max_zone": 3},
       48: {"prob_zone_gte_5": 0.4, "alert": "ORANGE", "max_zone": 5}, ...}
    """
    alerts = {}
    zones = zone_forecast["zone"].values

    for h in config.alert_horizons:
        window = zones[:min(h, len(zones))]
        n_dangerous = (window >= 5).sum()
        prob = n_dangerous / len(window) if len(window) > 0 else 0.0
        max_zone = int(window.max()) if len(window) > 0 else 0

        if prob >= 0.7:
            level = "RED"
        elif prob >= 0.4:
            level = "ORANGE"
        elif prob >= 0.2:
            level = "YELLOW"
        else:
            level = "GREEN"

        alerts[h] = {
            "hours_ahead": h,
            "prob_zone_gte_5": round(float(prob), 3),
            "alert": level,
            "max_zone": max_zone,
            "dangerous_hours": int(n_dangerous),
        }

    return alerts


def evaluate_weather_forecaster(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_scaler,
    config: Config,
) -> dict:
    """
    Evaluate Weather GRU forecast accuracy.

    METRICS COMPUTED:
      1. Per-variable MAE and RMSE (in REAL units, not scaled):
         - Ta_C: MAE in °C (e.g., 1.5°C = pretty good for 7-day forecast)
         - RH: MAE in fraction (e.g., 0.05 = 5% humidity error)
         - SSRD: MAE in W/m² (e.g., 50 W/m² = reasonable)
      2. Per-horizon MAE (how does accuracy degrade over 7 days?)
      3. EHI-N* zone accuracy (do predicted zones match actual zones?)

    Note: WHY INVERSE TRANSFORM?
      The model trains on MinMaxScaled (0–1) data. To compute meaningful
      errors in real units (°C, W/m²), we must inverse-transform predictions
      and targets back to their original scale. Without this step, zone
      computation would receive scaled values (0–0.5 instead of 35–40°C),
      which would produce incorrect zone assignments.
    """
    print("\n--- Weather GRU Evaluation ---")

    y_pred = model.predict(X_test, verbose=0)

    target_names = config.forecast_targets
    metrics = {}

    for t_idx, target in enumerate(target_names):
        # MAE and RMSE per horizon (in scaled space)
        errors = y_pred[:, :, t_idx] - y_test[:, :, t_idx]
        mae_per_hour = np.mean(np.abs(errors), axis=0)
        rmse_per_hour = np.sqrt(np.mean(errors ** 2, axis=0))

        # Report at key horizons
        for h in config.alert_horizons:
            if h - 1 < len(mae_per_hour):
                print(f"  {target} @ +{h}h: MAE={mae_per_hour[h-1]:.3f}, RMSE={rmse_per_hour[h-1]:.3f}")

        metrics[target] = {
            "mae_overall": float(np.mean(mae_per_hour)),
            "rmse_overall": float(np.sqrt(np.mean(errors ** 2))),
            "mae_24h": float(mae_per_hour[23]) if len(mae_per_hour) > 23 else None,
            "mae_168h": float(mae_per_hour[-1]),
        }

    # Evaluate EHI-N* zone accuracy using REAL units (inverse-transformed)
    print("\n  Evaluating EHI-N* zone accuracy on test samples...")
    n_samples = min(100, len(X_test))
    zone_correct = 0
    zone_total = 0
    for i in range(n_samples):
        # Inverse-transform to real units for zone computation
        pred_real = target_scaler.inverse_transform(
            y_pred[i].reshape(-1, len(target_names))
        )
        true_real = target_scaler.inverse_transform(
            y_test[i].reshape(-1, len(target_names))
        )
        pred_zones = forecast_to_ehi_zones(pred_real, config)
        for h in range(len(true_real)):
            true_Ta_K = true_real[h, 0] + 273.15
            true_RH = np.clip(true_real[h, 1], 0.0, 1.0)
            # Compute real solar heat load (Qs) from SSRD if available
            true_SSRD = float(true_real[h, 2]) if true_real.shape[1] > 2 else 0.0
            true_SSRD = max(0.0, true_SSRD)
            # Estimate solar altitude from SSRD magnitude
            true_solar_alt = float(np.arcsin(np.clip(true_SSRD / 1000, 0, 1)) * 180 / np.pi) if true_SSRD > 20 else 0.0
            true_Qs = ssrd_to_Qs(true_SSRD, true_solar_alt)
            true_zone = compute_zone(
                float(true_Ta_K), float(true_RH),
                config.Qm, config.body_height_m, config.body_mass_kg,
                Qs=true_Qs,
            )
            if pred_zones["zone"].iloc[h] == true_zone:
                zone_correct += 1
            zone_total += 1

    zone_acc = zone_correct / zone_total if zone_total > 0 else 0
    print(f"  Zone accuracy (sample): {zone_acc:.4f} ({zone_correct}/{zone_total})")
    metrics["zone_accuracy"] = float(zone_acc)

    return metrics


def plot_forecast_example(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_scaler,
    config: Config,
    sample_idx: int = 0,
):
    """Plot a single forecast example: predicted vs actual T, RH, SSRD + zones."""
    os.makedirs(config.output_dir, exist_ok=True)

    y_pred_scaled = model.predict(X_test[sample_idx : sample_idx + 1], verbose=0)[0]
    y_true_scaled = y_test[sample_idx]

    # Inverse-transform to real units for plotting and zone computation
    n_targets = len(config.forecast_targets)
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, n_targets))
    y_true = target_scaler.inverse_transform(y_true_scaled.reshape(-1, n_targets))

    hours = np.arange(1, config.forecast_horizon + 1)
    target_names = config.forecast_targets
    units = {"Ta_C": "°C", "RH": "fraction (0–1)", "SSRD": "W/m²"}
    full_names = {
        "Ta_C": "2m Temperature",
        "RH": "Relative Humidity",
        "SSRD": "Surface Solar Radiation Downwards",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot T, RH, SSRD
    for idx, target in enumerate(target_names):
        ax = axes[idx // 2, idx % 2]
        ax.plot(hours, y_true[:, idx], "b-", label="Actual (ERA5)", alpha=0.7, linewidth=1.2)
        ax.plot(hours, y_pred[:, idx], "r--", label="Predicted (Weather GRU)", alpha=0.7, linewidth=1.2)
        ax.set_xlabel("Forecast Lead Time (hours)")
        ax.set_ylabel(f"{full_names.get(target, target)} ({units.get(target, '')})")
        ax.set_title(f"{full_names.get(target, target)} — Predicted vs Actual (real units)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Mark day boundaries
        for d in [24, 48, 72, 96, 120, 144, 168]:
            if d <= config.forecast_horizon:
                ax.axvline(d, color="gray", linestyle=":", alpha=0.3)
                if d <= 168:
                    ax.text(d, ax.get_ylim()[1], f"Day {d//24}", ha="center",
                            va="bottom", fontsize=7, color="gray")

    # Plot EHI-N* zones
    ax = axes[1, 1]
    pred_zones = forecast_to_ehi_zones(y_pred, config)
    true_Ta_K = y_true[:, 0] + 273.15
    true_RH = y_true[:, 1]
    # Compute real Qs from actual SSRD for true zone computation
    true_SSRD = np.clip(y_true[:, 2], 0, 1500) if y_true.shape[1] > 2 else np.zeros(len(y_true))
    true_solar_alts = np.where(true_SSRD > 20,
                               np.arcsin(np.clip(true_SSRD / 1000, 0, 1)) * 180 / np.pi, 0.0)
    true_Qs_arr = np.array([ssrd_to_Qs(float(true_SSRD[h]), float(true_solar_alts[h]))
                            for h in range(len(y_true))])
    true_zones = np.array([
        compute_zone(float(true_Ta_K[h]), float(true_RH[h]),
                      config.Qm, config.body_height_m, config.body_mass_kg,
                      Qs=float(true_Qs_arr[h]))
        for h in range(len(y_true))
    ])

    ax.step(hours, true_zones, "b-", where="mid", label="Actual Zone (ERA5 → EHI-N*)", alpha=0.7, linewidth=1.5)
    ax.step(hours, pred_zones["zone"].values, "r--", where="mid", label="Predicted Zone (GRU → EHI-N*)", alpha=0.7, linewidth=1.5)
    ax.axhspan(6, 6.5, alpha=0.15, color="purple", label="Zone 6 (Extreme)")
    ax.axhspan(5, 6, alpha=0.1, color="red", label="Zone 5 (Very Hot)")
    ax.axhspan(4, 5, alpha=0.1, color="orange", label="Zone 4 (Hot)")
    ax.axhspan(3, 4, alpha=0.07, color="gold")
    ax.set_xlabel("Forecast Lead Time (hours)")
    ax.set_ylabel("EHI-N* Physiological Zone (1–6)")
    ax.set_title(f"EHI-N* Zone Forecast — MET Level {config.met_level} ({config.Qm} W/m²)")
    ax.set_ylim(0.5, 6.5)
    ax.set_yticks([1, 2, 3, 4, 5, 6])
    ax.set_yticklabels(["1 (Cool)", "2 (Comfort)", "3 (Warm)", "4 (Hot)", "5 (Very Hot)", "6 (Extreme)"])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"HeatRadar — 7-Day Weather GRU Forecast → EHI-N* Zone Prediction\n"
        f"Kolkata ({config.kolkata_lat}°N, {config.kolkata_lon}°E) | "
        f"Lookback: {config.lookback}h | MET {config.met_level}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(config.output_dir, "forecast_example.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def save_forecast_csv(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    df: pd.DataFrame,
    train_mask: np.ndarray,
    target_scaler,
    config: Config,
    n_samples: int = 20,
):
    """
    Save predicted vs actual forecasts as CSV for verification.

    Each row = one (sample, hour_ahead) pair with predicted and actual
    T, RH, SSRD, EHI-N* zone, and alert level. Lets you go back after
    a week and check if the forecast was right.
    """
    os.makedirs(config.output_dir, exist_ok=True)

    y_pred_all = model.predict(X_test, verbose=0)
    n_samples = min(n_samples, len(X_test))

    # Figure out test start timestamps
    test_times = df.index[~train_mask]
    # Account for lookback offset — first test sequence starts at lookback
    seq_offset = config.lookback

    rows = []
    for i in range(n_samples):
        y_pred_sample = y_pred_all[i]
        y_true_sample = y_test[i]

        # Inverse-transform to real units if target_scaler available
        if target_scaler is not None:
            pred_real = target_scaler.inverse_transform(
                y_pred_sample.reshape(-1, len(config.forecast_targets))
            )
            true_real = target_scaler.inverse_transform(
                y_true_sample.reshape(-1, len(config.forecast_targets))
            )
        else:
            pred_real = y_pred_sample
            true_real = y_true_sample

        # Get zones using real-unit values (not scaled)
        pred_zones_df = forecast_to_ehi_zones(pred_real, config)
        for h in range(config.forecast_horizon):
            true_Ta_K = true_real[h, 0] + 273.15
            true_RH = np.clip(true_real[h, 1], 0.0, 1.0)
            # Compute real solar heat load (Qs) from actual SSRD
            true_SSRD = float(true_real[h, 2]) if true_real.shape[1] > 2 else 0.0
            true_SSRD = max(0.0, true_SSRD)
            true_solar_alt = float(np.arcsin(np.clip(true_SSRD / 1000, 0, 1)) * 180 / np.pi) if true_SSRD > 20 else 0.0
            true_Qs = ssrd_to_Qs(true_SSRD, true_solar_alt)
            true_zone = compute_zone(
                float(true_Ta_K), float(true_RH),
                config.Qm, config.body_height_m, config.body_mass_kg,
                Qs=true_Qs,
            )

            # Try to get actual timestamp
            test_seq_idx = i + seq_offset
            if test_seq_idx < len(test_times):
                forecast_start = test_times[test_seq_idx]
                forecast_time = forecast_start + pd.Timedelta(hours=h + 1)
            else:
                forecast_time = pd.NaT

            rows.append({
                "sample_id": i,
                "forecast_start": forecast_start if test_seq_idx < len(test_times) else pd.NaT,
                "forecast_time": forecast_time,
                "hours_ahead": h + 1,
                "pred_Ta_C": round(float(pred_real[h, 0]), 2),
                "actual_Ta_C": round(float(true_real[h, 0]), 2),
                "pred_RH": round(float(pred_real[h, 1]), 4),
                "actual_RH": round(float(true_real[h, 1]), 4),
                "pred_SSRD": round(float(pred_real[h, 2]), 1) if pred_real.shape[1] > 2 else None,
                "actual_SSRD": round(float(true_real[h, 2]), 1) if true_real.shape[1] > 2 else None,
                "pred_zone": int(pred_zones_df["zone"].iloc[h]),
                "actual_zone": int(true_zone),
                "zone_correct": int(pred_zones_df["zone"].iloc[h]) == int(true_zone),
                "pred_alert": pred_zones_df["alert"].iloc[h] if "alert" in pred_zones_df else None,
            })

    csv_df = pd.DataFrame(rows)
    csv_path = os.path.join(config.output_dir, "forecast_predictions.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({len(csv_df)} rows, {n_samples} samples × {config.forecast_horizon}h)")

    # Summary stats
    zone_acc = csv_df["zone_correct"].mean()
    ta_mae = (csv_df["pred_Ta_C"] - csv_df["actual_Ta_C"]).abs().mean()
    rh_mae = (csv_df["pred_RH"] - csv_df["actual_RH"]).abs().mean()
    print(f"  CSV summary — Zone accuracy: {zone_acc:.2%}, T MAE: {ta_mae:.2f}°C, RH MAE: {rh_mae:.4f}")

    # --- Verification plot: zone predictions by day ---
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Panel 1: Zone accuracy by forecast lead time
    ax = axes[0]
    hourly_acc = csv_df.groupby("hours_ahead")["zone_correct"].mean()
    ax.bar(hourly_acc.index, hourly_acc.values, color="steelblue", alpha=0.6, width=1)
    # Add daily averages
    for d in range(1, 8):
        day_mask = (csv_df["hours_ahead"] > (d-1)*24) & (csv_df["hours_ahead"] <= d*24)
        day_acc = csv_df.loc[day_mask, "zone_correct"].mean()
        ax.axhline(day_acc, xmin=(d-1)/7, xmax=d/7, color="red", linewidth=2)
        ax.text(d * 24 - 12, day_acc + 0.02, f"Day {d}: {day_acc:.0%}", ha="center",
                fontsize=9, color="red", fontweight="bold")
    ax.set_xlabel("Forecast Lead Time (hours)")
    ax.set_ylabel("Zone Prediction Accuracy")
    ax.set_title("EHI-N* Zone Accuracy by Forecast Lead Time — Does Accuracy Degrade Over 7 Days?")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    for d in [24, 48, 72, 96, 120, 144]:
        ax.axvline(d, color="gray", linestyle=":", alpha=0.3)

    # Panel 2: Temperature error by lead time
    ax = axes[1]
    csv_df["ta_error"] = csv_df["pred_Ta_C"] - csv_df["actual_Ta_C"]
    hourly_mae = csv_df.groupby("hours_ahead")["ta_error"].apply(lambda x: x.abs().mean())
    hourly_bias = csv_df.groupby("hours_ahead")["ta_error"].mean()
    ax.fill_between(hourly_mae.index, -hourly_mae.values, hourly_mae.values,
                     alpha=0.2, color="steelblue", label="MAE envelope")
    ax.plot(hourly_bias.index, hourly_bias.values, "r-", linewidth=1.2, label="Bias (pred − actual)")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Forecast Lead Time (hours)")
    ax.set_ylabel("Temperature Error (°C)")
    ax.set_title("Temperature Forecast Error Over 7 Days — Bias & MAE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    for d in [24, 48, 72, 96, 120, 144]:
        ax.axvline(d, color="gray", linestyle=":", alpha=0.3)

    plt.suptitle(
        f"HeatRadar Forecast Verification — {n_samples} Test Samples\n"
        f"Overall Zone Accuracy: {zone_acc:.1%} | T MAE: {ta_mae:.2f}°C",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(config.output_dir, "forecast_verification.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    return csv_df


# =============================================================================
# Section 11: Open-Meteo Inference (fill ERA5 gap with live data)
# =============================================================================

def fetch_openmeteo_recent(
    lat: float,
    lon: float,
    hours: int = 240,
) -> pd.DataFrame:
    """
    Fetch the most recent `hours` of T2m, RH, SSRD from Open-Meteo API.

    This fills the ERA5 5-day gap so the model can forecast from "now".
    Open-Meteo provides free, no-key access to GFS/ECMWF recent observations.

    Returns DataFrame with columns: [Ta_C, RH, SSRD] indexed by datetime.
    """
    import urllib.request

    # Open-Meteo forecast API with past_hours for recent observations
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,relative_humidity_2m,shortwave_radiation"
        f"&past_hours={hours}"
        f"&forecast_hours=0"
        f"&timezone=auto"
    )

    print(f"  Fetching {hours}h of recent data from Open-Meteo...")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        raise RuntimeError(f"Open-Meteo fetch failed: {e}")

    hourly = data["hourly"]
    times = pd.to_datetime(hourly["time"])

    df = pd.DataFrame({
        "Ta_C": hourly["temperature_2m"],
        "RH": np.array(hourly["relative_humidity_2m"]) / 100.0,  # % → 0-1
        "SSRD": hourly["shortwave_radiation"],
    }, index=times)

    df.index.name = "datetime"

    # Drop NaN
    df = df.dropna()
    print(f"  Got {len(df)} hours: {df.index[0]} → {df.index[-1]}")
    print(f"  T: {df['Ta_C'].min():.1f}–{df['Ta_C'].max():.1f}°C")

    return df


def run_inference(config: Config):
    """
    Real-time inference mode — LIVE 7-day forecast.

    This is what you'd run in production to generate actual heat warnings.
    Unlike training (which uses historical ERA5 data), inference mode:

    1. Fetches the LAST 240 hours of weather from Open-Meteo API
       (free, no API key needed). This fills the ERA5 5-day gap because
       ERA5 reanalysis is always ~5 days behind real-time.

    2. Applies the SAME preprocessing used during training:
       - Solar geometry (PySolar)
       - Diurnal residuals (using the SAVED training mean, not computed
         from this data — that would be data leakage)
       - Cyclical time features

    3. Runs the trained Weather GRU model → 168h forecast of T, RH, SSRD

    4. Applies EHI-N* physics to each predicted hour → zone (1-6)

    5. Computes alerts: P(Zone ≥ 5) at +24h, +48h, +72h, +120h, +168h

    IMPORTANT: Requires a trained model in config.output_dir. Run training
    first with: python heatradar_nowcast.py --mode forecast --data_dir ./data
    """
    print("=" * 70)
    print("HeatRadar Nowcast — LIVE INFERENCE")
    print("=" * 70)

    # 1. Fetch recent data
    df = fetch_openmeteo_recent(config.kolkata_lat, config.kolkata_lon, config.lookback)

    # Add Kelvin
    df["Ta_K"] = df["Ta_C"] + 273.15

    # 2. Solar features
    df = compute_solar_features(df, config)

    # 3. LOWESS residuals — use the training diurnal mean from saved config
    config_path = os.path.join(config.output_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            saved = json.load(f)
        if "diurnal_T_mean" in saved:
            diurnal_T = {int(k): v for k, v in saved["diurnal_T_mean"].items()}
            diurnal_RH = {int(k): v for k, v in saved["diurnal_RH_mean"].items()}
            df["T_resid"] = df["Ta_C"] - df.index.hour.map(diurnal_T)
            df["RH_resid"] = df["RH"] - df.index.hour.map(diurnal_RH)
        else:
            # Fallback: compute from this data (not ideal)
            diurnal_T = df.groupby(df.index.hour)["Ta_C"].mean()
            diurnal_RH = df.groupby(df.index.hour)["RH"].mean()
            df["T_resid"] = df["Ta_C"] - df.index.hour.map(diurnal_T)
            df["RH_resid"] = df["RH"] - df.index.hour.map(diurnal_RH)
    else:
        diurnal_T = df.groupby(df.index.hour)["Ta_C"].mean()
        diurnal_RH = df.groupby(df.index.hour)["RH"].mean()
        df["T_resid"] = df["Ta_C"] - df.index.hour.map(diurnal_T)
        df["RH_resid"] = df["RH"] - df.index.hour.map(diurnal_RH)

    # 4. Cyclical time
    df = compute_cyclical_time_features(df)

    # 5. Load model and scaler
    model, scaler, _ = load_model_artifacts(config.output_dir, prefix="weather_gru")

    # 6. Scale input features
    input_cols = config.forecast_input_cols
    for col in input_cols:
        if col not in df.columns:
            df[col] = 0.0

    X_live = scaler.transform(df[input_cols].values)

    # Take the last lookback window
    if len(X_live) < config.lookback:
        raise ValueError(
            f"Not enough data: got {len(X_live)} hours, need {config.lookback}"
        )
    X_input = X_live[-config.lookback:].reshape(1, config.lookback, len(input_cols))

    # 7. Predict
    print("\n--- Forecasting 7 days ahead ---")
    y_pred = model.predict(X_input, verbose=0)[0]  # (horizon, 3)

    # Inverse-scale the targets (first 3 columns of the scaler correspond to Ta_C, RH, SSRD)
    # We need a target scaler — load it
    import joblib
    target_scaler_path = os.path.join(config.output_dir, "target_scaler.pkl")
    if os.path.exists(target_scaler_path):
        target_scaler = joblib.load(target_scaler_path)
        y_pred_flat = y_pred.reshape(-1, config.n_forecast_targets)
        y_pred_flat = target_scaler.inverse_transform(y_pred_flat)
        y_pred = y_pred_flat.reshape(config.forecast_horizon, config.n_forecast_targets)

    # 8. EHI-N* zones
    last_time = df.index[-1]
    zone_forecast = forecast_to_ehi_zones(y_pred, config, start_datetime=last_time)

    # 9. Alerts
    alerts = compute_alert_levels(zone_forecast, config)

    print("\n" + "=" * 70)
    print(f"  FORECAST from {last_time} + 168h")
    print(f"  MET level: {config.met_level} ({config.Qm} W/m²)")
    print("=" * 70)

    for h, alert in sorted(alerts.items()):
        emoji = {"RED": "!!!", "ORANGE": "! ", "YELLOW": "~ ", "GREEN": "  "}
        print(
            f"  {emoji.get(alert['alert'], '  ')} +{h:3d}h ({h//24}d): "
            f"{alert['alert']:6s} | P(Zone≥5)={alert['prob_zone_gte_5']:.1%} | "
            f"Max Zone={alert['max_zone']}"
        )

    # Save forecast
    os.makedirs(config.output_dir, exist_ok=True)
    forecast_path = os.path.join(config.output_dir, "latest_forecast.json")
    forecast_data = {
        "generated_at": str(pd.Timestamp.now()),
        "forecast_from": str(last_time),
        "met_level": config.met_level,
        "location": {"lat": config.kolkata_lat, "lon": config.kolkata_lon},
        "alerts": alerts,
        "hourly": zone_forecast.to_dict(orient="records"),
    }
    with open(forecast_path, "w") as f:
        json.dump(forecast_data, f, indent=2, default=str)
    print(f"\n  Saved forecast: {forecast_path}")

    return zone_forecast, alerts


# =============================================================================
# Section 12: Main Pipelines
# =============================================================================
#
# PIPELINE EXECUTION ORDER (for --mode full or running all three):
#
#   1. AUTOENCODER (unsupervised)
#      - Trains on NORMAL weather windows only
#      - Learns what "normal" looks like
#      - Outputs: latent_z (compressed state), anomaly_score (reconstruction error)
#      - These become INPUT FEATURES for the classifier and Weather GRU
#
#   2. CLASSIFIER (GRU — supervised binary classification)
#      - Consumes autoencoder outputs + EHI zones + residuals
#      - Predicts P(extreme) for the current hour
#      - Useful for real-time monitoring, less useful for advance warning
#
#   3. WEATHER GRU (multi-step forecasting — PRIMARY MODEL)
#      - Consumes autoencoder outputs + raw weather + residuals + time features
#      - Forecasts T, RH, SSRD for each of the next 168 hours
#      - EHI-N* physics then converts each hour to a zone (1-6)
#      - This is the main product: "Zone 5 expected in 72 hours"
#
# WHY THIS ORDER?
#   The autoencoder must run first because its outputs (latent_z, anomaly_score)
#   are used as input features by both the classifier and the Weather GRU.
#   Without the autoencoder, those columns would be zeros — the models would
#   still work but miss the anomaly detection signal.
# =============================================================================

def _load_and_preprocess(config: Config) -> Tuple[pd.DataFrame, np.ndarray]:
    """Shared data loading and feature engineering for all pipelines."""
    print("\n--- Loading ERA5 Data (T2m, D2m, SSRD) ---")
    df = load_era5_data(config)

    # Feature engineering: solar → LOWESS → EHI-N* → cyclical → labels
    df, train_mask = build_feature_matrix(df, config)

    return df, train_mask


def _run_autoencoder_and_populate(df: pd.DataFrame, train_mask: np.ndarray,
                                   config: Config) -> pd.DataFrame:
    """
    Run the autoencoder and populate latent_z + anomaly_score columns in df.

    This must run BEFORE the classifier or Weather GRU so they can use
    the autoencoder's outputs as input features.

    Returns the updated DataFrame with filled latent_z and anomaly_score columns.
    """
    print("\n" + "=" * 70)
    print("Step 1: Running Autoencoder (to produce features for downstream models)")
    print("=" * 70)

    # Scale autoencoder features
    ae_cols = ["T_resid", "RH_resid", "SSRD", "hour_sin", "hour_cos"]
    ae_scaled, ae_scaler = scale_features(df, train_mask, ae_cols)
    labels = df["extreme"].values

    # Create 24h windows
    X_all, y_all = create_sequences(ae_scaled, labels, config.ae_window)
    X_train, y_train, X_test, y_test = temporal_train_test_split(
        X_all, y_all, train_mask, config.ae_window
    )

    # Train on normal windows only
    normal_mask = y_train == 0
    X_train_normal = X_train[normal_mask]
    print(f"  Training on {len(X_train_normal)} normal windows "
          f"(excluded {(~normal_mask).sum()} extreme)")

    # Build, train, evaluate
    model = build_autoencoder(config.ae_window, X_train.shape[2], config)
    history = train_autoencoder(model, X_train_normal, X_test, config)
    metrics = evaluate_autoencoder(model, X_train_normal, X_test, y_test, config)

    # Plots
    plot_training_history(history, config, prefix="autoencoder")
    plot_autoencoder_errors(model, X_train_normal, X_test, y_test, config)

    # Save autoencoder model
    save_model_artifacts(model, ae_scaler, config, metrics, prefix="autoencoder")

    # ---- Populate latent_z and anomaly_score for ALL data ----
    # Run the encoder on all sequences to get the bottleneck representation
    print("\n  Populating latent_z and anomaly_score for downstream models...")

    # Get encoder output (bottleneck layer)
    encoder = Model(inputs=model.input, outputs=model.layers[1].output)

    # Process all windows
    all_pred = model.predict(X_all, verbose=0)
    all_errors = np.mean((X_all - all_pred) ** 2, axis=(1, 2))
    all_latent = encoder.predict(X_all, verbose=0)

    # Map back to DataFrame indices
    # Each sequence i covers data[i:i+ae_window], label at i+ae_window
    for i in range(len(X_all)):
        idx = config.ae_window + i
        if idx < len(df):
            df.iloc[idx, df.columns.get_loc("anomaly_score")] = float(all_errors[i])
            # Use mean of latent vector as a scalar summary
            df.iloc[idx, df.columns.get_loc("latent_z")] = float(np.mean(all_latent[i]))

    ae_filled = (df["anomaly_score"] != 0).sum()
    print(f"  Filled {ae_filled}/{len(df)} rows with autoencoder features")

    return df


def run_forecast_pipeline(config: Config):
    """
    Main forecast pipeline — the PRIMARY model.

    FULL FLOW:
      1. Load ERA5 data (T2m, D2m, SSRD) → ~60k-100k hourly records
      2. Preprocess: solar geometry, diurnal residuals, EHI zones, time features
      3. Run autoencoder to populate anomaly_score + latent_z features
      4. Scale all 12 input features (fit on train only)
      5. Create forecast sequences: 240h lookback → 168h horizon
      6. Train Weather GRU (50 epochs, early stopping at patience=10)
      7. Evaluate: per-variable MAE/RMSE + EHI zone accuracy
      8. Generate plots: training history, forecast example, verification CSV
      9. Save everything: model, scalers, config, metrics

    After this completes, you can run --mode inference for live forecasting.
    """
    print("=" * 70)
    print("HeatRadar Nowcast — Weather GRU Forecast Pipeline")
    print("=" * 70)
    print(f"  MET level: {config.met_level} ({config.Qm} W/m²)")
    print(f"  Lookback: {config.lookback}h ({config.lookback // 24} days)")
    print(f"  Horizon: {config.forecast_horizon}h ({config.forecast_horizon // 24} days)")

    df, train_mask = _load_and_preprocess(config)

    # Run autoencoder first to populate latent_z and anomaly_score
    # These are input features for the Weather GRU (features 11 and 12)
    df = _run_autoencoder_and_populate(df, train_mask, config)

    # Save the training diurnal means for inference later
    train_df = df.loc[train_mask]
    diurnal_T_mean = train_df.groupby(train_df.index.hour)["Ta_C"].mean().to_dict()
    diurnal_RH_mean = train_df.groupby(train_df.index.hour)["RH"].mean().to_dict()

    # Scale input features
    print("\n--- Scaling Input Features ---")
    input_cols = config.forecast_input_cols
    for col in input_cols:
        if col not in df.columns:
            df[col] = 0.0
    input_scaled, input_scaler = scale_features(df, train_mask, input_cols)

    # Scale target features (Ta_C, RH, SSRD)
    target_cols = config.forecast_targets
    target_scaled, target_scaler = scale_features(df, train_mask, target_cols)

    # Create forecast sequences
    print("\n--- Creating Forecast Sequences ---")
    print(f"  Input: {config.lookback}h window of {len(input_cols)} features")
    print(f"  Target: {config.forecast_horizon}h forecast of {target_cols}")

    X, y = create_forecast_sequences(
        input_scaled, target_scaled,
        config.lookback, config.forecast_horizon
    )
    print(f"  X shape: {X.shape}")  # (N, lookback, n_input)
    print(f"  y shape: {y.shape}")  # (N, horizon, n_targets)

    # Temporal split
    print("\n--- Temporal Split ---")
    # The label position for sequence i is at i + lookback (start of forecast)
    n_sequences = len(X)
    label_positions = train_mask[config.lookback : config.lookback + n_sequences]
    assert len(label_positions) == n_sequences

    X_train = X[label_positions]
    y_train = y[label_positions]
    X_test = X[~label_positions]
    y_test = y[~label_positions]

    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}")

    # Build model
    print("\n--- Building Weather GRU ---")
    model = build_weather_forecaster(
        config.lookback,
        len(input_cols),
        len(target_cols),
        config.forecast_horizon,
        config,
    )
    model.summary()

    # Train
    print("\n--- Training ---")
    history = train_weather_forecaster(model, X_train, y_train, X_test, y_test, config)

    # Evaluate
    metrics = evaluate_weather_forecaster(model, X_test, y_test, target_scaler, config)

    # Plots
    print("\n--- Generating Plots ---")
    plot_training_history(history, config, prefix="weather_gru")
    plot_forecast_example(model, X_test, y_test, target_scaler, config, sample_idx=0)
    save_forecast_csv(model, X_test, y_test, df, train_mask, target_scaler, config)

    # Save
    print("\n--- Saving Artifacts ---")
    save_model_artifacts(model, input_scaler, config, metrics, prefix="weather_gru")

    # Also save the target scaler and diurnal means for inference
    import joblib
    joblib.dump(target_scaler, os.path.join(config.output_dir, "target_scaler.pkl"))

    # Save diurnal means into config
    config_path = os.path.join(config.output_dir, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    config_dict["diurnal_T_mean"] = {str(k): v for k, v in diurnal_T_mean.items()}
    config_dict["diurnal_RH_mean"] = {str(k): v for k, v in diurnal_RH_mean.items()}
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    print(f"  Saved target scaler: {config.output_dir}/target_scaler.pkl")

    # Run a sample forecast through EHI-N*
    print("\n--- Sample EHI-N* Forecast ---")
    sample_pred = model.predict(X_test[:1], verbose=0)[0]
    # Inverse scale targets
    sample_pred_flat = target_scaler.inverse_transform(
        sample_pred.reshape(-1, len(target_cols))
    ).reshape(config.forecast_horizon, len(target_cols))

    zone_forecast = forecast_to_ehi_zones(sample_pred_flat, config)
    alerts = compute_alert_levels(zone_forecast, config)

    for h, alert in sorted(alerts.items()):
        print(
            f"  +{h:3d}h ({h//24}d): {alert['alert']:6s} | "
            f"P(Zone≥5)={alert['prob_zone_gte_5']:.1%} | Max Zone={alert['max_zone']}"
        )

    print("\n" + "=" * 70)
    print("FORECAST PIPELINE COMPLETE")
    for target in target_cols:
        m = metrics[target]
        print(f"  {target}: MAE={m['mae_overall']:.3f}, RMSE={m['rmse_overall']:.3f}")
    print(f"  Zone accuracy: {metrics['zone_accuracy']:.4f}")
    print(f"  Output: {config.output_dir}/")
    print("=" * 70)

    return model, metrics


def run_classifier_pipeline(config: Config):
    """
    GRU classifier pipeline — binary extreme heat prediction.

    With lead_time > 0 (default 48h), this is a FORECASTING classifier:
    given the past `lookback` hours of weather, predict whether conditions
    will be extreme `lead_time` hours in the future. This transforms the
    original nowcasting model into an early warning system.

    NOTE ON AUTOENCODER FEATURES:
      When run standalone (--mode classifier), the autoencoder hasn't run,
      so latent_z and anomaly_score will be zeros. The classifier still works
      but won't have the anomaly detection signal. For best results, use
      --mode full which runs autoencoder FIRST, populating those columns.
    """
    print("=" * 70)
    lt = config.classifier_lead_time
    if lt > 0:
        print(f"HeatRadar — GRU Classifier Pipeline (predicting {lt}h ahead)")
    else:
        print("HeatRadar — GRU Classifier Pipeline (nowcasting)")
    print("=" * 70)
    print(f"  MET level: {config.met_level} ({config.Qm} W/m²)")
    print(f"  Lookback: {config.lookback}h, Lead time: {lt}h")
    print(f"  Split year: {config.split_year}")

    df, train_mask = _load_and_preprocess(config)

    # Scale GRU features
    print("\n--- Scaling GRU Features ---")
    feature_cols = config.feature_cols
    # Ensure all feature cols exist (latent_z and anomaly_score are 0 without AE)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    scaled_data, scaler = scale_features(df, train_mask, feature_cols)
    labels = df["extreme"].values

    # Sequences — with lead_time offset for forecasting
    lt = config.classifier_lead_time
    print(f"\n--- Creating Sequences (lead_time={lt}h) ---")
    if lt > 0:
        print(f"  Classifier will predict {lt}h ({lt // 24:.0f} days) ahead")
    X, y = create_sequences(scaled_data, labels, config.lookback, lead_time=lt)
    print(f"  X shape: {X.shape}, y shape: {y.shape}")

    # Split
    print("\n--- Temporal Split ---")
    X_train, y_train, X_test, y_test = temporal_train_test_split(
        X, y, train_mask, config.lookback, lead_time=lt
    )

    # Build & train
    print("\n--- Building GRU Classifier ---")
    model = build_classifier(X_train.shape[1], X_train.shape[2], config)
    model.summary()

    print("\n--- Training ---")
    history = train_classifier(model, X_train, y_train, X_test, y_test, config)

    # Evaluate
    metrics = evaluate_classifier(model, X_test, y_test, config)

    # Plots
    print("\n--- Generating Plots ---")
    plot_training_history(history, config, prefix="classifier")

    y_prob = model.predict(X_test, verbose=0).flatten()
    opt_t = metrics["optimal_threshold"]
    y_pred = (y_prob >= opt_t).astype(int)

    plot_confusion_matrix(y_test, y_pred, config)
    plot_roc_pr_curves(y_test, y_prob, config)

    # Save
    print("\n--- Saving Artifacts ---")
    save_model_artifacts(model, scaler, config, metrics, prefix="classifier")

    print("\n" + "=" * 70)
    print("CLASSIFIER PIPELINE COMPLETE")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  Recall:  {metrics['recall_at_optimal']:.4f}")
    print(f"  F1:      {metrics['f1_at_optimal']:.4f}")
    print(f"  Output:  {config.output_dir}/")
    print("=" * 70)

    return model, scaler, metrics


def run_autoencoder_pipeline(config: Config):
    """
    Standalone autoencoder pipeline — unsupervised anomaly detection.

    Trains the GRU autoencoder on 24-hour windows of NORMAL weather only.
    Evaluates its ability to distinguish extreme from normal events via
    reconstruction error. Saves the model and generates diagnostic plots.

    NOTE: This standalone version does NOT populate latent_z/anomaly_score
    back into the DataFrame. For that, use run_forecast_pipeline() which
    calls _run_autoencoder_and_populate() internally, or use --mode full.
    """
    print("=" * 70)
    print("HeatRadar Nowcast — GRU Autoencoder Pipeline")
    print("=" * 70)
    print(f"  MET level: {config.met_level} ({config.Qm} W/m²)")
    print(f"  Window: {config.ae_window}h")

    df, train_mask = _load_and_preprocess(config)

    # Scale — use a subset of features for autoencoder
    print("\n--- Scaling Features ---")
    ae_cols = ["T_resid", "RH_resid", "SSRD", "hour_sin", "hour_cos"]
    scaled_data, scaler = scale_features(df, train_mask, ae_cols)
    labels = df["extreme"].values

    # Create sequences
    print("\n--- Creating Sequences ---")
    X_all, y_all = create_sequences(scaled_data, labels, config.ae_window)
    print(f"  X shape: {X_all.shape}, y shape: {y_all.shape}")

    # Split
    print("\n--- Temporal Split ---")
    X_train, y_train, X_test, y_test = temporal_train_test_split(
        X_all, y_all, train_mask, config.ae_window
    )

    # Train on NORMAL windows only
    normal_mask = y_train == 0
    X_train_normal = X_train[normal_mask]
    print(f"  Training on {len(X_train_normal)} normal windows "
          f"(excluded {(~normal_mask).sum()} extreme)")

    # Build & train
    print("\n--- Building GRU Autoencoder ---")
    model = build_autoencoder(config.ae_window, X_train.shape[2], config)
    model.summary()

    print("\n--- Training ---")
    history = train_autoencoder(model, X_train_normal, X_test, config)

    # Evaluate
    metrics = evaluate_autoencoder(model, X_train_normal, X_test, y_test, config)

    # Plots
    print("\n--- Generating Plots ---")
    plot_training_history(history, config, prefix="autoencoder")
    plot_autoencoder_errors(model, X_train_normal, X_test, y_test, config)

    # Save
    print("\n--- Saving Artifacts ---")
    save_model_artifacts(model, scaler, config, metrics, prefix="autoencoder")

    print("\n" + "=" * 70)
    print("AUTOENCODER PIPELINE COMPLETE")
    print(f"  Anomaly threshold: {metrics['anomaly_threshold']:.6f}")
    print(f"  Output: {config.output_dir}/")
    print("=" * 70)

    return model, scaler, metrics


def run_full_pipeline(config: Config):
    """
    Full pipeline: ALL three models in correct dependency order.

    ORDER MATTERS:
      1. Load & preprocess data ONCE (shared across all models)
      2. Autoencoder → produces anomaly_score + latent_z, populates df
      3. Classifier  → uses populated autoencoder features + EHI zones
      4. Weather GRU → uses populated autoencoder features + weather data

    The autoencoder MUST run first because both the classifier and Weather GRU
    consume its outputs (anomaly_score, latent_z) as input features.

    Pipeline:
      ERA5 (T2m, D2m, SSRD) → PySolar + RH → LOWESS
      → Autoencoder → {anomaly_score, latent_z} populated into df
      → GRU Classifier → P(extreme) [reads autoencoder features from df]
      → Weather GRU → 168h forecast → EHI-N* zones [reads AE features from df]
    """
    print("=" * 70)
    print("HeatRadar Nowcast — Full Pipeline (Autoencoder → Classifier → Weather GRU)")
    print("=" * 70)

    # 1. Load and preprocess data ONCE (all models share the same df)
    df, train_mask = _load_and_preprocess(config)

    # 2. Run autoencoder FIRST and populate latent_z + anomaly_score into df
    #    This way both the classifier and Weather GRU get real AE features
    df = _run_autoencoder_and_populate(df, train_mask, config)

    # 3. Classifier (legacy model) — now has real anomaly_score + latent_z
    print("\n" + "=" * 70)
    print("Step 2: Training GRU Classifier (legacy architecture)")
    print("=" * 70)

    feature_cols = config.feature_cols
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    scaled_data, cls_scaler = scale_features(df, train_mask, feature_cols)
    labels = df["extreme"].values

    lt = config.classifier_lead_time
    print(f"  Classifier lead_time={lt}h ({lt // 24:.0f} days ahead)")
    X, y = create_sequences(scaled_data, labels, config.lookback, lead_time=lt)
    X_train, y_train, X_test, y_test = temporal_train_test_split(
        X, y, train_mask, config.lookback, lead_time=lt
    )

    cls_model = build_classifier(X_train.shape[1], X_train.shape[2], config)
    cls_model.summary()
    cls_history = train_classifier(cls_model, X_train, y_train, X_test, y_test, config)
    cls_metrics = evaluate_classifier(cls_model, X_test, y_test, config)

    plot_training_history(cls_history, config, prefix="classifier")
    y_prob = cls_model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= cls_metrics["optimal_threshold"]).astype(int)
    plot_confusion_matrix(y_test, y_pred, config)
    plot_roc_pr_curves(y_test, y_prob, config)
    save_model_artifacts(cls_model, cls_scaler, config, cls_metrics, prefix="classifier")

    # 4. Weather GRU (primary model) — also has real anomaly_score + latent_z
    print("\n" + "=" * 70)
    print("Step 3: Training Weather GRU (7-day forecaster)")
    print("=" * 70)

    # Save training diurnal means for inference
    train_df = df.loc[train_mask]
    diurnal_T_mean = train_df.groupby(train_df.index.hour)["Ta_C"].mean().to_dict()
    diurnal_RH_mean = train_df.groupby(train_df.index.hour)["RH"].mean().to_dict()

    # Scale input and target features
    input_cols = config.forecast_input_cols
    for col in input_cols:
        if col not in df.columns:
            df[col] = 0.0
    input_scaled, input_scaler = scale_features(df, train_mask, input_cols)
    target_cols = config.forecast_targets
    target_scaled, target_scaler = scale_features(df, train_mask, target_cols)

    # Create forecast sequences
    X_fc, y_fc = create_forecast_sequences(
        input_scaled, target_scaled, config.lookback, config.forecast_horizon
    )
    n_seq = len(X_fc)
    label_positions = train_mask[config.lookback : config.lookback + n_seq]
    X_fc_train, y_fc_train = X_fc[label_positions], y_fc[label_positions]
    X_fc_test, y_fc_test = X_fc[~label_positions], y_fc[~label_positions]
    print(f"  Train: X={X_fc_train.shape}, y={y_fc_train.shape}")
    print(f"  Test:  X={X_fc_test.shape}, y={y_fc_test.shape}")

    # Build, train, evaluate
    fc_model = build_weather_forecaster(
        config.lookback, len(input_cols), len(target_cols),
        config.forecast_horizon, config
    )
    fc_model.summary()
    fc_history = train_weather_forecaster(
        fc_model, X_fc_train, y_fc_train, X_fc_test, y_fc_test, config
    )
    fc_metrics = evaluate_weather_forecaster(
        fc_model, X_fc_test, y_fc_test, target_scaler, config
    )

    # Plots and artifacts
    plot_training_history(fc_history, config, prefix="weather_gru")
    plot_forecast_example(fc_model, X_fc_test, y_fc_test, target_scaler, config)
    save_forecast_csv(fc_model, X_fc_test, y_fc_test, df, train_mask, target_scaler, config)
    save_model_artifacts(fc_model, input_scaler, config, fc_metrics, prefix="weather_gru")

    import joblib
    joblib.dump(target_scaler, os.path.join(config.output_dir, "target_scaler.pkl"))
    config_path = os.path.join(config.output_dir, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    config_dict["diurnal_T_mean"] = {str(k): v for k, v in diurnal_T_mean.items()}
    config_dict["diurnal_RH_mean"] = {str(k): v for k, v in diurnal_RH_mean.items()}
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("FULL PIPELINE COMPLETE")
    print(f"  Classifier — ROC-AUC: {cls_metrics['roc_auc']:.4f}, F1: {cls_metrics['f1_at_optimal']:.4f}")
    for target in target_cols:
        m = fc_metrics[target]
        print(f"  Weather GRU — {target}: MAE={m['mae_overall']:.3f}, RMSE={m['rmse_overall']:.3f}")
    print(f"  Zone accuracy: {fc_metrics['zone_accuracy']:.4f}")
    print(f"  Output: {config.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    config = parse_args()

    # ---- NAMESPACE OUTPUT BY MET LEVEL ----
    # Ensures runs for different MET levels don't overwrite each other.
    # ./output → ./output/met4  (or met3, met5, etc.)
    if not os.path.basename(config.output_dir).startswith("met"):
        config.output_dir = os.path.join(config.output_dir, f"met{config.met_level}")

    # ---- REPRODUCIBILITY: Set ALL random seeds ----
    # Neural networks use random numbers everywhere: weight initialization,
    # dropout masks, batch shuffling. Without fixed seeds, you get different
    # results every run. With seed=42, EVERYONE gets identical results.
    #
    # We set seeds for ALL sources of randomness:
    #   - numpy: array operations, random sampling
    #   - tensorflow: weight init, dropout, batch order
    #   - random: Python's built-in random module
    #   - PYTHONHASHSEED: Python dict/set ordering (affects data loading)
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    import random
    random.seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)

    # ---- LOAD EHI LOOKUP TABLES ----
    load_lookup_table(config)

    # ---- DISPATCH TO PIPELINE ----
    # Usage examples:
    #   python heatradar_nowcast.py --mode forecast --data_dir ./data
    #       → Runs autoencoder + Weather GRU (primary model)
    #   python heatradar_nowcast.py --mode full --data_dir ./data
    #       → Runs autoencoder → classifier → Weather GRU (all three)
    #   python heatradar_nowcast.py --mode inference
    #       → Live 7-day forecast using trained model + Open-Meteo API
    #   python heatradar_nowcast.py --mode autoencoder
    #       → Trains only the autoencoder (anomaly detection)
    #   python heatradar_nowcast.py --mode classifier
    #       → Trains GRU classifier (predicts 48h ahead by default)
    #   python heatradar_nowcast.py --mode classifier --lead_time 72
    #       → Classifier predicts 3 days ahead
    if config.mode == "forecast":
        run_forecast_pipeline(config)
    elif config.mode == "inference":
        run_inference(config)
    elif config.mode == "classifier":
        run_classifier_pipeline(config)
    elif config.mode == "autoencoder":
        run_autoencoder_pipeline(config)
    elif config.mode == "full":
        run_full_pipeline(config)
    else:
        print(f"Unknown mode: {config.mode}")
        sys.exit(1)
