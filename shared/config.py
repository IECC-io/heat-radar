"""
Heat-Radar shared configuration.

Central source of truth for body parameters, MET levels,
zone definitions, and physical constants used across
both nowcast and longterm pipelines.
"""

# === Indian adult body parameters ===
BODY_MASS_KG = 65        # kg (Indian adult average)
BODY_HEIGHT_M = 1.64     # m (Indian adult average)
BODY_SURFACE_AREA_M2 = 1.71  # m² (DuBois formula)

# === Physical constants ===
CPC = 3492.0             # J/kg/K, specific heat capacity of body
SIGMA = 5.67e-8          # W/m²/K⁴, Stefan-Boltzmann constant

# === MET levels ===
# MET × 58.15 = W/m²
MET_LEVELS = {
    1: {'watts': 58,  'label': 'Resting'},
    2: {'watts': 116, 'label': 'Light Activity'},
    3: {'watts': 174, 'label': 'Moderate Work'},
    4: {'watts': 233, 'label': 'Heavy Work'},
    5: {'watts': 291, 'label': 'Very Heavy Work'},
    6: {'watts': 349, 'label': 'Extreme Labor'},
}

# === EHI-N* Zone definitions ===
ZONES = {
    1: {'name': 'Comfortable',    'color': '#2196f3', 'region': 'phi'},
    2: {'name': 'Mild Stress',    'color': '#4caf50', 'region': 'Rf (oversaturated)'},
    3: {'name': 'Moderate',       'color': '#ffeb3b', 'region': 'Rf (unsaturated)'},
    4: {'name': 'Sweating',       'color': '#ff9800', 'region': 'Rs'},
    5: {'name': 'Max Sweating',   'color': '#f44336', 'region': 'Rs*'},
    6: {'name': 'Hyperdanger',    'color': '#9c27b0', 'region': 'dTcdt'},
}

# === Geographic bounds (India) ===
INDIA_LAT_MIN = 6.0
INDIA_LAT_MAX = 38.0
INDIA_LON_MIN = 66.0
INDIA_LON_MAX = 100.0

# === Kolkata (nowcast default) ===
KOLKATA_LAT = 22.5726
KOLKATA_LON = 88.3639
