"""
Generate EHI-N* Lookup Tables
=============================

Pre-computes EHI values and physiological zones across a grid of
(temperature, humidity, solar load) for each MET level (1-6).

Output: .npz files that the nowcast pipeline can load instead of
requiring Numba/NumbaMinpack at runtime.

Usage:
  python generate_lookup_tables.py
      → Generates tables for all 6 MET levels with default body params
  python generate_lookup_tables.py --met_levels 4
      → Only MET level 4
  python generate_lookup_tables.py --validate
      → Generate + validate against direct computation
  python generate_lookup_tables.py --output_dir ./my_tables

Requires: numpy, heatindex_ek.py (with Numba/NumbaMinpack)
"""

import argparse
import os
import sys
import time
import numpy as np

# Import heatindex_ek from the EHI-Validation repo
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EHI_SRC = os.path.join(SCRIPT_DIR, "..", "EHI-Validation", "src")
if os.path.isdir(EHI_SRC):
    sys.path.insert(0, EHI_SRC)

from heatindex_ek import (
    modifiedheatindex,
    find_eqvar,
    pvstar,
    cpc,
    Pa0,
)

# -------------------------------------------------------------------------
# Grid definitions
# -------------------------------------------------------------------------
# Temperature: 250 K (-23 C) to 330 K (57 C), step 0.25 K = 321 points
TA_MIN, TA_MAX, TA_STEP = 250.0, 330.0, 0.25
# Relative humidity: 0.0 to 1.0, step 0.005 = 201 points
RH_MIN, RH_MAX, RH_STEP = 0.0, 1.0, 0.005
# Solar heat load (for zones only): 0 to 200 W/m2, step 5 = 41 points
QS_MIN, QS_MAX, QS_STEP = 0.0, 200.0, 5.0

MET_LEVELS = {
    1: 65,    # Resting
    2: 130,   # Low activity
    3: 200,   # Moderate activity
    4: 240,   # High activity (outdoor labor)
    5: 290,   # Very high activity
    6: 400,   # Extreme activity
}


def make_grids():
    Ta_grid = np.arange(TA_MIN, TA_MAX + TA_STEP / 2, TA_STEP)
    RH_grid = np.arange(RH_MIN, RH_MAX + RH_STEP / 2, RH_STEP)
    Qs_grid = np.arange(QS_MIN, QS_MAX + QS_STEP / 2, QS_STEP)
    return Ta_grid, RH_grid, Qs_grid


def compute_zone_scalar(Ta_K, RH, Qm, H, M, Qs):
    """Compute EHI-N* zone (1-6) for a single point."""
    A = 0.202 * (M ** 0.425) * (H ** 0.725)
    C = M * cpc / A

    eqvar_name, _ = find_eqvar(Ta_K, RH, Qm, Qs, A, C)

    zone_map = {"phi": 1, "Rf": 2, "Rs": 4, "Rs*": 5, "dTcdt": 6}
    zone = zone_map.get(eqvar_name, 0)

    if eqvar_name == "Rf":
        ehi_shade = float(modifiedheatindex(
            np.float64(Ta_K), np.float64(RH),
            np.float64(Qm), np.float64(0.0),
            np.float64(H), np.float64(M)
        ))
        if Pa0 > pvstar(ehi_shade):
            zone = 2
        else:
            zone = 3

    return zone


def generate_ehi_table(Ta_grid, RH_grid, Qm, mrt, H, M):
    """Compute 2D EHI table for a given (Qm, mrt) combination."""
    n_ta = len(Ta_grid)
    n_rh = len(RH_grid)
    table = np.zeros((n_ta, n_rh), dtype=np.float64)

    for j, rh_val in enumerate(RH_grid):
        ta_arr = Ta_grid.copy()
        rh_arr = np.full_like(ta_arr, rh_val)
        qm_arr = np.full_like(ta_arr, Qm)
        mrt_arr = np.full_like(ta_arr, mrt)
        h_arr = np.full_like(ta_arr, H)
        m_arr = np.full_like(ta_arr, M)
        table[:, j] = modifiedheatindex(ta_arr, rh_arr, qm_arr, mrt_arr, h_arr, m_arr)

    return table


def generate_zone_table(Ta_grid, RH_grid, Qs_grid, Qm, H, M):
    """Compute 3D zone table (Ta x RH x Qs)."""
    n_ta = len(Ta_grid)
    n_rh = len(RH_grid)
    n_qs = len(Qs_grid)
    total = n_ta * n_rh * n_qs
    table = np.zeros((n_ta, n_rh, n_qs), dtype=np.int8)

    done = 0
    t0 = time.time()
    for i, ta_val in enumerate(Ta_grid):
        for j, rh_val in enumerate(RH_grid):
            for k, qs_val in enumerate(Qs_grid):
                table[i, j, k] = compute_zone_scalar(
                    ta_val, rh_val, Qm, H, M, qs_val
                )
                done += 1
        # Progress update every 10 Ta steps
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (total - done) / rate if rate > 0 else 0
            print(f"    Zone progress: {done}/{total} "
                  f"({100*done/total:.1f}%) "
                  f"~{remaining/60:.1f} min remaining")

    return table


def generate_tables_for_met(met_level, Ta_grid, RH_grid, Qs_grid,
                            H, M, output_dir):
    """Generate all lookup tables for a single MET level."""
    Qm = float(MET_LEVELS[met_level])
    print(f"\n{'='*60}")
    print(f"MET level {met_level} (Qm = {Qm} W/m²)")
    print(f"{'='*60}")

    # EHI shade (mrt=0)
    print(f"  Computing EHI shade table ({len(Ta_grid)} x {len(RH_grid)})...")
    t0 = time.time()
    ehi_shade = generate_ehi_table(Ta_grid, RH_grid, Qm, 0.0, H, M)
    print(f"    Done in {time.time()-t0:.1f}s "
          f"(range: {ehi_shade.min()-273.15:.1f} to {ehi_shade.max()-273.15:.1f} °C)")

    # EHI sun (mrt=1)
    print(f"  Computing EHI sun table ({len(Ta_grid)} x {len(RH_grid)})...")
    t0 = time.time()
    ehi_sun = generate_ehi_table(Ta_grid, RH_grid, Qm, 1.0, H, M)
    print(f"    Done in {time.time()-t0:.1f}s "
          f"(range: {ehi_sun.min()-273.15:.1f} to {ehi_sun.max()-273.15:.1f} °C)")

    # Zone table (3D: Ta x RH x Qs)
    n_zone_points = len(Ta_grid) * len(RH_grid) * len(Qs_grid)
    print(f"  Computing zone table "
          f"({len(Ta_grid)} x {len(RH_grid)} x {len(Qs_grid)} "
          f"= {n_zone_points:,} points)...")
    t0 = time.time()
    zone_table = generate_zone_table(Ta_grid, RH_grid, Qs_grid, Qm, H, M)
    print(f"    Done in {time.time()-t0:.1f}s")

    # Zone distribution
    unique, counts = np.unique(zone_table, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    print(f"    Zone distribution: {dist}")

    # Save
    filename = f"ehi_lookup_met{met_level}.npz"
    filepath = os.path.join(output_dir, filename)
    np.savez_compressed(
        filepath,
        Ta_grid=Ta_grid,
        RH_grid=RH_grid,
        Qs_grid=Qs_grid,
        ehi_shade=ehi_shade,
        ehi_sun=ehi_sun,
        zone_table=zone_table,
        Qm=np.float64(Qm),
        H=np.float64(H),
        M=np.float64(M),
    )
    size_mb = os.path.getsize(filepath) / 1e6
    print(f"  Saved: {filepath} ({size_mb:.1f} MB)")

    return ehi_shade, ehi_sun, zone_table


def validate_tables(met_level, output_dir, H, M, n_samples=1000):
    """Validate lookup tables against direct Numba computation."""
    from scipy.interpolate import RegularGridInterpolator

    filepath = os.path.join(output_dir, f"ehi_lookup_met{met_level}.npz")
    data = np.load(filepath)
    Ta_grid = data["Ta_grid"]
    RH_grid = data["RH_grid"]
    Qs_grid = data["Qs_grid"]
    Qm = float(data["Qm"])

    ehi_shade_interp = RegularGridInterpolator(
        (Ta_grid, RH_grid), data["ehi_shade"],
        method="linear", bounds_error=False, fill_value=None
    )
    ehi_sun_interp = RegularGridInterpolator(
        (Ta_grid, RH_grid), data["ehi_sun"],
        method="linear", bounds_error=False, fill_value=None
    )
    zone_interp = RegularGridInterpolator(
        (Ta_grid, RH_grid, Qs_grid), data["zone_table"].astype(np.float64),
        method="nearest", bounds_error=False, fill_value=None
    )

    rng = np.random.default_rng(42)
    Ta_test = rng.uniform(Ta_grid.min(), Ta_grid.max(), n_samples)
    RH_test = rng.uniform(0.01, 0.99, n_samples)
    Qs_test = rng.uniform(0, 150, n_samples)

    print(f"\nValidating MET {met_level} with {n_samples} random samples...")

    # EHI shade
    lookup_shade = ehi_shade_interp(np.column_stack([Ta_test, RH_test]))
    direct_shade = np.array([
        float(modifiedheatindex(
            np.float64(t), np.float64(r), np.float64(Qm),
            np.float64(0.0), np.float64(H), np.float64(M)
        ))
        for t, r in zip(Ta_test, RH_test)
    ])
    shade_err = np.abs(lookup_shade - direct_shade)
    print(f"  EHI shade — max err: {shade_err.max():.4f} K, "
          f"mean err: {shade_err.mean():.4f} K")

    # EHI sun
    lookup_sun = ehi_sun_interp(np.column_stack([Ta_test, RH_test]))
    direct_sun = np.array([
        float(modifiedheatindex(
            np.float64(t), np.float64(r), np.float64(Qm),
            np.float64(1.0), np.float64(H), np.float64(M)
        ))
        for t, r in zip(Ta_test, RH_test)
    ])
    sun_err = np.abs(lookup_sun - direct_sun)
    print(f"  EHI sun   — max err: {sun_err.max():.4f} K, "
          f"mean err: {sun_err.mean():.4f} K")

    # Zones
    lookup_zones = np.round(zone_interp(
        np.column_stack([Ta_test, RH_test, Qs_test])
    )).astype(int)
    direct_zones = np.array([
        compute_zone_scalar(t, r, Qm, H, M, q)
        for t, r, q in zip(Ta_test, RH_test, Qs_test)
    ])
    zone_match = np.mean(lookup_zones == direct_zones)
    print(f"  Zones     — accuracy: {zone_match*100:.1f}%")

    mismatches = np.where(lookup_zones != direct_zones)[0]
    if len(mismatches) > 0:
        print(f"    {len(mismatches)} mismatches (expected near zone boundaries)")
        for idx in mismatches[:5]:
            print(f"      Ta={Ta_test[idx]-273.15:.1f}°C, "
                  f"RH={RH_test[idx]:.2f}, Qs={Qs_test[idx]:.0f} → "
                  f"lookup={lookup_zones[idx]}, direct={direct_zones[idx]}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate EHI-N* lookup tables for the nowcast pipeline"
    )
    parser.add_argument(
        "--met_levels", type=int, nargs="+", default=list(MET_LEVELS.keys()),
        help="Which MET levels to generate (default: all 1-6)"
    )
    parser.add_argument(
        "--height", type=float, default=1.72,
        help="Body height in meters (default: 1.72)"
    )
    parser.add_argument(
        "--mass", type=float, default=70.0,
        help="Body mass in kg (default: 70.0)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: ./lookup_tables/)"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run validation after generation"
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(SCRIPT_DIR, "lookup_tables")
    os.makedirs(output_dir, exist_ok=True)

    Ta_grid, RH_grid, Qs_grid = make_grids()
    print(f"Grid: Ta {len(Ta_grid)} pts [{TA_MIN}-{TA_MAX} K], "
          f"RH {len(RH_grid)} pts [{RH_MIN}-{RH_MAX}], "
          f"Qs {len(Qs_grid)} pts [{QS_MIN}-{QS_MAX} W/m²]")
    print(f"Body: H={args.height} m, M={args.mass} kg")
    print(f"Output: {output_dir}")

    t_total = time.time()
    for met in args.met_levels:
        if met not in MET_LEVELS:
            print(f"Unknown MET level {met}, skipping")
            continue
        generate_tables_for_met(
            met, Ta_grid, RH_grid, Qs_grid,
            args.height, args.mass, output_dir
        )

    print(f"\nTotal generation time: {(time.time()-t_total)/60:.1f} minutes")

    if args.validate:
        for met in args.met_levels:
            if met in MET_LEVELS:
                validate_tables(met, output_dir, args.height, args.mass)


if __name__ == "__main__":
    main()
