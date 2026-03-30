"""
Download ERA5 SSRD (Surface Solar Radiation Downwards) from ARCO-ERA5 on Google Cloud.

Merges SSRD into existing NetCDF files that only have Ta, Td, RH.

Uses gcsfs for Google Cloud Storage access (no auth required for public data).
Opens the Zarr store ONCE, then selects SSRD only to avoid loading full metadata.
"""

import numpy as np
import xarray as xr
import gcsfs
import os
import sys

DATA_DIR = "./data"
ERA5_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# Kolkata — single grid point (nearest to 22.57°N, 88.36°E)
KOLKATA_LAT = 22.5   # nearest 0.25° grid point
KOLKATA_LON = 88.25  # nearest 0.25° grid point

YEARS = range(2005, 2025)
MONTHS = [3, 4, 5, 6, 7]


def download_ssrd():
    print("=" * 60)
    print("Downloading ERA5 SSRD for Kolkata from ARCO-ERA5")
    print("=" * 60)

    # Connect to Google Cloud Storage
    print("Connecting to Google Cloud Storage...")
    fs = gcsfs.GCSFileSystem(token="anon")
    store = fs.get_mapper(ERA5_URL)

    # Open Zarr store — only load SSRD variable to avoid huge metadata load
    print("Opening ARCO-ERA5 Zarr store (SSRD only)...")
    ds = xr.open_zarr(
        store,
        consolidated=True,
    )

    # Find the SSRD variable name
    ssrd_var = None
    for name in ["surface_solar_radiation_downwards", "ssrd"]:
        if name in ds:
            ssrd_var = name
            break

    if ssrd_var is None:
        # List available vars to help debug
        print(f"ERROR: SSRD not found. Available variables:")
        for v in sorted(ds.data_vars):
            print(f"  - {v}")
        sys.exit(1)

    print(f"  Found: '{ssrd_var}'")

    # Select just SSRD at Kolkata grid point
    print(f"  Selecting grid point: {KOLKATA_LAT}°N, {KOLKATA_LON}°E")
    ssrd_kolkata = ds[ssrd_var].sel(
        latitude=KOLKATA_LAT,
        longitude=KOLKATA_LON,
        method="nearest",
    )

    updated = 0
    skipped = 0
    errors = 0

    for year in YEARS:
        for month in MONTHS:
            filename = f"{DATA_DIR}/era5_kolkata_{year}_{month:02d}.nc"

            if not os.path.exists(filename):
                continue

            # Check if SSRD already exists and is valid
            with xr.open_dataset(filename) as existing:
                if "SSRD" in existing.data_vars:
                    if not np.all(np.isnan(existing["SSRD"].values)):
                        skipped += 1
                        continue

            print(f"\n  {year}-{month:02d}...", end=" ", flush=True)

            try:
                # Select this month's time range
                start = f"{year}-{month:02d}-01"
                if month == 12:
                    end = f"{year + 1}-01-01"
                else:
                    end = f"{year}-{month + 1:02d}-01"

                # Download just this month's SSRD (small chunk)
                ssrd_month = ssrd_kolkata.sel(time=slice(start, end)).compute()

                # Convert J/m² (accumulated per hour) to W/m² (÷ 3600)
                ssrd_wm2 = (ssrd_month / 3600.0).clip(min=0)

                print(f"got {len(ssrd_wm2)} hours, "
                      f"range: {float(ssrd_wm2.min()):.0f}–{float(ssrd_wm2.max()):.0f} W/m²")

                # Load existing file
                existing = xr.open_dataset(filename)

                # Align SSRD timestamps with existing data
                # Existing files may have different time coords, so match by nearest
                ssrd_aligned = ssrd_wm2.reindex(time=existing.time, method="nearest",
                                                 tolerance=np.timedelta64(1, "h"))

                # Drop spatial dims if they exist (we selected a single point)
                if "latitude" in ssrd_aligned.dims:
                    ssrd_aligned = ssrd_aligned.squeeze(["latitude", "longitude"], drop=True)

                existing["SSRD"] = ("time", ssrd_aligned.values)
                existing["SSRD"].attrs = {
                    "units": "W/m2",
                    "long_name": "Surface Solar Radiation Downwards",
                    "source": "ARCO-ERA5",
                }

                # Save
                existing.to_netcdf(filename + ".tmp")
                existing.close()
                os.replace(filename + ".tmp", filename)
                updated += 1

            except Exception as e:
                print(f"ERROR: {e}")
                errors += 1

    print(f"\n\n{'=' * 60}")
    print(f"Done! Updated: {updated}, Skipped (already had SSRD): {skipped}, Errors: {errors}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    download_ssrd()
