import os
import osmnx as ox
import sys
sys.path.append("../")
import geopandas as gpd

from load_dataset import download_buildings_osmnx, load_landsat_data
from density_functions import (
    points_to_cells, compute_density, attach_density_to_points,
    compute_volume_density, attach_volume_density_to_points,
)
from config import ADMINISTRATIVE_AREA, FILE_NAME


def run():
    # Load raw tables
    buildings_df = gpd.read_file("../data/osmnx_buildings.gpkg")
    landsat_gdf = load_landsat_data(FILE_NAME)

    # This gets the actual administrative polygon for Greater London
    london_gdf = ox.geocode_to_gdf(ADMINISTRATIVE_AREA)
    london_polygon = london_gdf.geometry.iloc[0]

    # Landsat: Missing Values and filtering bounds
    mask = landsat_gdf.within(london_polygon)
    landsat_gdf_clean = landsat_gdf[mask]
    print(landsat_gdf_clean.isna().mean())
    landsat_gdf_clean = landsat_gdf_clean.dropna()

    print('Landsat Before:', landsat_gdf.shape)
    print('Landsat After:', landsat_gdf_clean.shape)

    # Buildings: Missing Values and filtering bounds
    mask = buildings_df.within(london_polygon)
    buildings_df_clean = buildings_df[mask]
    print(buildings_df_clean.isna().mean())
    # buildings_df_clean = buildings_df_clean.dropna()

    print('Buildings Before:', buildings_df.shape)
    print('Buildings After:', buildings_df_clean.shape)

    cells = points_to_cells(landsat_gdf_clean)          # auto-infers lon/lat resolution separately
    density_grid = compute_density(buildings_df_clean, cells)
    volume_grid = compute_volume_density(buildings_df_clean, cells)

    london_urban_design_df = attach_density_to_points(landsat_gdf_clean, density_grid)
    london_urban_design_df = attach_volume_density_to_points(london_urban_design_df, volume_grid)

    # ======================
    # Sanity Checks
    # ======================

    ## 1. How many cells have building density > 1.0? (should be 0)
    pct_over = (london_urban_design_df['building_density'] > 1.0001).mean() * 100
    print(f"{pct_over:.2f}% of cells still exceed 1.0 after dissolving")

    if pct_over > 0:
        # worth inspecting these specific cells rather than blindly clipping
        problem_cells = london_urban_design_df[london_urban_design_df['building_density'] > 1.0001]
        print(problem_cells[['lon', 'lat', 'building_density']])

    # ======================
    # Save the final table — train_model.py and 02_data_exploration.py both
    # read this file from disk, so it needs to be persisted, not just held
    # in memory.
    # ======================
    out_gpkg = "../data/london_urban_design_data.gpkg"
    if os.path.exists(out_gpkg):
        # GPKG is a multi-layer SQLite container — writing to an existing file
        # can add a new layer alongside whatever's already there rather than
        # replacing it outright, and gpd.read_file() silently defaults to
        # GDAL's first-listed layer rather than the one just written. Removing
        # the file first guarantees a single, unambiguous layer on every run.
        os.remove(out_gpkg)
    london_urban_design_df.to_file(out_gpkg, driver="GPKG")
    print(f"Saved {out_gpkg}")

    out_csv = "../data/london_urban_design_data.csv"
    london_urban_design_df.drop(columns="geometry").to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    run()
