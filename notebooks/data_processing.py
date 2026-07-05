import osmnx as ox
import sys
sys.path.append("../")
import geopandas as gpd

from load_dataset import download_buildings_osmnx, load_landsat_data
from density_functions import points_to_cells, compute_density, attach_density_to_points
from config import ADMINISTRATIVE_AREA


# Download the buildings table
# download_buildings_osmnx(ADMINISTRATIVE_AREA, KEEP_COLS, req_timeout=600)


# Load raw tables
buildings_df = gpd.read_file("../data/osmnx_buildings.gpkg")
landsat_gdf = load_landsat_data("London_LST_NDVI_NDBI.tif")


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
london_urban_design_df = attach_density_to_points(landsat_gdf_clean, density_grid)


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