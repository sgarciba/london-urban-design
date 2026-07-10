# 🏙️ London Urban Design

Exploring how urban form relates to land surface temperature (LST) across London. Building footprints (OSM) and satellite-derived NDVI/NDBI/LST (Google Earth Engine Landsat data) are combined into a gridded dataset of building density and vegetation/built-up indices, then used to model LST with a Random Forest, testing both random and spatial train/test splits.

## 📂 Structure

- `notebooks/load_dataset.py` — download Landsat raster (GEE via Google Drive) and OSM building footprints
- `notebooks/density_functions.py` — build a raster-aligned grid and compute building density per cell
- `notebooks/data_processing.py` — clean, merge, and produce the final urban design dataset
- `notebooks/train_model.py` — train and evaluate LST prediction models
- `data/` — raw and processed datasets (large files gitignored)
- `figures/` — output maps and plots

## 📊 Model variables

| Variable | Role | Description | Source |
|---|---|---|---|
| `LST` | Target | Land Surface Temperature (°C) | Landsat/GEE |
| `NDVI` | Feature | Normalized Difference Vegetation Index — vegetation greenness, roughly -1 to 1 | Landsat/GEE |
| `NDBI` | Feature | Normalized Difference Built-up Index — built-up/impervious surface signal | Landsat/GEE |
| `MNDWI` | Feature | Modified Normalized Difference Water Index — water/moisture proxy | Landsat/GEE |
| `building_density` | Feature | Fraction of the pixel cell covered by OSM building footprint (0-1), after dissolving overlapping polygons to avoid double-counting | Computed — `density_functions.py: compute_density` |
| `building_volume` | Feature | Footprint area weighted by the OSM `building:levels` tag, summed per cell (m² × storeys) — a rough proxy for building mass/height, since footprint density alone can't tell a low sprawling building apart from a tower on the same footprint. Not a true height measurement — OSM's `height` tag is present on <1% of buildings, and `building:levels` coverage itself varies a lot by borough | Computed — `density_functions.py: compute_volume_density` |
