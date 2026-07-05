# 🏙️ London Urban Design

Exploring how urban form relates to land surface temperature (LST) across London. Building footprints (OSM) and satellite-derived NDVI/NDBI/LST (Google Earth Engine Landsat data) are combined into a gridded dataset of building density and vegetation/built-up indices, then used to model LST with a Random Forest, testing both random and spatial train/test splits.

## 📂 Structure

- `notebooks/load_dataset.py` — download Landsat raster (GEE via Google Drive) and OSM building footprints
- `notebooks/density_functions.py` — build a raster-aligned grid and compute building density per cell
- `notebooks/data_processing.py` — clean, merge, and produce the final urban design dataset
- `notebooks/train_model.py` — train and evaluate LST prediction models
- `data/` — raw and processed datasets (large files gitignored)
- `figures/` — output maps and plots
