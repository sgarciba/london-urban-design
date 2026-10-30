# 🏙️ London Urban Design

![CI](https://github.com/sgarciba/london-urban-design/actions/workflows/ci.yml/badge.svg)

Exploring how urban form relates to land surface temperature (LST) across London. Building footprints (OSM) and satellite-derived NDVI/NDBI/MNDWI/Albedo/LST (Google Earth Engine Landsat data) are combined into a gridded dataset of building density and vegetation/built-up/water/reflectance indices, then used to model LST with a Random Forest, testing both random and spatial train/test splits.

## 📂 Structure

- `notebooks/` — data pipeline and exploration
  - `pipeline.py` — CLI that runs the data steps (`download_landsat`, `download_buildings`, `process`)
  - `load_dataset.py` — download Landsat raster (GEE via Google Drive) and OSM building footprints
  - `density_functions.py` — build a raster-aligned grid and compute building density per cell
  - `data_processing.py` — clean, merge, and produce the final urban design dataset
  - `data_exploration.py` — per-borough maps, distributions and correlations
  - `train_model.py` — exploratory model selection (random vs. spatial CV)
- `regression_model/` — packaged, versioned LST model
  - `pipeline.py` — scikit-learn `Pipeline` (Random Forest regressor)
  - `train_pipeline.py` — trains on a borough-based spatial split and saves the model
  - `predict.py` — `make_prediction()` using the saved pipeline
  - `processing/` — dataset split (`split_dataset.py`) and model save/load (`data_manager.py`)
  - `trained_model/` — persisted model, `regression_model_output_v<VERSION>.pkl`
  - `VERSION` — model package version
- `src/app/` — FastAPI service serving the model
  - `main.py` — app with `/health` and `/predict` endpoints
  - `model.py` — load the pipeline and run a single prediction
  - `schemas.py` — Pydantic request/response models
  - `config.py` — model path and API model version
- `tests/` — pytest suite for the model (`test_model.py`) and the API (`test_api.py`), with shared fixtures in `conftest.py`
- `.github/workflows/ci.yml` — GitHub Actions CI (runs the tests)
- `config.py` — shared settings (dataset paths, features, target, training boroughs)
- `data/` — raw and processed datasets (gitignored, except the small `london_urban_design_sample.csv` used by tests)
- `figures/` — output maps and plots

## ⚙️ Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To download raw data, copy `credentials.py.example` to `credentials.py` and fill in your Google Drive OAuth client secrets path.

## 🔄 Data pipeline

Run from inside `notebooks/`:

```bash
cd notebooks
python pipeline.py --steps download_landsat,download_buildings,process
```

`process` is the default step. The `download_*` steps require an interactive Google OAuth login and re-fetch large files, so they are opt-in. The output dataset is written to `data/london_urban_design_data.gpkg` (plus a `.csv` copy and a 50-row `london_urban_design_sample.csv` for tests).

## 🧠 Training the model

```bash
python regression_model/train_pipeline.py
```

The model is trained on the boroughs listed in `TRAINED_BOROUGHTS` (`config.py`) and evaluated on the rest of London (spatial split), then saved to `regression_model/trained_model/regression_model_output_v<VERSION>.pkl`. Saving a new model removes older versions from that folder.

## 🚀 API

A FastAPI service loads the trained model once at startup and serves predictions.

```bash
uvicorn app.main:app --reload --app-dir src
```

Interactive docs are available at `http://127.0.0.1:8000/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `POST` | `/predict` | Predict LST (°C) for one grid cell |

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "NDVI": 0.25,
        "NDBI": 0.05,
        "MNDWI": -0.30,
        "ALBEDO": 0.15,
        "building_density": 0.40,
        "building_volume_density": 1.20
      }'
```

Example response (illustrative values):

```json
{"prediction": 31.4, "model_version": "0.1.0"}
```

All six features are required; a missing or non-numeric field returns `422 Unprocessable Entity`.

## 🧪 Tests

```bash
python -m pytest -v
```

`pyproject.toml` adds `src/` to the Python path, so tests import the app as `app`. Tests run on the committed model and on `data/london_urban_design_sample.csv`, a fixed-seed 50-row sample written by `data_processing.py` alongside the full dataset (which is gitignored). They don't need the full dataset or network access.

- `tests/conftest.py` — shared `sample_features` fixture loaded from the sample CSV
- `tests/test_model.py` — prediction output keys and type, plausible LST range (20–50 °C), and that higher NDVI predicts lower LST
- `tests/test_api.py` — `/health`, a valid `/predict` request, and a `422` response when a feature is missing

## 🔁 CI

GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on every push to `main` and on pull requests: it installs dependencies on Python 3.10 and runs the pytest suite against the committed model and sample data.

## 📊 Model variables

| Variable | Role | Description | Source |
|---|---|---|---|
| `LST` | Target | Land Surface Temperature (°C) | Landsat/GEE |
| `NDVI` | Feature | Normalized Difference Vegetation Index — vegetation greenness, roughly -1 to 1 | Landsat/GEE |
| `NDBI` | Feature | Normalized Difference Built-up Index — built-up/impervious surface signal | Landsat/GEE |
| `MNDWI` | Feature | Modified Normalized Difference Water Index — water/moisture proxy | Landsat/GEE |
| `ALBEDO` | Feature | Broadband surface albedo (Liang 2001 coefficients, adapted for Landsat 8) — surface reflectivity, roughly 0-1 | Landsat/GEE |
| `building_area` | Feature | Total OSM building footprint area per cell (m²), after dissolving overlapping polygons | Computed — `density_functions.py: compute_density` |
| `building_density` | Feature | Fraction of the pixel cell covered by OSM building footprint (0-1), after dissolving overlapping polygons to avoid double-counting | Computed — `density_functions.py: compute_density` |
| `building_volume` | Feature | Footprint area weighted by the OSM `building:levels` tag, summed per cell (m² × storeys) — a rough proxy for building mass/height, since footprint density alone can't tell a low sprawling building apart from a tower on the same footprint. Not a true height measurement — OSM's `height` tag is present on <1% of buildings, and `building:levels` coverage itself varies a lot by borough | Computed — `density_functions.py: compute_volume_density` |
| `building_volume_density` | Feature | `building_volume` divided by cell area — levels-weighted building density, distinguishing a tall tower from a low sprawling building with the same footprint | Computed — `density_functions.py: compute_volume_density` |
