from pathlib import Path

FOLDER_NAME = "GEE_Exports"
FILE_NAME = "London_Landsat_Features_2022.tif"

DATASET_PATH = "data/london_urban_design_data.gpkg"

ADMINISTRATIVE_AREA = "Greater London, United Kingdom"
KEEP_COLS = ["geometry", "building", "name", "building:levels", "height"]

TARGET_VAR = "LST"
TRAIN_COLS = [
    "NDVI",
    "NDBI",
    "MNDWI",
    "ALBEDO",
    "building_density",
    "building_volume_density",
]

TRAINED_BOROUGHTS = ["City of London", "Kensington and Chelsea", "Harrow"]

TRAINED_MODEL_DIR = Path("regression_model/trained_model")

if __name__ == "__main__":
    pass
