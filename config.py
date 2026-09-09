
from credentials import GOOGLE_DRIVE_CLIENT_SECRETS_JSON

FOLDER_NAME = "GEE_Exports"
FILE_NAME = "London_Landsat_Features_2022.tif"

ADMINISTRATIVE_AREA = "Greater London, United Kingdom"
KEEP_COLS = ["geometry", "building", "name", "building:levels", "height"]

TRAINED_BOROUGHTS = ["City of London", "Kensington and Chelsea", "Harrow"]

TRAINED_MODEL_DIR = "regression_model/trained_model"

if __name__ == "__main__":
    pass
