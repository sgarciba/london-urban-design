from regression_model.predict import make_prediction
import geopandas as gpd
from config import DATASET_PATH, TRAINED_BOROUGHTS
from regression_model.processing.split_dataset import borough_spatial_split


data = gpd.read_file(DATASET_PATH)

X_train, X_test, y_train, y_test = borough_spatial_split(data, TRAINED_BOROUGHTS)

result = make_prediction(input_data=X_test)

