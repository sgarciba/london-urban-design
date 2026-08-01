import geopandas as gpd
from config import DATASET_PATH, TRAINED_BOROUGHTS
from pipeline import lst_pipeline
from processing.data_manager import save_pipeline
from processing.split_dataset import borough_spatial_split

import sys
sys.path.append("../")

def run_training() -> None:
    """Train the model."""

    # read training data
    data = gpd.read_file(DATASET_PATH)

    # divide train and test
    X_train, X_test, y_train, y_test = borough_spatial_split(data, TRAINED_BOROUGHTS)

    # fit model
    lst_pipeline.fit(X_train, y_train)

    # persist trained model
    save_pipeline(pipeline_to_persist=lst_pipeline)


if __name__ == "__main__":
    run_training()