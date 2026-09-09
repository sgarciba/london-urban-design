import geopandas as gpd
from config import DATASET_PATH, TRAINED_BOROUGHTS
from regression_model.pipeline import lst_pipeline
from regression_model.processing.data_manager import save_pipeline
from regression_model.processing.split_dataset import borough_spatial_split

def run_training() -> None:
    """Train the model."""

    # 1. Read training data
    print(f"1. Reading training data from {DATASET_PATH}")
    data = gpd.read_file(DATASET_PATH)

    # 2. Split dataset
    print("2. Splitting dataset")
    X_train, X_test, y_train, y_test = borough_spatial_split(data, TRAINED_BOROUGHTS)

    # 3. Fit model
    print("3. Fitting model")
    lst_pipeline.fit(X_train, y_train)

    # 4. Saved trained model
    print("4. Saving trained model")
    save_pipeline(pipeline_to_persist=lst_pipeline)


if __name__ == "__main__":
    run_training()