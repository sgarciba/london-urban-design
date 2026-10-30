import pandas as pd
import pytest
from config import SAMPLE_DATASET_PATH, TRAIN_COLS


@pytest.fixture(scope="session")
def sample_features():
    data = pd.read_csv(SAMPLE_DATASET_PATH)
    X = data[TRAIN_COLS]
    return X.iloc[0].to_dict()
