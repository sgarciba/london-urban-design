import pytest
from app.model import load_model, predict
import geopandas as gpd
from config import DATASET_PATH, TRAIN_COLS

@pytest.fixture(scope="module")
def model():
    return load_model()


@pytest.fixture(scope="module")
def sample_london():
    data = gpd.read_file(DATASET_PATH)
    X = data[TRAIN_COLS]
    return X.iloc[0].to_dict() 


def test_predict_returns_expected_keys(model, sample_london):
    result = predict(model, sample_london)
    assert set(result) == {"prediction"}
    
    

def test_lst_is_positive_float(model, sample_london):
    lst = predict(model, sample_london)["prediction"]
    assert isinstance(lst, float)
    assert lst > 0


def test_lst_is_in_plausible_range(model, sample_london):
    lst = predict(model, sample_london)["prediction"]
    assert 20 <= lst <= 50


def test_higher_ndvi_predicts_lower_lst(model, sample_london):
    greener_area = {**sample_london, "NDVI": sample_london["NDVI"] + 3}
    base_lst = predict(model, sample_london)["prediction"]
    lower_lst = predict(model, greener_area)["prediction"]
    assert lower_lst < base_lst

    