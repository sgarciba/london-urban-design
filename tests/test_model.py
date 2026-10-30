import pytest
from app.model import load_model, predict

@pytest.fixture(scope="module")
def model():
    return load_model()


def test_predict_returns_expected_keys(model, sample_features):
    result = predict(model, sample_features)
    assert set(result) == {"prediction"}
    
    

def test_lst_is_positive_float(model, sample_features):
    lst = predict(model, sample_features)["prediction"]
    assert isinstance(lst, float)
    assert lst > 0


def test_lst_is_in_plausible_range(model, sample_features):
    lst = predict(model, sample_features)["prediction"]
    assert 20 <= lst <= 50


def test_higher_ndvi_predicts_lower_lst(model, sample_features):
    greener_area = {**sample_features, "NDVI": sample_features["NDVI"] + 3}
    base_lst = predict(model, sample_features)["prediction"]
    lower_lst = predict(model, greener_area)["prediction"]
    assert lower_lst < base_lst

    