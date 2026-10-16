import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import fetch_california_housing
from config import DATASET_PATH, TRAIN_COLS
import geopandas as gpd
from pathlib import Path

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c
    

@pytest.fixture(scope="module")
def valid_payload():
    data = gpd.read_file(DATASET_PATH)
    X = data[TRAIN_COLS]
    return X.iloc[0].to_dict() 


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    
    
def test_predict_valid_request(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    print(response)
    assert response.status_code == 200
    assert set(response.json()) == {"prediction", "model_version"}
    
    
def test_predict_missing_feature_returns_422(client, valid_payload):

    payload = valid_payload.copy()
    del payload["NDVI"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "NDVI" in response.text
    
    
