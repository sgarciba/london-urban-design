import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    
    
def test_predict_valid_request(client, sample_features):
    response = client.post("/predict", json=sample_features)
    print(response)
    assert response.status_code == 200
    assert set(response.json()) == {"prediction", "model_version"}
    
    
def test_predict_missing_feature_returns_422(client, sample_features):

    payload = sample_features.copy()
    del payload["NDVI"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "NDVI" in response.text
    
    
