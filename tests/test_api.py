import pytest
from fastapi.testclient import TestClient


class TestAPI:
    def test_health_endpoint(self):
        from api.main import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_predict_empty_text(self):
        from api.main import app
        client = TestClient(app)
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422

    def test_predict_too_long(self):
        from api.main import app
        client = TestClient(app)
        response = client.post("/predict", json={"text": "a" * 10001})
        assert response.status_code == 422
