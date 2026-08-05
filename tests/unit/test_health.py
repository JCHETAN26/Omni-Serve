from fastapi.testclient import TestClient

from gateway.main import create_app


def test_health_reports_degraded_without_engine():
    client = TestClient(create_app())
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_ready"] is False
    assert body["cache_ready"] is False
