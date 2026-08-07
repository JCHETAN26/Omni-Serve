"""HTTP surface: status codes, response shape, SSE framing."""

import json

from fastapi.testclient import TestClient

from gateway.engine.mock_engine import MockEngine
from gateway.main import create_app
from gateway.observability import Metrics
from gateway.service import ExtractionService
from tests.unit.test_service import DOCUMENT, TARGET, BrokenEngine, FakeCache


def client_for(engine=None, cache=None) -> TestClient:
    engine = engine or MockEngine(responses={DOCUMENT: TARGET})
    service = ExtractionService(engine, cache, Metrics())
    return TestClient(create_app(service=service))


def test_health_reports_ready_once_started():
    with client_for() as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["model_ready"] is True
    assert body["cache_ready"] is False


def test_health_reports_cache_when_configured():
    with client_for(cache=FakeCache()) as client:
        assert client.get("/health").json()["cache_ready"] is True


def test_extract_returns_a_validated_invoice():
    with client_for() as client:
        response = client.post("/v1/extract", json={"text": DOCUMENT})

    assert response.status_code == 200
    body = response.json()
    assert body["data"]["total"] == 40.59
    assert body["cached"] is False
    assert body["latency_ms"] >= 0


def test_extract_reports_a_cache_hit():
    with client_for(cache=FakeCache({DOCUMENT: TARGET})) as client:
        body = client.post("/v1/extract", json={"text": DOCUMENT}).json()

    assert body["cached"] is True
    assert body["latency_ms"] == 0.0


def test_bad_model_output_is_502_not_500():
    """The gateway worked; its upstream produced garbage."""
    with client_for(engine=BrokenEngine()) as client:
        response = client.post("/v1/extract", json={"text": DOCUMENT})

    assert response.status_code == 502


def test_empty_text_is_rejected_before_reaching_the_engine():
    engine = MockEngine(responses={DOCUMENT: TARGET})
    with client_for(engine=engine) as client:
        response = client.post("/v1/extract", json={"text": ""})

    assert response.status_code == 422
    assert engine.calls == 0


def test_stream_returns_sse_frames():
    with client_for() as client:
        response = client.post("/v1/extract", json={"text": DOCUMENT, "stream": True})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    frames = [f for f in response.text.split("\n\n") if f.strip()]
    assert frames[-1].startswith("event: done")

    payload = "".join(
        frame.split("data: ", 1)[1] for frame in frames if frame.startswith("event: token")
    )
    assert json.loads(payload) == TARGET


def test_metrics_endpoint_exposes_prometheus_text():
    with client_for() as client:
        client.post("/v1/extract", json={"text": DOCUMENT})
        response = client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    assert "# TYPE omniserve_requests_total counter" in body
    assert "omniserve_requests_total 1" in body
    assert "omniserve_ttft_ms_bucket" in body


def test_metrics_distinguish_cached_from_generated_requests():
    with client_for(cache=FakeCache({DOCUMENT: TARGET})) as client:
        client.post("/v1/extract", json={"text": DOCUMENT})
        client.post("/v1/extract", json={"text": "a different document"})
        body = client.get("/metrics").text

    assert 'omniserve_cache_hits_total{tier="exact"} 1' in body
    assert "omniserve_cache_misses_total 1" in body
