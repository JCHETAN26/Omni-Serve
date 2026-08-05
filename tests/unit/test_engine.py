import json

import pytest

from gateway.engine.base import ExtractionEngine, GenerationMetrics
from gateway.engine.mock_engine import MockEngine
from gateway.models.schemas import Invoice

TARGET = {
    "vendor": "Acme Supply Co.",
    "invoice_number": "INV-2024-0117",
    "invoice_date": "2024-01-17",
    "line_items": [],
    "subtotal": 37.5,
    "tax": 3.09,
    "total": 40.59,
    "currency": "USD",
}


async def test_mock_engine_satisfies_the_protocol():
    engine = MockEngine()
    assert isinstance(engine, ExtractionEngine)


async def test_engine_is_not_ready_until_started():
    engine = MockEngine()
    assert engine.ready is False

    await engine.start()
    assert engine.ready is True

    await engine.stop()
    assert engine.ready is False


async def test_generate_returns_schema_valid_json():
    engine = MockEngine(responses={"doc": TARGET})
    await engine.start()

    result = await engine.generate("doc")

    assert json.loads(result.text) == TARGET
    Invoice.model_validate(json.loads(result.text))


async def test_stream_reassembles_into_the_same_payload():
    engine = MockEngine(responses={"doc": TARGET})
    await engine.start()

    chunks = [chunk async for chunk in engine.stream("doc")]

    assert len(chunks) > 1  # actually streamed, not one blob
    assert json.loads("".join(chunks)) == TARGET


async def test_unknown_document_falls_back_to_valid_json():
    engine = MockEngine()
    await engine.start()

    result = await engine.generate("never seen")

    Invoice.model_validate(json.loads(result.text))


async def test_metrics_capture_ttft_and_totals():
    engine = MockEngine(responses={"doc": TARGET}, ttft_seconds=0.01)
    await engine.start()

    result = await engine.generate("doc")
    snapshot = result.metrics.snapshot()

    assert snapshot["ttft_ms"] >= 10.0
    assert snapshot["total_ms"] >= snapshot["ttft_ms"]
    assert snapshot["completion_tokens"] > 0


def test_ttft_is_none_before_the_first_token():
    metrics = GenerationMetrics()

    assert metrics.ttft_ms is None
    assert metrics.snapshot()["ttft_ms"] is None


def test_first_token_mark_is_not_overwritten():
    metrics = GenerationMetrics()
    metrics.mark_first_token()
    first = metrics.first_token_at
    metrics.mark_first_token()

    assert metrics.first_token_at == first


def test_tokens_per_second_needs_a_finished_generation():
    metrics = GenerationMetrics()
    assert metrics.tokens_per_second == 0.0

    metrics.mark_first_token()
    metrics.completion_tokens = 40
    metrics.finish()
    assert metrics.tokens_per_second > 0.0


async def test_call_count_tracks_gpu_work():
    """Phase 6 asserts cache hits skip the engine; this is how it will check."""
    engine = MockEngine()
    await engine.start()

    await engine.generate("a")
    [chunk async for chunk in engine.stream("b")]

    assert engine.calls == 2


def test_vllm_worker_imports_without_cuda():
    """The lazy-import boundary: constructing must not require vllm."""
    import sys

    from gateway.engine.vllm_worker import VLLMEngine

    engine = VLLMEngine(model="meta-llama/Llama-3.1-8B-Instruct")

    assert engine.ready is False
    assert "vllm" not in sys.modules


async def test_vllm_worker_refuses_to_generate_before_start():
    from gateway.engine.vllm_worker import VLLMEngine

    engine = VLLMEngine(model="x")

    with pytest.raises(RuntimeError, match="start"):
        async for _ in engine._iterate("doc"):
            pass
