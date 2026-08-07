"""Routing behaviour: what actually reaches the GPU, and what gets cached."""

import json

import pytest

from gateway.cache.semantic_cache import CacheHit
from gateway.engine.mock_engine import MockEngine
from gateway.models.schemas import Invoice
from gateway.observability import Metrics
from gateway.service import ExtractionService, InvalidModelOutput

DOCUMENT = "*** ACME SUPPLY CO. ***\nTOT $40.59"
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


class FakeCache:
    def __init__(self, seeded: dict | None = None, tier: str = "exact") -> None:
        self.store = dict(seeded or {})
        self.tier = tier
        self.writes: list[tuple[str, dict]] = []

    async def get(self, text: str):
        if text in self.store:
            return CacheHit(self.store[text], self.tier, 1.0)
        return None

    async def set(self, text: str, value: dict) -> None:
        self.store[text] = value
        self.writes.append((text, value))


class BrokenEngine(MockEngine):
    async def generate(self, document: str):
        result = await super().generate(document)
        result.text = "I could not find an invoice."
        return result

    async def stream(self, document: str):
        yield "not json at all"


async def build(engine=None, cache=None):
    engine = engine or MockEngine(responses={DOCUMENT: TARGET})
    await engine.start()
    return ExtractionService(engine, cache, Metrics()), engine


async def test_miss_calls_the_engine_and_returns_a_validated_invoice():
    service, engine = await build()

    outcome = await service.extract(DOCUMENT)

    assert isinstance(outcome.invoice, Invoice)
    assert outcome.invoice.total == 40.59
    assert outcome.cached is False
    assert engine.calls == 1


async def test_cache_hit_never_touches_the_engine():
    """The entire point of Phase 5 — assert the GPU is actually skipped."""
    cache = FakeCache({DOCUMENT: TARGET})
    service, engine = await build(cache=cache)

    outcome = await service.extract(DOCUMENT)

    assert outcome.cached is True
    assert outcome.tier == "exact"
    assert engine.calls == 0


async def test_use_cache_false_bypasses_a_warm_cache():
    cache = FakeCache({DOCUMENT: TARGET})
    service, engine = await build(cache=cache)

    outcome = await service.extract(DOCUMENT, use_cache=False)

    assert outcome.cached is False
    assert engine.calls == 1


async def test_successful_generation_is_written_back():
    cache = FakeCache()
    service, _ = await build(cache=cache)

    await service.extract(DOCUMENT)

    assert len(cache.writes) == 1
    assert cache.writes[0][1]["total"] == 40.59


async def test_invalid_output_raises_and_is_not_cached():
    """A bad generation must not become a permanently served bad answer."""
    cache = FakeCache()
    service, _ = await build(engine=BrokenEngine(), cache=cache)

    with pytest.raises(InvalidModelOutput):
        await service.extract(DOCUMENT)

    assert cache.writes == []
    assert service.metrics.get("omniserve_invalid_output_total") == 1


async def test_metrics_separate_hits_misses_and_tiers():
    cache = FakeCache({DOCUMENT: TARGET}, tier="semantic")
    service, _ = await build(cache=cache)

    await service.extract(DOCUMENT)
    await service.extract("some other document")

    assert service.metrics.get("omniserve_cache_hits_total", tier="semantic") == 1
    assert service.metrics.get("omniserve_cache_misses_total") == 1
    assert service.metrics.get("omniserve_requests_total") == 2


async def test_ttft_is_recorded_for_engine_requests():
    engine = MockEngine(responses={DOCUMENT: TARGET}, ttft_seconds=0.01)
    service, _ = await build(engine=engine)

    await service.extract(DOCUMENT)

    histogram = service.metrics.histogram("omniserve_ttft_ms")
    assert histogram is not None and histogram.observations == 1


async def test_stream_emits_tokens_then_exactly_one_terminal_event():
    service, _ = await build()

    events = [event async for event in service.stream(DOCUMENT)]

    assert events[-1]["event"] == "done"
    assert [e["event"] for e in events].count("done") == 1
    assert all(e["event"] == "token" for e in events[:-1])


async def test_streamed_tokens_reassemble_into_the_payload():
    service, _ = await build()

    events = [event async for event in service.stream(DOCUMENT)]
    body = "".join(e["data"] for e in events if e["event"] == "token")

    assert json.loads(body) == TARGET


async def test_stream_cache_hit_skips_the_engine():
    cache = FakeCache({DOCUMENT: TARGET})
    service, engine = await build(cache=cache)

    events = [event async for event in service.stream(DOCUMENT)]

    assert engine.calls == 0
    assert events[-1]["data"]["cached"] is True


async def test_stream_terminates_with_error_rather_than_silence():
    """A stream that just stops is indistinguishable from a hung connection."""
    cache = FakeCache()
    service, _ = await build(engine=BrokenEngine(), cache=cache)

    events = [event async for event in service.stream(DOCUMENT)]

    assert events[-1]["event"] == "error"
    assert cache.writes == []


async def test_service_works_without_a_cache_configured():
    service, engine = await build(cache=None)

    outcome = await service.extract(DOCUMENT)

    assert outcome.cached is False
    assert engine.calls == 1
