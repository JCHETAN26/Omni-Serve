"""Request orchestration: cache, then engine, then validate.

Kept out of the route handlers so the routing decisions — which are the part
with real behaviour — are testable without HTTP, and so Phase 7 can drive this
directly when isolating gateway overhead from network time.
"""

import json
from dataclasses import dataclass
from typing import AsyncIterator, Protocol

from pydantic import ValidationError

from gateway.engine.base import ExtractionEngine
from gateway.models.schemas import Invoice
from gateway.observability import Metrics


class ResponseCache(Protocol):
    async def get(self, text: str): ...

    async def set(self, text: str, value: dict) -> None: ...


class InvalidModelOutput(RuntimeError):
    """The engine returned something that isn't a valid Invoice.

    Should be impossible once constrained decoding is on — which is exactly why
    it is worth surfacing loudly rather than papering over. If this fires in
    production, the grammar is not actually applied.
    """

    def __init__(self, raw: str) -> None:
        super().__init__("engine returned output that is not a valid Invoice")
        self.raw = raw


@dataclass
class ExtractionOutcome:
    invoice: Invoice
    cached: bool
    tier: str | None
    latency_ms: float
    ttft_ms: float | None


class ExtractionService:
    def __init__(
        self,
        engine: ExtractionEngine,
        cache: ResponseCache | None = None,
        metrics: Metrics | None = None,
    ) -> None:
        self.engine = engine
        self.cache = cache
        self.metrics = metrics or Metrics()

    # ---- helpers ---------------------------------------------------------

    def _parse(self, raw: str) -> Invoice:
        try:
            return Invoice.model_validate_json(raw)
        except (ValidationError, ValueError) as exc:
            self.metrics.increment("omniserve_invalid_output_total")
            raise InvalidModelOutput(raw) from exc

    async def _lookup(self, document: str):
        if self.cache is None:
            return None
        hit = await self.cache.get(document)
        if hit is not None:
            self.metrics.increment("omniserve_cache_hits_total", tier=hit.tier)
        else:
            self.metrics.increment("omniserve_cache_misses_total")
        return hit

    async def _store(self, document: str, invoice: Invoice) -> None:
        """Only ever called with a validated invoice.

        Caching unvalidated output would let one bad generation be served
        repeatedly — a transient failure promoted to a persistent one.
        """
        if self.cache is not None:
            await self.cache.set(document, json.loads(invoice.model_dump_json()))

    # ---- entry points ----------------------------------------------------

    async def extract(self, document: str, use_cache: bool = True) -> ExtractionOutcome:
        self.metrics.increment("omniserve_requests_total")

        if use_cache:
            hit = await self._lookup(document)
            if hit is not None:
                invoice = Invoice.model_validate(hit.value)
                self.metrics.observe("omniserve_request_duration_ms", 0.0, source="cache")
                return ExtractionOutcome(invoice, True, hit.tier, 0.0, None)

        result = await self.engine.generate(document)
        invoice = self._parse(result.text)

        metrics = result.metrics
        self.metrics.observe("omniserve_request_duration_ms", metrics.total_ms, source="engine")
        if metrics.ttft_ms is not None:
            self.metrics.observe("omniserve_ttft_ms", metrics.ttft_ms)
        self.metrics.increment("omniserve_completion_tokens_total", metrics.completion_tokens)

        await self._store(document, invoice)
        return ExtractionOutcome(invoice, False, None, metrics.total_ms, metrics.ttft_ms)

    async def stream(self, document: str, use_cache: bool = True) -> AsyncIterator[dict]:
        """Yield SSE-shaped events: many `token`, then exactly one `done` or `error`.

        Callers rely on a terminal event always arriving — a stream that stops
        after tokens is indistinguishable from a hung connection.
        """
        self.metrics.increment("omniserve_requests_total")

        if use_cache:
            hit = await self._lookup(document)
            if hit is not None:
                payload = json.dumps(hit.value)
                yield {"event": "token", "data": payload}
                yield {"event": "done", "data": {"cached": True, "tier": hit.tier}}
                return

        chunks: list[str] = []
        async for chunk in self.engine.stream(document):
            chunks.append(chunk)
            yield {"event": "token", "data": chunk}

        raw = "".join(chunks)
        try:
            invoice = self._parse(raw)
        except InvalidModelOutput:
            yield {"event": "error", "data": {"reason": "invalid_model_output"}}
            return

        await self._store(document, invoice)
        yield {"event": "done", "data": {"cached": False, "tier": None}}
