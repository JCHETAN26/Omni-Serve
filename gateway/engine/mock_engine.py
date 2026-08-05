"""GPU-free engine for tests, local development, and honest load-test baselines.

Satisfies the same protocol as `VLLMEngine`, so the gateway, cache and SSE paths
can be exercised end to end without CUDA. Streaming is chunked and optionally
delayed so timing code sees a realistic shape rather than one instant burst.
"""

import asyncio
import json
from typing import AsyncIterator

from gateway.engine.base import GenerationMetrics, GenerationResult
from gateway.models.schemas import Invoice

FALLBACK = Invoice(
    vendor="Mock Vendor Inc.",
    invoice_number="INV-0000-00000",
    invoice_date="2024-01-01",
    line_items=[],
    subtotal=0.0,
    tax=0.0,
    total=0.0,
    currency="USD",
)


class MockEngine:
    def __init__(
        self,
        responses: dict[str, dict] | None = None,
        chunk_size: int = 16,
        ttft_seconds: float = 0.0,
        inter_token_seconds: float = 0.0,
    ) -> None:
        self.responses = responses or {}
        self.chunk_size = chunk_size
        self.ttft_seconds = ttft_seconds
        self.inter_token_seconds = inter_token_seconds
        self._ready = False
        self.calls = 0

    @property
    def ready(self) -> bool:
        return self._ready

    async def start(self) -> None:
        self._ready = True

    async def stop(self) -> None:
        self._ready = False

    def _payload(self, document: str) -> str:
        if document in self.responses:
            return json.dumps(self.responses[document], separators=(",", ":"))
        return FALLBACK.model_dump_json()

    async def generate(self, document: str) -> GenerationResult:
        metrics = GenerationMetrics()
        self.calls += 1

        if self.ttft_seconds:
            await asyncio.sleep(self.ttft_seconds)
        metrics.mark_first_token()

        text = self._payload(document)
        metrics.completion_tokens = max(1, len(text) // 4)
        metrics.prompt_tokens = max(1, len(document) // 4)
        metrics.finish()
        return GenerationResult(text=text, metrics=metrics)

    async def stream(self, document: str) -> AsyncIterator[str]:
        self.calls += 1
        if self.ttft_seconds:
            await asyncio.sleep(self.ttft_seconds)

        text = self._payload(document)
        for start in range(0, len(text), self.chunk_size):
            yield text[start : start + self.chunk_size]
            if self.inter_token_seconds:
                await asyncio.sleep(self.inter_token_seconds)
