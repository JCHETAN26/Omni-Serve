"""Engine interface shared by the vLLM worker and the mock.

The gateway depends on this protocol, never on vLLM directly, so routing,
caching and observability stay testable on a machine with no GPU — and so
Phase 7 can load-test the gateway itself without a model in the way.
"""

import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol, runtime_checkable


@dataclass
class GenerationMetrics:
    """Per-request timings. TTFT is the headline number for Phase 7."""

    started_at: float = field(default_factory=time.perf_counter)
    first_token_at: float | None = None
    finished_at: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def mark_first_token(self) -> None:
        if self.first_token_at is None:
            self.first_token_at = time.perf_counter()

    def finish(self) -> None:
        self.finished_at = time.perf_counter()

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_at is None:
            return None
        return (self.first_token_at - self.started_at) * 1000

    @property
    def total_ms(self) -> float:
        end = self.finished_at or time.perf_counter()
        return (end - self.started_at) * 1000

    @property
    def tokens_per_second(self) -> float:
        if not self.first_token_at or not self.finished_at:
            return 0.0
        decode_seconds = self.finished_at - self.first_token_at
        return self.completion_tokens / decode_seconds if decode_seconds > 0 else 0.0

    def snapshot(self) -> dict:
        return {
            "ttft_ms": round(self.ttft_ms, 2) if self.ttft_ms is not None else None,
            "total_ms": round(self.total_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tokens_per_second": round(self.tokens_per_second, 2),
        }


@dataclass
class GenerationResult:
    text: str
    metrics: GenerationMetrics


@runtime_checkable
class ExtractionEngine(Protocol):
    """Minimal surface the gateway needs from a model backend."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    @property
    def ready(self) -> bool: ...

    async def generate(self, document: str) -> GenerationResult: ...

    def stream(self, document: str) -> AsyncIterator[str]: ...
