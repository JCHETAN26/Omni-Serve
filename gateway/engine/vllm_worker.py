"""vLLM AsyncEngine with grammar-constrained JSON decoding (Phase 4).

vLLM is imported lazily inside `start()`, so this module — and everything that
routes through it — imports fine on a laptop. Requires CUDA to actually run.

Constrained decoding masks logits against a grammar compiled from the Invoice
schema, which makes syntactically invalid JSON unrepresentable rather than
merely unlikely. Worth being precise about what that buys: the output is
guaranteed to parse and to match the schema's shape and types. It is not
guaranteed to be *correct* — the model can still emit a well-formed wrong total.
Schema validity rate goes to 100%; field F1 is what the fine-tune is for.
"""

import asyncio
from typing import Any, AsyncIterator

from gateway.engine.base import GenerationMetrics, GenerationResult
from gateway.models.schemas import Invoice
from gateway.prompt import build_messages


class VLLMEngine:
    def __init__(
        self,
        model: str,
        adapter_path: str | None = None,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        include_schema: bool = False,
        guided_backend: str = "outlines",
        enforce_schema: bool = True,
    ) -> None:
        self.model = model
        self.adapter_path = adapter_path
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.include_schema = include_schema
        self.guided_backend = guided_backend
        self.enforce_schema = enforce_schema

        self._engine: Any = None
        self._tokenizer: Any = None
        self._request_counter = 0

    @property
    def ready(self) -> bool:
        return self._engine is not None

    # ---- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        args = AsyncEngineArgs(
            model=self.model,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enable_lora=self.adapter_path is not None,
            # One adapter, but vLLM needs the rank ceiling declared up front.
            max_lora_rank=16,
            disable_log_requests=True,
        )
        self._engine = AsyncLLMEngine.from_engine_args(args)
        self._tokenizer = await self._engine.get_tokenizer()

    async def stop(self) -> None:
        self._engine = None
        self._tokenizer = None

    # ---- request construction -------------------------------------------

    def _sampling_params(self) -> Any:
        from vllm import SamplingParams

        params: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if self.enforce_schema:
            from vllm.sampling_params import GuidedDecodingParams

            params["guided_decoding"] = GuidedDecodingParams(
                json=Invoice.model_json_schema(),
                backend=self.guided_backend,
            )

        return SamplingParams(**params)

    def _lora_request(self) -> Any:
        if self.adapter_path is None:
            return None
        from vllm.lora.request import LoRARequest

        return LoRARequest("omniserve", 1, self.adapter_path)

    def _prompt(self, document: str) -> str:
        """Render via the tokenizer's chat template — the same path training used."""
        messages = build_messages(document, self.include_schema)
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _next_request_id(self) -> str:
        self._request_counter += 1
        return f"omniserve-{self._request_counter}"

    # ---- generation ------------------------------------------------------

    async def _iterate(self, document: str):
        if not self.ready:
            raise RuntimeError("VLLMEngine.start() must be awaited before use")

        generator = self._engine.generate(
            self._prompt(document),
            self._sampling_params(),
            self._next_request_id(),
            lora_request=self._lora_request(),
        )
        async for output in generator:
            yield output

    async def generate(self, document: str) -> GenerationResult:
        metrics = GenerationMetrics()
        text = ""

        async for output in self._iterate(document):
            completion = output.outputs[0]
            if completion.text and not text:
                metrics.mark_first_token()
            text = completion.text
            metrics.completion_tokens = len(completion.token_ids)
            metrics.prompt_tokens = len(output.prompt_token_ids or [])

        metrics.finish()
        return GenerationResult(text=text, metrics=metrics)

    async def stream(self, document: str) -> AsyncIterator[str]:
        """Yield only the newly generated text on each step.

        vLLM hands back the cumulative completion every step, so the gateway
        would re-send the whole prefix on every SSE frame without this diff.
        """
        emitted = 0
        async for output in self._iterate(document):
            text = output.outputs[0].text
            if len(text) > emitted:
                yield text[emitted:]
                emitted = len(text)
                await asyncio.sleep(0)
