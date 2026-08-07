"""OmniServe FastAPI gateway (Phase 6).

Wires the cache, engine and metrics behind three endpoints. The engine is
chosen at startup: a real vLLM worker when `--model` is given, otherwise the
mock, so the gateway is runnable and load-testable on a machine with no GPU.
"""

import argparse
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse

from gateway import __version__
from gateway.engine.mock_engine import MockEngine
from gateway.models.schemas import ExtractRequest, ExtractResponse, HealthResponse
from gateway.observability import Metrics
from gateway.service import ExtractionService, InvalidModelOutput

DEFAULT_SETTINGS = {
    "model": None,
    "adapter": None,
    "redis_url": None,
    "enforce_schema": True,
}


def build_engine(settings: dict):
    if not settings.get("model"):
        return MockEngine()

    from gateway.engine.vllm_worker import VLLMEngine

    return VLLMEngine(
        model=settings["model"],
        adapter_path=settings.get("adapter"),
        enforce_schema=settings.get("enforce_schema", True),
    )


async def build_cache(settings: dict):
    if not settings.get("redis_url"):
        return None

    from gateway.cache.semantic_cache import SemanticCache

    cache = SemanticCache(redis_url=settings["redis_url"])
    await cache.connect()
    return cache


def create_app(settings: dict | None = None, service: ExtractionService | None = None) -> FastAPI:
    """`service` is injectable so tests drive the real routes without a GPU."""
    settings = {**DEFAULT_SETTINGS, **(settings or {})}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is not None:
            app.state.service = service
            app.state.engine = service.engine
            app.state.cache = service.cache
            # The app owns engine lifecycle even when the service is injected;
            # otherwise /health reports degraded for a perfectly good engine.
            if not service.engine.ready:
                await service.engine.start()
        else:
            engine = build_engine(settings)
            await engine.start()
            cache = await build_cache(settings)
            app.state.engine = engine
            app.state.cache = cache
            app.state.service = ExtractionService(engine, cache, app.state.metrics)
        yield
        engine = getattr(app.state, "engine", None)
        if engine is not None:
            await engine.stop()
        cache = getattr(app.state, "cache", None)
        if cache is not None and hasattr(cache, "close"):
            await cache.close()

    app = FastAPI(title="OmniServe", version=__version__, lifespan=lifespan)
    app.state.metrics = service.metrics if service is not None else Metrics()
    app.state.settings = settings

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        engine = getattr(request.app.state, "engine", None)
        cache = getattr(request.app.state, "cache", None)
        model_ready = bool(engine is not None and engine.ready)
        return HealthResponse(
            status="ok" if model_ready else "degraded",
            version=__version__,
            model_ready=model_ready,
            cache_ready=cache is not None,
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics(request: Request) -> str:
        return request.app.state.metrics.render_prometheus()

    @app.post("/v1/extract", response_model=ExtractResponse)
    async def extract(request: Request, payload: ExtractRequest):
        service: ExtractionService = request.app.state.service

        if payload.stream:
            return StreamingResponse(_sse(service, payload), media_type="text/event-stream")

        try:
            outcome = await service.extract(payload.text, use_cache=payload.use_cache)
        except InvalidModelOutput as exc:
            # 502, not 500: the gateway worked, its upstream produced garbage.
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        return ExtractResponse(
            data=outcome.invoice,
            cached=outcome.cached,
            latency_ms=round(outcome.latency_ms, 2),
            ttft_ms=round(outcome.ttft_ms, 2) if outcome.ttft_ms is not None else None,
        )

    return app


async def _sse(service: ExtractionService, payload: ExtractRequest):
    async for event in service.stream(payload.text, use_cache=payload.use_cache):
        data = event["data"]
        body = data if isinstance(data, str) else json.dumps(data)
        yield f"event: {event['event']}\ndata: {body}\n\n"


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(prog="gateway")
    parser.add_argument("--model-path", dest="model", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--redis-url", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        create_app({"model": args.model, "adapter": args.adapter, "redis_url": args.redis_url}),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
