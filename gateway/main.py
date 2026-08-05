"""OmniServe FastAPI gateway.

Phase 1 scaffold: app factory plus health check. The semantic cache (Phase 5),
vLLM engine (Phase 4), and /v1/extract + /metrics endpoints (Phase 6) plug in
here as they land.
"""

import argparse

from fastapi import FastAPI

from gateway import __version__
from gateway.models.schemas import HealthResponse


def create_app() -> FastAPI:
    app = FastAPI(title="OmniServe", version=__version__)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        model_ready = getattr(app.state, "engine", None) is not None
        cache_ready = getattr(app.state, "cache", None) is not None
        return HealthResponse(
            status="ok" if model_ready else "degraded",
            version=__version__,
            model_ready=model_ready,
            cache_ready=cache_ready,
        )

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(prog="gateway")
    parser.add_argument("--model-path", default=None, help="Path to the fine-tuned SLM / adapter.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    app.state.model_path = args.model_path
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
