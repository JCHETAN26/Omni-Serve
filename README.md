# OmniServe

High-throughput LLM gateway and fine-tuned SLM engine for structured JSON extraction.

OmniServe replaces proprietary-API calls in document-extraction pipelines with a
domain-adapted small language model served behind a caching, schema-enforcing
gateway. Three optimizations do the work:

- **QLoRA fine-tuning** of a 7B/8B base model for field-extraction accuracy.
- **vLLM + constrained decoding** — PagedAttention and continuous batching for
  throughput, Outlines grammar masking so invalid JSON is unrepresentable.
- **Redis semantic cache** — near-duplicate prompts skip the GPU entirely.

Full engineering spec: [Build-plan.md](./Build-plan.md).

## Status

| Phase | Scope | State |
| --- | --- | --- |
| 1 | Repo layout, CI, branch protection | ✅ scaffolded |
| 2 | Synthetic dataset + baseline eval | ⬜ not started |
| 3 | QLoRA fine-tuning (Unsloth) | ⬜ not started |
| 4 | vLLM engine + constrained decoding | ⬜ not started |
| 5 | Redis semantic cache | ⬜ not started |
| 6 | Gateway endpoints + observability | ⬜ not started |
| 7 | Benchmarks (accuracy, TTFT, throughput) | ⬜ not started |

Today the gateway exposes `/health` only; `/v1/extract` and `/metrics` arrive in Phase 6.

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

pytest tests/unit
uvicorn gateway.main:app --reload
curl localhost:8000/health
```

Optional dependency groups keep heavy stacks out of the default install:
`.[cache]` (Redis + embeddings), `.[engine]` (vLLM + Outlines), `.[training]`
(PyTorch + PEFT), `.[observability]` (OpenTelemetry + Prometheus).

## Layout

```text
gateway/          FastAPI app
  cache/          Redis semantic caching layer      (Phase 5)
  engine/         vLLM worker + Outlines masking    (Phase 4)
  models/         Pydantic schemas — extraction source of truth
data/             Synthetic dataset generation      (Phase 2)
training/         QLoRA scripts and adapters        (Phase 3)
benchmarks/       Locust load tests + eval metrics  (Phase 7)
docker/           Dockerfile and compose setup
tests/            unit/ and integration/ suites
```

## Contributing

`main` is protected: no direct pushes. Open a PR, get one approval, and land
green CI (`black --check .`, `flake8 .`, `pytest`).
