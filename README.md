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
| 2 | Synthetic dataset + baseline eval | 🟡 pipeline built, baseline run pending GPU |
| 3 | QLoRA fine-tuning (Unsloth) | 🟡 script ready, training run pending GPU |
| 4 | vLLM engine + constrained decoding | 🟡 built, needs CUDA to run |
| 5 | Redis semantic cache | ✅ built and tested |
| 6 | Gateway endpoints + observability | ✅ built and tested |
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
`.[data]` (OpenAI client), `.[cache]` (Redis + redisvl), `.[embeddings]`
(sentence-transformers), `.[engine]` (vLLM + Outlines), `.[training]` (PyTorch +
PEFT), `.[observability]` (OpenTelemetry + Prometheus).

## Building the dataset

Ground-truth invoices are generated programmatically, then rendered into
unstructured text. Labels are correct by construction — the LLM only writes
prose, it never authors the JSON it would then be graded against.

```bash
# free and deterministic: template renderers, no API key, ~1s for 10k
python -m data.generate_dataset --count 10000 --offline --noise 0.01

# richer text via GPT-4o (costs money; renders text only, not labels)
OPENAI_API_KEY=... python -m data.generate_dataset --count 10000

python -m data.split_dataset          # 8500 / 1000 / 500, seeded
```

The 500-record test split is frozen: baseline and fine-tuned scores are only
comparable if both were measured on the same records.

## Fine-tuning

Needs a CUDA GPU. Unsloth and torch are imported lazily inside `train()`, so the
config and data plumbing stay importable and testable without one.

**On Colab:** open [`training/colab_finetune.ipynb`](training/colab_finetune.ipynb)
— it installs, builds the dataset, measures the untuned baseline, trains, scores
the tuned model on the same records, and saves the adapter to Drive.

**Locally:**

```bash
pip install -e ".[training]" && pip install unsloth
python -m training.train_qlora --max-seq-length 1024 --epochs 2 \
    --output training/adapters/omniserve-slm-8b
```

Pass `--max-seq-length 1024`: the longest example in this dataset is ~640 tokens,
and the 4096 default will OOM an 8B on a 16GB T4 for no benefit.

Defaults follow the build plan: r=16, α=32, batch 2 × grad-accum 4 (effective 8),
lr 2e-4, 10 warmup steps. Training scores the assistant turn only — without that
the model spends most of its loss learning to echo the invoice it was handed.

## The API

```bash
uvicorn gateway.main:app                      # MockEngine, no GPU needed
python -m gateway.main --model-path ./training/adapters/omniserve-slm-8b \
    --redis-url redis://localhost:6379        # vLLM + cache
```

| Endpoint | Behaviour |
| --- | --- |
| `POST /v1/extract` | Cache lookup, then engine on a miss. `stream: true` returns SSE. |
| `GET /health` | Engine and cache readiness. |
| `GET /metrics` | Prometheus text: request counts, cache hits by tier, TTFT and duration histograms. |

Three behaviours worth knowing:

- **A cache hit never reaches the engine.** That's the point of Phase 5, so it's
  asserted directly rather than inferred from timing.
- **Invalid model output is a 502, and is never cached.** The gateway worked;
  its upstream produced garbage. Caching it would promote one bad generation
  into a permanently served wrong answer.
- **A stream always terminates** with exactly one `done` or `error` event. A
  stream that simply stops is indistinguishable from a hung connection.

## Serving

The gateway depends on the `ExtractionEngine` protocol, never on vLLM directly,
so two backends satisfy it:

- **`VLLMEngine`** — AsyncLLMEngine, PagedAttention, continuous batching, LoRA
  adapter loading, and grammar-constrained decoding. Needs CUDA. vLLM is
  imported inside `start()`, so the module imports fine without a GPU.
- **`MockEngine`** — no GPU, satisfies the same protocol, streams in chunks with
  configurable TTFT. Used by the tests, and by Phase 7 as a no-model baseline.

### What constrained decoding does and doesn't buy

Outlines compiles a grammar from the `Invoice` schema and masks logits during
decode, so syntactically invalid JSON becomes unrepresentable rather than
unlikely. Schema validity rate goes to 100%.

It does **not** make the output correct. The model can still emit a well-formed
wrong total. Constrained decoding fixes the *invalid JSON syntax rate*; field F1
is what the fine-tune is for. Reporting them as one number would overstate both.

Because the grammar masks properties in the schema's declaration order,
`build_completion` serializes training targets in that same order. A model
trained on sorted keys would fight the mask on every key token — see
`test_completion_key_order_matches_the_schema_grammar`.

## Caching

Two tiers, because similarity alone is unsafe for extraction.

**Tier 1 — exact.** SHA-256 of the normalized document. Always correct, and it
catches the hit pattern that dominates in practice: retries, duplicate
submissions, batch reprocessing.

**Tier 2 — semantic.** Vector search gated on *both* a cosine threshold and
agreement on every number in the document. That second condition is not
optional. Measured with all-MiniLM-L6-v2:

| Pair | Cosine | Above 0.95? |
| --- | --- | --- |
| Same template, different amounts — **different invoices** | 0.9673 | yes |
| Same vendor, different qty + amounts — **different invoices** | 0.9716 | yes |
| Same document, reflowed and uppercased — **real duplicate** | 1.0 | yes |

Similarity cannot tell the first two from the third, so a threshold-only cache
returns one invoice's totals for another and writes a wrong number to your
database with no error raised. The numeric fingerprint separates them.

```python
cache = SemanticCache(redis_url="redis://localhost:6379", threshold=0.95)
await cache.connect()
hit = await cache.get(document)          # CacheHit(value, tier, similarity) | None
await cache.set(document, extracted)
```

`SemanticCache(require_numeric_match=False)` restores the naive behaviour, and
`tests/integration/test_cache_redis.py` has a test proving it serves wrong data.
The default embedder is `HashEmbedder` (no torch); install `.[embeddings]` and
pass `SentenceTransformerEmbedder()` for real similarity.

## The prompt contract

`gateway/prompt.py` is the single source of truth for prompts, imported by
training, evaluation, and serving alike. A fine-tune is only as good as the
agreement between the prompt it trained on and the one it's served with, and
drift between them surfaces as a bad eval rather than an error.

It has two variants, and which one you use matters:

- **`--include-schema`** spells out the target JSON schema. The untuned baseline
  needs this — it has no idea what fields you want.
- **default (no schema)** omits it. The fine-tune has internalized the schema,
  and dropping it saves ~400 tokens of prefill per request, which is TTFT.

The baseline is therefore given the *more* informative prompt, so any win the
fine-tune shows is understated rather than inflated.

## Measuring accuracy

`benchmarks/eval_accuracy.py` targets any OpenAI-compatible endpoint, so one
script scores the baseline, the tuned adapter, and the gateway.

```bash
# baseline — schema in prompt
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8001
python -m benchmarks.eval_accuracy --model meta-llama/Llama-3.1-8B-Instruct \
    --tag baseline --include-schema

# tuned — schema omitted, matching training
python -m benchmarks.eval_accuracy --model omniserve-slm-8b --tag tuned
```

`eval_local.py` is the same evaluation without a server — it loads the model
in-process, for notebooks and single-GPU boxes where vLLM can't run alongside
training. Identical prompts and scoring, so the numbers are comparable.

Reports field-level precision/recall/F1, exact-match rate, invalid-JSON-syntax
rate, and schema-validity rate to `benchmarks/results/`.

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
