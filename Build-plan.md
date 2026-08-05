# OmniServe -- High-Throughput LLM Gateway & Fine-Tuned SLM Engine
## Complete Build Plan & Engineering Specification

---

## 1. Project Overview & Problem Statement

### The Problem
Enterprise software pipelines extracting structured JSON from raw unstructured data (e.g., invoices, support logs, medical records) using proprietary APIs (like GPT-4o) face three major production bottlenecks:
1. **High API Costs:** Running millions of tokens through proprietary LLMs is cost-prohibitive at scale.
2. **High & Unpredictable Latency:** Standard API calls incur high Time-To-First-Token (TTFT) and lack local caching for repetitive requests.
3. **Schema Instability:** Non-deterministic model outputs frequently fail strict JSON schema validation, causing downstream database ingestion errors and requiring expensive retry logic.

### The Solution
**OmniServe** is an end-to-end, high-throughput LLM gateway paired with a domain-adapted Small Language Model (SLM). It combines custom QLoRA fine-tuning for high field-extraction accuracy with a high-performance backend serving layer (vLLM), Redis semantic caching, logit-masking for guaranteed schema outputs, and full OpenTelemetry observability.

---

## 2. System Architecture

```text
                              [ Client Requests ]
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │   FastAPI Gateway   │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │ Redis Semantic Cache│
                            └─────┬───────────┬───┘
                                  │           │
                      Cache Hit   │           │  Cache Miss
                     (Similarity) │           │
                                  ▼           ▼
                     ┌─────────────────┐ ┌───────────────────────────┐
                     │ Fast Stream     │ │ vLLM Engine               │
                     │ Cached Response │ │  - Fine-Tuned SLM Adapter │
                     └─────────────────┘ │  - Outlines Grammar Mask  │
                                         │  - PagedAttention KV Cache│
                                         └────────────┬──────────────┘
                                                      │
                                                      ▼
                                         [ OpenTelemetry / Phoenix ]
                                         (TTFT, Latency, Token Metrics)
```

---

## 3. Tech Stack & Prerequisites

* **Base Model:** Llama-3.1-8B-Instruct or Qwen-2.5-7B-Instruct
* **Fine-Tuning Framework:** PyTorch, Unsloth, Hugging Face `transformers`, `peft` (QLoRA), `datasets`
* **Inference & Serving Engine:** vLLM (PagedAttention, Continuous Batching)
* **API Gateway & Routing:** Python 3.11+, FastAPI, `asyncio`, Uvicorn
* **Caching Layer:** Redis, `redisvl` (Redis Vector Library), `sentence-transformers`
* **Constrained Decoding:** Outlines / Instructor (Grammar-guided JSON logit masking)
* **Observability & Benchmarking:** OpenTelemetry, Arize Phoenix / LangSmith, Locust (Load Testing)
* **DevOps & CI/CD:** GitHub Actions, Docker, PyTest, Black/Flake8

---

## 4. Phase-by-Phase Implementation Blueprint

### Phase 1: Environment Setup & Repository Rules
* **Goal:** Establish repo structure, branch protection, and automated CI.
* **Tasks:**
  1. Initialize repository layout:

     ```text
     omniserve/
     ├── .github/workflows/     # CI pipelines (linting, tests, build)
     ├── data/                  # Synthetic dataset generation scripts
     ├── training/              # Unsloth / QLoRA training scripts & adapters
     ├── gateway/               # FastAPI application & middleware
     │   ├── cache/             # Redis semantic caching layer
     │   ├── engine/            # vLLM worker & Outlines schema enforcement
     │   └── models/            # Pydantic schemas
     ├── benchmarks/            # Locust load tests & eval metrics scripts
     ├── docker/                # Gateway Dockerfile & compose setup
     ├── pyproject.toml
     └── README.md
     ```

  2. **Git Workflow Configuration (NO Direct Pushes to Main):**
     * Configure GitHub repository branch protection rules on `main`.
     * Require Pull Requests (PRs) with at least 1 approval before merging.
     * Require CI status checks to pass before merging.

---

### Phase 2: Synthetic Dataset Generation & Baseline Eval Setup
* **Goal:** Construct a dataset for fine-tuning and establish pre-fine-tuning baseline metrics.
* **Tasks:**
  1. **Dataset Generation:** Write a script (`data/generate_dataset.py`) using a seed schema (e.g., extracting invoice vendor, date, line items, total amount, and tax) to generate 10,000 synthetic unstructured text → target JSON pairs using GPT-4o.
  2. **Data Splitting:** Split dataset into 8,500 train records, 1,000 validation records, and 500 test evaluation records.
  3. **Baseline Evaluation (Un-tuned Base SLM):**
     * Prompt the base un-tuned SLM (e.g., Llama-3.1-8B) on the 500 test evaluation records.
     * Record **Field-Level F1 Accuracy** and **JSON Schema Validity Rate** (baseline benchmark).

---

### Phase 3: Domain SLM Fine-Tuning (Unsloth + QLoRA)
* **Goal:** Adapt the 7B/8B base model to extract structured data with high precision.
* **Tasks:**
  1. Write the training script (`training/train_qlora.py`) using **Unsloth** for memory-efficient 4-bit QLoRA training.
  2. Set LoRA target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
  3. Configure hyper-parameters:
     * Rank (*r*) = 16, Alpha (*α*) = 32
     * Batch size = 2 per device (with gradient accumulation steps = 4)
     * Learning rate = 2e-4, Warmup steps = 10, Epochs = 1-2
  4. Run training on GPU instance (e.g., RunPod or Google Colab T4/A100).
  5. Save LoRA adapters (`/training/adapters/`) and upload/merge artifacts.

---

### Phase 4: High-Throughput Serving Engine (vLLM & Constrained Decoding)
* **Goal:** Serve the fine-tuned model with sub-second latency and guaranteed valid JSON outputs.
* **Tasks:**
  1. Integrate **vLLM AsyncEngine** (`gateway/engine/vllm_worker.py`) to leverage PagedAttention and dynamic batching.
  2. Implement **Constrained Decoding** using Outlines / Instructor:
     * Pass target Pydantic schemas into the generation pipeline.
     * Mask token logits during generation so the model physically cannot emit invalid JSON syntax.
  3. Build streaming response hooks to support real-time token streaming over FastAPI Server-Sent Events (SSE).

---

### Phase 5: Semantic Caching Layer (Redis + Vector Search)
* **Goal:** Bypass GPU compute entirely for identical or semantically similar queries.
* **Tasks:**
  1. Build a Redis cache client (`gateway/cache/semantic_cache.py`) using `redisvl`.
  2. Generate embeddings for incoming text payloads using a fast local embedding model (e.g., `all-MiniLM-L6-v2`).
  3. Implement Cosine Similarity Threshold check (≥ 0.95):
     * **Cache Hit:** If embedding similarity exceeds threshold, immediately return cached JSON response from Redis (skipping vLLM entirely).
     * **Cache Miss:** Forward prompt to vLLM engine, stream response to client, and asynchronously store prompt embedding + output in Redis.

---

### Phase 6: FastAPI Gateway Integration & Observability
* **Goal:** Wrap serving, caching, and model components in a unified, monitored API.
* **Tasks:**
  1. Create FastAPI endpoints:
     * `POST /v1/extract`: Primary structured extraction endpoint.
     * `GET /health`: Health checks and model readiness.
     * `GET /metrics`: OpenTelemetry / Prometheus metric endpoint.
  2. Instrument tracing with OpenTelemetry / Arize Phoenix:
     * Measure **TTFT (Time-To-First-Token)**, **Tokens/Sec**, **Cache Hit/Miss Ratio**, and **Total Request Duration**.

---

### Phase 7: Benchmarking & Metric Collection Framework
* **Goal:** Systematically run tests to calculate baseline vs. optimized numbers for evaluation.
* **Tasks:**
  1. **Accuracy & Quality Benchmark (`benchmarks/eval_accuracy.py`):**
     * Compare un-tuned base model vs. fine-tuned SLM on the 500 test records.
     * Calculate: Field Extraction F1 Score, Field-Level Exact Match Rate, and Invalid JSON Syntax Rate.
  2. **Latency & Throughput Load Testing (`benchmarks/locustfile.py`):**
     * Run Locust load test simulating 10, 25, and 50 concurrent users sending parallel extraction requests.
     * Benchmark **Baseline Gateway** (Hugging Face pipeline / no cache) vs. **Optimized Gateway** (vLLM + Redis Cache + Outlines).
     * Measure and export:
       * Time-To-First-Token (p50, p99)
       * Total Throughput (Requests per Second / Tokens per Second)
       * Peak Memory Overhead

---

## 5. CI/CD & Quality Assurance Pipeline

Every contribution **MUST** follow PR-based submission. Direct commits to `main` are blocked.

### GitHub Actions Workflow (`.github/workflows/ci.yml`)
1. **Linting & Code Style:** Run `black --check .` and `flake8 .`
2. **Unit Tests:** Execute `pytest tests/unit/` verifying:
   * Pydantic schema validation.
   * Redis cache hit/miss logic mock.
   * FastAPI routing behavior.
3. **Integration Tests:** Spin up Dockerized Redis container service in CI and verify end-to-end API payloads against mock model responses.
4. **Automated PR Enforcement:** PRs can only merge when all workflow steps pass.

---

## 6. How to Run & Reproduce the Project Locally

```bash
# 1. Clone repository
git clone https://github.com/JCHETAN26/Omni-Serve.git
cd Omni-Serve

# 2. Setup virtual environment
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# 3. Start Redis Cache
docker run -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

# 4. Start OmniServe API Gateway
python -m gateway.main --model-path ./training/adapters/omniserve-slm-7b

# 5. Run Evaluation & Load Testing Suites
python benchmarks/eval_accuracy.py
locust -f benchmarks/locustfile.py --headless -u 25 -r 5 --run-time 2m --host http://localhost:8000
```

---

This complete blueprint outlines the problem, architecture, fine-tuning strategy, infrastructure optimizations, evaluation loops, and PR-driven CI workflow.
