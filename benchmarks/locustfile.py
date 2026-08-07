"""Locust load profile for the OmniServe gateway.

    locust -f benchmarks/locustfile.py --headless -u 25 -r 5 --run-time 2m \\
        --host http://localhost:8000

Environment:
    OMNISERVE_CORPUS          test split to draw documents from
    OMNISERVE_DUPLICATE_RATIO fraction of requests that repeat an earlier one
    OMNISERVE_STREAM_SHARE    fraction of requests using SSE (for TTFT)

TTFT is measured on the streaming path only. A non-streaming request has no
observable first token — its latency is entirely generation time — so reporting
one number for both would blend two different quantities.
"""

import json
import os
import time
from pathlib import Path

from locust import HttpUser, between, events, task

from benchmarks.workload import build_requests, load_documents

CORPUS = Path(os.environ.get("OMNISERVE_CORPUS", "data/generated/test.jsonl"))
DUPLICATE_RATIO = float(os.environ.get("OMNISERVE_DUPLICATE_RATIO", "0.3"))
STREAM_SHARE = float(os.environ.get("OMNISERVE_STREAM_SHARE", "0.5"))
REQUESTS_PER_USER = int(os.environ.get("OMNISERVE_REQUESTS_PER_USER", "500"))

_documents = load_documents(CORPUS) if CORPUS.exists() else ["fallback document"]


@events.test_start.add_listener
def _announce(environment, **_):
    print(
        f"corpus={CORPUS} docs={len(_documents)} "
        f"duplicate_ratio={DUPLICATE_RATIO} stream_share={STREAM_SHARE}"
    )


class OmniServeUser(HttpUser):
    wait_time = between(0.0, 0.05)

    def on_start(self) -> None:
        # Per-user seed: identical sequences across users would make every user
        # after the first a guaranteed cache hit and the hit rate meaningless.
        seed = int(time.time() * 1000) % 100000 + id(self) % 1000
        self.requests = build_requests(_documents, REQUESTS_PER_USER, DUPLICATE_RATIO, seed=seed)
        self.cursor = 0

    def _next_document(self) -> str:
        document = self.requests[self.cursor % len(self.requests)]
        self.cursor += 1
        return document

    @task(1)
    def extract_json(self) -> None:
        payload = {"text": self._next_document(), "stream": False}
        with self.client.post(
            "/v1/extract", json=payload, catch_response=True, name="extract"
        ) as response:
            if response.status_code != 200:
                response.failure(f"status {response.status_code}")
                return
            body = response.json()
            # Tag hits separately so the report can show what the cache did
            # rather than only that the average got faster.
            events.request.fire(
                request_type="POST",
                name="extract:cached" if body.get("cached") else "extract:generated",
                response_time=body.get("latency_ms", 0.0),
                response_length=len(response.content),
                exception=None,
                context={},
            )
            response.success()

    @task(1)
    def extract_stream(self) -> None:
        payload = {"text": self._next_document(), "stream": True}
        started = time.perf_counter()
        first_token_at = None
        saw_terminal = False

        with self.client.post(
            "/v1/extract",
            json=payload,
            stream=True,
            catch_response=True,
            name="extract:stream",
        ) as response:
            if response.status_code != 200:
                response.failure(f"status {response.status_code}")
                return

            for line in response.iter_lines():
                if not line:
                    continue
                text = line.decode() if isinstance(line, bytes) else line
                if text.startswith("event: token") and first_token_at is None:
                    first_token_at = time.perf_counter()
                elif text.startswith("event: done") or text.startswith("event: error"):
                    saw_terminal = True

            if not saw_terminal:
                # A stream that ends without a terminal event is a bug, not a
                # slow response — surface it instead of averaging it away.
                response.failure("stream ended without done/error")
                return
            response.success()

        if first_token_at is not None:
            events.request.fire(
                request_type="SSE",
                name="ttft",
                response_time=(first_token_at - started) * 1000,
                response_length=0,
                exception=None,
                context={},
            )


def weight_tasks() -> None:
    """Apply STREAM_SHARE to the task weights."""
    stream_weight = max(1, int(round(STREAM_SHARE * 10)))
    OmniServeUser.extract_stream.locust_task_weight = stream_weight
    OmniServeUser.extract_json.locust_task_weight = max(1, 10 - stream_weight)


weight_tasks()


def _entry_summary(entry) -> dict:
    return {
        "requests": entry.num_requests,
        "failures": entry.num_failures,
        "p50_ms": entry.get_response_time_percentile(0.50),
        "p95_ms": entry.get_response_time_percentile(0.95),
        "p99_ms": entry.get_response_time_percentile(0.99),
        "rps": round(entry.total_rps, 2),
    }


def summarize_stats(environment) -> dict:
    """Extract the numbers the report cares about from Locust's stats.

    `stats.total` is a separate object, not a member of `stats.entries` — read
    only the entries and the aggregate row silently comes back as zeros while
    Locust's own console output shows real traffic.
    """
    entries = {}
    for key, entry in environment.stats.entries.items():
        name = key[0] if isinstance(key, tuple) else key
        entries[name] = _entry_summary(entry)

    entries["Aggregated"] = _entry_summary(environment.stats.total)
    return entries


@events.test_stop.add_listener
def _dump(environment, **_):
    destination = os.environ.get("OMNISERVE_STATS_OUT")
    if destination:
        Path(destination).write_text(json.dumps(summarize_stats(environment), indent=2) + "\n")
        print(f"stats -> {destination}")
