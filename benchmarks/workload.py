"""Request corpus construction and latency statistics.

Split out from the Locust file because both are easy to get quietly wrong and
neither is testable inside a running load test.

The duplicate ratio is the number that decides this benchmark's honesty. Cache
hit rate is a direct function of how often the workload repeats a document, so
a benchmark that picks a flattering ratio and doesn't say so is measuring its
own assumptions. It is a required argument, and it is echoed into every report.
"""

import json
import math
import random
from pathlib import Path


def load_documents(path: Path, limit: int | None = None) -> list[str]:
    with path.open() as handle:
        documents = [json.loads(line)["text"] for line in handle if line.strip()]
    return documents[:limit] if limit else documents


def build_requests(
    documents: list[str], count: int, duplicate_ratio: float, seed: int = 0
) -> list[str]:
    """Produce `count` requests where `duplicate_ratio` of them repeat an earlier one.

    Duplicates are drawn from documents already issued, which is what makes them
    cacheable — sampling from the whole corpus would produce repeats the cache
    has never seen and understate hit rate.
    """
    if not 0.0 <= duplicate_ratio <= 1.0:
        raise ValueError("duplicate_ratio must be between 0.0 and 1.0")
    if not documents:
        raise ValueError("no documents to build a workload from")

    # Without this the document cycle wraps and every extra request becomes an
    # unrequested duplicate: asking for 2000 requests at ratio 0.3 from a
    # 50-document corpus silently yields a 0.975 duplicate rate, which would
    # inflate cache hit rate — the headline number — with no warning at all.
    required_unique = math.ceil(count * (1.0 - duplicate_ratio))
    if len(documents) < required_unique:
        raise ValueError(
            f"corpus too small: {count} requests at duplicate_ratio "
            f"{duplicate_ratio} needs {required_unique} unique documents, "
            f"have {len(documents)}. Lower --requests-per-user, raise the "
            f"ratio, or generate a bigger test split."
        )

    rng = random.Random(seed)
    issued: list[str] = []
    requests: list[str] = []
    cursor = 0  # advances only on a new document

    for _ in range(count):
        if issued and rng.random() < duplicate_ratio:
            requests.append(rng.choice(issued))
            continue
        # Indexing by request position instead would skip a corpus document
        # every time a duplicate was drawn, using fewer unique documents than
        # the ratio implies and quietly raising the real duplicate rate again.
        document = documents[cursor % len(documents)]
        cursor += 1
        issued.append(document)
        requests.append(document)

    return requests


def percentile(samples: list[float], q: float) -> float | None:
    """Nearest-rank percentile: the smallest value at or above rank ceil(q*N).

    `ceil`, not `round(q*N + 0.5)` — the latter lands one rank high whenever
    q*N is a whole number, reporting p99 of 1..100 as 100 instead of 99. Small
    on paper, but it over-reports every tail latency at exactly the boundaries
    people quote.
    """
    if not samples:
        return None
    ordered = sorted(samples)
    rank = max(1, min(len(ordered), math.ceil(q * len(ordered))))
    return ordered[rank - 1]


def summarize(samples: list[float]) -> dict:
    if not samples:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "mean": None, "max": None}
    return {
        "count": len(samples),
        "p50": round(percentile(samples, 0.50), 2),
        "p95": round(percentile(samples, 0.95), 2),
        "p99": round(percentile(samples, 0.99), 2),
        "mean": round(sum(samples) / len(samples), 2),
        "max": round(max(samples), 2),
    }
