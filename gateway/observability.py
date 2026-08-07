"""In-process metrics with Prometheus text exposition.

Deliberately dependency-free rather than pulling `prometheus_client` or the
OpenTelemetry SDK into the base install: the exposition format is a few lines,
and the gateway should not need a 30MB dependency tree to report six numbers.
`.[observability]` exists for when you want real OTel export — the collector
interface below is small enough to swap.

Histogram buckets are cumulative (`le`), per the Prometheus text format, which
is what makes p50/p99 computable by the scraper rather than by us.
"""

import threading
from dataclasses import dataclass, field

# Latency buckets in milliseconds. Dense below 1s because that is where the
# TTFT story lives — a cache hit and a cold generation differ by ~1000x, and
# uniform buckets would put both in one bin.
LATENCY_BUCKETS_MS = (1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)


@dataclass
class Histogram:
    buckets: tuple[float, ...] = LATENCY_BUCKETS_MS
    counts: list[int] = field(default_factory=list)
    total: float = 0.0
    observations: int = 0

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0] * len(self.buckets)

    def observe(self, value: float) -> None:
        self.observations += 1
        self.total += value
        for index, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[index] += 1

    def quantile(self, q: float) -> float | None:
        """Bucket-bounded estimate, for /health-style summaries.

        Returns the upper bound of the bucket containing the quantile, so it
        over-reports rather than under-reports. Real percentiles come from the
        scraper; this is a convenience, not a substitute.
        """
        if not self.observations:
            return None
        target = q * self.observations
        for bound, count in zip(self.buckets, self.counts):
            if count >= target:
                return float(bound)
        return float(self.buckets[-1])


class Metrics:
    """Thread-safe counters and histograms for one gateway process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, int] = {}
        self.histograms: dict[str, Histogram] = {}

    def increment(self, name: str, amount: int = 1, **labels: str) -> None:
        with self._lock:
            key = _key(name, labels)
            self.counters[key] = self.counters.get(key, 0) + amount

    def observe(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            key = _key(name, labels)
            if key not in self.histograms:
                self.histograms[key] = Histogram()
            self.histograms[key].observe(value)

    def get(self, name: str, **labels: str) -> int:
        return self.counters.get(_key(name, labels), 0)

    def histogram(self, name: str, **labels: str) -> Histogram | None:
        return self.histograms.get(_key(name, labels))

    def snapshot(self) -> dict:
        """Human-readable summary, for logs and the health payload."""
        with self._lock:
            return {
                "counters": dict(self.counters),
                "latency": {
                    name: {
                        "count": hist.observations,
                        "mean_ms": (
                            round(hist.total / hist.observations, 2) if hist.observations else None
                        ),
                        "p50_ms": hist.quantile(0.50),
                        "p99_ms": hist.quantile(0.99),
                    }
                    for name, hist in self.histograms.items()
                },
            }

    def render_prometheus(self) -> str:
        """Text exposition format 0.0.4."""
        lines: list[str] = []
        with self._lock:
            for key, value in sorted(self.counters.items()):
                name, labels = _split(key)
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name}{labels} {value}")

            for key, hist in sorted(self.histograms.items()):
                name, labels = _split(key)
                lines.append(f"# TYPE {name} histogram")
                cumulative = 0
                for bound, count in zip(hist.buckets, hist.counts):
                    cumulative = count
                    lines.append(f"{name}_bucket{_with_le(labels, bound)} {cumulative}")
                lines.append(f"{name}_bucket{_with_le(labels, '+Inf')} {hist.observations}")
                lines.append(f"{name}_sum{labels} {hist.total}")
                lines.append(f"{name}_count{labels} {hist.observations}")

        return "\n".join(lines) + "\n"


def _key(name: str, labels: dict[str, str]) -> str:
    if not labels:
        return name
    rendered = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return f"{name}{{{rendered}}}"


def _split(key: str) -> tuple[str, str]:
    if "{" not in key:
        return key, ""
    name, _, rest = key.partition("{")
    return name, "{" + rest


def _with_le(labels: str, bound) -> str:
    if not labels:
        return f'{{le="{bound}"}}'
    return labels[:-1] + f',le="{bound}"}}'
