"""Orchestrate Locust runs and emit a provenance-stamped report.

    python -m benchmarks.run_loadtest --users 10 25 50 --duration 60s \\
        --duplicate-ratio 0.3 --tag optimized

Every report records which engine served it, whether the cache was attached, and
the duplicate ratio the workload used. Those three facts determine whether the
numbers mean anything, and a report that omits them is a report that can be
quoted misleadingly — including by accident, months later.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

MOCK_ENGINE_WARNING = (
    "Served by MockEngine: these numbers measure gateway, cache and network "
    "overhead only. They say nothing about model throughput or real TTFT."
)


def probe_gateway(host: str) -> dict:
    """Ask the gateway what it is before trusting anything it reports."""
    import urllib.request

    with urllib.request.urlopen(f"{host}/health", timeout=5) as response:
        return json.loads(response.read())


class MemorySampler(threading.Thread):
    """Poll RSS of the gateway process. `ps` avoids a psutil dependency."""

    def __init__(self, pid: int, interval: float = 0.5) -> None:
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self.samples: list[float] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(self.pid)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                value = output.stdout.strip()
                if value:
                    self.samples.append(int(value) / 1024)  # KB -> MB
            except (subprocess.CalledProcessError, ValueError):
                pass
            self._stop.wait(self.interval)

    def stop(self) -> dict:
        self._stop.set()
        self.join(timeout=2)
        if not self.samples:
            return {"peak_mb": None, "mean_mb": None, "samples": 0}
        return {
            "peak_mb": round(max(self.samples), 1),
            "mean_mb": round(sum(self.samples) / len(self.samples), 1),
            "samples": len(self.samples),
        }


def run_one(args, users: int, out_dir: Path) -> dict:
    stats_path = out_dir / f"locust-{args.tag}-{users}u.json"
    env = {
        **os.environ,
        "OMNISERVE_CORPUS": str(args.corpus),
        "OMNISERVE_DUPLICATE_RATIO": str(args.duplicate_ratio),
        "OMNISERVE_STREAM_SHARE": str(args.stream_share),
        "OMNISERVE_STATS_OUT": str(stats_path),
    }

    command = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        "benchmarks/locustfile.py",
        "--headless",
        "-u",
        str(users),
        "-r",
        str(max(1, users // 5)),
        "--run-time",
        args.duration,
        "--host",
        args.host,
        "--only-summary",
    ]

    sampler = MemorySampler(args.gateway_pid) if args.gateway_pid else None
    if sampler:
        sampler.start()

    print(f"\n=== {users} users, {args.duration} ===")
    started = time.perf_counter()
    result = subprocess.run(command, env=env)
    elapsed = time.perf_counter() - started

    memory = sampler.stop() if sampler else {"peak_mb": None, "mean_mb": None, "samples": 0}
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    return {
        "users": users,
        "duration_s": round(elapsed, 1),
        "exit_code": result.returncode,
        "memory": memory,
        "stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="run_loadtest")
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--users", type=int, nargs="+", default=[10, 25, 50])
    parser.add_argument("--duration", default="60s")
    parser.add_argument(
        "--duplicate-ratio",
        type=float,
        required=True,
        help="Fraction of requests repeating an earlier document. Drives cache "
        "hit rate, so it is required and recorded rather than defaulted.",
    )
    parser.add_argument("--stream-share", type=float, default=0.5)
    parser.add_argument("--corpus", type=Path, default=Path("data/generated/test.jsonl"))
    parser.add_argument("--tag", required=True, help="baseline | optimized | ...")
    parser.add_argument("--gateway-pid", type=int, default=None, help="For memory sampling.")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    health = probe_gateway(args.host)
    print(f"gateway: engine={health.get('engine')} cache={health.get('cache_ready')}")

    runs = [run_one(args, users, args.out) for users in args.users]

    report = {
        "tag": args.tag,
        "host": args.host,
        "engine": health.get("engine"),
        "cache_enabled": health.get("cache_ready"),
        "duplicate_ratio": args.duplicate_ratio,
        "stream_share": args.stream_share,
        "runs": runs,
    }
    if health.get("engine") == "MockEngine":
        report["warning"] = MOCK_ENGINE_WARNING

    path = args.out / f"loadtest-{args.tag}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n{'users':>7}{'rps':>10}{'p50 ms':>10}{'p99 ms':>10}{'peak MB':>10}")
    for run in runs:
        aggregated = run["stats"].get("Aggregated") or {}
        print(
            f"{run['users']:>7}{aggregated.get('rps', 0):>10}"
            f"{aggregated.get('p50_ms', 0):>10}{aggregated.get('p99_ms', 0):>10}"
            f"{run['memory']['peak_mb'] or 0:>10}"
        )

    if "warning" in report:
        print(f"\nWARNING: {report['warning']}")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
