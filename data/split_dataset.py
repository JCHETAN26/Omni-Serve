"""Deterministic train/validation/test split (Phase 2).

    python -m data.split_dataset --train 8500 --val 1000 --test 500

The split is seeded and shuffled once. The test slice must stay frozen across
the whole project — baseline and fine-tuned numbers are only comparable if both
were measured on identical records.
"""

import argparse
import json
import random
import sys
from pathlib import Path


def split_records(
    records: list[dict], train: int, val: int, test: int, seed: int = 0
) -> dict[str, list[dict]]:
    total = train + val + test
    if len(records) < total:
        sys.exit(f"Need {total} records for the requested split, dataset has {len(records)}.")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    return {
        "train": shuffled[:train],
        "val": shuffled[train : train + val],
        "test": shuffled[train + val : total],
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="split_dataset")
    parser.add_argument("--input", type=Path, default=Path("data/generated/dataset.jsonl"))
    parser.add_argument("--outdir", type=Path, default=Path("data/generated"))
    parser.add_argument("--train", type=int, default=8500)
    parser.add_argument("--val", type=int, default=1000)
    parser.add_argument("--test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"{args.input} not found. Run data.generate_dataset first.")

    with args.input.open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    splits = split_records(records, args.train, args.val, args.test, args.seed)

    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = args.outdir / f"{name}.jsonl"
        with path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        print(f"{name:>5}: {len(rows):>5} -> {path}")


if __name__ == "__main__":
    main()
