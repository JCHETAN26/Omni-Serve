"""Synthetic dataset generation (Phase 2).

Ground-truth invoices are built programmatically by `invoice_factory`; GPT-4o is
used only to render each one as realistic unstructured text. This keeps labels
exact and halves token spend versus asking the model for both halves.

    # free, no API key, deterministic — template renderers
    python -m data.generate_dataset --count 10000 --offline

    # GPT-4o rendering
    OPENAI_API_KEY=... python -m data.generate_dataset --count 10000

Output: JSONL of {"id", "text", "target", "style"} at data/generated/dataset.jsonl
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

from gateway.models.schemas import Invoice

from data.invoice_factory import make_invoices
from data.renderers import render

RENDER_PROMPT = """\
You are producing training data for a document-extraction model.

Rewrite the invoice below as a realistic {style}. Vary the wording, ordering and \
layout naturally — real documents are messy.

Hard requirements:
- Every value below must appear in your output, exactly once, unaltered.
- Do not invent, drop, round or recompute any figure.
- Output only the document text. No preamble, no commentary, no code fences.

Invoice data:
{payload}"""

STYLES = [
    "scanned invoice with a plain-text line-item table",
    "email from accounts receivable to a customer",
    "terse point-of-sale receipt printout",
    "paragraph of prose summarizing the charges in a records system",
    "fax cover sheet followed by billing details",
]


def _record(index: int, invoice: Invoice, text: str, style: str) -> dict:
    return {
        "id": f"inv-{index:06d}",
        "text": text,
        "target": json.loads(invoice.model_dump_json()),
        "style": style,
    }


def generate_offline(count: int, seed: int, noise: float) -> list[dict]:
    rng = random.Random(seed)
    invoices = make_invoices(count, seed=seed)
    return [
        _record(i, invoice, render(invoice, rng, noise), "template")
        for i, invoice in enumerate(invoices)
    ]


async def _render_one(client, model: str, index: int, invoice: Invoice, rng, sem) -> dict | None:
    style = rng.choice(STYLES)
    prompt = RENDER_PROMPT.format(style=style, payload=invoice.model_dump_json(indent=2))

    async with sem:
        for attempt in range(4):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=1.0,
                )
                text = (response.choices[0].message.content or "").strip()
                if text:
                    return _record(index, invoice, text, style)
            except Exception as exc:  # noqa: BLE001 - transient API failure
                if attempt == 3:
                    print(f"  ! {invoice.invoice_number} failed: {exc}", file=sys.stderr)
                    return None
                await asyncio.sleep(2**attempt)
    return None


async def generate_with_llm(count: int, seed: int, model: str, concurrency: int) -> list[dict]:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        sys.exit("openai package not installed. pip install -e '.[data]' (or use --offline)")

    client = AsyncOpenAI()
    rng = random.Random(seed)
    sem = asyncio.Semaphore(concurrency)
    invoices = make_invoices(count, seed=seed)

    tasks = [_render_one(client, model, i, invoice, rng, sem) for i, invoice in enumerate(invoices)]

    records = []
    for done in asyncio.as_completed(tasks):
        record = await done
        if record is not None:
            records.append(record)
        if len(records) % 250 == 0 and records:
            print(f"  rendered {len(records)}/{count}")

    records.sort(key=lambda r: r["id"])
    return records


def main() -> None:
    parser = argparse.ArgumentParser(prog="generate_dataset")
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline", action="store_true", help="Template renderers, no API calls.")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--noise", type=float, default=0.0, help="OCR noise rate (offline only).")
    parser.add_argument("--out", type=Path, default=Path("data/generated/dataset.jsonl"))
    args = parser.parse_args()

    if args.offline:
        records = generate_offline(args.count, args.seed, args.noise)
    else:
        records = asyncio.run(
            generate_with_llm(args.count, args.seed, args.model, args.concurrency)
        )

    # Cheap insurance against a renderer that silently mangles its own labels.
    for record in records:
        Invoice.model_validate(record["target"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    dropped = args.count - len(records)
    print(
        f"Wrote {len(records)} records to {args.out}" + (f" ({dropped} dropped)" if dropped else "")
    )


if __name__ == "__main__":
    main()
