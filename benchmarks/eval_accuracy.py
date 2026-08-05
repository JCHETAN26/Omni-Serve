"""Accuracy benchmark against any OpenAI-compatible endpoint (Phase 2 baseline, Phase 7 compare).

The same script measures every configuration in the project — point it at a vLLM
server running the untuned base model for the baseline, at the fine-tuned adapter
later, or at the OmniServe gateway itself.

    # baseline: untuned base model served by vLLM
    vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8001
    python -m benchmarks.eval_accuracy --endpoint http://localhost:8001/v1 \\
        --model meta-llama/Llama-3.1-8B-Instruct --tag baseline
"""

import argparse
import asyncio
import json
from pathlib import Path

from benchmarks.metrics import score
from gateway.models.schemas import Invoice

EXTRACT_PROMPT = """\
Extract the invoice below into JSON matching this schema:

{schema}

Return only the JSON object. No explanation, no code fences.

Document:
{document}"""


async def _predict(client, model: str, document: str, schema: str, sem, max_tokens: int) -> str:
    async with sem:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": EXTRACT_PROMPT.format(schema=schema, document=document),
                        }
                    ],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception:  # noqa: BLE001 - transient; a failure scores as unparseable
                if attempt == 2:
                    return ""
                await asyncio.sleep(2**attempt)
    return ""


async def run(args) -> dict:
    from openai import AsyncOpenAI

    with args.dataset.open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if args.limit:
        records = records[: args.limit]

    client = AsyncOpenAI(base_url=args.endpoint, api_key=args.api_key)
    schema = json.dumps(Invoice.model_json_schema(), indent=2)
    sem = asyncio.Semaphore(args.concurrency)

    print(f"Evaluating {args.model} on {len(records)} records...")
    predictions = await asyncio.gather(
        *(
            _predict(client, args.model, record["text"], schema, sem, args.max_tokens)
            for record in records
        )
    )

    results = score(list(predictions), [record["target"] for record in records])
    results["tag"] = args.tag
    results["model"] = args.model
    return results


def main() -> None:
    parser = argparse.ArgumentParser(prog="eval_accuracy")
    parser.add_argument("--dataset", type=Path, default=Path("data/generated/test.jsonl"))
    parser.add_argument("--endpoint", default="http://localhost:8001/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="not-needed", help="Placeholder for local servers.")
    parser.add_argument("--tag", default="baseline", help="Label for this run in the report.")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()

    results = asyncio.run(run(args))

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"accuracy-{args.tag}.json"
    path.write_text(json.dumps(results, indent=2) + "\n")

    print()
    for key, value in results.items():
        print(f"  {key:>26}: {value}")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
