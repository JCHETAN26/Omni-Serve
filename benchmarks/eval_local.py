"""In-process accuracy evaluation, for notebooks and single-GPU boxes.

`eval_accuracy.py` talks to an OpenAI-compatible server, which assumes vLLM is
running. In Colab you have one T4 and it is busy training, so this path loads
the model directly and generates in batches instead.

Same prompts (`gateway.prompt`) and same scoring (`benchmarks.metrics`) as the
served path, so numbers from the two are comparable.

    python -m benchmarks.eval_local --model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \\
        --tag baseline --include-schema --limit 500
"""

import argparse
import json
from pathlib import Path

from benchmarks.metrics import score
from gateway.prompt import build_messages


def load_model(model_name: str, adapter: str | None, max_seq_length: int):
    """Load a 4-bit model for inference, preferring Unsloth when available."""
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter or model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    except ImportError:
        pass

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def generate_batch(
    model, tokenizer, documents: list[str], include_schema: bool, max_new_tokens: int
) -> list[str]:
    import torch

    prompts = [
        tokenizer.apply_chat_template(
            build_messages(document, include_schema), tokenize=False, add_generation_prompt=True
        )
        for document in documents
    ]

    # Left padding: with right padding the generated continuation starts after
    # the pad run and the decode slice below would return the wrong span.
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    tokenizer.padding_side = previous_side
    prompt_length = inputs["input_ids"].shape[1]
    return tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)


def run(args) -> dict:
    with args.dataset.open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if args.limit:
        records = records[: args.limit]

    model, tokenizer = load_model(args.model, args.adapter, args.max_seq_length)

    predictions: list[str] = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        predictions.extend(
            generate_batch(
                model,
                tokenizer,
                [record["text"] for record in batch],
                args.include_schema,
                args.max_new_tokens,
            )
        )
        print(f"  {min(start + args.batch_size, len(records))}/{len(records)}")

    results = score(predictions, [record["target"] for record in records])
    results["tag"] = args.tag
    results["model"] = args.adapter or args.model
    results["include_schema"] = args.include_schema

    # Raw output is the only way to tell "the model is bad" from "the harness is
    # wrong". A 0.0 F1 means nothing until you have read what it actually said.
    args.out.mkdir(parents=True, exist_ok=True)
    raw_path = args.out / f"predictions-{args.tag}.jsonl"
    with raw_path.open("w") as handle:
        for record, prediction in zip(records, predictions):
            handle.write(
                json.dumps(
                    {"id": record["id"], "prediction": prediction, "target": record["target"]}
                )
                + "\n"
            )
    print(f"Raw predictions -> {raw_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(prog="eval_local")
    parser.add_argument("--dataset", type=Path, default=Path("data/generated/test.jsonl"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory, if any.")
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--include-schema", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()

    results = run(args)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"accuracy-{args.tag}.json"
    path.write_text(json.dumps(results, indent=2) + "\n")

    print()
    for key, value in results.items():
        print(f"  {key:>26}: {value}")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
