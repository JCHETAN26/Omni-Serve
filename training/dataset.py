"""Dataset loading and chat formatting for QLoRA training.

Deliberately free of torch/unsloth imports so the formatting logic — the part
that silently ruins a fine-tune when it's wrong — is unit-testable on a laptop
with no GPU.
"""

import json
from pathlib import Path
from typing import Any, Callable

from gateway.prompt import build_completion, build_messages

# Marks where the assistant turn begins. Everything before it is context the
# model should condition on but not be scored for reproducing.
RESPONSE_MARKER = "<|start_header_id|>assistant<|end_header_id|>"


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def to_conversation(record: dict, include_schema: bool = False) -> list[dict[str, str]]:
    """One dataset record -> full chat turn list including the gold answer."""
    return build_messages(record["text"], include_schema) + [
        {"role": "assistant", "content": build_completion(record["target"])}
    ]


def format_records(
    records: list[dict],
    apply_chat_template: Callable[..., str],
    include_schema: bool = False,
) -> list[str]:
    """Render records to training strings via the tokenizer's own chat template.

    Taking `apply_chat_template` as an argument rather than a tokenizer keeps
    this testable, and lets the same code train Llama or Qwen without caring
    which special tokens either one uses.
    """
    return [
        apply_chat_template(to_conversation(record, include_schema), tokenize=False)
        for record in records
    ]


def build_hf_dataset(path: Path, apply_chat_template: Callable[..., str], **kwargs) -> Any:
    """Load a split and wrap it as a Hugging Face Dataset of {"text": ...}."""
    from datasets import Dataset

    texts = format_records(load_jsonl(path), apply_chat_template, **kwargs)
    return Dataset.from_dict({"text": texts})
