"""Config and formatting tests for Phase 3.

Nothing here touches a GPU: `training.train_qlora` imports unsloth/torch lazily
inside `train()`, so the hyper-parameters and data plumbing stay testable in CI.
"""

import argparse
import json

import pytest

from data.generate_dataset import generate_offline
from training.dataset import format_records, load_jsonl, to_conversation
from training.train_qlora import TARGET_MODULES, TrainConfig, build_config


def fake_chat_template(messages, tokenize=False):
    """Stand-in for tokenizer.apply_chat_template with Llama-3-style markers."""
    parts = [f"<|start_header_id|>{m['role']}<|end_header_id|>\n{m['content']}" for m in messages]
    return "".join(parts) + "<|eot_id|>"


def test_defaults_match_the_build_plan():
    config = TrainConfig()

    assert (config.lora_r, config.lora_alpha) == (16, 32)
    assert (config.batch_size, config.grad_accum) == (2, 4)
    assert config.learning_rate == 2e-4
    assert config.warmup_steps == 10
    assert config.target_modules == TARGET_MODULES
    assert len(config.target_modules) == 7


def test_effective_batch_size_is_reported_correctly():
    assert TrainConfig(batch_size=2, grad_accum=4).effective_batch_size == 8
    assert TrainConfig(batch_size=4, grad_accum=8).effective_batch_size == 32


def test_target_modules_are_not_shared_between_configs():
    first, second = TrainConfig(), TrainConfig()
    first.target_modules.append("lm_head")

    assert "lm_head" not in second.target_modules


def test_build_config_defaults_to_completion_only():
    args = argparse.Namespace(
        model="m",
        train_path="t",
        val_path="v",
        output="o",
        max_seq_length=4096,
        lora_r=16,
        lora_alpha=32,
        batch_size=2,
        grad_accum=4,
        learning_rate=2e-4,
        warmup_steps=10,
        epochs=2.0,
        seed=1,
        include_schema=False,
        train_on_prompt=False,
        merge_16bit=False,
    )

    assert build_config(args).completion_only is True
    args.train_on_prompt = True
    assert build_config(args).completion_only is False


def test_format_records_produces_one_string_per_record():
    records = generate_offline(5, seed=0, noise=0.0)
    texts = format_records(records, fake_chat_template)

    assert len(texts) == 5
    for text, record in zip(texts, records):
        assert record["text"] in text
        assert text.endswith("<|eot_id|>")


def test_formatted_text_contains_the_target_json():
    records = generate_offline(3, seed=1, noise=0.0)
    texts = format_records(records, fake_chat_template)

    for text, record in zip(texts, records):
        assert json.dumps(record["target"], separators=(",", ":"), sort_keys=True) in text


def test_response_marker_is_present_for_completion_masking():
    from training.dataset import RESPONSE_MARKER

    records = generate_offline(1, seed=0, noise=0.0)
    text = format_records(records, fake_chat_template)[0]

    assert RESPONSE_MARKER in text
    assert text.index(RESPONSE_MARKER) > text.index("<|start_header_id|>user<|end_header_id|>")


def test_schema_flag_threads_through_to_the_prompt():
    records = generate_offline(1, seed=0, noise=0.0)

    plain = format_records(records, fake_chat_template, include_schema=False)[0]
    with_schema = format_records(records, fake_chat_template, include_schema=True)[0]

    assert len(with_schema) > len(plain)
    assert "properties" in with_schema


def test_load_jsonl_roundtrip(tmp_path):
    records = generate_offline(4, seed=0, noise=0.0)
    path = tmp_path / "split.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))

    assert load_jsonl(path) == records


def test_conversation_has_no_trailing_whitespace_drift():
    record = generate_offline(1, seed=0, noise=0.0)[0]
    answer = to_conversation(record)[-1]["content"]

    assert answer == answer.strip()


@pytest.mark.parametrize("include_schema", [True, False])
def test_conversation_shape_is_stable(include_schema):
    record = generate_offline(1, seed=0, noise=0.0)[0]
    conversation = to_conversation(record, include_schema)

    assert [m["role"] for m in conversation] == ["system", "user", "assistant"]
