import json

from gateway.prompt import build_completion, build_messages, example_text, field_spec
from training.dataset import to_conversation

DOCUMENT = "*** ACME SUPPLY CO. ***\nRef INV-2024-0117\nTOT $40.59"
TARGET = {
    "vendor": "Acme Supply Co.",
    "invoice_number": "INV-2024-0117",
    "invoice_date": "2024-01-17",
    "line_items": [],
    "subtotal": 37.5,
    "tax": 3.09,
    "total": 40.59,
    "currency": "USD",
}


def test_messages_carry_system_and_document():
    messages = build_messages(DOCUMENT)

    assert [m["role"] for m in messages] == ["system", "user"]
    assert DOCUMENT in messages[1]["content"]


def test_schema_variant_is_strictly_longer():
    without = build_messages(DOCUMENT)[1]["content"]
    with_schema = build_messages(DOCUMENT, include_schema=True)[1]["content"]

    assert "invoice_number" in with_schema
    assert "invoice_number" not in without
    assert len(with_schema) > len(without)


def test_field_spec_lists_every_model_field():
    from gateway.models.schemas import Invoice

    spec = field_spec()

    for name in Invoice.model_fields:
        assert name in spec
    assert "YYYY-MM-DD" in spec  # date format is spelled out, not inferred


def test_schema_variant_carries_a_worked_example():
    """The example is the fix for schema echo — assert it is actually there."""
    content = build_messages(DOCUMENT, include_schema=True)[1]["content"]
    example = json.loads(example_text())

    assert example_text() in content
    assert example["vendor"] == "Example Trading Co."
    assert "Do not repeat these" in content


def test_baseline_prompt_stays_far_cheaper_than_the_raw_schema():
    """Raw model_json_schema() was ~1700 chars and got echoed back verbatim."""
    from gateway.models.schemas import Invoice

    raw_schema = json.dumps(Invoice.model_json_schema())

    assert len(field_spec()) + len(example_text()) < len(raw_schema)


def test_completion_is_compact_and_stable():
    completion = build_completion(TARGET)

    assert " " not in completion.replace("Acme Supply Co.", "").replace("INV-2024-0117", "")
    assert json.loads(completion) == TARGET
    # Key order must not depend on dict insertion order.
    shuffled = dict(reversed(list(TARGET.items())))
    assert build_completion(shuffled) == completion


def test_completion_key_order_matches_the_schema_grammar():
    """Training order must equal the order Outlines will mask against.

    Outlines compiles its grammar from the schema's property order, so a model
    trained on any other order fights the mask on every key token at serve time.
    """
    from gateway.models.schemas import Invoice

    completion_keys = list(json.loads(build_completion(TARGET)))
    schema_keys = list(Invoice.model_json_schema()["properties"])

    assert completion_keys == schema_keys
    assert completion_keys != sorted(completion_keys)  # the bug this replaced


def test_conversation_ends_with_the_gold_answer():
    conversation = to_conversation({"text": DOCUMENT, "target": TARGET})

    assert [m["role"] for m in conversation] == ["system", "user", "assistant"]
    assert json.loads(conversation[-1]["content"]) == TARGET
