"""The prompt contract, shared by training, evaluation, and serving.

This lives in `gateway` (the always-installed core) rather than `training` so
importing it never drags in torch. Every component that talks to the model must
build prompts here — a fine-tune is only as good as the prompt agreement between
training and inference, and a silent drift between the two shows up as a
mysteriously bad eval rather than as an error.

Two variants exist on purpose:

- `include_schema=True` gives a compact field list plus a worked example. An
  untuned model has no idea what fields we want, so the baseline needs this to
  be measured fairly.
- `include_schema=False` omits both. A fine-tuned model has internalized the
  schema, and dropping them saves ~130 tokens of prefill per request — which is
  TTFT, the metric Phase 7 reports.

So the baseline is deliberately given the more informative prompt. Any win the
fine-tune shows is therefore understated, not inflated.

The first baseline run used the raw `model_json_schema()` here and scored 0.0
field F1 at a 0.73 parse rate: the model echoed the schema back instead of
extracting, and the echo was long enough to hit the token cap. A spec with no
example shows a model what fields exist but never what an answer looks like.
"""

from functools import lru_cache

from gateway.models.schemas import Invoice

SYSTEM = (
    "You extract structured invoice data from unstructured documents. "
    "You reply with a single JSON object and nothing else."
)

WITH_SCHEMA = """\
Extract the invoice below into a JSON object with exactly these fields:

{fields}

Answer with the extracted values, like this:

{example}

Return only the JSON object for the document below. Do not repeat these
instructions or the field list. No explanation, no code fences.

Document:
{document}"""

WITHOUT_SCHEMA = """\
Extract the invoice below into JSON.

Document:
{document}"""

# A worked example, not just a spec. Handed a bare JSON Schema and no example, an
# untuned model frequently echoes the schema back — which parses as JSON, scores
# zero on every field, and looks like total incompetence rather than a prompt
# that never showed it what an answer looks like.
EXAMPLE = Invoice(
    vendor="Example Trading Co.",
    invoice_number="INV-2024-00042",
    invoice_date="2024-03-08",
    line_items=[
        {"description": "Steel Bracket", "quantity": 4.0, "unit_price": 12.5, "amount": 50.0}
    ],
    subtotal=50.0,
    tax=4.0,
    total=54.0,
    currency="USD",
)

_TYPE_HINTS = {
    "invoice_date": "string, YYYY-MM-DD",
    "currency": "string, 3-letter code",
    "line_items": (
        "array of objects, each with description (string), quantity (number), "
        "unit_price (number), amount (number)"
    ),
}


@lru_cache(maxsize=1)
def field_spec() -> str:
    """Compact field list generated from the model, so it cannot drift.

    Deliberately not `model_json_schema()`: the full JSON Schema is ~425 tokens
    of nested `$defs` and `anyOf`, which is both the thing models echo and
    enough output to hit the token cap when they do.
    """
    lines = []
    for name, field in Invoice.model_fields.items():
        hint = _TYPE_HINTS.get(name)
        if hint is None:
            annotation = field.annotation
            hint = "number" if annotation in (float, int) else "string"
        lines.append(f"- {name} ({hint})")
    return "\n".join(lines)


@lru_cache(maxsize=1)
def example_text() -> str:
    return EXAMPLE.model_dump_json()


def build_messages(document: str, include_schema: bool = False) -> list[dict[str, str]]:
    """Chat messages for one extraction request."""
    if include_schema:
        content = WITH_SCHEMA.format(fields=field_spec(), example=example_text(), document=document)
    else:
        content = WITHOUT_SCHEMA.format(document=document)
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    ]


def build_completion(target: dict) -> str:
    """The exact assistant turn the model is trained to produce.

    Serialized through the Pydantic model rather than `json.dumps`, which fixes
    the key order to the schema's declaration order — `vendor` first, not
    `currency`. That is not cosmetic: Outlines derives its grammar from the same
    schema and masks logits in property order, so a model trained on sorted keys
    would fight the mask on every key token at serve time. Wrong order costs
    quality and decode speed, and shows up only as a bad post-training eval.

    Pydantic also emits compact separators — whitespace tokens are tokens not
    spent being right.
    """
    return Invoice.model_validate(target).model_dump_json()
