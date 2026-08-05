"""The prompt contract, shared by training, evaluation, and serving.

This lives in `gateway` (the always-installed core) rather than `training` so
importing it never drags in torch. Every component that talks to the model must
build prompts here — a fine-tune is only as good as the prompt agreement between
training and inference, and a silent drift between the two shows up as a
mysteriously bad eval rather than as an error.

Two variants exist on purpose:

- `include_schema=True` spells out the target JSON schema. An untuned model has
  no idea what fields we want, so the baseline needs this to be measured fairly.
- `include_schema=False` omits it. A fine-tuned model has internalized the
  schema, and dropping it saves ~400 tokens of prefill per request — which is
  TTFT, the metric Phase 7 reports.

So the baseline is deliberately given the more informative prompt. Any win the
fine-tune shows is therefore understated, not inflated.
"""

import json
from functools import lru_cache

from gateway.models.schemas import Invoice

SYSTEM = (
    "You extract structured invoice data from unstructured documents. "
    "You reply with a single JSON object and nothing else."
)

WITH_SCHEMA = """\
Extract the invoice below into JSON matching this schema:

{schema}

Return only the JSON object. No explanation, no code fences.

Document:
{document}"""

WITHOUT_SCHEMA = """\
Extract the invoice below into JSON.

Document:
{document}"""


@lru_cache(maxsize=1)
def schema_text() -> str:
    return json.dumps(Invoice.model_json_schema(), indent=2)


def build_messages(document: str, include_schema: bool = False) -> list[dict[str, str]]:
    """Chat messages for one extraction request."""
    if include_schema:
        content = WITH_SCHEMA.format(schema=schema_text(), document=document)
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
