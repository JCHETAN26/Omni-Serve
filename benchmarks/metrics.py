"""Extraction quality metrics (Phase 7 math, needed now for the Phase 2 baseline).

Field-level scoring flattens each record to `path -> value` pairs and compares
them as sets, so a model that emits five of seven fields correctly is scored on
what it got right rather than pass/fail on the whole record.
"""

import json
import re
from typing import Any

from pydantic import ValidationError

from gateway.models.schemas import Invoice

SCALAR_FIELDS = [
    "vendor",
    "invoice_number",
    "invoice_date",
    "subtotal",
    "tax",
    "total",
    "currency",
]
ITEM_FIELDS = ["description", "quantity", "unit_price", "amount"]

_WHITESPACE = re.compile(r"\s+")


def normalize(value: Any) -> Any:
    """Collapse cosmetic differences that shouldn't count as extraction errors."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    text = _WHITESPACE.sub(" ", str(value).strip())
    # "1,240.50" and "$1240.5" are the same extraction; casing never matters.
    stripped = text.replace(",", "").lstrip("$€£").removeprefix("CA$")
    try:
        return round(float(stripped), 2)
    except ValueError:
        return text.casefold()


def flatten(record: dict) -> dict[str, Any]:
    """Invoice dict -> {field_path: normalized value}, skipping absent fields."""
    flat = {}
    for field in SCALAR_FIELDS:
        if record.get(field) is not None:
            flat[field] = normalize(record[field])

    items = record.get("line_items") or []
    if isinstance(items, list):
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            for field in ITEM_FIELDS:
                if item.get(field) is not None:
                    flat[f"line_items[{index}].{field}"] = normalize(item[field])
    return flat


def parse_prediction(raw: str) -> tuple[dict | None, bool, bool]:
    """Return (record, parsed_ok, schema_valid) for one raw model output."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)

    # Untuned models routinely wrap JSON in prose; salvage the outermost object.
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None, False, False
        text = text[start : end + 1]

    try:
        record = json.loads(text)
    except json.JSONDecodeError:
        return None, False, False
    if not isinstance(record, dict):
        return None, False, False

    try:
        Invoice.model_validate(record)
    except ValidationError:
        return record, True, False
    return record, True, True


def score(predictions: list[str], targets: list[dict]) -> dict[str, float]:
    """Aggregate field F1, exact-match, and JSON health over a test split."""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must be the same length")

    tp = fp = fn = 0
    exact = parsed = schema_ok = 0

    for raw, target in zip(predictions, targets):
        gold = flatten(target)
        record, ok, valid = parse_prediction(raw)
        parsed += ok
        schema_ok += valid

        if record is None:
            fn += len(gold)  # unparseable output misses every field
            continue

        pred = flatten(record)
        for path, value in pred.items():
            if path in gold and gold[path] == value:
                tp += 1
            else:
                fp += 1
        fn += sum(1 for path in gold if gold.get(path) != pred.get(path))
        exact += pred == gold

    total = len(targets)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "field_precision": round(precision, 4),
        "field_recall": round(recall, 4),
        "field_f1": round(f1, 4),
        "exact_match_rate": round(exact / total, 4) if total else 0.0,
        "json_parse_rate": round(parsed / total, 4) if total else 0.0,
        "invalid_json_syntax_rate": round(1 - parsed / total, 4) if total else 0.0,
        "schema_validity_rate": round(schema_ok / total, 4) if total else 0.0,
        "n": total,
    }
