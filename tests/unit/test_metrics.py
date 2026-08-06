import json

from benchmarks.metrics import diagnose, flatten, normalize, parse_prediction, score

TARGET = {
    "vendor": "Acme Supply Co.",
    "invoice_number": "INV-2024-0117",
    "invoice_date": "2024-01-17",
    "line_items": [
        {"description": "Widget A", "quantity": 3.0, "unit_price": 12.5, "amount": 37.5}
    ],
    "subtotal": 37.5,
    "tax": 3.09,
    "total": 40.59,
    "currency": "USD",
}


def test_normalize_collapses_cosmetic_differences():
    assert normalize("  Acme   Supply ") == normalize("acme supply")
    assert normalize("$1,240.50") == normalize(1240.5)
    assert normalize("40.590") == 40.59


def test_flatten_expands_line_items_and_skips_missing():
    flat = flatten({"vendor": "Acme", "tax": None, "line_items": [{"description": "Widget A"}]})

    assert flat["vendor"] == "acme"
    assert flat["line_items[0].description"] == "widget a"
    assert "tax" not in flat


def test_parse_prediction_strips_fences_and_prose():
    record, ok, valid = parse_prediction(f"Here you go:\n```json\n{json.dumps(TARGET)}\n```")

    assert (ok, valid) == (True, True)
    assert record["vendor"] == "Acme Supply Co."


def test_parse_prediction_flags_valid_json_that_breaks_schema():
    record, ok, valid = parse_prediction(json.dumps({**TARGET, "total": "not a number"}))

    assert ok is True
    assert valid is False
    assert record is not None


def test_parse_prediction_rejects_unrecoverable_output():
    assert parse_prediction("I could not find an invoice.") == (None, False, False)


def test_score_perfect_prediction():
    results = score([json.dumps(TARGET)], [TARGET])

    assert results["field_f1"] == 1.0
    assert results["exact_match_rate"] == 1.0
    assert results["schema_validity_rate"] == 1.0
    assert results["invalid_json_syntax_rate"] == 0.0


def test_score_partial_credit_beats_zero():
    wrong = {**TARGET, "total": 99.99, "vendor": "Wrong Vendor"}
    results = score([json.dumps(wrong)], [TARGET])

    assert 0.0 < results["field_f1"] < 1.0
    assert results["exact_match_rate"] == 0.0
    assert results["schema_validity_rate"] == 1.0


def test_score_counts_unparseable_output_as_total_miss():
    results = score(["sorry, no idea"], [TARGET])

    assert results["field_f1"] == 0.0
    assert results["invalid_json_syntax_rate"] == 1.0
    assert results["json_parse_rate"] == 0.0


def test_score_penalizes_hallucinated_extra_line_items():
    extra = {**TARGET, "line_items": TARGET["line_items"] + [dict(TARGET["line_items"][0])]}
    results = score([json.dumps(extra)], [TARGET])

    assert results["field_precision"] < 1.0
    assert results["field_recall"] == 1.0


def test_diagnose_flags_zero_f1_with_parseable_output():
    """The exact pattern the first baseline run produced."""
    warning = diagnose({"field_f1": 0.0, "json_parse_rate": 0.73, "schema_validity_rate": 0.0})

    assert warning is not None
    assert "predictions" in warning


def test_diagnose_flags_valid_json_that_is_never_an_invoice():
    warning = diagnose({"field_f1": 0.2, "json_parse_rate": 0.9, "schema_validity_rate": 0.0})

    assert warning is not None


def test_diagnose_stays_quiet_on_a_plausibly_weak_model():
    assert diagnose({"field_f1": 0.31, "json_parse_rate": 0.8, "schema_validity_rate": 0.7}) is None


def test_diagnose_stays_quiet_when_nothing_parsed():
    """Zero F1 with zero parsing is a real failure, not a harness bug."""
    assert diagnose({"field_f1": 0.0, "json_parse_rate": 0.0, "schema_validity_rate": 0.0}) is None
