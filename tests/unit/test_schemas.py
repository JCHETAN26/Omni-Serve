import pytest
from pydantic import ValidationError

from gateway.models.schemas import ExtractRequest, Invoice

VALID_INVOICE = {
    "vendor": "Acme Supply Co.",
    "invoice_number": "INV-2024-0117",
    "invoice_date": "2024-01-17",
    "line_items": [
        {"description": "Widget A", "quantity": 3, "unit_price": 12.5, "amount": 37.5},
    ],
    "subtotal": 37.5,
    "tax": 3.09,
    "total": 40.59,
    "currency": "USD",
}


def test_invoice_parses_valid_payload():
    invoice = Invoice.model_validate(VALID_INVOICE)

    assert invoice.vendor == "Acme Supply Co."
    assert invoice.line_items[0].quantity == 3
    assert invoice.invoice_date.year == 2024


def test_invoice_rejects_non_positive_quantity():
    payload = {**VALID_INVOICE, "line_items": [{**VALID_INVOICE["line_items"][0], "quantity": 0}]}

    with pytest.raises(ValidationError):
        Invoice.model_validate(payload)


def test_invoice_rejects_malformed_currency():
    with pytest.raises(ValidationError):
        Invoice.model_validate({**VALID_INVOICE, "currency": "DOLLARS"})


def test_extract_request_defaults():
    request = ExtractRequest(text="Invoice from Acme...")

    assert request.schema_name == "invoice"
    assert request.use_cache is True
    assert request.stream is False


def test_extract_request_rejects_empty_text():
    with pytest.raises(ValidationError):
        ExtractRequest(text="")
