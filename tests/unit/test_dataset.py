import random

import pytest

from data.generate_dataset import generate_offline
from data.invoice_factory import make_invoice, make_invoices
from data.renderers import add_ocr_noise, render
from data.split_dataset import split_records
from gateway.models.schemas import Invoice


def test_invoice_arithmetic_is_consistent():
    for invoice in make_invoices(200, seed=7):
        assert invoice.subtotal == pytest.approx(
            sum(i.amount for i in invoice.line_items), abs=0.01
        )
        assert invoice.total == pytest.approx(invoice.subtotal + invoice.tax, abs=0.01)
        for item in invoice.line_items:
            assert item.amount == pytest.approx(item.quantity * item.unit_price, abs=0.01)


def test_generation_is_reproducible_from_seed():
    assert make_invoices(20, seed=3) == make_invoices(20, seed=3)
    assert make_invoices(20, seed=3) != make_invoices(20, seed=4)


def test_rendered_text_carries_the_total():
    rng = random.Random(0)
    for _ in range(50):
        invoice = make_invoice(rng)
        text = render(invoice, rng)
        assert f"{invoice.total:,.2f}" in text


def test_ocr_noise_never_corrupts_numeric_tokens():
    rng = random.Random(1)
    text = "Sterling Supply Co. TOTAL 1580.55 USD"
    noisy = add_ocr_noise(text, rng, rate=1.0)

    assert "1580.55" in noisy
    assert noisy != text  # letters were corrupted


def test_offline_records_validate_against_schema():
    records = generate_offline(25, seed=0, noise=0.0)

    assert len(records) == 25
    assert len({r["id"] for r in records}) == 25
    for record in records:
        assert record["text"].strip()
        Invoice.model_validate(record["target"])


def test_split_is_disjoint_and_correctly_sized():
    records = generate_offline(60, seed=0, noise=0.0)
    splits = split_records(records, train=40, val=12, test=8, seed=0)

    assert [len(splits[k]) for k in ("train", "val", "test")] == [40, 12, 8]
    ids = [r["id"] for rows in splits.values() for r in rows]
    assert len(set(ids)) == 60


def test_split_is_stable_across_runs():
    records = generate_offline(60, seed=0, noise=0.0)
    first = split_records(records, 40, 12, 8, seed=0)["test"]
    second = split_records(records, 40, 12, 8, seed=0)["test"]

    assert [r["id"] for r in first] == [r["id"] for r in second]


def test_split_refuses_to_silently_undersize():
    records = generate_offline(10, seed=0, noise=0.0)

    with pytest.raises(SystemExit):
        split_records(records, train=8500, val=1000, test=500)
