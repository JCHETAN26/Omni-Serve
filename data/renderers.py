"""Template renderers: Invoice -> unstructured text.

Used directly by `--offline` generation, and as the few-shot style reference for
GPT-4o rendering. Style variety matters more than realism per sample: a model
trained only on tidy tables collapses on OCR noise and email prose.
"""

import random

from gateway.models.schemas import Invoice

SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "CAD": "CA$"}

# Character confusions typical of OCR over scanned invoices.
OCR_CONFUSIONS = {"0": "O", "1": "l", "5": "S", "8": "B", "S": "5", "O": "0"}


def _fmt(amount: float, currency: str) -> str:
    return f"{SYMBOLS.get(currency, '')}{amount:,.2f}"


def _date_variants(invoice: Invoice, rng: random.Random) -> str:
    d = invoice.invoice_date
    return rng.choice(
        [
            d.isoformat(),
            d.strftime("%m/%d/%Y"),
            d.strftime("%d %B %Y"),
            d.strftime("%B %-d, %Y"),
            d.strftime("%d-%b-%Y"),
        ]
    )


def render_table(invoice: Invoice, rng: random.Random) -> str:
    lines = [
        f"{invoice.vendor}",
        f"INVOICE {invoice.invoice_number}",
        f"Date: {_date_variants(invoice, rng)}",
        "",
        "DESCRIPTION                QTY    UNIT PRICE      AMOUNT",
        "-" * 58,
    ]
    for item in invoice.line_items:
        lines.append(
            f"{item.description:<26}{item.quantity:>5.0f}"
            f"{_fmt(item.unit_price, invoice.currency):>14}"
            f"{_fmt(item.amount, invoice.currency):>14}"
        )
    lines += [
        "-" * 58,
        f"{'Subtotal':>45}{_fmt(invoice.subtotal, invoice.currency):>13}",
        f"{'Tax':>45}{_fmt(invoice.tax, invoice.currency):>13}",
        f"{'TOTAL DUE':>45}{_fmt(invoice.total, invoice.currency):>13}",
        "",
        f"All amounts in {invoice.currency}.",
    ]
    return "\n".join(lines)


def render_email(invoice: Invoice, rng: random.Random) -> str:
    items = "; ".join(
        f"{item.quantity:.0f}x {item.description} at {_fmt(item.unit_price, invoice.currency)} "
        f"({_fmt(item.amount, invoice.currency)})"
        for item in invoice.line_items
    )
    return (
        f"Subject: Invoice {invoice.invoice_number} from {invoice.vendor}\n\n"
        f"Hi there,\n\n"
        f"Please find our invoice dated {_date_variants(invoice, rng)} attached. "
        f"We billed the following: {items}.\n\n"
        f"That comes to {_fmt(invoice.subtotal, invoice.currency)} before tax, with "
        f"{_fmt(invoice.tax, invoice.currency)} tax applied, for a total of "
        f"{_fmt(invoice.total, invoice.currency)} {invoice.currency} due on receipt.\n\n"
        f"Thanks,\nAccounts Receivable\n{invoice.vendor}"
    )


def render_receipt(invoice: Invoice, rng: random.Random) -> str:
    lines = [
        f"*** {invoice.vendor.upper()} ***",
        f"Ref {invoice.invoice_number}  {_date_variants(invoice, rng)}",
        "",
    ]
    for item in invoice.line_items:
        lines.append(f"{item.description}")
        lines.append(
            f"  {item.quantity:.0f} @ {_fmt(item.unit_price, invoice.currency)}"
            f" = {_fmt(item.amount, invoice.currency)}"
        )
    lines += [
        "",
        f"SUB {_fmt(invoice.subtotal, invoice.currency)}",
        f"TAX {_fmt(invoice.tax, invoice.currency)}",
        f"TOT {_fmt(invoice.total, invoice.currency)} ({invoice.currency})",
        "THANK YOU FOR YOUR BUSINESS",
    ]
    return "\n".join(lines)


def render_prose(invoice: Invoice, rng: random.Random) -> str:
    items = ", ".join(
        f"{item.description} (qty {item.quantity:.0f}, {_fmt(item.amount, invoice.currency)})"
        for item in invoice.line_items
    )
    return (
        f"Record of charges from {invoice.vendor}, reference {invoice.invoice_number}, "
        f"issued {_date_variants(invoice, rng)}. The order covered {items}. "
        f"Net of tax the balance was {_fmt(invoice.subtotal, invoice.currency)}; tax added "
        f"{_fmt(invoice.tax, invoice.currency)}, bringing the amount payable to "
        f"{_fmt(invoice.total, invoice.currency)} in {invoice.currency}."
    )


RENDERERS = [render_table, render_email, render_receipt, render_prose]


def add_ocr_noise(text: str, rng: random.Random, rate: float = 0.01) -> str:
    """Corrupt a small fraction of characters the way a scan would.

    Tokens containing digits are left alone. Corrupting them would desync the
    text from its own label (an amount read as "l240.50" is unrecoverable), and
    the point of this noise is teaching robustness, not manufacturing bad labels.
    Vendor and description strings are still fair game, so enabling this does
    trade a little label fidelity for it — hence the 0.0 default.
    """
    tokens = text.split(" ")
    for i, token in enumerate(tokens):
        if any(ch.isdigit() for ch in token):
            continue
        tokens[i] = "".join(
            OCR_CONFUSIONS[ch] if ch in OCR_CONFUSIONS and rng.random() < rate else ch
            for ch in token
        )
    return " ".join(tokens)


def render(invoice: Invoice, rng: random.Random, noise: float = 0.0) -> str:
    """Render one invoice in a randomly chosen style."""
    text = rng.choice(RENDERERS)(invoice, rng)
    return add_ocr_noise(text, rng, noise) if noise else text
