"""Deterministic synthetic invoice generation.

Ground truth is constructed here, not by the LLM. Every record is arithmetically
consistent and validates against `gateway.models.schemas.Invoice` before it ever
reaches the renderer, so training labels are correct by construction rather than
by trusting a model to agree with its own prose.
"""

import random
from datetime import date, timedelta

from gateway.models.schemas import Invoice, LineItem

VENDOR_STEMS = [
    "Acme",
    "Northwind",
    "Contoso",
    "Fabrikam",
    "Globex",
    "Initech",
    "Umbra",
    "Cascade",
    "Ironwood",
    "Blue Harbor",
    "Sterling",
    "Meridian",
    "Pinnacle",
    "Redwood",
    "Copperline",
    "Vantage",
    "Halcyon",
    "Brightpath",
    "Kestrel",
]

VENDOR_SUFFIXES = [
    "Supply Co.",
    "Industries",
    "Logistics LLC",
    "Partners",
    "Group",
    "Manufacturing",
    "Systems Inc.",
    "Trading Co.",
    "Holdings",
    "& Sons",
]

PRODUCTS = [
    ("Widget A", 8.0, 45.0),
    ("Widget B", 10.0, 60.0),
    ("Steel Bracket", 3.5, 22.0),
    ("Hex Bolt (100pk)", 12.0, 30.0),
    ("Copper Tubing 3m", 25.0, 90.0),
    ("Industrial Adhesive", 15.0, 55.0),
    ("Safety Goggles", 9.0, 28.0),
    ("Nitrile Gloves (box)", 14.0, 40.0),
    ("Cable Harness", 30.0, 120.0),
    ("Pressure Valve", 55.0, 210.0),
    ("Bearing Assembly", 40.0, 175.0),
    ("Consulting Hours", 85.0, 250.0),
    ("Freight Surcharge", 20.0, 95.0),
    ("Packaging Materials", 6.0, 35.0),
    ("Calibration Service", 120.0, 400.0),
]

CURRENCIES = ["USD"] * 8 + ["EUR", "GBP", "CAD"]

# Jurisdiction-plausible rates; 0.0 appears so the model learns tax-exempt cases.
TAX_RATES = [0.0, 0.045, 0.0625, 0.07, 0.0825, 0.0875, 0.095, 0.13, 0.19, 0.20]


def _money(value: float) -> float:
    return round(value + 1e-9, 2)


def make_invoice(rng: random.Random) -> Invoice:
    """Build one arithmetically consistent invoice."""
    vendor = f"{rng.choice(VENDOR_STEMS)} {rng.choice(VENDOR_SUFFIXES)}"

    issued = date(2023, 1, 1) + timedelta(days=rng.randint(0, 1095))
    prefix = rng.choice(["INV", "INV", "IN", "BILL", "SI"])
    invoice_number = f"{prefix}-{issued.year}-{rng.randint(1, 99999):05d}"

    line_items = []
    for description, low, high in rng.sample(PRODUCTS, rng.randint(1, 6)):
        quantity = float(rng.randint(1, 40))
        unit_price = _money(rng.uniform(low, high))
        line_items.append(
            LineItem(
                description=description,
                quantity=quantity,
                unit_price=unit_price,
                amount=_money(quantity * unit_price),
            )
        )

    subtotal = _money(sum(item.amount for item in line_items))
    tax = _money(subtotal * rng.choice(TAX_RATES))

    return Invoice(
        vendor=vendor,
        invoice_number=invoice_number,
        invoice_date=issued,
        line_items=line_items,
        subtotal=subtotal,
        tax=tax,
        total=_money(subtotal + tax),
        currency=rng.choice(CURRENCIES),
    )


def make_invoices(count: int, seed: int = 0) -> list[Invoice]:
    """Generate `count` invoices reproducibly from `seed`."""
    rng = random.Random(seed)
    return [make_invoice(rng) for _ in range(count)]
