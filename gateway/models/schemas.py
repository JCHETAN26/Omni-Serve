"""Extraction schemas.

These models are the single source of truth for the project: the dataset
generator (Phase 2) emits records that validate against `Invoice`, and the
serving engine (Phase 4) compiles the same model into a token-level grammar
mask so the SLM cannot emit structurally invalid JSON.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str
    quantity: float = Field(gt=0)
    unit_price: float = Field(ge=0)
    amount: float = Field(ge=0)


class Invoice(BaseModel):
    vendor: str
    invoice_number: str
    invoice_date: date
    line_items: list[LineItem]
    subtotal: float = Field(ge=0)
    tax: float = Field(ge=0)
    total: float = Field(ge=0)
    currency: str = Field(default="USD", min_length=3, max_length=3)


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, description="Raw unstructured document text.")
    schema_name: Literal["invoice"] = "invoice"
    stream: bool = False
    use_cache: bool = True


class ExtractResponse(BaseModel):
    data: Invoice
    cached: bool = False
    latency_ms: float
    ttft_ms: float | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    model_ready: bool
    cache_ready: bool
    # Which backend is actually serving. Reported so a benchmark run can stamp
    # its own provenance and never present MockEngine numbers as GPU numbers.
    engine: str | None = None
