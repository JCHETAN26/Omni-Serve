"""Guard and key tests — no Redis, no network.

These cover the logic that decides whether a cache hit is safe to serve, which
is the part that can silently corrupt extracted data if it's wrong.
"""

import importlib.util

import pytest

from gateway.cache.embeddings import HashEmbedder
from gateway.cache.fingerprint import (
    exact_key,
    normalize,
    numeric_fingerprint,
    numeric_tokens,
)

INVOICE_A = """*** GLOBEX SUPPLY CO. ***
Ref INV-2024-0117  January 19, 2024
Safety Goggles
  24 @ $17.62 = $422.88
SUB $422.88
TAX $0.00
TOT $422.88 (USD)"""

# Same vendor, same template, same layout — different amounts.
INVOICE_B = INVOICE_A.replace("17.62", "21.40").replace("422.88", "513.60")


def test_normalize_collapses_whitespace_and_case():
    assert normalize("  ACME   Supply\n\nCo. ") == "acme supply co."


def test_exact_key_ignores_formatting_but_not_content():
    assert exact_key("Total  $40.59") == exact_key("total $40.59")
    assert exact_key("Total $40.59") != exact_key("Total $40.60")


def test_numeric_tokens_normalize_separators_and_precision():
    assert numeric_tokens("$1,240.50") == numeric_tokens("1240.5")
    assert numeric_tokens("qty 3") == ["3"]


def test_numeric_tokens_are_order_insensitive():
    assert numeric_tokens("10 then 20") == numeric_tokens("20 then 10")


def test_fingerprint_separates_invoices_that_differ_only_in_amounts():
    """The whole reason the guard exists."""
    assert numeric_fingerprint(INVOICE_A) != numeric_fingerprint(INVOICE_B)


def test_fingerprint_survives_reformatting_of_one_document():
    reflowed = INVOICE_A.replace("\n", "   ").replace("  ", " ").upper()

    assert numeric_fingerprint(reflowed) == numeric_fingerprint(INVOICE_A)


def test_guard_separates_documents_that_embed_similarly():
    """Similarity is high; the guard is what actually keeps them apart.

    HashEmbedder scores these ~0.71 — it over-weights the changed number
    tokens. MiniLM, the model the cache actually uses, scores the same pair
    0.9673 (see test_minilm_collides_above_threshold), which is above the
    0.95 threshold. Either way the fingerprints must differ.
    """
    embedder = HashEmbedder()
    a, b = embedder.embed(normalize(INVOICE_A)), embedder.embed(normalize(INVOICE_B))
    cosine = sum(x * y for x, y in zip(a, b))

    assert cosine > 0.5  # substantially similar
    assert numeric_fingerprint(INVOICE_A) != numeric_fingerprint(INVOICE_B)  # guard catches it


@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None,
    reason="sentence-transformers not installed (pip install -e '.[embeddings]')",
)
def test_minilm_collides_above_threshold():
    """Characterization test for the real encoder — the reason the guard exists.

    Opt-in: it downloads ~90MB, so CI skips it and the guard's correctness is
    gated by the tests above instead. Run it before changing `threshold`.
    """
    from gateway.cache.embeddings import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder()
    a, b = embedder.embed(INVOICE_A), embedder.embed(INVOICE_B)
    cosine = sum(x * y for x, y in zip(a, b))

    assert cosine > 0.95, "if this drops, the guard may no longer be load-bearing"
    assert numeric_fingerprint(INVOICE_A) != numeric_fingerprint(INVOICE_B)


def test_hash_embedder_is_deterministic_and_unit_norm():
    embedder = HashEmbedder()
    vector = embedder.embed("acme supply co")

    assert vector == embedder.embed("acme supply co")
    assert abs(sum(x * x for x in vector) - 1.0) < 1e-9
    assert len(vector) == embedder.dims


def test_hash_embedder_separates_unrelated_documents():
    embedder = HashEmbedder()
    a = embedder.embed("safety goggles nitrile gloves packaging")
    b = embedder.embed("consulting hours calibration service freight")

    assert sum(x * y for x, y in zip(a, b)) < 0.5
