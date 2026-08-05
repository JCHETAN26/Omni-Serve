"""Cache keys and the numeric guard.

The guard is the load-bearing idea here. Embedding similarity measures whether
two documents *read* alike, which is not the question a cache for an extraction
pipeline needs answered. Serving one invoice's JSON for another writes wrong
numbers into the database and raises no error anywhere.

Measured with all-MiniLM-L6-v2, the model this cache uses:

    same template, different amounts      cosine 0.9673
    same vendor, different qty + amounts  cosine 0.9716
    same document, reflowed + uppercased  cosine 1.0

The first two are *different invoices* and both clear the 0.95 threshold, so
similarity alone would serve them as hits. Only the third is a real hit.

So a semantic hit must additionally agree on every number in the document. That
narrows the semantic tier to what it is genuinely good at — the same document
re-submitted with different whitespace, casing, line wrapping or OCR noise —
and takes the dangerous case off the table.
"""

import hashlib
import re

_WHITESPACE = re.compile(r"\s+")
# Matches 1,234.56 / 1234.56 / 42 — currency symbols and separators are stripped
# so "$1,240.50" and "1240.5" fingerprint identically.
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def normalize(text: str) -> str:
    """Collapse formatting noise that shouldn't produce a distinct cache entry."""
    return _WHITESPACE.sub(" ", text.strip()).casefold()


def exact_key(text: str) -> str:
    """Stable digest of the normalized document, for the exact-match tier."""
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def numeric_tokens(text: str) -> list[str]:
    """Every number in the document, normalized and sorted.

    Sorted rather than positional: reordered line items with identical values
    are the same extraction, but a changed value is not.
    """
    tokens = []
    for raw in _NUMBER.findall(text):
        cleaned = raw.replace(",", "")
        try:
            number = float(cleaned)
        except ValueError:
            continue
        # 40.5 and 40.50 are one value; 40 stays "40.0" so int/float agree.
        tokens.append(f"{number:.4f}".rstrip("0").rstrip("."))
    return sorted(tokens)


def numeric_fingerprint(text: str) -> str:
    """Digest of the document's numbers, for tagging and comparing cache entries."""
    return hashlib.sha256("|".join(numeric_tokens(text)).encode()).hexdigest()[:32]
