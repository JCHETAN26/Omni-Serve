"""Pluggable embedders for the semantic cache.

`sentence-transformers` pulls torch, so it is imported lazily and the cache
depends on a protocol rather than a concrete model. That keeps cache logic
testable without a 400MB install, and leaves room to swap MiniLM for a domain
encoder later without touching the cache.
"""

import hashlib
import math
from typing import Protocol, runtime_checkable

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIMS = 384


@runtime_checkable
class Embedder(Protocol):
    @property
    def dims(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Production embedder. Loads the model on first use, not at import."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dims(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return [float(x) for x in vector]


class HashEmbedder:
    """Deterministic bag-of-words embedder — no torch, no downloads.

    Real semantics are not the point: this exists so cache wiring can be tested
    and so `docker compose up` works before anyone has downloaded a model. It is
    genuinely similarity-bearing (shared tokens raise cosine), just crude.
    """

    def __init__(self, dims: int = 64) -> None:
        self._dims = dims

    @property
    def dims(self) -> int:
        return self._dims

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dims
        for token in text.casefold().split():
            digest = hashlib.sha1(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self._dims
            vector[index] += 1.0

        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector] if norm else vector
