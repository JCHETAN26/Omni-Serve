"""Two-tier response cache backed by Redis (Phase 5).

Tier 1 — exact. SHA-256 of the normalized document. Always correct, sub-millisecond,
and it catches the hit pattern that actually dominates production: retries,
duplicate submissions, and batch reprocessing of the same files.

Tier 2 — semantic. Vector search over document embeddings, gated on a cosine
threshold *and* agreement on every number in the document. See `fingerprint` for
why the second condition is not optional.

Both tiers skip the GPU entirely on a hit, which is the whole point.
"""

import json
from dataclasses import dataclass
from typing import Any

from redisvl.index import AsyncSearchIndex
from redisvl.query import VectorQuery
from redisvl.redis.utils import array_to_buffer

from gateway.cache.embeddings import Embedder, HashEmbedder
from gateway.cache.fingerprint import exact_key, numeric_fingerprint, normalize

EXACT_PREFIX = "omniserve:exact"
INDEX_NAME = "omniserve-cache"
INDEX_PREFIX = "omniserve:sem"


@dataclass(frozen=True)
class CacheHit:
    value: dict
    tier: str  # "exact" | "semantic"
    similarity: float


class SemanticCache:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        embedder: Embedder | None = None,
        threshold: float = 0.95,
        ttl_seconds: int | None = 86_400,
        require_numeric_match: bool = True,
        semantic_enabled: bool = True,
    ) -> None:
        self.redis_url = redis_url
        self.embedder = embedder or HashEmbedder()
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.require_numeric_match = require_numeric_match
        self.semantic_enabled = semantic_enabled
        self._index: AsyncSearchIndex | None = None
        self.stats = {"exact_hits": 0, "semantic_hits": 0, "misses": 0, "guard_rejections": 0}

    # ---- lifecycle -------------------------------------------------------

    def _schema(self) -> dict:
        return {
            "index": {
                "name": INDEX_NAME,
                "prefix": INDEX_PREFIX,
                "storage_type": "hash",
            },
            "fields": [
                {"name": "digits", "type": "tag"},
                {"name": "response", "type": "text"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": self.embedder.dims,
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                        "datatype": "float32",
                    },
                },
            ],
        }

    async def connect(self) -> None:
        self._index = AsyncSearchIndex.from_dict(self._schema(), redis_url=self.redis_url)
        await self._index.create(overwrite=False)

    async def close(self) -> None:
        if self._index is not None:
            await self._index.disconnect()
            self._index = None

    @property
    def index(self) -> AsyncSearchIndex:
        if self._index is None:
            raise RuntimeError("SemanticCache.connect() must be awaited before use")
        return self._index

    async def _client(self):
        """The raw redis client, for the exact tier's plain GET/SET.

        `client` is populated once connected; the private accessor is the
        documented way to force lazy connection and is the fallback only.
        """
        return self.index.client or await self.index._get_client()

    # ---- reads -----------------------------------------------------------

    async def get(self, text: str) -> CacheHit | None:
        client = await self._client()

        raw = await client.get(f"{EXACT_PREFIX}:{exact_key(text)}")
        if raw is not None:
            self.stats["exact_hits"] += 1
            return CacheHit(json.loads(raw), "exact", 1.0)

        if not self.semantic_enabled:
            self.stats["misses"] += 1
            return None

        hit = await self._semantic_get(text)
        if hit is None:
            self.stats["misses"] += 1
        else:
            self.stats["semantic_hits"] += 1
        return hit

    async def _semantic_get(self, text: str) -> CacheHit | None:
        vector = self.embedder.embed(normalize(text))
        query = VectorQuery(
            vector=vector,
            vector_field_name="embedding",
            return_fields=["response", "digits"],
            num_results=1,
            dtype="float32",
        )

        results = await self.index.query(query)
        if not results:
            return None

        match = results[0]
        similarity = 1.0 - float(match["vector_distance"])
        if similarity < self.threshold:
            return None

        # Similar prose, different numbers: the case that would corrupt data.
        if self.require_numeric_match and match.get("digits") != numeric_fingerprint(text):
            self.stats["guard_rejections"] += 1
            return None

        return CacheHit(json.loads(match["response"]), "semantic", similarity)

    # ---- writes ----------------------------------------------------------

    async def set(self, text: str, value: dict) -> None:
        payload = json.dumps(value)
        client = await self._client()

        key = f"{EXACT_PREFIX}:{exact_key(text)}"
        if self.ttl_seconds:
            await client.set(key, payload, ex=self.ttl_seconds)
        else:
            await client.set(key, payload)

        if not self.semantic_enabled:
            return

        record = {
            "digits": numeric_fingerprint(text),
            "response": payload,
            "embedding": array_to_buffer(self.embedder.embed(normalize(text)), "float32"),
        }
        await self.index.load(
            [record], keys=[f"{INDEX_PREFIX}:{exact_key(text)}"], ttl=self.ttl_seconds
        )

    async def clear(self) -> None:
        client = await self._client()
        cursor = 0
        while True:
            cursor, keys = await client.scan(cursor, match=f"{EXACT_PREFIX}:*", count=500)
            if keys:
                await client.delete(*keys)
            if cursor == 0:
                break
        await self.index.clear()
        for key in self.stats:
            self.stats[key] = 0

    # ---- reporting -------------------------------------------------------

    def hit_ratio(self) -> float:
        hits = self.stats["exact_hits"] + self.stats["semantic_hits"]
        total = hits + self.stats["misses"]
        return round(hits / total, 4) if total else 0.0

    def snapshot(self) -> dict[str, Any]:
        return {**self.stats, "hit_ratio": self.hit_ratio(), "threshold": self.threshold}
