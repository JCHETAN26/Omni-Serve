"""Semantic cache against a real Redis Stack instance.

Skipped when no Redis is reachable, so the suite still runs on a laptop — except
when REQUIRE_REDIS=1, which CI sets. Without that, a broken Redis service would
turn this file into silent skips and the job would go green having tested
nothing.
"""

import os

import pytest

from gateway.cache.embeddings import HashEmbedder
from gateway.cache.semantic_cache import SemanticCache

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
REQUIRE_REDIS = os.environ.get("REQUIRE_REDIS") == "1"

INVOICE_A = """*** GLOBEX SUPPLY CO. ***
Ref INV-2024-0117  January 19, 2024
Safety Goggles
  24 @ $17.62 = $422.88
SUB $422.88
TAX $0.00
TOT $422.88 (USD)"""

# Byte-different, semantically identical: the case the cache should serve.
INVOICE_A_REFLOWED = INVOICE_A.replace("\n", "  \n ").upper()

# Different invoice, same template: the case that must never be served.
INVOICE_B = INVOICE_A.replace("17.62", "21.40").replace("422.88", "513.60")

RESULT_A = {"vendor": "Globex Supply Co.", "total": 422.88, "currency": "USD"}
RESULT_B = {"vendor": "Globex Supply Co.", "total": 513.60, "currency": "USD"}


@pytest.fixture
async def cache():
    instance = SemanticCache(redis_url=REDIS_URL, embedder=HashEmbedder(), ttl_seconds=60)
    try:
        await instance.connect()
    except Exception as exc:  # noqa: BLE001 - environment, not logic
        if REQUIRE_REDIS:
            pytest.fail(f"REQUIRE_REDIS=1 but Redis at {REDIS_URL} is unreachable: {exc}")
        pytest.skip(f"Redis unavailable at {REDIS_URL}: {exc}")

    await instance.clear()
    yield instance
    await instance.clear()
    await instance.close()


async def test_miss_on_unseen_document(cache):
    assert await cache.get(INVOICE_A) is None
    assert cache.stats["misses"] == 1


async def test_exact_hit_roundtrips_the_payload(cache):
    await cache.set(INVOICE_A, RESULT_A)
    hit = await cache.get(INVOICE_A)

    assert hit is not None
    assert hit.tier == "exact"
    assert hit.value == RESULT_A
    assert hit.similarity == 1.0


async def test_exact_tier_ignores_whitespace_and_case(cache):
    await cache.set(INVOICE_A, RESULT_A)
    hit = await cache.get("  " + INVOICE_A.upper() + "\n\n")

    assert hit is not None
    assert hit.tier == "exact"


async def test_semantic_tier_serves_a_reformatted_duplicate(cache):
    await cache.set(INVOICE_A, RESULT_A)
    hit = await cache.get(INVOICE_A_REFLOWED)

    assert hit is not None
    assert hit.value == RESULT_A
    assert hit.similarity >= cache.threshold


async def test_guard_refuses_a_different_invoice_on_the_same_template(cache):
    """The data-corruption case. A hit here would write 422.88 in place of 513.60."""
    await cache.set(INVOICE_A, RESULT_A)
    hit = await cache.get(INVOICE_B)

    assert hit is None
    assert cache.stats["guard_rejections"] >= 1


async def test_disabling_the_guard_reintroduces_the_bug(cache):
    """Documents exactly what the guard buys, and what turning it off costs."""
    cache.require_numeric_match = False
    cache.threshold = 0.5  # force the tier-2 path regardless of encoder
    await cache.set(INVOICE_A, RESULT_A)

    hit = await cache.get(INVOICE_B)

    assert hit is not None
    assert hit.value == RESULT_A  # wrong invoice's data, served without error


async def test_both_invoices_cached_independently(cache):
    await cache.set(INVOICE_A, RESULT_A)
    await cache.set(INVOICE_B, RESULT_B)

    assert (await cache.get(INVOICE_A)).value == RESULT_A
    assert (await cache.get(INVOICE_B)).value == RESULT_B


async def test_semantic_tier_can_be_switched_off(cache):
    cache.semantic_enabled = False
    await cache.set(INVOICE_A, RESULT_A)

    assert (await cache.get(INVOICE_A)).tier == "exact"
    assert await cache.get(INVOICE_A_REFLOWED) is None


async def test_clear_empties_both_tiers(cache):
    await cache.set(INVOICE_A, RESULT_A)
    await cache.clear()

    assert await cache.get(INVOICE_A) is None


async def test_stats_track_hit_ratio(cache):
    await cache.set(INVOICE_A, RESULT_A)
    await cache.get(INVOICE_A)
    await cache.get(INVOICE_A)
    await cache.get("a completely unrelated support ticket about login errors")

    snapshot = cache.snapshot()
    assert snapshot["exact_hits"] == 2
    assert snapshot["misses"] == 1
    assert snapshot["hit_ratio"] == pytest.approx(2 / 3, abs=0.001)


async def test_ttl_is_applied_to_exact_entries(cache):
    from gateway.cache.fingerprint import exact_key

    await cache.set(INVOICE_A, RESULT_A)
    client = await cache._client()
    ttl = await client.ttl(f"omniserve:exact:{exact_key(INVOICE_A)}")

    assert 0 < ttl <= 60
