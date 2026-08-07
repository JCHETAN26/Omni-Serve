import pytest

from benchmarks.workload import build_requests, percentile, summarize

DOCS = [f"invoice number {i}" for i in range(50)]


def test_zero_duplicate_ratio_never_repeats():
    requests = build_requests(DOCS, count=50, duplicate_ratio=0.0)

    assert len(requests) == 50
    assert len(set(requests)) == 50


def test_full_duplicate_ratio_repeats_almost_everything():
    requests = build_requests(DOCS, count=50, duplicate_ratio=1.0)

    # The first request has nothing to repeat, so exactly one is unique.
    assert len(set(requests)) == 1


def test_duplicate_ratio_lands_near_the_requested_rate():
    corpus = [f"invoice number {i}" for i in range(2000)]
    requests = build_requests(corpus, count=2000, duplicate_ratio=0.3, seed=1)
    unique_share = len(set(requests)) / len(requests)

    assert 0.6 < unique_share < 0.78  # ~0.7 unique for a 0.3 duplicate rate


def test_corpus_too_small_is_refused_rather_than_silently_inflating_duplicates():
    """Wrapping the document cycle would turn 0.3 into 0.975 with no warning."""
    with pytest.raises(ValueError, match="corpus too small"):
        build_requests(DOCS, count=2000, duplicate_ratio=0.3)


def test_corpus_exactly_large_enough_is_accepted():
    requests = build_requests(DOCS, count=50, duplicate_ratio=0.0)

    assert len(requests) == 50


def test_duplicates_are_drawn_from_documents_already_issued():
    """Repeats must be cacheable — sampling the whole corpus would not be."""
    requests = build_requests(DOCS, count=50, duplicate_ratio=0.5, seed=2)

    # New documents can only enter via the corpus branch, which walks the
    # corpus in order. So first-occurrences must appear in corpus order — a
    # duplicate can never introduce a document the cache has not seen.
    first_seen = list(dict.fromkeys(requests))

    assert first_seen == DOCS[: len(first_seen)]


def test_workload_is_reproducible_from_seed():
    corpus = [f"invoice number {i}" for i in range(100)]

    assert build_requests(corpus, 100, 0.3, seed=7) == build_requests(corpus, 100, 0.3, seed=7)
    assert build_requests(corpus, 100, 0.3, seed=7) != build_requests(corpus, 100, 0.3, seed=8)


def test_invalid_ratio_is_rejected():
    with pytest.raises(ValueError):
        build_requests(DOCS, 10, duplicate_ratio=1.5)
    with pytest.raises(ValueError):
        build_requests(DOCS, 10, duplicate_ratio=-0.1)


def test_empty_corpus_is_rejected():
    with pytest.raises(ValueError):
        build_requests([], 10, 0.3)


def test_percentile_uses_nearest_rank():
    samples = [float(i) for i in range(1, 101)]

    assert percentile(samples, 0.50) == 50.0
    assert percentile(samples, 0.99) == 99.0
    assert percentile(samples, 1.0) == 100.0


def test_percentile_handles_single_and_empty_samples():
    assert percentile([], 0.5) is None
    assert percentile([42.0], 0.99) == 42.0


def test_summarize_reports_the_expected_shape():
    result = summarize([10.0, 20.0, 30.0, 40.0])

    assert result["count"] == 4
    assert result["max"] == 40.0
    assert result["mean"] == 25.0
    assert result["p50"] is not None


def test_summarize_of_nothing_is_not_zero():
    """Empty must be None, not 0.0 — a zero p99 reads as a fast system."""
    result = summarize([])

    assert result["count"] == 0
    assert result["p99"] is None
    assert result["mean"] is None
