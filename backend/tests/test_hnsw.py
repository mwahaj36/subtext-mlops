import numpy as np
import pytest

from engine.hnsw import DistanceMetric, HNSWIndex


def test_distance_metrics():
    u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    w = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    assert DistanceMetric.cosine(u, w) == pytest.approx(0.0, abs=1e-5)
    assert DistanceMetric.cosine(u, v) == pytest.approx(1.0, abs=1e-5)
    assert DistanceMetric.euclidean(u, w) == pytest.approx(0.0, abs=1e-5)
    assert DistanceMetric.euclidean(u, v) == pytest.approx(np.sqrt(2.0), abs=1e-5)


def test_hnsw_insertion_and_search():
    dim = 64
    index = HNSWIndex(dim=dim, m=8, ef_construction=32, ef_search=32, seed=42)

    rng = np.random.default_rng(123)
    vectors = {}
    for i in range(50):
        vec = rng.normal(size=dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        node_id = f"item_{i}"
        vectors[node_id] = vec
        index.add_point(node_id, vec)

    assert index.size() == 50

    # Search for an exact indexed vector
    query_id = "item_10"
    results = index.search(vectors[query_id], k=5)
    assert len(results) == 5
    # The exact vector must be top result with sim ~ 1.0
    top_id, top_sim = results[0]
    assert top_id == query_id
    assert top_sim == pytest.approx(1.0, abs=1e-4)


def test_hnsw_recall_against_brute_force():
    dim = 32
    index = HNSWIndex(dim=dim, m=8, ef_construction=32, ef_search=32, seed=42)
    rng = np.random.default_rng(999)

    items = []
    for i in range(100):
        v = rng.normal(size=dim).astype(np.float32)
        v /= np.linalg.norm(v)
        items.append((f"id_{i}", v))
        index.add_point(f"id_{i}", v)

    # Pick query
    q = rng.normal(size=dim).astype(np.float32)
    q /= np.linalg.norm(q)

    # Brute-force cosine similarity
    brute_force = []
    for nid, v in items:
        sim = float(np.dot(q, v))
        brute_force.append((nid, sim))
    brute_force.sort(key=lambda x: x[1], reverse=True)
    exact_top_5 = {nid for nid, _ in brute_force[:5]}

    hnsw_results = index.search(q, k=5)
    hnsw_top_5 = {nid for nid, _ in hnsw_results}

    # At least 3 out of top 5 must match (>= 60% recall on random 32-D vectors)
    overlap = len(exact_top_5.intersection(hnsw_top_5))
    assert overlap >= 3
