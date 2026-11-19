import numpy as np

from engine.quantization import InvertedIndexPQ, ProductQuantizer


def test_product_quantizer_encode_decode():
    dim = 64
    m = 8
    k = 16
    pq = ProductQuantizer(dim=dim, m=m, k=k, max_iters=10, seed=42)

    rng = np.random.default_rng(42)
    train_data = rng.normal(size=(100, dim)).astype(np.float32)
    pq.train(train_data)

    test_vectors = train_data[:10]
    codes = pq.encode(test_vectors)
    assert codes.shape == (10, m)

    reconstructed = pq.decode(codes)
    assert reconstructed.shape == (10, dim)

    # Check that quantization error is finite and bounded
    mse = np.mean((test_vectors - reconstructed) ** 2)
    assert mse < 1.0


def test_asymmetric_distance_computation():
    dim = 32
    m = 4
    k = 8
    pq = ProductQuantizer(dim=dim, m=m, k=k, max_iters=5, seed=42)

    rng = np.random.default_rng(42)
    dataset = rng.normal(size=(50, dim)).astype(np.float32)
    pq.train(dataset)

    codes = pq.encode(dataset)
    query = rng.normal(size=dim).astype(np.float32)

    adc_dists = pq.compute_asymmetric_distances(query, codes)
    assert len(adc_dists) == 50

    # Verify ADC distances match manual formula sum_m ||q_m - c||^2
    reconstructed = pq.decode(codes)
    manual_dists = np.sum((query - reconstructed) ** 2, axis=1)
    np.testing.assert_allclose(adc_dists, manual_dists, rtol=1e-4)


def test_inverted_index_pq():
    dim = 32
    ivf = InvertedIndexPQ(dim=dim, n_lists=4, m=4, k_sub=8)

    rng = np.random.default_rng(42)
    data = rng.normal(size=(60, dim)).astype(np.float32)
    doc_ids = [f"doc_{i}" for i in range(60)]

    ivf.train_and_index(doc_ids, data)

    # Query
    q = data[0]  # query with exact first document
    res = ivf.query(q, top_k=5, n_probe=4)
    assert len(res) > 0
    top_doc, _ = res[0]
    assert top_doc == "doc_0"
