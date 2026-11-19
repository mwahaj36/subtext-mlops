import numpy as np
import pytest

import concepts
from search_projection import linear_project, make_centered_similarity, normalize
from search_rerank import cosine_similarity


def axes():
    return np.eye(3, 768)


def test_normalize_768_and_zero():
    vector = np.arange(768, dtype=float)
    unit = normalize(vector)
    assert unit.shape == (768,)
    assert np.linalg.norm(unit) == pytest.approx(1)
    np.testing.assert_allclose(normalize(unit), unit)
    np.testing.assert_array_equal(normalize(np.zeros(768)), np.zeros(768))


def test_centering_removes_shared_corpus_direction(tmp_path):
    x, y, shared = axes()
    path = tmp_path / "corpus_mean.npy"
    np.save(path, shared * 10)
    similarity = make_centered_similarity(x + shared * 10, np.load(path))
    assert similarity(x + shared * 10) == pytest.approx(1)
    assert similarity(y + shared * 10) == pytest.approx(0)
    assert similarity(-x + shared * 10) == pytest.approx(-1)
    assert similarity(shared * 10) == 0
    assert similarity(None) == 0
    assert make_centered_similarity(shared, shared)(x) == 0


def test_linear_projection_matches_affine_mapper(tmp_path):
    vectors = np.arange(2 * 768).reshape(2, 768)
    coef = np.eye(3, 768)
    intercept = np.array([10, 20, 30])
    path = tmp_path / "linear_projection.npz"
    np.savez(path, coef=coef, intercept=intercept)
    projected = linear_project(vectors, path)
    assert projected.shape == (2, 3)
    np.testing.assert_array_equal(projected, vectors[:, :3] + intercept)


@pytest.mark.parametrize("left,right,expected", [(0, 0, 1), (0, 1, 0), (1, 2, 0)])
def test_orthogonal_concept_axes(left, right, expected):
    vectors = axes()
    assert cosine_similarity(vectors[left], vectors[right]) == pytest.approx(expected)
    assert cosine_similarity(np.zeros(768), vectors[left]) == 0


def test_concept_cosine_and_orthogonal_decomposition():
    x, y, z = axes()
    query = normalize(3 * x + 4 * y)
    parallel = np.dot(query, x) * x
    residual = query - parallel
    assert np.dot(parallel, residual) == pytest.approx(0)
    np.testing.assert_allclose(parallel + residual, query)
    index = concepts.ConceptIndex(["obsession", "music", "movie"], axes(), np.zeros(768), np.ones(3), 10, None)
    sims = index.query_similarities(query)
    np.testing.assert_allclose(sims, [0.6, 0.8, 0])
    np.testing.assert_array_equal(index.query_similarities(np.zeros(768)), np.zeros(3))
    assert index.doc_score(sims, ["unknown"]) is None
    assert index.doc_score(sims, ["music"]) > index.doc_score(sims, ["obsession"])
    assert index.keyword_indices(["music", "music", "movie", "unknown"]) == [1]
    anchor = index.anchor(["obsession"])
    assert anchor.score(["obsession"]) == pytest.approx(1)
    assert anchor.score(["music"]) == pytest.approx(0)
    assert anchor.score(["unknown"]) is None
    assert index.anchor(["unknown"]) is None


def test_keyword_space_has_its_own_mean():
    vectors = np.array([[1, 1, 0], [0, 1, 1]], dtype=float)
    centered, mean = concepts.centre_keyword_space(vectors)
    np.testing.assert_allclose(mean, concepts._unit(vectors).mean(axis=0))
    np.testing.assert_allclose(np.linalg.norm(centered, axis=1), 1)
    assert np.dot(centered[0], centered[1]) == pytest.approx(-1)


def test_optional_artifacts_load_mean_and_quantized_vectors(tmp_path, monkeypatch):
    path = tmp_path / "concept_index.npz"
    mean_path = tmp_path / "corpus_mean.npy"
    np.savez(path, vocab=["obsession", "music", "movie"], vectors_q=axes().astype(np.int8), scale=1.0, kmu=np.zeros(768), df=np.ones(3), n_docs=10)
    np.save(mean_path, np.full(768, 0.01))
    monkeypatch.setattr(concepts, "INDEX_PATH", str(path))
    monkeypatch.setattr(concepts, "MEAN_PATH", str(mean_path))
    monkeypatch.setattr(concepts, "_loaded", False)
    monkeypatch.setattr(concepts, "_index", None)
    assert concepts.get_index().vecs.shape == (3, 768)
    np.testing.assert_allclose(concepts.corpus_mean(), 0.01)


def test_missing_artifacts_disable_concepts(tmp_path, monkeypatch):
    monkeypatch.setattr(concepts, "INDEX_PATH", str(tmp_path / "missing.npz"))
    monkeypatch.setattr(concepts, "_loaded", False)
    monkeypatch.setattr(concepts, "_index", None)
    assert concepts.get_index() is None
    assert concepts.corpus_mean() is None


def test_pool_normalization_and_quality_prior():
    np.testing.assert_array_equal(concepts.zscore([4, 4, 4]), [0, 0, 0])
    values = concepts.zscore([1, 2, 3])
    assert values.mean() == pytest.approx(0)
    assert values.std() == pytest.approx(1)
    assert concepts.zscore([]).size == 0
    assert concepts.quality_prior(10, 1) < concepts.quality_prior(8, 10000)
    assert concepts.quality_prior(None, None) == pytest.approx(0.63)
