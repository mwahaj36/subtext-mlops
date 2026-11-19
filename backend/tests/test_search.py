from types import SimpleNamespace
import random

import numpy as np
import pytest

import concepts
import search
import search_rerank
from search_rules import detect_negation, parse_exclusions, word_in_text


def movie(id, title, vector=None, **extra):
    return {
        "_id": str(id), "title": title, "release_year": 2000,
        "genres": ["Drama"], "vote_average": 8, "vote_count": 2000,
        "overview": "A story of obsession", "keywords": ["obsession"],
        "original_language": "en", "$vector": np.eye(1, 768)[0].tolist(),
    } | ({"$vector": vector.tolist()} if vector is not None else {}) | extra


class Collection:
    def __init__(self, docs):
        self.docs = docs

    def find(self, filter=None, **kwargs):
        if filter and "title" in filter:
            return [d for d in self.docs if d["title"] in filter["title"].get("$in", [])]
        return list(self.docs)

    def find_one(self, filter):
        return next(iter(self.find(filter=filter)), None)


@pytest.fixture
def engine(monkeypatch):
    class Doc(list):
        ents = []

    def nlp(text):
        return Doc(SimpleNamespace(text=word, ent_type_="") for word in text.split())

    monkeypatch.setattr(search, "nlp", nlp)
    monkeypatch.setattr(search_rerank, "nlp", nlp)
    monkeypatch.setattr(search, "PEOPLE_SET", set())
    monkeypatch.setattr(search, "embed", lambda text: np.eye(1, 768)[0])
    monkeypatch.setattr(concepts, "get_index", lambda: None)
    monkeypatch.setattr(concepts, "corpus_mean", lambda: None)

    def setup(docs, concept_index=False):
        monkeypatch.setattr(search, "collection", Collection(docs))
        if concept_index:
            index = concepts.ConceptIndex(["obsession", "music"], np.eye(2, 768), np.zeros(768), np.ones(2), 100, None)
            monkeypatch.setattr(concepts, "get_index", lambda: index)
        return search.search

    return setup


def test_public_api_exports():
    assert search.score_movie is search_rerank.score_movie
    assert callable(search.embed)
    assert callable(search.compute_dynamic_concept_weight)
    assert callable(search.cosine_similarity)


@pytest.mark.parametrize("concept_index", [False, True])
def test_concept_scoring_handles_unknown_keywords_and_preserves_order(engine, concept_index):
    docs = [movie(1, "First"), movie(2, "Second", np.eye(1, 768, k=1)[0], keywords=["music"]), movie(3, "Unknown", keywords=["unknown"])]
    run = engine(docs, concept_index)
    first = run("obsession", debug=True)
    second = run("obsession", debug=True)
    assert first == second
    assert first
    scores = [r["score"] for r in first]
    assert scores == sorted(scores, reverse=True)
    assert all(np.isfinite(r["score"]) for r in first)
    assert first[0]["id"] == "1"
    if concept_index:
        assert first[0]["breakdown"]["concept"] > 0


def test_deduplication_franchise_limit_and_k(engine):
    docs = [movie(i, title) for i, title in enumerate(["Saga", "Saga 2", "Saga 3", "Other", "Other", "Another"])]
    results = engine(docs)("obsession", k=4)
    assert len(results) == 4
    titles = [r["title"] for r in results]
    assert len(titles) == len(set(titles))
    assert sum(title.startswith("Saga") for title in titles) <= 2


def test_watchlist_and_explicit_exclusions(engine):
    run = engine([movie(1, "First"), movie(2, "Second"), movie(3, "Third")])
    assert [r["id"] for r in run("obsession", watchlist_ids=["1", "2"], exclude_ids=["1"])] == ["2"]


def test_illness_filter_requires_document_evidence(engine):
    run = engine([movie(1, "Relevant", overview="A woman living with cancer"), movie(2, "Unrelated", overview="A teacher retires")])
    results = run("cancer")
    assert [r["id"] for r in results] == ["1"]


def test_empty_candidate_pool(engine):
    assert engine([])("obsession") == []


def test_zero_embedding_is_finite(engine, monkeypatch):
    run = engine([movie(1, "First")])
    monkeypatch.setattr(search, "embed", lambda text: np.zeros(768))
    results = run("obsession")
    assert all(np.isfinite(r["score"]) for r in results)


def test_for_you_is_repeatable_and_does_not_reseed_global_rng(engine):
    run = engine([movie(1, "First"), movie(2, "Second")])
    vector = np.eye(1, 768)[0]
    before = random.getstate()
    first = run("", taste_vector=vector, user_id="user")
    assert first == run("", taste_vector=vector, user_id="user")
    assert random.getstate() == before
    assert all(r["mode"] == "TASTE_DISCOVERY" for r in first)


def test_sibling_anchor_pinned_once(engine):
    run = engine([movie(1, "Anchor"), movie(2, "Sibling")], concept_index=True)
    results = run("movies like Anchor", k=2)
    assert results[0]["id"] == "1"
    assert results[0]["score"] == 1
    assert len({r["id"] for r in results}) == len(results)


def test_tonal_guard_penalizes_missing_requested_genre():
    vector = np.eye(1, 768)[0]
    comedy = movie(1, "Comedy", genres=["Comedy"])
    horror = movie(2, "Horror", genres=["Horror"])
    good = search.score_movie("comedy", vector, vector, comedy, "GLOBAL", [], relevance=0.8)[0]
    bad = search.score_movie("comedy", vector, vector, horror, "GLOBAL", [], relevance=0.8)[0]
    assert good > bad
    assert search.score_movie("no horror", vector, vector, horror, "GLOBAL", [], relevance=0.8)[0] < search.score_movie("no horror", vector, vector, comedy, "GLOBAL", [], relevance=0.8)[0]


@pytest.mark.parametrize("mode", ["GLOBAL", "PERSON_STRICT", "SIBLING_DISCOVERY"])
def test_display_normalization_is_monotonic(mode):
    scores = [search.normalize_score(x, mode) for x in np.linspace(-1, 3, 50)]
    assert scores == sorted(scores)
    assert all(0 <= x <= 1 for x in scores)


def test_exclusions_and_word_boundaries():
    assert detect_negation("no horror", "horror")
    assert not word_in_text("war", "warm")
    terms, genres, cleaned = parse_exclusions("obsession without horror")
    assert "Horror" in genres
    assert "obsession" in cleaned


def test_intent_gate_uses_query_type(engine):
    engine([])
    assert search.compute_dynamic_concept_weight("starring someone", "GLOBAL") == 0.05
    assert search.compute_dynamic_concept_weight("rainy neon", "GLOBAL") == 0.55
    assert search.compute_dynamic_concept_weight("obsession", "GLOBAL") == 0.35
