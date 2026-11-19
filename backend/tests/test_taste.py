import asyncio
from datetime import date, datetime

import numpy as np
import pytest

import taste


class FrozenDatetime(datetime):
    @classmethod
    def now(cls):
        return cls(2026, 10, 25)


class Connection:
    def __init__(self, rows):
        self.rows = rows
        self.closed = False
        self.sql = ""

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql, params):
        self.sql = sql
        assert params == ("user",)

    def fetchall(self):
        # Match the actual database WHERE clause: aspirational films never enter.
        assert "interaction_type = 'watched'" in self.sql
        return [row for row, kind in self.rows if kind == "watched"]

    def close(self):
        self.closed = True


def compute(monkeypatch, rows, vectors):
    connection = Connection(rows)
    monkeypatch.setattr(taste, "get_db_connection", lambda: connection)
    monkeypatch.setattr(taste, "datetime", FrozenDatetime)
    monkeypatch.setattr(taste, "_compute_top_genres", lambda vector: [])

    async def fetch(ids):
        return {i: vectors[i] for i in ids if i in vectors}

    monkeypatch.setattr(taste, "fetch_vectors_batch", fetch)
    result = asyncio.run(taste.compute_taste_vector("user"))
    assert connection.closed
    return result


@pytest.mark.parametrize("rating,liked,watched,weight", [
    (5, False, None, 1),
    (2.5, False, None, 0.25),
    (2.5, True, None, 0.75),
    (5, True, "2026-10-24", 1.8),
    (5, False, date(2026, 10, 24), 1.2),
    (5, False, "2026-04-28", 1),
    (5, False, "2026-04-29", 1.2),
    (5, False, "invalid", 1),
    (None, False, None, 1),
])
def test_quadratic_rating_likes_and_six_month_boundary(monkeypatch, rating, liked, watched, weight):
    x, y = np.eye(2, 768)
    rows = [((1, rating, liked, watched), "watched")]
    rows.extend(((i, 5, False, None), "watched") for i in range(2, 11))
    vectors = {i: y for i in range(2, 11)} | {1: x}
    result = compute(monkeypatch, rows, vectors)
    expected = x * weight + y * 9
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(result["vector"], expected)
    assert result["movie_count"] == 10
    assert np.linalg.norm(result["vector"]) == pytest.approx(1)


def test_watchlist_excluded_and_missing_vectors_skipped(monkeypatch):
    x, y = np.eye(2, 768)
    rows = [((i, 5, False, None), "watched") for i in range(10)]
    rows.append(((99, 5, True, "2026-10-24"), "watchlist"))
    result = compute(monkeypatch, rows, {i: x for i in range(5)} | {99: y})
    np.testing.assert_array_equal(result["vector"], x)
    assert result["movie_count"] == 5


@pytest.mark.parametrize("movies,vectors", [(9, 9), (10, 4)])
def test_insufficient_history_or_vectors(monkeypatch, movies, vectors):
    rows = [((i, 5, False, None), "watched") for i in range(movies)]
    assert compute(monkeypatch, rows, {i: np.ones(768) for i in range(vectors)}) is None
