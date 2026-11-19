"""
query_log.py — durable search-query logging for retrieval research.

Why Postgres and not a file: the Hugging Face Space runs in a container with an
ephemeral filesystem. Anything written to disk there is destroyed on the next
rebuild or restart, so a JSONL log on the Space would appear to work and then
silently vanish. Neon is already attached and durable, so the log survives
deploys and can be pulled from anywhere.

Privacy: the query text and coarse result shape are recorded. No user id, no IP,
no session identifier — `has_user` is a single boolean so we can tell
personalised traffic from anonymous without identifying anyone. That is the
minimum needed to report "N queries from real traffic" in a paper, and nothing
more.

Failure policy: logging must never break or slow a search. Every path is wrapped,
every error is swallowed, and the write happens off the request thread.
"""

import asyncio
import json
import os

from database import get_db_connection

_ENABLED = os.getenv("QUERY_LOG_ENABLED", "1") != "0"
_table_ready = False

_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
    id          BIGSERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    query       TEXT        NOT NULL,
    k           INTEGER,
    n_results   INTEGER,
    mode        TEXT,
    latency_ms  INTEGER,
    top_titles  JSONB,
    has_user    BOOLEAN     NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS query_log_ts_idx ON query_log (ts DESC);
"""


def ensure_table():
    """Idempotent. Called once lazily rather than at import, so a database
    hiccup at boot can't stop the API from starting."""
    global _table_ready
    if _table_ready:
        return True
    conn = get_db_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA)
            conn.commit()
        _table_ready = True
        return True
    except Exception as e:
        print(f"[QUERY_LOG] table init failed: {e}")
        return False
    finally:
        conn.close()


def _write_sync(query, k, n_results, mode, latency_ms, top_titles, has_user):
    if not ensure_table():
        return
    conn = get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO query_log
                   (query, k, n_results, mode, latency_ms, top_titles, has_user)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (query[:500], k, n_results, mode, latency_ms,
                 json.dumps(top_titles[:5], ensure_ascii=False), has_user),
            )
            conn.commit()
    except Exception as e:
        print(f"[QUERY_LOG] insert failed: {e}")
    finally:
        conn.close()


def log_search(query, results, k=None, latency_ms=None, has_user=False):
    """Fire-and-forget. Returns immediately; the insert runs on a worker thread
    so a slow database never adds latency to a search response."""
    if not _ENABLED:
        return
    try:
        q = (query or "").strip()
        if not q:
            return  # empty query = "For You" mode, nothing to learn from
        payload = dict(
            query=q,
            k=k,
            n_results=len(results) if results is not None else 0,
            mode=(results[0].get("mode") if results else None),
            latency_ms=latency_ms,
            top_titles=[r.get("title") for r in (results or [])[:5]],
            has_user=bool(has_user),
        )
        asyncio.create_task(asyncio.to_thread(_write_sync, **payload))
    except Exception:
        pass  # never let logging surface to the caller
