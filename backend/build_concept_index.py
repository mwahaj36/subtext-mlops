"""
build_concept_index.py — offline build for the keyword-concept layer.

Scans the AstraDB movie collection, collects the TMDB keyword vocabulary with
document frequencies, embeds each keyword once, and writes:

    concept_index.npz   vocab / float16 vectors / df / n_docs
    corpus_mean.npy     mean document vector, used to centre similarities

Run after a large enrichment batch; the vocabulary is stable enough that this
is a monthly-ish job, not a per-sync one.

    python build_concept_index.py                # full rebuild
    python build_concept_index.py --limit 40000  # sample (vocabulary saturates early)

Keywords appearing in fewer than --min-df films are dropped: they are mostly
typos and one-off tags, and they inflate the artifact for no ranking benefit.
"""

import argparse
import os
import time

import numpy as np

import concepts
from search import collection, embed

_DIR = os.path.dirname(os.path.abspath(__file__))


def scan(limit=None, min_df=2):
    """Collect keyword document-frequencies and the mean document vector."""
    df = {}
    vec_sum = None
    n_docs = 0
    n_vecs = 0
    t0 = time.time()

    cursor = collection.find(
        filter={},
        projection={"keywords": 1, "$vector": 1},
        limit=limit,
        request_timeout_ms=120000,
        timeout_ms=0,
    )
    for doc in cursor:
        n_docs += 1
        for kw in (doc.get("keywords") or []):
            if isinstance(kw, str) and kw.strip():
                k = kw.strip().lower()
                df[k] = df.get(k, 0) + 1
        v = doc.get("$vector")
        if v:
            arr = np.asarray(v, dtype=np.float64)
            vec_sum = arr if vec_sum is None else vec_sum + arr
            n_vecs += 1
        if n_docs % 10000 == 0:
            print(f"  {n_docs} docs | {len(df)} keywords | "
                  f"{n_docs / (time.time() - t0):.0f}/s", flush=True)

    kept = {k: c for k, c in df.items() if c >= min_df}
    print(f"scanned {n_docs} docs in {time.time() - t0:.0f}s | "
          f"{len(df)} keywords -> {len(kept)} with df>={min_df}")
    mean = (vec_sum / n_vecs) if (n_vecs and vec_sum is not None) else None
    return kept, mean, n_docs


def build(df_map, mean_vec, n_docs, embed_fn=None):
    embed_fn = embed_fn or embed
    vocab = sorted(df_map)
    print(f"embedding {len(vocab)} keywords...")
    out = np.zeros((len(vocab), 768), dtype=np.float32)
    t0 = time.time()
    for i, kw in enumerate(vocab):
        out[i] = embed_fn(kw)
        if i and i % 2000 == 0:
            rate = i / (time.time() - t0)
            print(f"  {i}/{len(vocab)} | {rate:.0f}/s | "
                  f"eta {(len(vocab) - i) / max(rate, 1e-6) / 60:.1f}min", flush=True)

    # Centre once at build time, then quantise to int8: 0.0024 max cosine
    # error, identical top-50 rankings, half the bytes in git.
    centred, kmu = concepts.centre_keyword_space(out)
    scale = float(np.abs(centred).max()) / 127.0
    quant = np.clip(np.round(centred / scale), -127, 127).astype(np.int8)

    idx_path = os.path.join(_DIR, "concept_index.npz")
    np.savez_compressed(
        idx_path,
        vocab=np.array(vocab, dtype=object),
        vectors_q=quant,
        scale=np.float32(scale),
        kmu=kmu.astype(np.float32),
        df=np.array([df_map[k] for k in vocab], dtype=np.int32),
        n_docs=np.int32(n_docs),
    )
    print(f"wrote {idx_path} ({os.path.getsize(idx_path) / 1e6:.1f} MB)")

    if mean_vec is not None:
        mean_path = os.path.join(_DIR, "corpus_mean.npy")
        np.save(mean_path, mean_vec.astype(np.float32))
        print(f"wrote {mean_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-df", type=int, default=2)
    args = ap.parse_args()

    df_map, mean_vec, n_docs = scan(args.limit, args.min_df)
    build(df_map, mean_vec, n_docs)
