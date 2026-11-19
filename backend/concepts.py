"""
concepts.py — keyword-concept layer for search.

Why this exists
---------------
The stored movie vectors are built from a long blob (overview twice, keywords
twice, genres, ten cast names, director, review text). Mean-pooled, that blob
is dominated by boilerplate every film shares, so the whole 130k-film corpus
collapses into a narrow cosine band: measured on this database, a random pair
of films sits at 0.742 +/- 0.035. Black Swan scores 0.769 against Whiplash
while Bad Boys II scores 0.794. The dense vector cannot tell those apart.

The TMDB keyword lists can. Whiplash carries 'obsession', 'perfectionist',
'perfection', 'public humiliation'; Black Swan carries 'perfectionism',
'self destructiveness', 'obsession'; Bad Boys II carries 'drug lord',
'miami', 'buddy cop'. Same query, same corpus, and the ordering comes out
right — keyword space separates by 0.06 where dense space separates by 0.005.

So we embed the keyword *vocabulary* once, offline, and at query time ask
"which concepts is this query about, and does this film actually carry them?"
That turns a vibe query into the same kind of structured-evidence lookup that
already makes person and illness queries the strongest paths in the engine.

Artifacts (built by build_concept_index.py, both optional):
    concept_index.npz  — vocab, float16 keyword vectors, document frequencies
    corpus_mean.npy    — mean document vector, for centering

Everything degrades to a no-op when the artifacts are absent, so a deploy that
does not ship them keeps the previous behaviour instead of failing.
"""

import os
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(_DIR, "concept_index.npz")
MEAN_PATH = os.path.join(_DIR, "corpus_mean.npy")

# Concepts weaker than this are noise rather than signal — the query simply
# isn't about them. Set against the centred distribution, where unrelated
# keyword pairs sit at 0.00 +/- 0.09 and genuinely related ones run 0.35-0.50.
MIN_CONCEPT_SIM = 0.22
# How many of a film's best-matching keywords count toward its concept score.
# 1 rewards a single lucky tag; the whole list drowns the signal in filler.
TOP_KEYWORDS_PER_DOC = 4

# TMDB keywords that describe how a film was made or packaged rather than what
# it is about. They match on production vocabulary — "movies about obsession"
# was expanding to 'movie' and 'erotic movie' purely because the query said
# "movies", which dragged in Eyes Wide Shut and The Voyeurs.
META_KEYWORDS = {
    "movie", "film", "feature film", "short film", "short", "silent film",
    "b movie", "cult film", "independent film", "live action", "remake",
    "sequel", "prequel", "reboot", "spin off", "woman director",
    "duringcreditsstinger", "aftercreditsstinger", "direct to video",
    "tv movie", "3d", "imax", "black and white", "based on novel or book",
    "based on comic", "based on play or musical", "based on video game",
    "based on tv series", "based on short film", "based on manga",
    "adapted from", "anthology", "ensemble cast", "nudity", "erotic movie",
}

# Words that say nothing about subject matter. Stripped before expansion so
# the concept layer reads what the user means, not how they phrased it.
QUERY_FILLER = {
    "movie", "movies", "film", "films", "show", "shows", "cinema", "flick",
    "flicks", "about", "with", "that", "which", "some", "any", "good", "best",
    "great", "top", "recommend", "recommendations", "suggest", "give", "me",
    "want", "watch", "watching", "looking", "for", "something", "like", "the",
    "a", "an", "of", "and", "or", "please", "kind", "sort", "type", "feel",
    "feels", "really", "very", "quite", "to", "make", "makes", "get", "gets",
    "is", "are", "be", "im", "i'm", "need", "find", "show", "watchable",
}


def strip_filler(query: str) -> str:
    """Drop phrasing words before concept expansion. Falls back to the original
    when a query is nothing but filler ("something good to watch")."""
    kept = [w for w in query.lower().split() if w.strip(",.!?'\"") not in QUERY_FILLER]
    return " ".join(kept) if kept else query

_index = None
_loaded = False


def _unit(a, axis=1):
    return a / np.maximum(np.linalg.norm(a, axis=axis, keepdims=True), 1e-9)


def centre_keyword_space(vecs):
    """Centre the keyword space on its own mean, and return (centred, mean).

    Short-string embeddings are just as anisotropic as document ones: raw, two
    random TMDB keywords sit at 0.707 +/- 0.035 and 'perfectionist' scores
    0.735 against 'drug lord', so max-similarity saturates and ranks nothing.
    Centred, random pairs fall to -0.002 +/- 0.087, 'perfectionist' vs 'drug
    lord' drops to 0.053, while 'jazz' vs 'ballet' holds at 0.480."""
    unit = _unit(np.asarray(vecs, dtype=np.float32))
    kmu = unit.mean(axis=0)
    return _unit(unit - kmu), kmu


class ConceptIndex:
    def __init__(self, vocab, vecs, kmu, df, n_docs, mu):
        self.vocab = list(vocab)
        self.kw2i = {k: i for i, k in enumerate(self.vocab)}
        self.df = df
        self.vecs = vecs      # [V, 768] float32, centred and L2-normalised
        self.kmu = kmu
        self.n_docs = max(int(n_docs), 1)
        self.mu = mu
        # Rare keywords carry more information than ones half the corpus has.
        # Squashed to [0.5, 1.0] so IDF tilts the ranking without dominating it.
        raw_idf = np.log((self.n_docs + 1.0) / (df.astype(np.float64) + 1.0))
        hi = max(float(raw_idf.max()), 1e-6)
        self.idf = (0.5 + 0.5 * (raw_idf / hi)).astype(np.float32)
        # Production metadata is muted rather than deleted, so an index built
        # before this list existed still behaves.
        self.is_meta = np.array([v in META_KEYWORDS for v in self.vocab])

    def query_similarities(self, qvec):
        """Cosine of the query against every keyword in the vocabulary. One
        [V, 768] matmul, ~5ms for a 20k vocabulary — done once per search.

        Deliberately *not* mean-centered: self.mu is the mean of long document
        blobs, and keyword vectors are short strings. Centering short text by a
        document mean mixes the two regimes — it collapsed sibling concept
        scores from ~0.75 to ~0.09 and put Blade Runner in the results for
        "movies like Whiplash". Centering stays where both sides are documents."""
        q = np.asarray(qvec, dtype=np.float32)
        nq = np.linalg.norm(q)
        if nq == 0:
            return np.zeros(len(self.vocab), dtype=np.float32)
        # Same centring the vocabulary got, so query and keywords are compared
        # in one space.
        q = q / nq - self.kmu
        n = np.linalg.norm(q)
        if n == 0:
            return np.zeros(len(self.vocab), dtype=np.float32)
        sims = self.vecs @ (q / n)
        sims[self.is_meta] = 0.0
        return sims

    def refine(self, qsims, topk=16, keep_frac=0.6, blend=0.6):
        """Re-aim the query at the coherent core of what it matched.

        The embedder has no idea which phrases are idioms. "movies to make me
        ugly cry" expands to tears, crying, sorrow, sadness, angst — and also
        to 'cry wolf' (a false-alarm idiom), 'ugliness', 'ugly girl' and
        'crying man', because 'ugly' and 'cry' get read as separate words. Those
        junk concepts are what put Salo, Martyrs and Antichrist in the results
        for a tearjerker query.

        The junk gives itself away: the real concepts agree with each other
        while the strays agree with nothing. So take the top matches, keep the
        ones with above-median agreement, and rebuild the query from those
        (classic pseudo-relevance feedback). No idiom list, no per-query rules —
        it works the same on any query."""
        if len(qsims) == 0:
            return qsims
        n = min(topk, len(qsims))
        idx = np.argpartition(-qsims, n - 1)[:n]
        idx = idx[qsims[idx] > 0]
        if len(idx) < 4:
            return qsims

        V = self.vecs[idx]
        agreement = V @ V.T
        # Mean agreement with the *other* candidates.
        centrality = (agreement.sum(axis=1) - 1.0) / max(len(idx) - 1, 1)
        cut = np.quantile(centrality, 1.0 - keep_frac)
        core = idx[centrality >= cut]
        if len(core) < 2:
            return qsims

        w = qsims[core].astype(np.float32)
        direction = (self.vecs[core] * w[:, None]).sum(axis=0)
        norm = np.linalg.norm(direction)
        if norm == 0:
            return qsims
        refined = self.vecs @ (direction / norm)
        # Blend rather than replace: the original query still has a vote, so a
        # query whose matches genuinely are diffuse doesn't get force-focused.
        out = blend * refined + (1.0 - blend) * qsims
        out[self.is_meta] = 0.0
        return out

    def keyword_indices(self, keywords, limit=None):
        seen, idx = set(), []
        for k in (keywords or []):
            i = self.kw2i.get(k if isinstance(k, str) else str(k))
            if i is not None and i not in seen and not self.is_meta[i]:
                seen.add(i)
                idx.append(i)
        if limit is not None:
            # Keep the most distinctive tags: 'perfectionist' says far more
            # about a film than 'drama'.
            idx.sort(key=lambda i: -self.idf[i])
            idx = idx[:limit]
        return idx

    def anchor(self, keywords, limit=18):
        """Set-to-set comparator for 'movies like X'.

        Averaging a film's keywords into one centroid destroys what makes it
        that film: Whiplash's centroid expands to 'music, musician, action,
        drama, trouble' and Parasite's to 'lifestyle, family, home, business',
        which match everything and rank nothing. Comparing the two keyword
        *sets* keeps the rare tags decisive."""
        idx = self.keyword_indices(keywords, limit=limit)
        if not idx:
            return None
        return AnchorConcepts(self, idx)

    def expand(self, qsims, topn=10):
        """The concepts this query is actually about, best first."""
        if topn <= 0:
            return []
        n = min(topn, len(qsims))
        idx = np.argpartition(-qsims, n - 1)[:n]
        idx = idx[np.argsort(-qsims[idx])]
        return [(self.vocab[i], float(qsims[i])) for i in idx
                if qsims[i] >= MIN_CONCEPT_SIM]

    def doc_score(self, qsims, doc_keywords):
        """How strongly this film's own keywords carry the query's concepts.

        Mean of its best TOP_KEYWORDS_PER_DOC matches, IDF-weighted. Returns
        None when the film has no keywords we know about, so the caller can
        fall back to dense similarity instead of scoring it a hard zero."""
        if not doc_keywords:
            return None
        hits = []
        for kw in doc_keywords:
            i = self.kw2i.get(kw if isinstance(kw, str) else str(kw))
            if i is not None:
                hits.append(qsims[i] * self.idf[i])
        if not hits:
            return None
        hits.sort(reverse=True)
        return float(np.mean(hits[:TOP_KEYWORDS_PER_DOC]))


class AnchorConcepts:
    """Soft set overlap between an anchor film's keywords and a candidate's.

    For each anchor concept, how well does the candidate cover it? IDF-weighted
    so rare tags dominate. The anchor-to-vocabulary similarities are computed
    once per search (~1MB, one matmul), leaving per-candidate scoring to a
    slice and a max."""

    def __init__(self, index, idx):
        self.index = index
        self.idx = idx
        self.sims = index.vecs[idx] @ index.vecs.T   # [n_anchor, V]
        self.w = index.idf[idx].astype(np.float64)
        self.wsum = float(self.w.sum()) or 1.0

    def top_concepts(self, n=10):
        """The anchor's own distinctive tags — used to seed retrieval."""
        return [self.index.vocab[i] for i in self.idx[:n]]

    def score(self, doc_keywords):
        cols = self.index.keyword_indices(doc_keywords)
        if not cols:
            return None
        best = self.sims[:, cols].max(axis=1)
        return float((best * self.w).sum() / self.wsum)


def _load():
    global _index, _loaded
    if _loaded:
        return _index
    _loaded = True
    try:
        if not os.path.exists(INDEX_PATH):
            print("[CONCEPTS] concept_index.npz not found — concept layer disabled.")
            return None
        d = np.load(INDEX_PATH, allow_pickle=True)
        if "vectors_q" in d:
            # int8, already centred at build time. Quantisation costs at most
            # 0.0024 of cosine and leaves top-50 rankings identical, for half
            # the file — worth it for an artifact that lives in git.
            vecs = _unit(d["vectors_q"].astype(np.float32) * float(d["scale"]))
            kmu = d["kmu"].astype(np.float32)
        else:
            vecs, kmu = centre_keyword_space(d["vectors"])
        mu = None
        if os.path.exists(MEAN_PATH):
            mu = np.load(MEAN_PATH).astype(np.float32)
        _index = ConceptIndex(d["vocab"], vecs, kmu, d["df"], int(d["n_docs"]), mu)
        print(f"[CONCEPTS] loaded {len(_index.vocab)} keyword concepts "
              f"(doc centering {'on' if mu is not None else 'off'}).")
    except Exception as e:
        print(f"[CONCEPTS] failed to load concept index: {e}")
        _index = None
    return _index


def get_index():
    return _load()


def corpus_mean():
    idx = _load()
    return idx.mu if idx is not None else None


# Bayesian prior params. C is roughly the corpus mean rating; m is the vote
# count at which a film's own average starts to be believed.
_PRIOR_MEAN = 6.3
_PRIOR_VOTES = 400.0


def quality_prior(vote_average, vote_count):
    """Shrink thin ratings toward the corpus mean.

    Needed because relevance now has real dynamic range: the old quality term
    (vote_average/10 * 0.07, a span of ~0.03) was calibrated against a semantic
    signal that only moved ~0.13, and once relevance spans ~0.9 it stops
    counting for anything. Without this, "slow burn paranoia" answers with
    Sharp Corner (2025) and Snuff (1976) — films that carry the right keyword
    and nothing else — ahead of The Conversation."""
    try:
        v = max(float(vote_count or 0.0), 0.0)
        r = float(vote_average or 0.0)
    except (TypeError, ValueError):
        return _PRIOR_MEAN / 10.0
    if r <= 0:
        return _PRIOR_MEAN / 10.0
    weighted = (v / (v + _PRIOR_VOTES)) * r + (_PRIOR_VOTES / (v + _PRIOR_VOTES)) * _PRIOR_MEAN
    return weighted / 10.0


def zscore(values):
    """Rank the pool against itself.

    Raw cosines here live in a band roughly 0.70-0.90, so a 'semantic * 0.90'
    term only ever moves a score by ~0.13 — less than a single vibe boost
    (0.15) or genre penalty (0.50), which is why keyword rules were quietly
    outvoting semantics. Normalising within the candidate pool restores real
    dynamic range without hand-tuned constants, and adapts per query."""
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return a
    sd = a.std()
    if sd < 1e-9:
        return np.zeros_like(a)
    return (a - a.mean()) / sd
