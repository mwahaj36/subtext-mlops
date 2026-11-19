"""Search orchestration and backwards-compatible public imports."""
import re
import time
import math
import concepts
from search_projection import make_centered_similarity as make_centered_similarity
import numpy as np
from search_runtime import (
    MODEL_ID as MODEL_ID,
    get_model as get_model,
    embed as embed,
    nlp as nlp,
    db as db,
    collection as collection,
    PEOPLE_SET as PEOPLE_SET,
    PEOPLE_INDEX_PATH as PEOPLE_INDEX_PATH,
    ASTRA_TOKEN as ASTRA_TOKEN,
    ASTRA_ENDPOINT as ASTRA_ENDPOINT,
)
from search_rules import (
    CORE_GENRES as CORE_GENRES,
    VIBE_RULES as VIBE_RULES,
    SENSORY_EXPANSIONS as SENSORY_EXPANSIONS,
    word_in_text as word_in_text,
    expand_query_for_embedding as expand_query_for_embedding,
    ILLNESS_KEYWORDS as ILLNESS_KEYWORDS,
    detect_negation as detect_negation,
    detect_illness_intent as detect_illness_intent,
    illness_matches_doc as illness_matches_doc,
    _NEG_PATTERNS as _NEG_PATTERNS,
    _NEG_STOP as _NEG_STOP,
    _NEG_SKIP as _NEG_SKIP,
    parse_exclusions as parse_exclusions,
    doc_mentions_term as doc_mentions_term,
    TEXTURE_KEYWORD_MAP as TEXTURE_KEYWORD_MAP,
    build_texture_enrichment as build_texture_enrichment,
    normalize_score as normalize_score,
    smart_title as smart_title,
    get_base_title as get_base_title,
)
from search_rerank import (
    cosine_similarity as cosine_similarity,
    compute_genre_penalty as compute_genre_penalty,
    compute_genre_alignment_penalty as compute_genre_alignment_penalty,
    CONCEPT_WEIGHT as CONCEPT_WEIGHT,
    compute_dynamic_concept_weight as compute_dynamic_concept_weight,
    QUALITY_WEIGHT as QUALITY_WEIGHT,
    SIBLING_GENRE_WEIGHT as SIBLING_GENRE_WEIGHT,
    CLASH_GENRES as CLASH_GENRES,
    compute_vibe_purge_penalty as compute_vibe_purge_penalty,
    compute_multi_vibe_bonus as compute_multi_vibe_bonus,
    score_movie as score_movie,
)

def strip_names(text: str):
    doc = nlp(text)
    return " ".join(["CHARACTER" if token.ent_type_ == "PERSON" else token.text for token in doc])


# =============================================================================
# AstraDB-COMPATIBLE TITLE LOOKUP
# =============================================================================

def find_movie_by_title(title: str):
    variants = list(set([
        title,
        title.title(),
        title.capitalize(),
        smart_title(title),
        title.lower(),
        title.upper(),
    ]))
    return collection.find_one(filter={"title": {"$in": variants}})


def find_movie_by_title_with_vector(title: str, proj: dict, year: int = None):
    variants = list(set([
        title,
        title.title(),
        title.capitalize(),
        smart_title(title),
        title.lower(),
        title.upper(),
        title.replace(" ", ""),
        " ".join(re.findall(r'[A-Z][a-z]*|[a-z]+', title.title()))
    ]))

    f = {"title": {"$in": variants}}
    if year:
        f["release_year"] = year

    proj_with_vec = {**proj, "$vector": 1, "vote_count": 1}
    candidates = list(collection.find(
        filter=f,
        projection=proj_with_vec,
        limit=5
    ))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x.get("vote_count", 0), reverse=True)
    doc = candidates[0]

    if not doc:
        return None

    if "$vector" in doc and doc["$vector"]:
        if np.all(np.array(doc["$vector"]) == 0):
            del doc["$vector"]

    if "$vector" not in doc:
        print(f"📡 Generating missing vector for: {doc['title']}")
        vec = embed(doc["title"] + " " + doc.get("overview", ""))
        if vec is not None and not np.all(vec == 0):
            doc["$vector"] = vec.tolist()

    return doc


# =============================================================================
# RESOLUTION & INTENT LAYERS
# =============================================================================

def resolve_entities(query):
    q = query.lower()
    entities = []

    words = q.replace(",", " ").replace(".", " ").split()
    for n in [3, 2]:
        for i in range(len(words) - n + 1):
            window = " ".join(words[i:i + n])
            if window in PEOPLE_SET:
                entities.append(window.title())

    if not entities:
        doc = nlp(query)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities.append(ent.text)

    if not entities:
        matches = re.findall(r'\b[A-Z][a-z]+\s[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b', query)
        entities.extend(matches)

    return list(set(entities))


def classify_entity_intent(query, entities):
    q = query.lower()
    filmography_triggers = [
        "movies with", "movies starring", "films starring",
        "movies by", "films by", "directed by",
        "starring", "films of", "movies of", "filmography"
    ]
    has_person = len(entities) > 0
    is_explicit = any(t in q for t in filmography_triggers)
    if has_person and is_explicit:
        return "PERSON_STRICT"
    if has_person:
        return "PERSON_VIBE"
    return "GLOBAL"


def infer_query_focus(q_low: str, anchor_genres: list) -> str:
    if "romance" in q_low or "romantic" in q_low or "love story" in q_low:
        return "Romance"
    if "horror" in q_low or "scary" in q_low:
        return "Horror"
    if "comedy" in q_low or "funny" in q_low:
        return "Comedy"
    if "action" in q_low:
        return "Action"
    if "thriller" in q_low:
        return "Thriller"
    if anchor_genres:
        return anchor_genres[0]
    return ""


# =============================================================================
# SCORING HELPERS
# =============================================================================



def search(
    query: str,
    k: int = 10,
    min_year: int = None,
    max_year: int = None,
    language: str = None,
    genres: list = None,
    exclude_genres: list = None,
    min_vote: float = None,
    debug: bool = False,
    taste_vector: object = None,
    taste_blend: float = None,
    watchlist_ids: list = None,
    exclude_ids: list = None,
    user_top_genres: list = None,
    user_id: str = None,
):
    # Handle "For You" mode — empty query with taste vector
    is_for_you = False
    if not query.strip() and taste_vector is not None:
        is_for_you = True
        query = "cinematic experience"
        taste_blend = 1.0
    
    q_low = query.lower()

    # --- AUTO-FILTER PARSING ---
    if not min_year and not max_year:
        decade_match = re.search(r'\b(19|20)?(\d0)s\b', q_low)
        if decade_match:
            century = decade_match.group(1) or "19"
            decade = decade_match.group(2)
            min_year = int(f"{century}{decade}")
            max_year = min_year + 9
            print(f"📅 [AUTO-FILTER] Detected decade: {min_year}-{max_year}")
        elif "1900s" in q_low:
            min_year, max_year = 1900, 1909
        elif any(w in q_low for w in ["old movies", "classic movies", "vintage movies"]):
            max_year = 1980
            print("📅 [AUTO-FILTER] Detected 'old' intent: max_year=1980")

    if not language:
        lang_map = {
            "hindi": "hi", "korean": "ko", "japanese": "ja",
            "french": "fr", "spanish": "es", "german": "de",
            "italian": "it", "chinese": "zh", "tamil": "ta",
            "telugu": "te", "malayalam": "ml", "kannada": "kn",
            "arabic": "ar", "russian": "ru"
        }
        for lang_name, lang_code in lang_map.items():
            if f"{lang_name} movies" in q_low or f"{lang_name} film" in q_low or q_low.startswith(f"{lang_name} "):
                language = lang_code
                print(f"🌐 [AUTO-FILTER] Detected language: {lang_name} ({lang_code})")
                break

    repetition_log = {}
    start_time = time.time()

    vibe_match = None
    for v_name, rules in VIBE_RULES.items():
        if any(word_in_text(w, q_low) for w in rules["keywords"]):
            vibe_match = v_name
            break
    if vibe_match:
        print(f"🎭 [VIBE DETECTED] Query has {vibe_match} tone. Applying constraints.")

    similarity_waiver = any(word_in_text(kw, q_low) for kw in ["like", "similar", "vibe", "style", "-esque"])

    # -------------------------------------------------------------------------
    # 1. INTENT RESOLUTION
    # -------------------------------------------------------------------------
    entities = resolve_entities(query)
    mode = classify_entity_intent(query, entities)
    illness_intent = detect_illness_intent(query)

    proj = {
        "title": 1, "overview": 1, "genres": 1, "vote_average": 1,
        "vote_count": 1, "release_year": 1, "poster_path": 1,
        "keywords": 1, "tagline": 1, "cast_names": 1, "director": 1,
        "popularity": 1, "$vector": 1
    }

    # -------------------------------------------------------------------------
    # 2. ANCHOR RESOLUTION
    # -------------------------------------------------------------------------
    search_vec = None
    reference_title = None
    ref_doc = None
    anchor_id = None
    anchor_genres = []
    query_focus = ""

    like_triggers = ["movies like", "movie like", "films like", "film like", "similar to", "reminds me of"]
    suffix_triggers = ["-esque", "esque", " like movies", " like movie", " vibe movies"]

    for trigger in like_triggers:
        if trigger in q_low:
            reference_title = query[q_low.find(trigger) + len(trigger):].strip().strip("'\"")
            mode = "SIBLING_DISCOVERY"
            break

    if not reference_title:
        genre_like_match = re.search(r"\b(movies|films|comedies|dramas|thrillers|horror|action|sci-fi|animated|romance|shows)\s+like\s+(.*)", q_low)
        if genre_like_match:
            reference_title = genre_like_match.group(2).strip().strip("'\"")
            mode = "SIBLING_DISCOVERY"

    if not reference_title:
        for s_trig in suffix_triggers:
            if s_trig in q_low:
                reference_title = query[:q_low.find(s_trig)].strip().strip("'\"")
                mode = "SIBLING_DISCOVERY"
                break

    # Genre Shield: Skip anchoring if query is a broad genre (e.g. "Mystery" or "Action movies")
    is_broad_genre = any(genre == q_low.replace("movies", "").replace("films", "").replace("movie", "").replace("film", "").strip() 
                         for genre in CORE_GENRES)

    if not reference_title and not is_broad_genre:
        auto_ref = find_movie_by_title(query)
        if auto_ref:
            reference_title = auto_ref["title"]
            mode = "SIBLING_DISCOVERY"

    # FIX: Apply sensory query expansion before embedding for GLOBAL queries.
    # SIBLING_DISCOVERY uses anchor text enrichment instead, so we skip there.
    vibe_anchor_text = query
    embedding_query = expand_query_for_embedding(query) if mode == "GLOBAL" else query

    # Content exclusions. Only for GLOBAL: when a title resolved (SIBLING mode)
    # the words belong to the title, not to a negation — "No Country for Old
    # Men" must not be read as excluding "country".
    excluded_terms, excluded_genres = [], []
    if mode == "GLOBAL":
        excluded_terms, excluded_genres, cleaned_query = parse_exclusions(query)
        if excluded_terms or excluded_genres:
            print(f"🚫 [EXCLUDE] terms={excluded_terms} genres={excluded_genres} "
                  f"| embedding: {cleaned_query!r}")
            embedding_query = expand_query_for_embedding(cleaned_query)

    if reference_title and len(reference_title) >= 2:
        anchor_year = None
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', reference_title)
        if year_match:
            anchor_year = int(year_match.group(1))
            reference_title = reference_title.replace(str(anchor_year), "").strip()

        proj_with_year = {**proj, "release_year": 1}
        ref_doc = find_movie_by_title_with_vector(reference_title, proj_with_year, year=anchor_year)

        if not ref_doc:
            prefix = reference_title.strip().title()
            if len(prefix) >= 2:
                prefix_next = prefix[:-1] + chr(ord(prefix[-1]) + 1)
                f_prefix = {"title": {"$gte": prefix, "$lt": prefix_next}}
                if anchor_year:
                    f_prefix["release_year"] = anchor_year
                candidates = list(collection.find(
                    filter=f_prefix,
                    limit=10,
                    projection=proj
                ))
                if candidates:
                    candidates.sort(key=lambda x: x.get("vote_count", 0), reverse=True)
                    ref_doc = candidates[0]

        if ref_doc:
            anchor_id = ref_doc.get("_id")
            print(f"--- [ANCHOR FOUND] --- {ref_doc['title']} ({ref_doc.get('release_year', 'N/A')}) | id={anchor_id} | votes={ref_doc.get('vote_count', 0)}")

            anchor_overview = strip_names(ref_doc.get("overview", ""))
            anchor_genres_str = ", ".join(ref_doc.get("genres", []))
            anchor_keywords = ref_doc.get("keywords", [])
            texture = build_texture_enrichment(anchor_keywords)
            vibe_anchor_text = (
                f"{anchor_overview} "
                f"This movie is a {anchor_genres_str}. "
                f"{texture}"
            ).strip()

            db_vector = ref_doc.get("$vector")
            search_vec = np.array(db_vector) if db_vector else embed(vibe_anchor_text)
            anchor_genres = ref_doc.get("genres", [])

    query_focus = infer_query_focus(q_low, anchor_genres)

    if search_vec is None:
        # Use expanded query for embedding (sensory expansions applied here)
        search_vec = embed(embedding_query)

    # --- PERSONALIZATION ---
    # If in For You mode (empty query), the search vector is 100% user taste.
    # If searching with a query, we blend the query vector and the user taste vector 
    # but ensure the query remains dominant (max taste weight is capped).
    taste_vec_arr = None
    if taste_vector is not None and taste_blend is not None and taste_blend > 0:
        print(f"🧬 [PERSONALIZATION] Blending query vector with user taste (blend={taste_blend})")
        taste_vec_arr = np.array(taste_vector, dtype=float)
        taste_vec_arr = taste_vec_arr / (np.linalg.norm(taste_vec_arr) + 1e-9)
        
        search_vec = search_vec / (np.linalg.norm(search_vec) + 1e-9)
        if is_for_you:
            search_vec = taste_vec_arr
        else:
            # Scale taste weight to max out at 0.5 so query is always dominant
            taste_weight = taste_blend * 0.5
            search_vec = search_vec + taste_vec_arr * taste_weight
            search_vec = search_vec / (np.linalg.norm(search_vec) + 1e-9)

    # -------------------------------------------------------------------------
    # 2b. CONCEPT EXPANSION
    # Which TMDB keywords is this query actually about? One matmul against the
    # keyword vocabulary; drives both retrieval (pool E) and scoring.
    # -------------------------------------------------------------------------
    cindex = concepts.get_index()
    concept_sims = None      # GLOBAL: query -> vocabulary similarities
    anchor_concepts = None   # SIBLING: anchor keyword set -> candidate keywords
    query_concepts = []
    if cindex is not None:
        try:
            if mode == "SIBLING_DISCOVERY" and ref_doc is not None:
                # Compare the anchor in concept space, not document space:
                # Whiplash's keywords (obsession, perfectionist, public
                # humiliation) are what make Black Swan its sibling, while its
                # document vector sits closer to Bad Boys II.
                anchor_concepts = cindex.anchor(ref_doc.get("keywords"))
            if anchor_concepts is not None:
                query_concepts = [(c, 1.0) for c in anchor_concepts.top_concepts(10)]
            else:
                # Expand from the query's subject matter, not its phrasing.
                stripped = concepts.strip_filler(query)
                concept_vec = embed(stripped) if stripped != query.lower() else search_vec
                concept_sims = cindex.refine(cindex.query_similarities(concept_vec))
                query_concepts = cindex.expand(concept_sims, topn=12)
            if query_concepts:
                print(f"🧩 [CONCEPTS] {', '.join(c for c, _ in query_concepts[:6])}")
        except Exception as e:
            print(f"⚠️ [CONCEPTS] expansion failed: {e}")
            concept_sims = None
            anchor_concepts = None

    # -------------------------------------------------------------------------
    # 3. FILTERS
    # -------------------------------------------------------------------------
    # We do NOT include watchlist_ids or exclude_ids here because AstraDB has a 100-item limit for $in/$nin.
    # We will handle these via in-memory filtering later.
    filters = [{"genres": {"$nin": ["Documentary", "TV Movie"] + excluded_genres}}]
    if min_year:
        filters.append({"release_year": {"$gte": min_year}})
    if max_year:
        filters.append({"release_year": {"$lte": max_year}})
    if language:
        filters.append({"original_language": language})
    if min_vote:
        filters.append({"vote_average": {"$gte": min_vote}})

    if mode == "PERSON_STRICT" and entities and not similarity_waiver:
        p = entities[0]
        filters.append({
            "$or": [
                {"cast_names": {"$in": [p]}},
                {"director": p}
            ]
        })

    search_filter = {"$and": filters} if filters else {}

    # -------------------------------------------------------------------------
    # 4. HYBRID RETRIEVAL
    # -------------------------------------------------------------------------
    candidate_map = {}

    # PATH A: Watchlist-Only Mode (Fetch everything in watchlist and score it)
    if watchlist_ids is not None and len(watchlist_ids) > 0:
        print(f"📥 [SEARCH] Watchlist-Only Mode: Fetching {len(watchlist_ids)} items in batches...")
        watchlist_docs = []
        # Batch by 100 to stay under AstraDB limits
        for i in range(0, len(watchlist_ids), 100):
            batch = watchlist_ids[i:i+100]
            try:
                batch_docs = list(collection.find(
                    filter={"_id": {"$in": batch}},
                    projection=proj
                ))
                watchlist_docs.extend(batch_docs)
            except Exception as e:
                print(f"⚠️ Batch fetch failed: {e}")
        
        candidate_map = {doc["_id"]: doc for doc in watchlist_docs}
        
    else:
        # PATH B: Global Discovery Mode
        try:
            # We fetch a larger pool (150) to allow for in-memory exclusion of watched movies
            pool_a = list(collection.find(
                filter=search_filter,
                sort={"$vector": search_vec.tolist()},
                limit=150, 
                projection=proj
            ))

            pool_b = []
            if mode == "SIBLING_DISCOVERY" and ref_doc:
                pool_b = list(collection.find(
                    filter={**search_filter, "vote_average": {"$gte": 7.5}},
                    sort={"$vector": search_vec.tolist()},
                    limit=50,
                    projection=proj
                ))
                
            pool_c = []
            if mode == "SIBLING_DISCOVERY" and ref_doc:
                kw_list = ref_doc.get("keywords", [])[:5]
                if kw_list:
                    pool_c = list(collection.find(
                        filter={**search_filter, "keywords": {"$in": kw_list}},
                        limit=30,
                        projection=proj
                    ))

            pool_d = []
            if ref_doc and ref_doc.get("director"):
                pool_d = list(collection.find(
                    filter={**search_filter, "director": ref_doc["director"]},
                    limit=20,
                    projection=proj
                ))

            # POOL E — concept retrieval.
            # The dense ANN pool is the hard ceiling on quality: no amount of
            # re-ranking recovers a film it never returned. Measured on this
            # corpus, Black Swan sits outside the top 300 for "movies like
            # Whiplash" (0.808 vs a 0.863 pool floor) while Bad Boys II scores
            # 0.803 — the dense space cannot rank them apart, so the right
            # answers are simply absent. Pulling films that literally carry the
            # query's concepts as TMDB keywords puts them back in contention.
            pool_e = []
            if query_concepts:
                try:
                    pool_e = list(collection.find(
                        filter={**search_filter,
                                "keywords": {"$in": [c for c, _ in query_concepts[:10]]},
                                "vote_count": {"$gte": 25}},
                        sort={"$vector": search_vec.tolist()},
                        limit=120,
                        projection=proj
                    ))
                except Exception as e:
                    print(f"⚠️ [CONCEPTS] concept pool fetch failed: {e}")

            candidate_map = {doc["_id"]: doc
                             for doc in pool_a + pool_b + pool_c + pool_d + pool_e}

        except Exception as e:
            print(f"❌ Database Retrieval Error: {e}")
            return []

    s_norm = np.linalg.norm(search_vec)

    def fast_sim(v):
        if v is None:
            return 0.0
        v_arr = np.array(v, dtype=float)
        n = np.linalg.norm(v_arr)
        return float(np.dot(search_vec, v_arr) / (s_norm * n)) if n > 0 and s_norm > 0 else 0.0

    # -------------------------------------------------------------------------
    # 4b. RELEVANCE — dense + concept evidence, normalised across the pool
    #
    # Raw cosine here spans only ~0.70-0.90, so the old `semantic * 0.90` term
    # could move a score by at most ~0.13 while a single vibe_boost was 0.15
    # and a genre penalty 0.50: the keyword rules were outvoting semantics
    # outright, which is how Bad Boys II reached rank 6 for "slow burn
    # paranoia" from retrieval position 160. Z-scoring both signals within the
    # candidate pool gives each real dynamic range and self-calibrates per
    # query, instead of relying on absolute cosine thresholds that don't hold
    # across query styles.
    # -------------------------------------------------------------------------
    pool = list(candidate_map.values())
    mu = concepts.corpus_mean()

    sim_fn = make_centered_similarity(search_vec, mu) if mu is not None else fast_sim
    dense_sims = [sim_fn(d.get("$vector")) for d in pool]

    if (concept_sims is not None or anchor_concepts is not None) and cindex is not None:
        if anchor_concepts is not None:
            raw = [anchor_concepts.score(d.get("keywords")) for d in pool]
        else:
            raw = [cindex.doc_score(concept_sims, d.get("keywords")) for d in pool]
        known = [x for x in raw if x is not None]
        # A film with no keywords we recognise is unproven, not disproven —
        # give it the pool average so it rides on its dense score alone.
        fill = float(np.mean(known)) if known else 0.0
        concept_scores = [fill if x is None else x for x in raw]
        dyn_concept_weight = compute_dynamic_concept_weight(query, mode)
        z = (concepts.zscore(concept_scores) * dyn_concept_weight
             + concepts.zscore(dense_sims) * (1.0 - dyn_concept_weight))
    else:
        concept_scores = [0.0] * len(pool)
        z = concepts.zscore(dense_sims)

    # Quality prior, normalised in the same space so it stays proportionate to
    # relevance instead of being swamped by it.
    z = z + QUALITY_WEIGHT * concepts.zscore(
        [concepts.quality_prior(d.get("vote_average"), d.get("vote_count")) for d in pool]
    )

    # Sibling searches also need domain coherence. Whiplash's distinctive tags
    # include 'emotional abuse', 'verbal abuse' and 'public humiliation', which
    # match Misery and Natural Born Killers as strongly as they match Sound of
    # Metal — tags alone can't tell an abusive-mentor drama from a torture
    # thriller. Genre overlap can. The existing dna_boost meant to do this is
    # 0.01-0.02, calibrated back when the semantic term only moved 0.13.
    if mode == "SIBLING_DISCOVERY" and anchor_genres:
        aset = set(anchor_genres)
        jaccard = []
        for d in pool:
            g = set(d.get("genres") or [])
            jaccard.append(len(aset & g) / len(aset | g) if (aset | g) else 0.0)
        z = z + SIBLING_GENRE_WEIGHT * concepts.zscore(jaccard)

    # Logistic squash back to (0,1) so downstream bonuses/penalties, the
    # score > 0.1 gate and normalize_score() all keep working on a sane range.
    relevance = {}
    for d, zi, cs, ds in zip(pool, z, concept_scores, dense_sims):
        relevance[str(d["_id"])] = {
            "rel": 1.0 / (1.0 + math.exp(-max(-8.0, min(8.0, float(zi))))),
            "concept": float(cs), "dense": float(ds),
        }

    candidates = sorted(
        pool,
        key=lambda x: relevance[str(x["_id"])]["rel"],
        reverse=True
    )

    # -------------------------------------------------------------------------
    # 5. IN-MEMORY FILTERING (Bypasses AstraDB 100-item limit)
    # -------------------------------------------------------------------------
    # Pre-calculate exclusion sets for O(1) lookup
    watchlist_set = set(watchlist_ids) if watchlist_ids is not None else None
    exclude_set = set(exclude_ids) if exclude_ids is not None else set()

    def apply_filters(apply_content_exclusion=True):
        out, seen, dropped = [], set(), 0
        for doc in candidates:
            doc_id = str(doc["_id"])
            if doc_id in seen:
                continue

            # 1. Exclusion filter (Watched movies)
            if doc_id in exclude_set:
                continue

            # 2. Inclusion filter (Watchlist only)
            if watchlist_set is not None and doc_id not in watchlist_set:
                continue

            # 3. Content exclusion ("without aliens") — drop films that carry
            #    the excluded subject in their title, overview or keywords.
            if apply_content_exclusion and excluded_terms and \
                    any(doc_mentions_term(t, doc) for t in excluded_terms):
                dropped += 1
                continue

            seen.add(doc_id)
            out.append(doc)

            # Raised from 150: the concept pool adds candidates that rank poorly
            # by dense similarity, and a 150 cap would discard exactly those.
            if len(out) >= max(260, k * 3):
                break
        return out, dropped

    filtered_candidates, dropped_by_exclusion = apply_filters()
    if dropped_by_exclusion:
        print(f"🚫 [EXCLUDE] dropped {dropped_by_exclusion} films carrying {excluded_terms}")

    # An over-eager exclusion is worse than none: if the negation parse
    # misfired and gutted the pool, rank without it rather than return nothing.
    if excluded_terms and len(filtered_candidates) < min(k, 5):
        print(f"⚠️ [EXCLUDE] only {len(filtered_candidates)} candidates survived "
              f"{excluded_terms} — ignoring the content exclusion for this query.")
        filtered_candidates, _ = apply_filters(apply_content_exclusion=False)

    candidates = filtered_candidates

    # -------------------------------------------------------------------------
    # 5. SCORING
    # -------------------------------------------------------------------------
    scored_results = []
    anchor_rating = ref_doc.get("vote_average", 7.0) if (mode == "SIBLING_DISCOVERY" and ref_doc) else 7.0

    for doc in candidates:
        if anchor_id and doc.get("_id") == anchor_id:
            continue

        doc_v = doc.get("$vector")
        if doc_v is None:
            continue

        if illness_intent and not illness_matches_doc(illness_intent, doc):
            continue

        rel = relevance.get(str(doc["_id"]), {})
        score, reason, breakdown = score_movie(
            vibe_anchor_text, search_vec, doc_v, doc, mode, entities,
            anchor_genres=anchor_genres, anchor_rating=anchor_rating,
            ref_doc=ref_doc, query_focus=query_focus,
            user_top_genres=user_top_genres, relevance=rel.get("rel")
        )
        if debug:
            breakdown["concept"] = round(rel.get("concept", 0.0), 4)
            breakdown["dense"] = round(rel.get("dense", 0.0), 4)

        if score > 0.1:
            display_score = normalize_score(score, mode)

            # Taste blending — adjust score based on user taste similarity
            taste_sim = 0.0
            if taste_vec_arr is not None and taste_blend and taste_blend > 0:
                taste_sim = cosine_similarity(taste_vec_arr, doc_v)
                if is_for_you:
                    display_score = round(max(0.0, taste_sim), 4)
                else:
                    # Blend the display score and taste similarity, but cap the taste influence weight 
                    # at 0.4 so the search query remains the primary driver.
                    taste_weight = taste_blend * 0.4
                    query_weight = 1.0 - taste_weight
                    blended_score = query_weight * display_score + taste_weight * max(0.0, taste_sim)
                    display_score = round(blended_score, 4)

            # DAILY DRIFT: If it's a "For You" discovery query, add a daily user-specific jitter
            if is_for_you and user_id:
                try:
                    import random
                    from datetime import datetime
                    # Seed with user_id and current date so it changes daily but stays consistent for the user during the day
                    day_str = datetime.now().strftime("%Y-%m-%d")
                    # Use a stable hash of user_id + date + doc_id to keep the jitter consistent for this specific movie today
                    seed_str = f"{user_id}_{day_str}_{str(doc.get('_id'))}"
                    jitter = random.Random(seed_str).uniform(-0.04, 0.04)
                    display_score = round(max(0.01, display_score + jitter), 4)
                except Exception:
                    pass

            result = {
                "id": str(doc.get("_id")),
                "title": doc.get("title"),
                "year": doc.get("release_year", 0),
                "score": display_score,
                "ranking_score": round(float(score), 4),
                "reason": reason if not is_for_you else "Matches your cinematic taste profile.",
                "mode": "TASTE_DISCOVERY" if is_for_you else mode,
                "genres": doc.get("genres", []),
                "vote": doc.get("vote_average", 0),
                "poster_path": doc.get("poster_path"),
            }
            # Add taste match percentage for badge display
            if taste_vector is not None and taste_sim > 0:
                result["taste_match"] = round(taste_sim * 100)
            if debug:
                result["breakdown"] = breakdown
            scored_results.append(result)

    scored_results.sort(key=lambda x: x["score"], reverse=True)

    # -------------------------------------------------------------------------
    # 6. DEDUP + FRANCHISE CAP + GENRE DIVERSITY
    # -------------------------------------------------------------------------
    # The genre-diversity nudge that used to live here has been removed rather
    # than repaired. It multiplied score by 0.82 *inside* the selection loop —
    # after the sort — so it rewrote the score clients display without ever
    # changing the order, which is how a response could show 0.6959 ranked
    # above 0.8394 (normalize_score(0.887) = 0.8487, x0.82 = 0.6959).
    #
    # Making it actually apply turned out to be worse than leaving it dead: it
    # keys on the exact genre tuple, so it penalises the plain-Drama films that
    # are the right answers for a mood query and promotes whatever carries an
    # unusual genre mix. On "movies to make me ugly cry" it pushed Ordinary
    # People and Close out and pulled Antichrist to #9. Ten dramas is the
    # correct answer to a query about grief; diversity is not a virtue here.
    final_output = []
    seen = set()
    franchise_counts = {}
    title_counts = {}

    for r in scored_results:
        base = get_base_title(r["title"])
        movie_key = (r["title"], r["year"])

        # Two films with the *same* title (a remake, or "How to Train Your
        # Dragon" 2010 and 2025) read as a duplicate bug to a user even though
        # they're different records, so allow only one per exact title.
        if title_counts.get(r["title"], 0) >= 1:
            continue

        if movie_key not in seen and franchise_counts.get(base, 0) < 2:
            final_output.append(r)
            seen.add(movie_key)
            franchise_counts[base] = franchise_counts.get(base, 0) + 1
            title_counts[r["title"]] = title_counts.get(r["title"], 0) + 1
            repetition_log[r["title"]] = repetition_log.get(r["title"], 0) + 1

        if len(final_output) >= k:
            break

    # -------------------------------------------------------------------------
    # 7. ANCHOR PIN (SIBLING_DISCOVERY)
    # -------------------------------------------------------------------------
    if mode == "SIBLING_DISCOVERY" and ref_doc:
        passes = True
        ry = ref_doc.get("release_year", 0)
        va = ref_doc.get("vote_average", 0)
        ol = ref_doc.get("original_language", "")

        if min_year and ry < min_year:
            passes = False
        if max_year and ry > max_year:
            passes = False
        if min_vote and va < min_vote:
            passes = False
        if language and ol != language:
            passes = False

        if passes:
            anchor_result = {
                "id": str(ref_doc.get("_id")),
                "title": ref_doc.get("title"),
                "year": ry,
                "score": 1.0,
                "reason": "This is the movie you searched for.",
                "mode": mode,
                "genres": ref_doc.get("genres", []),
                "vote": va,
                "poster_path": ref_doc.get("poster_path"),
            }
            if debug:
                anchor_result["breakdown"] = {"note": "anchor — pinned to #1, not scored"}

            final_output = [r for r in final_output if r["id"] != str(ref_doc.get("_id"))]
            final_output = [anchor_result] + final_output[:k - 1]

    latency = int((time.time() - start_time) * 1000)
    print(
        f"[{mode}] '{query}' | "
        f"illness={illness_intent} | "
        f"anchor={ref_doc['title'] if ref_doc else None} | "
        f"candidates={len(candidates)} | "
        f"scored={len(scored_results)} | "
        f"latency={latency}ms"
    )
    return final_output
