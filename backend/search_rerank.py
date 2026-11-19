"""Candidate scoring and tonal guards; no network or model initialization."""
import re
import numpy as np
from search_runtime import nlp
from search_rules import (
    VIBE_RULES,
    word_in_text,
    detect_negation,
)

def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_genre_penalty(doc_genres: list, kill_genres: list, boost_genres: list) -> float:
    kill_hits = sum(1 for g in doc_genres if g in kill_genres)
    boost_hits = sum(1 for g in doc_genres if g in boost_genres)
    return max(0, kill_hits - boost_hits) * 0.50


def compute_genre_alignment_penalty(anchor_genres: list, doc_genres: list, query_focus: str) -> float:
    if not anchor_genres:
        return 0.0
    missing = set(anchor_genres) - set(doc_genres)
    penalty = 0.0
    for g in missing:
        if query_focus and g.lower() == query_focus.lower():
            penalty += 0.25
        else:
            penalty += 0.08
    return penalty


# How much of the relevance signal comes from keyword-concept evidence rather
# Base default weight of the keyword-concept evidence layer.
CONCEPT_WEIGHT = 0.55

def compute_dynamic_concept_weight(query: str, mode: str) -> float:
    """
    Dynamically gates the concept evidence weight to resolve the Intent Asymmetry (S8):
    - Subjective, affective, sensory, and negation queries route strongly to concept evidence (weight 0.55).
    - Entity-dense queries (specific actors, directors, cast) route directly to dense vectors (weight 0.05)
      to prevent 21,822 atmospheric keywords from diluting the entity signal.
    - Default / thematic queries use a balanced weight (0.35).
    """
    q_low = query.lower()

    # 1. Entity / Person patterns: actor or director lookups
    person_indicators = [
        "starring", "directed by", "director", "actor", "actress",
        "played by", "roles", "featuring", "cast"
    ]
    if any(p in q_low for p in person_indicators):
        print("🎯 [INTENT GATE] Entity/Person intent detected -> Gating concept weight to 0.05 (Dense dominant)")
        return 0.05

    try:
        doc = nlp(query)
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        if persons and len(query.split()) <= 6:
            print(f"🎯 [INTENT GATE] Named Person detected ({persons}) -> Gating concept weight to 0.05")
            return 0.05
    except Exception:
        pass

    # 2. Subjective / Affective / Sensory / Negation patterns
    subjective_indicators = [
        "feel", "vibe", "vibes", "mood", "cry", "sad", "tearjerker", "cozy", "warm hug",
        "neon", "rainy", "slow burn", "paranoia", "sun-bleached", "aesthetic", "gloomy",
        "no ", "not ", "without ", "none ", "never ", "zero jump", "unravel", "fall apart",
        "devastating", "heartbreaking", "existential", "nostalgic", "wholesome", "depressing"
    ]
    if any(s in q_low for s in subjective_indicators):
        print("🎯 [INTENT GATE] Subjective/Affective/Negation intent detected -> Concept weight 0.55 (Hybrid activated)")
        return 0.55

    # 3. Default / Thematic discovery
    return 0.35

# Weight of the Bayesian quality prior, in the same normalised space as
# relevance. Enough to break ties toward films people actually rate highly,
# not enough to turn every query into a popularity chart.
QUALITY_WEIGHT = 0.35

# Genre agreement with the anchor film, sibling searches only.
SIBLING_GENRE_WEIGHT = 0.30

CLASH_GENRES = {"Thriller", "Horror", "War", "Science Fiction", "Action", "Crime", "Mystery"}

def compute_vibe_purge_penalty(anchor_genres: list, doc_genres: list) -> float:
    if not anchor_genres:
        return 0.0
    anchor_set = set(anchor_genres)
    is_romance_anchor = "Romance" in anchor_set or "Musical" in anchor_set
    if not is_romance_anchor:
        return 0.0

    anchor_clash_identity = anchor_set.intersection(CLASH_GENRES)
    doc_set = set(doc_genres)
    true_clashes = doc_set.intersection(CLASH_GENRES) - anchor_clash_identity

    if not true_clashes:
        return 0.0

    if "Music" in doc_genres or "Musical" in doc_genres:
        return 0.15

    return 0.60


def compute_multi_vibe_bonus(q_low: str, doc_genres: list) -> float:
    """
    Rewards films that satisfy multiple vibe dimensions simultaneously.
    A film matching 'glamorous + self-destructive + great soundtrack' should
    rank above one that only matches 'glamorous'.
    Adds 0.10 per additional vibe matched beyond the first (capped at 0.30).
    """
    matched_count = 0
    for vibe, rules in VIBE_RULES.items():
        matched_keywords = [w for w in rules["keywords"] if word_in_text(w, q_low)]
        if not matched_keywords:
            continue
        all_negated = all(detect_negation(q_low, w) for w in matched_keywords)
        if all_negated:
            continue
        if any(g in doc_genres for g in rules["boost_genres"]):
            matched_count += 1
    if matched_count >= 2:
        return min(0.10 * (matched_count - 1), 0.30)
    return 0.0


# =============================================================================
# MAIN SCORING FUNCTION
# =============================================================================

def score_movie(
    query, query_vec, doc_vec, doc, mode, entities,
    anchor_genres=None, anchor_rating=7.0, ref_doc=None, query_focus="",
    user_top_genres=None, relevance=None
):
    q_low = query.lower()
    doc_genres = doc.get("genres", [])

    # `relevance` is the pool-normalised dense+concept score computed in
    # search(); it spans (0,1) properly instead of the ~0.13 of usable range
    # raw cosine gives. Falls back to raw cosine when scored outside search().
    if relevance is not None:
        semantic = relevance
    else:
        semantic = cosine_similarity(query_vec, doc_vec) if doc_vec is not None else 0.0

    penalty = 0.0
    genre_penalty = 0.0
    vibe_boost = 0.0
    alignment_penalty = 0.0
    mood_penalty = 0.0
    person_penalty = 0.0
    multi_vibe_bonus = 0.0

    # --- Mood Purity (SIBLING_DISCOVERY) ---
    if mode == "SIBLING_DISCOVERY":
        is_anchor_romance = any(g in {"Romance", "Drama"} for g in (anchor_genres or []))
        if is_anchor_romance:
            doc_overview = doc.get("overview", "").lower()
            clash_kw = ["stoner", "slapstick", "gross-out", "buddy comedy"]
            if any(k in doc_overview for k in clash_kw) or (
                "Comedy" in doc_genres and "Drama" not in doc_genres
            ):
                alignment_penalty += 0.50

    # --- Genre Alignment ---
    if anchor_genres and mode != "PERSON_STRICT":
        alignment_penalty += compute_genre_alignment_penalty(anchor_genres, doc_genres, query_focus)

    # --- Hard Tonal Filter ---
    tonal_penalty = 0.0
    explicit_genres = ["Comedy", "Horror", "Action", "Romance", "Science Fiction", "Documentary", "Animation", "Thriller", "Crime", "Mystery"]
    for eg in explicit_genres:
        is_romcom = any(x in q_low for x in ["romcom", "rom-com", "rom com"])
        is_mystery = any(x in q_low for x in ["mystery", "whodunnit", "whodunit", "detective"])

        # POSITIVE INTENT: user wants this genre
        if (re.search(rf"\b{eg.lower()}s?\b", q_low) or (eg == "Comedy" and ("comedies" in q_low or is_romcom)) or (eg == "Romance" and is_romcom) or (eg == "Mystery" and is_mystery)) and eg not in doc_genres:
            # Waiver: don't penalize if the intent was actually negated
            if not detect_negation(q_low, eg.lower()) and not (eg == "Comedy" and detect_negation(q_low, "comedy")):
                tonal_penalty += 0.85

        # NEGATIVE INTENT: user explicitly forbids a genre/style
        if detect_negation(q_low, eg.lower()) or (eg == "Comedy" and detect_negation(q_low, "comedy")):
            if eg in doc_genres:
                penalty += 0.50

    # --- Hard Rom-com Block ---
    is_romcom_negated = any(detect_negation(q_low, x) for x in ["romcom", "rom-com", "rom com"])
    if is_romcom_negated and ("Romance" in doc_genres and "Comedy" in doc_genres):
        penalty += 1.50 # Hard block for rom-coms

    # --- Vibe Rules + Multi-vibe bonus ---
    for vibe, rules in VIBE_RULES.items():
        matched_keywords = [w for w in rules["keywords"] if word_in_text(w, q_low)]
        if matched_keywords:
            all_negated = all(detect_negation(q_low, w) for w in matched_keywords)
            if not all_negated:
                # If rom-com is forbidden, don't let funny/emotional rules boost a rom-com
                if is_romcom_negated and ("Romance" in doc_genres and "Comedy" in doc_genres):
                    continue

                if any(g in doc_genres for g in rules["boost_genres"]):
                    vibe_boost += 0.15
                genre_penalty += compute_genre_penalty(doc_genres, rules["kill_genres"], rules["boost_genres"])

    # Multi-vibe intersection bonus — rewards matching all dimensions of a compound vibe query
    multi_vibe_bonus = compute_multi_vibe_bonus(q_low, doc_genres)
    vibe_boost += multi_vibe_bonus

    # --- Vibe Purge (anchor-aware, self-penalty safe) ---
    penalty += compute_vibe_purge_penalty(anchor_genres or [], doc_genres)

    # --- Quality signals ---
    vote_raw = float(doc.get("vote_average", 5.0))
    vote_count = float(doc.get("vote_count", 0))
    quality_score = vote_raw / 10.0

    p_raw = float(doc.get("popularity", 0))
    p_score = min(np.log1p(p_raw) / 8.0, 1.0)

    # --- Mood Guard ---
    cynical_kw = ["dystopian", "absurdist", "satire", "dark comedy", "surreal", "survival", "experimental"]
    if anchor_genres and ("Romance" in anchor_genres or "Musical" in anchor_genres):
        if any(k in doc.get("overview", "").lower() for k in cynical_kw):
            mood_penalty += 0.50

    # SADNESS GUARD: Pure comedies without Drama/Thriller get penalized for dark/existential queries
    sad_triggers = ["sad", "existential", "melancholy", "depressing", "tragic", "stare at the ceiling", "unravel", "fall apart"]
    if any(t in q_low for t in sad_triggers):
        if "Comedy" in doc_genres and "Drama" not in doc_genres and "Thriller" not in doc_genres:
            mood_penalty += 0.40

    # --- Masterpiece Bonus ---
    masterpiece_bonus = 0.05 if (vote_count > 5000 and vote_raw >= 7.8) else 0.0

    # --- Legacy Boost ---
    legacy_boost = np.log10(vote_count) * 0.01 if vote_count > 2000 else 0.0

    # --- Person Alignment ---
    if entities:
        similarity_triggers = ["like", "similar to", "reminds me of", "-esque", "vibe"]
        is_similarity_query = any(t in q_low for t in similarity_triggers)

        cast = [str(c).lower() for c in (doc.get("cast_names") or [])]
        director = str(doc.get("director", "")).lower()
        has_match = False
        for ent in entities:
            e_low = ent.lower()
            if e_low in director or any(e_low == c for c in cast):
                has_match = True
                break

        if not has_match and not is_similarity_query:
            person_penalty = 0.50

    # --- DNA Boost & Purity ---
    dna_boost = 0.0
    purity_penalty = 0.0
    if anchor_genres:
        anchor_set = set(anchor_genres)
        doc_set = set(doc_genres)
        overlap = len(doc_set.intersection(anchor_set))
        if overlap >= 3:
            dna_boost = 0.02
        elif overlap == 2:
            dna_boost = 0.01
        purity_penalty = len(doc_set - anchor_set) * 0.01

    # --- Keyword Overlap Bonus ---
    keyword_bonus = 0.0
    kw_overlap = 0
    if ref_doc and ref_doc.get("keywords") and doc.get("keywords"):
        anchor_kw = set(ref_doc["keywords"])
        doc_kw = set(doc.get("keywords", []))
        kw_overlap = len(anchor_kw.intersection(doc_kw))
        cap = 0.20 if mode == "SIBLING_DISCOVERY" else 0.10
        keyword_bonus = min(kw_overlap * 0.02, cap)

    if mode == "SIBLING_DISCOVERY" and kw_overlap == 0:
        purity_penalty += 0.15

    # --- Director Signal ---
    if mode == "SIBLING_DISCOVERY" and ref_doc and doc.get("director") and ref_doc.get("director"):
        if doc["director"] == ref_doc["director"]:
            dna_boost += 0.15

    # --- Final Score ---
    score = 0.0

    # Explicit Top Genre Boost
    # If the user has identified favorite genres, give those movies a tiny nudge
    if user_top_genres:
        for g_data in user_top_genres:
            if g_data["genre"] in doc_genres:
                # 0.02 to 0.05 boost depending on how much they like the genre
                vibe_boost += (g_data["affinity"] / 100.0) * 0.05

    if mode == "SIBLING_DISCOVERY":
        score = (
            (semantic * 2.0)
            + dna_boost
            + keyword_bonus
            + masterpiece_bonus
            - purity_penalty
            + (quality_score * 0.05)
            + vibe_boost
            - genre_penalty
            - penalty
            - mood_penalty
            - alignment_penalty
            - tonal_penalty
        )
    elif mode == "PERSON_STRICT":
        constraint_match = 0.0
        cast = doc.get("cast_names", [])
        director = str(doc.get("director", "")).lower()
        if isinstance(cast, str):
            cast = [c.lower().strip() for c in cast.split(",")]
        else:
            cast = [str(c).lower() for c in (cast or [])]
        for ent in entities:
            if ent.lower() in director or any(ent.lower() == c for c in cast):
                constraint_match = 1.0
                break
        score = (
            (semantic * 0.40)
            + (quality_score * 0.30)
            + (p_score * 0.10)
            + (constraint_match * 0.20)
            + legacy_boost
            - penalty
        )
    else:  # GLOBAL / PERSON_VIBE
        score = (
            (semantic * 0.90)
            + (quality_score * 0.07)
            + (p_score * 0.03)
            + legacy_boost
            + vibe_boost
            - genre_penalty
            - person_penalty
            - penalty
            - mood_penalty
        )

    # --- Reason ---
    matched_vibes = []
    for v_name, rules in VIBE_RULES.items():
        matched_keywords = [w for w in rules["keywords"] if word_in_text(w, q_low)]
        if not matched_keywords:
            continue
        if all(detect_negation(q_low, w) for w in matched_keywords):
            continue
        if any(g in doc_genres for g in rules["boost_genres"]):
            matched_vibes.append(v_name)

    if mode == "PERSON_STRICT":
        reason = f"Matches your interest in {', '.join(entities)}."
    elif mode == "SIBLING_DISCOVERY":
        if keyword_bonus >= 0.10:
            reason = "Strong thematic sibling — shares key story elements."
        elif dna_boost >= 0.02:
            reason = "Closely matches the genre DNA of your reference film."
        else:
            reason = "Thematic sibling that matches the style of the movie you referenced."
    elif matched_vibes:
        vibe_labels = {
            "emotional": "emotional",
            "scary": "scary",
            "funny": "funny",
            "action": "high-octane",
            "investigative": "investigative",
            "wholesome": "wholesome",
            "cozy": "cozy",
            "existential": "existential",
            "glamorous": "glamorous",
            "nostalgic": "nostalgic",
            "bittersweet_ending": "bittersweet",
            "unraveling": "emotionally intense",
            "sun_scorched": "sun-scorched and desolate",
            "gritty_urban": "urban and atmospheric",
            "paranoid": "paranoid",
            "dark_wit": "sharp and darkly funny",
            "analog_unhurried": "unhurried and analog",
        }
        labels = [vibe_labels.get(v, v) for v in matched_vibes]
        if len(labels) > 1:
            reason = f"Matches the {', '.join(labels[:-1])} and {labels[-1]} vibes of your search."
        else:
            reason = f"Captured the {labels[0]} essence of your request."
    elif semantic > 0.85:
        reason = "Direct semantic match to the core of your query."
    else:
        reason = "Matches the overall vibe of your search."

    breakdown = {
        "semantic": round(semantic, 4),
        "keyword_bonus": round(keyword_bonus, 4),
        "dna_boost": round(dna_boost, 4),
        "vibe_boost": round(vibe_boost, 4),
        "multi_vibe_bonus": round(multi_vibe_bonus, 4),
        "quality": round(quality_score, 4),
        "legacy": round(legacy_boost, 4),
        "total_penalty": round(penalty + genre_penalty + alignment_penalty + mood_penalty + purity_penalty + person_penalty, 4),
    }

    return score, reason, breakdown
