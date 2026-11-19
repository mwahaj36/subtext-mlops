"""Query rules, exclusions, and display-score helpers."""
import math
import re

CORE_GENRES = {
    "action", "adventure", "animation", "comedy", "crime", "documentary",
    "drama", "family", "fantasy", "history", "horror", "music", "mystery",
    "romance", "science fiction", "thriller", "war", "western", "sci-fi",
    "superhero", "k-drama", "anime", "noir", "neo-noir"
}

VIBE_RULES = {
    "emotional": {
        "keywords": ["cry", "sad", "bawl", "tearjerker", "devastating", "heartbreaking", "emotional", "beautiful", "romance", "romantic", "love story", "romcom", "rom-com", "rom com", "feels", "tragic", "downward spiral"],
        "boost_genres": ["Drama", "Romance"],
        "kill_genres": ["Action", "Adventure", "Horror", "Comedy", "Crime", "Mystery", "Science Fiction", "Animation", "Family"]
    },
    "scary": {
        "keywords": ["terrifying", "scary", "horror", "spooky", "jump scare", "creepy", "nightmare"],
        "boost_genres": ["Horror", "Thriller"],
        "kill_genres": ["Romance", "Comedy", "Family", "Animation"]
    },
    "funny": {
        "keywords": ["hilarious", "laugh", "funny", "comedy", "comedies", "lmao", "fun", "humor", "feel better", "cheer me up", "romcom", "rom-com", "rom com", "lol", "slaps", "unhinged", "dark humor", "chaotic"],
        "boost_genres": ["Comedy"],
        "kill_genres": ["Horror", "War", "Documentary", "Mystery", "Thriller"]
    },
    "action": {
        "keywords": ["action", "explosions", "badass", "fight", "cool", "thrilling", "intense", "adrenaline", "hype"],
        "boost_genres": ["Action", "Adventure", "Thriller"],
        "kill_genres": ["Romance", "Documentary", "Family", "Animation"]
    },
    "investigative": {
        "keywords": ["whodunnit", "whodunit", "detective", "mystery", "solve", "clues", "investigation", "crime", "murder mystery"],
        "boost_genres": ["Mystery", "Crime"],
        "kill_genres": ["Romance", "Musical", "Fantasy", "Animation", "Family"]
    },
    "wholesome": {
        "keywords": ["wholesome", "warm", "heartwarming", "feel-good", "sweet", "comfort", "genuine", "pure", "pure-hearted", "believe in people", "green flag"],
        "boost_genres": ["Family", "Comedy", "Romance"],
        "kill_genres": ["Horror", "War", "Crime", "Thriller"]
    },
    "cozy": {
        "keywords": ["cozy", "sunday afternoon", "comforting", "peaceful", "gentle", "soft", "vibes", "chill", "relaxing", "lowkey", "comfort movie"],
        "boost_genres": ["Animation", "Family", "Comedy", "Romance"],
        "kill_genres": ["Action", "Horror", "Thriller", "War", "Science Fiction"]
    },
    "existential": {
        "keywords": ["existential", "meaning of life", "stare at the ceiling", "philosophical", "deep", "thought-provoking", "crisis", "melancholy", "melancholic", "rent free"],
        "boost_genres": ["Drama", "Science Fiction"],
        "kill_genres": ["Action", "Comedy", "Family", "Animation"]
    },
    "glamorous": {
        "keywords": ["glamorous", "glamour", "main character", "fashion", "fame", "rich", "extravagant", "chic", "aesthetic", "glitzy", "slaps", "era", "self-destructive", "downward spiral", "messy"],
        "boost_genres": ["Drama", "Music", "Comedy"],
        "kill_genres": ["Action", "Horror", "War", "Animation", "Family"]
    },
    "nostalgic": {
        "keywords": ["nostalgic", "nostalgia", "throwback", "memory", "growing up", "retro", "vintage", "childhood", "bittersweet", "core"],
        "boost_genres": ["History", "Family", "Drama"],
        "kill_genres": []
    },
    # bittersweet endings / complicated emotions
    "bittersweet_ending": {
        "keywords": ["ended something", "quiet relief", "relief mixed", "ambivalent", "complicated feelings", "not sure how to feel", "mixed feelings", "letting go", "outgrown", "moving on"],
        "boost_genres": ["Drama", "Romance"],
        "kill_genres": ["Horror", "Action", "Comedy", "Family", "Animation", "Science Fiction"]
    },
    # watching someone fall apart
    "unraveling": {
        "keywords": ["unravel", "fall apart", "spiral", "deteriorate", "losing it", "breaking down", "can't look away", "self-destruct", "coming undone"],
        "boost_genres": ["Drama", "Thriller"],
        "kill_genres": ["Family", "Animation", "Comedy", "Romance"]
    },
    "sun_scorched": {
        "keywords": ["dusty", "sun-bleached", "middle of nowhere", "middle-of-nowhere", "heat you can", "scorching", "desert", "barren", "arid", "sun baked", "sun-baked"],
        "boost_genres": ["Western", "Drama", "Thriller", "Crime"],
        "kill_genres": ["Family", "Animation", "Comedy", "Romance", "Science Fiction", "Horror"]
    },
    # sensory — rainy neon city noir
    "gritty_urban": {
        "keywords": ["rainy city", "neon reflections", "wet pavement", "neon lights", "night city", "urban alienation", "nobody's where", "urban drift", "city at night"],
        "boost_genres": ["Drama", "Crime", "Thriller"],
        "kill_genres": ["Family", "Animation", "Romance", "Comedy", "Western"]
    },
    # paranoid / everyone is lying
    "paranoid": {
        "keywords": ["paranoid", "everyone might be lying", "conspiracy", "can't trust anyone", "surveillance", "deceived", "manipulation", "gaslit"],
        "boost_genres": ["Thriller", "Crime", "Mystery", "Drama"],
        "kill_genres": ["Family", "Animation", "Comedy", "Romance"]
    },
    # mean wit / dark satire
    "dark_wit": {
        "keywords": ["funny but mean", "wit that makes you wince", "mean funny", "dark satire", "sharp wit", "caustic", "biting comedy", "wince", "sardonic"],
        "boost_genres": ["Comedy", "Drama", "Thriller"],
        "kill_genres": ["Family", "Animation", "Horror", "War"]
    },
    # analog / unhurried / no irony
    "analog_unhurried": {
        "keywords": ["unhurried", "analog", "no irony", "different time", "made in a different", "slow cinema", "quiet film", "deliberate pace", "restrained"],
        "boost_genres": ["Drama"],
        "kill_genres": ["Action", "Science Fiction", "Horror", "Animation", "Comedy"]
    },
    "superhero": {
        "keywords": ["superhero", "super hero", "super-hero", "marvel", "dc", "superpowers", "super-powers", "masked vigilante", "vigilante", "comic book", "savior", "multiverse", "avengers", "justice league"],
        "boost_genres": ["Action", "Adventure", "Science Fiction", "Fantasy"],
        "kill_genres": ["Documentary", "History", "War"]
    },
}

# =============================================================================
# SENSORY QUERY EXPANSION
# For queries that are primarily aesthetic/sensory, we expand the text before
# embedding so the vector captures cinematic texture rather than literal words.
# "dusty sun-bleached" by itself doesn't sit near film embeddings — adding
# film-vocabulary context anchors it in the right part of the space.
# =============================================================================

SENSORY_EXPANSIONS = {
    "dusty":            "neo-western slow deliberate arid landscape minimalist isolation",
    "sun-bleached":     "western arthouse heat haze desolate wide open slow burn",
    "middle of nowhere":"western road movie isolation quiet desolate slow",
    "neon":             "urban noir nighttime alienation rain cinematic atmospheric",
    "wet pavement":     "noir atmospheric urban loneliness cinematic drifting",
    "rainy city":       "noir atmospheric urban nighttime drifting alienation",
    "rainy":            "atmospheric melancholic city night slow deliberate",
    "analog":           "unhurried classical restrained period texture no irony sincere",
    "unhurried":        "slow cinema deliberate arthouse quiet restrained",
    "no irony":         "sincere earnest classical unironic human",
    "unravel":          "psychological deterioration obsession disintegration intense",
    "fall apart":       "psychological deterioration slow disintegration character study",
    "paranoid":         "conspiracy surveillance deception institutional mistrust",
    "glamorous":        "stylish extravagant decadent fame ambition self-destruction",
    "self-destructive": "excess ambition unraveling tragic glamour downfall",
    "superhero":        "heroic journey mythic powers high stakes spectacle vigilante blockbuster justice",
}

def word_in_text(word: str, text: str) -> bool:
    """
    Checks if a word or phrase is present in the text with proper word boundaries.
    Handles leading/trailing non-alphanumeric characters (like '-esque' or '-baked').
    """
    if not word:
        return False
    start_boundary = r"\b" if word[0].isalnum() else ""
    end_boundary = r"\b" if word[-1].isalnum() else ""
    pattern = start_boundary + re.escape(word) + end_boundary
    return bool(re.search(pattern, text))


def expand_query_for_embedding(query: str) -> str:
    """
    Appends film-vocabulary context for sensory/aesthetic queries so that
    the embedding lands closer to relevant films rather than literal word matches.
    """
    q_low = query.lower()
    expansions = []
    for trigger, expansion in SENSORY_EXPANSIONS.items():
        if word_in_text(trigger, q_low):
            expansions.append(expansion)
    if expansions:
        # Deduplicate expansion words
        seen = set()
        deduped = []
        for chunk in expansions:
            for word in chunk.split():
                if word not in seen:
                    seen.add(word)
                    deduped.append(word)
        return query + " " + " ".join(deduped)
    return query


# =============================================================================
# ILLNESS HARD FILTER
# =============================================================================

ILLNESS_KEYWORDS = {
    "cancer": [
        "cancer", "tumor", "tumour", "chemotherapy", "chemo",
        "oncology", "leukemia", "leukaemia", "lymphoma",
        "carcinoma", "malignant", "terminal diagnosis"
    ],
    "alzheimer": ["alzheimer", "dementia", "memory loss", "memory disease"],
    "aids": ["aids", "hiv", "hiv-positive", "immune deficiency"],
    "blind": ["blind", "blindness", "sight loss", "visually impaired", "losing sight"],
    "deaf": ["deaf", "deafness", "hearing loss", "hearing impaired"],
    "paralysis": ["paralysis", "paralyzed", "paralyze", "paralyzing", "paralysed", "paralyse", "paralysing", "quadriplegic", "paraplegic", "wheelchair"],
    "diabetes": ["diabetes", "diabetic", "insulin"],
    "mental illness": ["mental illness", "schizophrenia", "bipolar", "psychosis", "psychiatric"],
}


def detect_negation(query: str, keyword: str) -> bool:
    q = query.lower()
    k = keyword.lower()
    negators = ["no ", "not ", "without ", "none ", "never ", "stop ", "zero "]
    for n in negators:
        # Use s? to catch plurals like 'rom-coms' or 'thrillers'
        if re.search(rf"\b{n}{k}s?\b", q):
            return True
    return False


def detect_illness_intent(query: str):
    q = query.lower()
    for illness, terms in ILLNESS_KEYWORDS.items():
        if any(word_in_text(t, q) for t in terms):
            return illness
    return None


def illness_matches_doc(illness_key: str, doc: dict) -> bool:
    terms = ILLNESS_KEYWORDS.get(illness_key, [])
    overview = doc.get("overview", "").lower()
    keywords = " ".join(doc.get("keywords", [])).lower()
    return any(t in (overview + " " + keywords) for t in terms)


# =============================================================================
# CONTENT EXCLUSION  ("sci-fi without aliens", "a heist movie with no guns")
#
# detect_negation() above only recognises the fixed genre vocabulary, so a
# negated *subject* used to be missed entirely — and worse than missed: the
# word still went into the embedding, so "sci-fi without aliens" pulled the
# query vector straight into alien films and returned Alien, Aliens, Alien3,
# District 9, Arrival and Contact. Here we pull the negated subject out of the
# query before embedding, and drop films that carry it as evidence.
# =============================================================================

_NEG_PATTERNS = [
    r"\bwithout\s+(?:any\s+|a\s+|an\s+|the\s+)?([a-z][a-z\-\s]{2,28})",
    r"\bwith\s+no\s+([a-z][a-z\-\s]{2,28})",
    r"\bno\s+([a-z][a-z\-\s]{2,28})",
    r"\b(?:but\s+)?not\s+(?:a\s+|an\s+|the\s+)?([a-z][a-z\-\s]{2,28})",
    r"\bexcept\s+(?:for\s+)?([a-z][a-z\-\s]{2,28})",
    r"\bminus\s+(?:the\s+)?([a-z][a-z\-\s]{2,28})",
]

# The captured span runs to the end of the string, so cut it at the first word
# that starts a new thought.
_NEG_STOP = {"and", "or", "but", "with", "that", "which", "just", "please",
             "movies", "movie", "films", "film", "shows", "like", "about"}

# Bare negations that aren't excluding a subject at all.
_NEG_SKIP = {"one", "idea", "matter", "longer", "sure", "way", "doubt",
             "spoilers", "subtitles", "sense", "good", "bad", "really"}


def parse_exclusions(query: str):
    """-> (content_terms, genre_names, text_with_negated_spans_removed)"""
    q_low = query.lower()
    terms, genres, spans = [], [], []

    for pat in _NEG_PATTERNS:
        for m in re.finditer(pat, q_low):
            phrase = m.group(1).strip()
            words = []
            for w in phrase.split():
                w = w.strip(",.;:!?'\"")
                if not w or w in _NEG_STOP:
                    break
                words.append(w)
                if len(words) == 3:
                    break
            if not words:
                continue
            term = " ".join(words)
            if term in _NEG_SKIP or len(term) < 3:
                continue
            # Genres keep flowing through the existing penalty path, and also
            # get excluded outright — "not horror" should mean not horror.
            canonical = {"sci-fi": "Science Fiction"}.get(
                term, term.title() if term in CORE_GENRES else None)
            if canonical:
                if canonical not in genres:
                    genres.append(canonical)
            elif term not in terms:
                terms.append(term)
            spans.append((m.start(), m.start(1) + len(term)))

    # Strip the negated spans so the embedding never sees the excluded subject.
    cleaned = query
    for start, end in sorted(spans, reverse=True):
        cleaned = cleaned[:start] + cleaned[end:]
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")

    return terms, genres, (cleaned or query)


def doc_mentions_term(term: str, doc: dict) -> bool:
    """Does this film actually carry the excluded subject?"""
    haystack = (
        (doc.get("title") or "") + " " +
        (doc.get("overview") or "") + " " +
        " ".join(str(k) for k in (doc.get("keywords") or []))
    ).lower()
    singular = term[:-1] if term.endswith("s") and len(term) > 4 else term
    return word_in_text(singular, haystack) or word_in_text(singular + "s", haystack)


# =============================================================================
# CINEMATIC TEXTURE TAGS
# =============================================================================

TEXTURE_KEYWORD_MAP = {
    "slow burn": "slow-burn restrained atmospheric deliberate",
    "atmospheric": "atmospheric immersive mood-driven",
    "surrealism": "surreal dreamlike nonlinear experimental",
    "nonlinear timeline": "nonlinear fragmented time structure",
    "unreliable narrator": "unreliable narrator deceptive perspective",
    "psychological thriller": "psychological manipulation power games tension",
    "mind game": "mind games psychological tension deception",
    "obsession": "obsessive compulsive fixation intensity",
    "dark comedy": "darkly comic ironic subversive",
    "bittersweet": "bittersweet melancholic hopeful ambivalent",
    "melancholy": "melancholic longing emotional weight",
    "wholesome": "heartwarming wholesome genuine warm",
    "cozy": "cozy comforting peaceful low-key",
    "nostalgic": "nostalgic sentimental bittersweet childhood",
    "glamorous": "glamorous stylish chic extravagant",
    "existential": "existential philosophical deep reflective",
    "female protagonist": "female-led woman-centred perspective",
    "lgbtq": "queer identity sexuality desire",
    "forbidden love": "forbidden desire repressed longing taboo",
    "class differences": "class tension social hierarchy power",
    "based on novel": "literary adapted source material",
    "twist ending": "twist revelation recontextualisation surprise",
    "period piece": "period historical costume detailed world",
    "foreign language": "foreign language subtitled international arthouse",
    # Additional texture tags
    "paranoia": "paranoid institutional distrust surveillance conspiracy",
    "ambivalent": "ambivalent complicated mixed emotion unresolved",
    "psychological deterioration": "unraveling disintegration obsession breakdown",
    "isolation": "isolation confined space solitude sparse",
    "sun": "arid heat desolate sun-scorched landscape",
    "rain": "rain atmospheric wet moody urban",
}


def build_texture_enrichment(keywords: list) -> str:
    enrichments = []
    kw_lower = [k.lower() for k in (keywords or [])]
    for trigger, descriptor in TEXTURE_KEYWORD_MAP.items():
        if any(trigger in k for k in kw_lower):
            enrichments.append(descriptor)
    return " ".join(enrichments)


# =============================================================================
# UTILS
# =============================================================================

def normalize_score(raw_score: float, mode: str) -> float:
    """
    Sigmoid normalization — preserves ranking order and avoids ceiling collapse.
    Replaces the old min(display_score, 1.0) hard cap which made genuinely
    different scores (0.95 vs 0.72) both appear as 1.0.
    Centered at 0.5, steepness 5 — gives ~0.95 at the top, ~0.50 at midpoint.
    """
    if mode == "SIBLING_DISCOVERY":
        raw_score = raw_score / 2.5
    else:
        raw_score = raw_score / 1.05
    return round(1 / (1 + math.exp(-5 * (raw_score - 0.5))), 4)


def smart_title(t):
    lowers = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 'from', 'by', 'in', 'of'}
    words = t.split()
    return " ".join([w.title() if i == 0 or w.lower() not in lowers else w.lower() for i, w in enumerate(words)])


def get_base_title(title: str):
    base = title.split(":")[0]
    base = re.sub(r'\s+(Part|Chapter|Vol|Volume)\s+[IVX0-9]+.*', '', base, flags=re.IGNORECASE)
    base = re.sub(r'\s+[2-9]$|\s+10$', '', base)
    return base.strip()
