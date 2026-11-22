"""Offline curated course movie catalog and versioned retrieval executor."""
from __future__ import annotations

import re
from typing import Any

COURSE_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "id": "496243",
        "title": "Parasite",
        "release_year": 2019,
        "original_language": "ko",
        "genres": ["Drama", "Thriller", "Comedy"],
        "director": "Bong Joon Ho",
        "cast_names": ["Song Kang-ho", "Lee Sun-kyun", "Cho Yeo-jeong"],
        "keywords": ["class divide", "con artist", "dark comedy", "tension", "family", "social satire", "slow burn"],
        "overview": "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan.",
        "vote_average": 8.5,
        "poster_path": None,
    },
    {
        "id": "11423",
        "title": "Memories of Murder",
        "release_year": 2003,
        "original_language": "ko",
        "genres": ["Crime", "Drama", "Mystery", "Thriller"],
        "director": "Bong Joon Ho",
        "cast_names": ["Song Kang-ho", "Kim Sang-kyung"],
        "keywords": ["detective", "investigation", "rain", "procedural", "slow burn", "obsession", "noir", "no gore"],
        "overview": "In 1986 Gyunggi Province, two detectives struggle with an unprecedented string of unsolved rural murders using contrasting methods.",
        "vote_average": 8.1,
        "poster_path": None,
    },
    {
        "id": "655977",
        "title": "The Host",
        "release_year": 2006,
        "original_language": "ko",
        "genres": ["Horror", "Drama", "Science Fiction"],
        "director": "Bong Joon Ho",
        "cast_names": ["Song Kang-ho", "Byun Hee-bong", "Park Hae-il"],
        "keywords": ["monster", "river", "family", "satire", "creature"],
        "overview": "A creature emerges from Seoul's Han River and abducts a vendor's daughter, forcing his dysfunctional family to mount a rescue.",
        "vote_average": 7.0,
        "poster_path": None,
    },
    {
        "id": "670",
        "title": "Decision to Leave",
        "release_year": 2022,
        "original_language": "ko",
        "genres": ["Crime", "Drama", "Mystery", "Romance", "Thriller"],
        "director": "Park Chan-wook",
        "cast_names": ["Tang Wei", "Park Hae-il"],
        "keywords": ["detective", "obsession", "noir", "slow burn", "mist", "melancholy", "rain", "psychological"],
        "overview": "An insomniac detective investigating a man's fatal fall from a mountain peak develops a tangled fixation on the dead man's mysterious widow.",
        "vote_average": 7.6,
        "poster_path": None,
    },
    {
        "id": "153",
        "title": "Lost in Translation",
        "release_year": 2003,
        "original_language": "en",
        "genres": ["Drama", "Romance"],
        "director": "Sofia Coppola",
        "cast_names": ["Bill Murray", "Scarlett Johansson"],
        "keywords": ["tokyo", "neon", "loneliness", "insomnia", "late night", "bittersweet", "quiet", "hotel", "melancholy"],
        "overview": "A faded movie star and a neglected newlywed form an understated bond while adrift in sleepless neon-lit Tokyo.",
        "vote_average": 7.4,
        "poster_path": None,
    },
    {
        "id": "152601",
        "title": "Her",
        "release_year": 2013,
        "original_language": "en",
        "genres": ["Drama", "Romance", "Science Fiction"],
        "director": "Spike Jonze",
        "cast_names": ["Joaquin Phoenix", "Scarlett Johansson", "Amy Adams"],
        "keywords": ["loneliness", "ai", "intimacy", "bittersweet", "warm colors", "soft", "futuristic", "gentle"],
        "overview": "A lonely letter writer falls in love with an adaptive operating system while navigating heartbreak and modern isolation.",
        "vote_average": 7.9,
        "poster_path": None,
    },
    {
        "id": "64690",
        "title": "Drive",
        "release_year": 2011,
        "original_language": "en",
        "genres": ["Crime", "Drama", "Thriller"],
        "director": "Nicolas Winding Refn",
        "cast_names": ["Ryan Gosling", "Carey Mulligan", "Oscar Isaac"],
        "keywords": ["neon", "synthwave", "late night", "stunt driver", "noir", "los angeles", "heist"],
        "overview": "A quiet Hollywood stuntman who moonlights as a getaway driver is drawn into a dangerous underworld conflict to protect his neighbor.",
        "vote_average": 7.6,
        "poster_path": None,
    },
    {
        "id": "335984",
        "title": "Blade Runner 2049",
        "release_year": 2017,
        "original_language": "en",
        "genres": ["Science Fiction", "Drama", "Mystery"],
        "director": "Denis Villeneuve",
        "cast_names": ["Ryan Gosling", "Harrison Ford", "Ana de Armas"],
        "keywords": ["neon", "rain", "dystopian", "memory", "slow burn", "identity", "cyberpunk", "noir"],
        "overview": "A young replicant blade runner uncovers a long-buried secret that leads him on a solitary search for Rick Deckard.",
        "vote_average": 7.6,
        "poster_path": None,
    },
    {
        "id": "329865",
        "title": "Arrival",
        "release_year": 2016,
        "original_language": "en",
        "genres": ["Drama", "Science Fiction", "Mystery"],
        "director": "Denis Villeneuve",
        "cast_names": ["Amy Adams", "Jeremy Renner", "Forest Whitaker"],
        "keywords": ["linguistics", "time", "grief", "memory", "mind-bending", "contemplative", "space", "aliens"],
        "overview": "A linguist recruited by the military to decode alien visitors confronts a revelation that reshapes her sense of time and loss.",
        "vote_average": 7.6,
        "poster_path": None,
    },
    {
        "id": "146233",
        "title": "Prisoners",
        "release_year": 2013,
        "original_language": "en",
        "genres": ["Crime", "Drama", "Mystery", "Thriller"],
        "director": "Denis Villeneuve",
        "cast_names": ["Hugh Jackman", "Jake Gyllenhaal", "Paul Dano"],
        "keywords": ["rain", "investigation", "disappearance", "morality", "slow burn", "bleak", "autumn"],
        "overview": "When two young girls vanish in suburban Pennsylvania, a desperate father and a methodical detective collide in a tense investigation.",
        "vote_average": 8.1,
        "poster_path": None,
    },
    {
        "id": "27205",
        "title": "Inception",
        "release_year": 2010,
        "original_language": "en",
        "genres": ["Action", "Science Fiction", "Adventure", "Mystery"],
        "director": "Christopher Nolan",
        "cast_names": ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Tom Hardy"],
        "keywords": ["dreams", "heist", "mind-bending", "labyrinth", "time", "subconscious"],
        "overview": "A thief who infiltrates dreams is offered a chance at redemption if he can plant an idea inside a target's subconscious.",
        "vote_average": 8.4,
        "poster_path": None,
    },
    {
        "id": "157336",
        "title": "Interstellar",
        "release_year": 2014,
        "original_language": "en",
        "genres": ["Adventure", "Drama", "Science Fiction"],
        "director": "Christopher Nolan",
        "cast_names": ["Matthew McConaughey", "Anne Hathaway", "Jessica Chastain"],
        "keywords": ["space odyssey", "relativity", "time", "family", "cosmos", "emotional", "epic"],
        "overview": "Astronauts travel through a wormhole near Saturn searching for a habitable planet while confronting time dilation and devotion.",
        "vote_average": 8.4,
        "poster_path": None,
    },
    {
        "id": "546554",
        "title": "Knives Out",
        "release_year": 2019,
        "original_language": "en",
        "genres": ["Comedy", "Crime", "Mystery", "Thriller"],
        "director": "Rian Johnson",
        "cast_names": ["Daniel Craig", "Ana de Armas", "Chris Evans"],
        "keywords": ["whodunit", "twist", "autumn", "mansion", "detective", "inheritance", "ensemble"],
        "overview": "A sharp detective unravels eccentric family secrets after a renowned crime novelist dies mysteriously at his estate.",
        "vote_average": 7.8,
        "poster_path": None,
    },
    {
        "id": "84365",
        "title": "Whiplash",
        "release_year": 2014,
        "original_language": "en",
        "genres": ["Drama", "Music"],
        "director": "Damien Chazelle",
        "cast_names": ["Miles Teller", "J.K. Simmons"],
        "keywords": ["jazz", "obsession", "perfectionism", "drums", "mentor", "intense", "late night"],
        "overview": "A driven young jazz drummer endures psychological pressure from an uncompromising conservatory instructor.",
        "vote_average": 8.4,
        "poster_path": None,
    },
    {
        "id": "44214",
        "title": "Black Swan",
        "release_year": 2010,
        "original_language": "en",
        "genres": ["Drama", "Thriller"],
        "director": "Darren Aronofsky",
        "cast_names": ["Natalie Portman", "Mila Kunis", "Vincent Cassel"],
        "keywords": ["obsession", "perfectionism", "ballet", "psychological", "double", "pressure"],
        "overview": "A ballerina's sanity unravels under the relentless pursuit of artistic perfection for a demanding production.",
        "vote_average": 7.7,
        "poster_path": None,
    },
    {
        "id": "843",
        "title": "In the Mood for Love",
        "release_year": 2000,
        "original_language": "zh",
        "genres": ["Drama", "Romance"],
        "director": "Wong Kar-wai",
        "cast_names": ["Tony Leung Chiu-wai", "Maggie Cheung"],
        "keywords": ["rain", "neon", "yearning", "slow burn", "hong kong", "1960s", "melancholy", "restraint"],
        "overview": "Two neighbors in 1962 Hong Kong form an aching, unspoken bond after suspecting their spouses of infidelity.",
        "vote_average": 8.1,
        "poster_path": None,
    },
    {
        "id": "11104",
        "title": "Chungking Express",
        "release_year": 1994,
        "original_language": "zh",
        "genres": ["Drama", "Comedy", "Romance"],
        "director": "Wong Kar-wai",
        "cast_names": ["Takeshi Kaneshiro", "Faye Wong", "Tony Leung Chiu-wai"],
        "keywords": ["late night", "neon", "hong kong", "dreamy", "urban", "whimsical", "90s"],
        "overview": "Two lovelorn Hong Kong policemen cross paths with enigmatic women amid late-night snack bars and blurred city lights.",
        "vote_average": 8.0,
        "poster_path": None,
    },
    {
        "id": "76",
        "title": "Before Sunrise",
        "release_year": 1995,
        "original_language": "en",
        "genres": ["Drama", "Romance"],
        "director": "Richard Linklater",
        "cast_names": ["Ethan Hawke", "Julie Delpy"],
        "keywords": ["vienna", "conversation", "intimate", "bittersweet", "walking", "one night", "90s"],
        "overview": "Two young travelers meet on a train and spend one fleeting night walking and talking through Vienna before dawn.",
        "vote_average": 8.0,
        "poster_path": None,
    },
    {
        "id": "80",
        "title": "Before Sunset",
        "release_year": 2004,
        "original_language": "en",
        "genres": ["Drama", "Romance"],
        "director": "Richard Linklater",
        "cast_names": ["Ethan Hawke", "Julie Delpy"],
        "keywords": ["paris", "aging", "regret", "bittersweet", "conversation", "second chance"],
        "overview": "Nine years after a single night in Vienna, two former strangers reunite in Paris with only hours before a flight.",
        "vote_average": 7.8,
        "poster_path": None,
    },
    {
        "id": "458737",
        "title": "First Reformed",
        "release_year": 2017,
        "original_language": "en",
        "genres": ["Drama", "Thriller"],
        "director": "Paul Schrader",
        "cast_names": ["Ethan Hawke", "Amanda Seyfried"],
        "keywords": ["faith", "aging", "grief", "character study", "quiet", "moral crisis", "winter"],
        "overview": "A solitary pastor at a small upstate church spirals through a crisis of faith, grief, and conscience.",
        "vote_average": 7.1,
        "poster_path": None,
    },
    {
        "id": "666277",
        "title": "Past Lives",
        "release_year": 2023,
        "original_language": "en",
        "genres": ["Drama", "Romance"],
        "director": "Celine Song",
        "cast_names": ["Greta Lee", "Teo Yoo", "John Magaro"],
        "keywords": ["bittersweet", "fate", "childhood", "quiet", "cry", "yearning", "immigration"],
        "overview": "Childhood friends separated by emigration reunite decades later in New York, reflecting on destiny and choices made.",
        "vote_average": 7.8,
        "poster_path": None,
    },
    {
        "id": "38",
        "title": "Eternal Sunshine of the Spotless Mind",
        "release_year": 2004,
        "original_language": "en",
        "genres": ["Science Fiction", "Drama", "Romance"],
        "director": "Michel Gondry",
        "cast_names": ["Jim Carrey", "Kate Winslet", "Kirsten Dunst"],
        "keywords": ["memory", "mind-bending", "heartbreak", "surreal", "fever dream", "bittersweet"],
        "overview": "After a painful breakup, a couple undergoes a procedure to erase each other from memory only to rediscover why they connected.",
        "vote_average": 8.1,
        "poster_path": None,
    },
    {
        "id": "129",
        "title": "Spirited Away",
        "release_year": 2001,
        "original_language": "ja",
        "genres": ["Animation", "Family", "Fantasy"],
        "director": "Hayao Miyazaki",
        "cast_names": ["Rumi Hiiragi", "Miyu Irino"],
        "keywords": ["spirits", "coming-of-age", "soothe", "magical", "whimsical", "bathhouse"],
        "overview": "A ten-year-old girl enters a spirit realm ruled by gods and witches, working at a mystical bathhouse to free her parents.",
        "vote_average": 8.5,
        "poster_path": None,
    },
    {
        "id": "128",
        "title": "Princess Mononoke",
        "release_year": 1997,
        "original_language": "ja",
        "genres": ["Animation", "Adventure", "Fantasy"],
        "director": "Hayao Miyazaki",
        "cast_names": ["Yoji Matsuda", "Yuriko Ishida"],
        "keywords": ["forest", "nature", "mythology", "90s", "epic", "conflict"],
        "overview": "A cursed young warrior travels westward and gets caught in a struggle between forest gods and an industrial iron town.",
        "vote_average": 8.3,
        "poster_path": None,
    },
    {
        "id": "8392",
        "title": "My Neighbor Totoro",
        "release_year": 1988,
        "original_language": "ja",
        "genres": ["Animation", "Family", "Fantasy"],
        "director": "Hayao Miyazaki",
        "cast_names": ["Noriko Hidaka", "Chika Sakamoto"],
        "keywords": ["cozy", "autumn", "comfort", "gentle", "soothe", "80s", "countryside", "warm"],
        "overview": "Two sisters move to the Japanese countryside and discover playful forest spirits while awaiting their mother's recovery.",
        "vote_average": 8.1,
        "poster_path": None,
    },
    {
        "id": "346",
        "title": "Seven Samurai",
        "release_year": 1954,
        "original_language": "ja",
        "genres": ["Action", "Drama"],
        "director": "Akira Kurosawa",
        "cast_names": ["Toshiro Mifune", "Takashi Shimura"],
        "keywords": ["1950s", "classic", "black-and-white", "village", "samurai", "honor"],
        "overview": "Impoverished farmers hire seven masterless samurai to defend their mountain village from marauding bandits.",
        "vote_average": 8.5,
        "poster_path": None,
    },
    {
        "id": "548",
        "title": "Rashomon",
        "release_year": 1950,
        "original_language": "ja",
        "genres": ["Crime", "Drama", "Mystery"],
        "director": "Akira Kurosawa",
        "cast_names": ["Toshiro Mifune", "Machiko Kyo"],
        "keywords": ["1950s", "classic", "black-and-white", "truth", "perspective", "forest", "rain"],
        "overview": "Four witnesses offer conflicting accounts of a crime in a rain-swept forest, questioning the nature of truth.",
        "vote_average": 8.1,
        "poster_path": None,
    },
    {
        "id": "194",
        "title": "Amélie",
        "release_year": 2001,
        "original_language": "fr",
        "genres": ["Comedy", "Romance"],
        "director": "Jean-Pierre Jeunet",
        "cast_names": ["Audrey Tautou", "Mathieu Kassovitz"],
        "keywords": ["cozy", "comfort", "paris", "whimsical", "warm", "autumn", "gentle", "feel-good"],
        "overview": "An imaginative Parisian waitress quietly orchestrates small miracles in the lives of strangers around her neighborhood.",
        "vote_average": 7.9,
        "poster_path": None,
    },
    {
        "id": "531428",
        "title": "Portrait of a Lady on Fire",
        "release_year": 2019,
        "original_language": "fr",
        "genres": ["Drama", "Romance", "History"],
        "director": "Céline Sciamma",
        "cast_names": ["Noémie Merlant", "Adèle Haenel"],
        "keywords": ["painting", "yearning", "slow burn", "sea", "intimate", "cry", "memory"],
        "overview": "On an isolated Brittany island in 1770, a painter is commissioned to paint a reluctant bride's portrait in secret.",
        "vote_average": 8.1,
        "poster_path": None,
    },
    {
        "id": "582",
        "title": "The Lives of Others",
        "release_year": 2006,
        "original_language": "de",
        "genres": ["Drama", "Thriller"],
        "director": "Florian Henckel von Donnersmarck",
        "cast_names": ["Ulrich Mühe", "Martina Gedeck", "Sebastian Koch"],
        "keywords": ["surveillance", "east berlin", "slow burn", "conscience", "quiet", "drama"],
        "overview": "In 1984 East Berlin, a Stasi agent tasked with bugging a playwright's apartment becomes quietly transformed by their lives.",
        "vote_average": 8.0,
        "poster_path": None,
    },
    {
        "id": "11216",
        "title": "Cinema Paradiso",
        "release_year": 1988,
        "original_language": "it",
        "genres": ["Drama", "Romance"],
        "director": "Giuseppe Tornatore",
        "cast_names": ["Philippe Noiret", "Salvatore Cascio"],
        "keywords": ["nostalgia", "comfort", "cry", "80s", "memory", "cinema", "bittersweet"],
        "overview": "A filmmaker recalls his Sicilian childhood friendship with the village theater's wise projectionist.",
        "vote_average": 8.4,
        "poster_path": None,
    },
    {
        "id": "25376",
        "title": "The Secret in Their Eyes",
        "release_year": 2009,
        "original_language": "es",
        "genres": ["Crime", "Drama", "Mystery", "Romance", "Thriller"],
        "director": "Juan José Campanella",
        "cast_names": ["Ricardo Darín", "Soledad Villamil"],
        "keywords": ["cold case", "obsession", "memory", "no gore", "twist", "twist ending", "argentina"],
        "overview": "A retired legal counselor writes a novel revisiting an unsolved Buenos Aires murder case from twenty-five years earlier.",
        "vote_average": 8.0,
        "poster_path": None,
    },
    {
        "id": "20453",
        "title": "3 Idiots",
        "release_year": 2009,
        "original_language": "hi",
        "genres": ["Comedy", "Drama"],
        "director": "Rajkumar Hirani",
        "cast_names": ["Aamir Khan", "R. Madhavan", "Sharman Joshi"],
        "keywords": ["friendship", "feel-good", "comfort", "college", "heartwarming"],
        "overview": "Two friends set off across India searching for a long-lost college roommate who taught them to question convention.",
        "vote_average": 8.0,
        "poster_path": None,
    },
    {
        "id": "346648",
        "title": "Paddington 2",
        "release_year": 2017,
        "original_language": "en",
        "genres": ["Adventure", "Comedy", "Family"],
        "director": "Paul King",
        "cast_names": ["Ben Whishaw", "Hugh Grant", "Sally Hawkins"],
        "keywords": ["cozy", "autumn", "comfort", "warm", "wholesome", "feel-good", "kindness"],
        "overview": "A polite marmalade-loving bear in London tries to clear his name after being framed for stealing a pop-up book.",
        "vote_average": 7.5,
        "poster_path": None,
    },
    {
        "id": "391713",
        "title": "Lady Bird",
        "release_year": 2017,
        "original_language": "en",
        "genres": ["Comedy", "Drama"],
        "director": "Greta Gerwig",
        "cast_names": ["Saoirse Ronan", "Laurie Metcalf", "Timothée Chalamet"],
        "keywords": ["autumn", "coming-of-age", "comfort", "mother daughter", "sacramento", "bittersweet"],
        "overview": "A willful high school senior in Sacramento navigates her final year of friendships, first loves, and tension with her mother.",
        "vote_average": 7.3,
        "poster_path": None,
    },
    {
        "id": "694",
        "title": "The Shining",
        "release_year": 1980,
        "original_language": "en",
        "genres": ["Horror", "Thriller"],
        "director": "Stanley Kubrick",
        "cast_names": ["Jack Nicholson", "Shelley Duvall"],
        "keywords": ["isolation", "gore", "blood", "hotel", "madness", "winter", "80s"],
        "overview": "A family oversees an isolated mountain hotel for the winter where a sinister presence drives the father into madness.",
        "vote_average": 8.2,
        "poster_path": None,
    },
)


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9\-']+", text.lower()) if len(t) >= 2}


def _passes_filters(doc: dict[str, Any], filters: dict[str, Any]) -> bool:
    year = int(doc["release_year"])
    if filters["min_year"] is not None and year < int(filters["min_year"]):
        return False
    if filters["max_year"] is not None and year > int(filters["max_year"]):
        return False
    if filters["language"] and doc["original_language"] != filters["language"]:
        return False
    if filters["min_vote"] is not None and float(doc["vote_average"]) < float(filters["min_vote"]):
        return False

    doc_genres = {g.lower() for g in doc.get("genres", [])}
    if any(g.lower() in doc_genres for g in filters["excluded_genres"]):
        return False
    if filters["genres"] and not any(g.lower() in doc_genres for g in filters["genres"]):
        return False

    blob = " ".join([doc["title"], doc["overview"], " ".join(doc["keywords"])]).lower()
    for term in filters["excluded_terms"]:
        if term.lower() in blob and f"no {term.lower()}" not in blob:
            return False

    if filters["directors"]:
        d_low = doc["director"].lower()
        if not any(d.lower() in d_low or d_low in d.lower() for d in filters["directors"]):
            return False

    if filters["actors"]:
        cast_low = [c.lower() for c in doc["cast_names"]]
        if not any(any(a.lower() in c or c in a.lower() for c in cast_low) for a in filters["actors"]):
            return False
    return True


def _build_applied_filters(
    prediction: dict[str, Any] | None,
    *,
    min_year: int | None,
    max_year: int | None,
    language: str | None,
    min_vote: float | None,
    retrieval_mode: str,
) -> dict[str, Any]:
    ext = (prediction or {}).get("extraction", {}) if retrieval_mode == "auto" else {}
    eff_min_y = min_year if min_year is not None else ext.get("year_min")
    eff_max_y = max_year if max_year is not None else ext.get("year_max")
    eff_lang = language or ext.get("language")
    return {
        "genres": list(ext.get("genres", [])),
        "excluded_genres": list(ext.get("excluded_genres", [])),
        "excluded_terms": list(ext.get("excluded_terms", [])),
        "actors": list(ext.get("actors", [])),
        "directors": list(ext.get("directors", [])),
        "reference_titles": list(ext.get("reference_titles", [])),
        "min_year": eff_min_y,
        "max_year": eff_max_y,
        "language": eff_lang,
        "min_vote": min_vote,
    }


def search_course_catalog(
    query: str,
    prediction: dict[str, Any] | None,
    *,
    k: int = 10,
    min_year: int | None = None,
    max_year: int | None = None,
    language: str | None = None,
    min_vote: float | None = None,
    retrieval_mode: str = "auto",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retrieve and rank offline catalog films via versioned route execution or baseline modes."""
    applied = _build_applied_filters(
        prediction,
        min_year=min_year,
        max_year=max_year,
        language=language,
        min_vote=min_vote,
        retrieval_mode=retrieval_mode,
    )
    route = (prediction or {}).get("selected_route", "dense_semantic") if retrieval_mode == "auto" else retrieval_mode
    q_tokens = _tokenize(query)
    ref_titles = [t.lower() for t in applied["reference_titles"]]
    anchor_docs = [d for d in COURSE_CATALOG if d["title"].lower() in ref_titles]
    anchor_kws = {kw.lower() for d in anchor_docs for kw in d["keywords"]}
    anchor_genres = {g.lower() for d in anchor_docs for g in d["genres"]}

    candidates = [d for d in COURSE_CATALOG if _passes_filters(d, applied)]
    scored: list[tuple[float, dict[str, Any], str]] = []
    for doc in candidates:
        doc_tokens = _tokenize(f"{doc['title']} {doc['overview']} {' '.join(doc['keywords'])}")
        kw_tokens = {kw.lower() for kw in doc["keywords"]}
        overlap = len(q_tokens & doc_tokens) / max(len(q_tokens), 1)
        kw_hits = sum(1 for kw in kw_tokens if any(t in kw for t in q_tokens)) / max(len(kw_tokens), 1)
        quality = (float(doc["vote_average"]) - 6.5) / 3.0

        if retrieval_mode == "dense_only":
            score = 0.55 + 0.35 * overlap + 0.10 * quality
            reason = "Dense semantic baseline overlap."
        elif retrieval_mode == "static_hybrid":
            score = 0.50 + 0.25 * overlap + 0.20 * kw_hits + 0.15 * quality
            reason = "Static hybrid lexical, keyword, and quality blend."
        elif route == "similarity_anchor" and anchor_docs:
            if doc["title"].lower() in ref_titles:
                score, reason = 1.0, "Reference film anchor."
            else:
                kw_sim = len(anchor_kws & kw_tokens) / max(len(anchor_kws | kw_tokens), 1)
                g_sim = len(anchor_genres & {g.lower() for g in doc["genres"]}) / max(len(anchor_genres), 1)
                score = 0.58 + 0.25 * kw_sim + 0.12 * g_sim + 0.10 * quality
                reason = f"Thematic similarity to {', '.join(applied['reference_titles'])}."
        elif route == "structured_filter":
            score = 0.70 + 0.20 * max(overlap, kw_hits) + 0.15 * quality
            reason = "Matched structured constraints and query concepts."
        elif route == "hybrid_concept":
            score = 0.60 + 0.30 * kw_hits + 0.20 * overlap + 0.15 * quality
            reason = "Matched affective, sensory, or thematic concepts."
        else:
            score = 0.56 + 0.30 * max(overlap, kw_hits) + 0.14 * quality
            reason = "Semantic discovery match."

        scored.append((min(0.9999, round(float(score), 4)), doc, reason))

    scored.sort(key=lambda item: (item[0], item[1]["vote_average"]), reverse=True)
    results = [
        {
            "id": str(doc["id"]),
            "title": doc["title"],
            "year": int(doc["release_year"]),
            "score": score,
            "reason": reason,
            "mode": route.upper(),
            "genres": list(doc["genres"]),
            "vote": float(doc["vote_average"]),
            "poster_path": doc["poster_path"],
        }
        for score, doc, reason in scored[: max(1, int(k))]
    ]
    return results, applied
