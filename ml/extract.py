"""Deterministic rule and dictionary baseline for bounded constraint extraction."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from backend.search_rules import (
    _NEG_PATTERNS,
    _NEG_SKIP,
    _NEG_STOP,
    word_in_text,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

_DIRECTOR_PATTERNS = (
    r"\bdirected\s+by\s+([A-ZÀ-Ÿ][A-Za-zÀ-ÿ'’\-\.]+(?:\s+[A-ZÀ-Ÿ][A-Za-zÀ-ÿ'’\-\.]+){1,3})",
    r"\bdirector\s+([A-ZÀ-Ÿ][A-Za-zÀ-ÿ'’\-\.]+(?:\s+[A-ZÀ-Ÿ][A-Za-zÀ-ÿ'’\-\.]+){1,3})",
    r"\b(?:from|by)\s+director\s+([A-ZÀ-Ÿ][A-Za-zÀ-ÿ'’\-\.]+(?:\s+[A-ZÀ-Ÿ][A-Za-zÀ-ÿ'’\-\.]+){1,3})",
)
_ACTOR_PATTERNS = (
    r"\b(?:starring|with|featuring|lead\s+by|led\s+by)\s+([A-ZÀ-Ÿ][A-Za-zÀ-ÿ'’\-\.]+(?:\s+[A-ZÀ-Ÿ][A-Za-zÀ-ÿ'’\-\.]+){1,3})",
)
_SIMILARITY_CUES = (
    r"\b(?:similar\s+to|reminiscent\s+of|vibes\s+(?:of|like)|mood\s+of|feels\s+like|like)\s+(?:the\s+movie\s+|the\s+film\s+)?"
)
_SIMILARITY_PATTERN = (
    _SIMILARITY_CUES
    + r"([A-Z0-9][A-Za-z0-9:'’\-\s]{1,50}?)(?=\s*(?:,|but\b|without\b|with\b|no\b|and\b|or\b|from\b|in\b|between\b|directed\b|starring\b|featuring\b|released\b|that\b|which\b|$))"
)
_DECADE_MAP = {
    "1920s": (1920, 1929), "20s": (1920, 1929),
    "1930s": (1930, 1939), "30s": (1930, 1939),
    "1940s": (1940, 1949), "40s": (1940, 1949),
    "1950s": (1950, 1959), "50s": (1950, 1959),
    "1960s": (1960, 1969), "60s": (1960, 1969),
    "1970s": (1970, 1979), "70s": (1970, 1979),
    "1980s": (1980, 1989), "80s": (1980, 1989),
    "1990s": (1990, 1999), "90s": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2029),
}


def load_lexicons(
    people_path: Path | None = None,
    galaxy_path: Path | None = None,
) -> tuple[set[str], dict[str, str]]:
    """Load lowercase people set and lowercase->canonical title map if files exist."""
    p_path = people_path or (REPO_ROOT / "backend" / "people_index.json")
    g_path = galaxy_path or (REPO_ROOT / "backend" / "galaxy_points.json")
    people: set[str] = set()
    titles: dict[str, str] = {}
    if p_path.exists():
        with p_path.open("r", encoding="utf-8") as f:
            people = {str(x).strip().lower() for x in json.load(f) if str(x).strip()}
    if g_path.exists():
        with g_path.open("r", encoding="utf-8") as f:
            for item in json.load(f):
                t = str(item.get("t", "")).strip()
                if t and len(t) >= 3:
                    titles.setdefault(t.lower(), t)
    return people, titles


def _normalize_genre(term: str, aliases: dict[str, str]) -> str | None:
    key = term.strip().lower()
    if key in aliases:
        return aliases[key]
    if key.endswith("s") and key[:-1] in aliases:
        return aliases[key[:-1]]
    return None


def _extract_year_bounds(text: str, min_y: int, max_y: int) -> tuple[int | None, int | None]:
    q = text.lower()
    m = re.search(r"\bbetween\s+(19\d{2}|20\d{2})\s+and\s+(19\d{2}|20\d{2})\b", q)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        if min_y <= y1 <= y2 <= max_y:
            return y1, y2
    m = re.search(r"\b(19\d{2}|20\d{2})\s*[-–]\s*(19\d{2}|20\d{2})\b", q)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        if min_y <= y1 <= y2 <= max_y:
            return y1, y2
    m = re.search(
        r"\b(?:after|since)\s+(19\d{2}|20\d{2})\b|\bfrom\s+(19\d{2}|20\d{2})\s+(?:onwards?|or\s+later|to\s+(?:now|present))\b",
        q,
    )
    if m:
        y = int(m.group(1) or m.group(2))
        if min_y <= y <= max_y:
            return y, None
    m = re.search(r"\b(?:before|prior\s+to|pre-)\s*(19\d{2}|20\d{2})\b", q)
    if m:
        y = int(m.group(1))
        if min_y <= y <= max_y:
            return None, y
    for token, bounds in _DECADE_MAP.items():
        if re.search(rf"\b{re.escape(token)}\b", q):
            return bounds
    m = re.search(r"\b(?:from|in|released\s+in)\s+(19\d{2}|20\d{2})\b", q)
    if m:
        y = int(m.group(1))
        if min_y <= y <= max_y:
            return y, y
    return None, None


def _parse_negations(text: str, genre_aliases: dict[str, str]) -> tuple[list[str], list[str], str]:
    q_low = text.lower()
    terms: list[str] = []
    genres: list[str] = []
    spans: list[tuple[int, int]] = []
    for pat in _NEG_PATTERNS:
        for m in re.finditer(pat, q_low):
            phrase = m.group(1).strip()
            words: list[str] = []
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
            if term in _NEG_SKIP or len(term) < 3 or any(term.startswith(sk + " ") for sk in _NEG_SKIP):
                continue
            canonical = _normalize_genre(term, genre_aliases)
            if canonical:
                if canonical not in genres:
                    genres.append(canonical)
            elif term not in terms:
                terms.append(term)
            spans.append((m.start(), m.start(1) + len(term)))

    cleaned = text
    for start, end in sorted(spans, reverse=True):
        cleaned = cleaned[:start] + " " + cleaned[end:]
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return sorted(genres), sorted(terms), cleaned


def extract_constraints(
    text: str,
    taxonomy: dict[str, Any],
    people_set: set[str] | None = None,
    title_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Extract bounded entity, attribute, reference-title, and exclusion fields."""
    genre_aliases: dict[str, str] = taxonomy.get("genre_aliases", {})
    lang_map: dict[str, list[str]] = taxonomy.get("canonical_languages", {})
    y_bounds = taxonomy.get("year_bounds", {"min": 1888, "max": 2030})

    excluded_genres, excluded_terms, positive_text = _parse_negations(text, genre_aliases)
    pos_low = positive_text.lower()

    genres: list[str] = []
    for alias in sorted(genre_aliases, key=len, reverse=True):
        if word_in_text(alias, pos_low):
            canon = genre_aliases[alias]
            if canon not in genres and canon not in excluded_genres:
                genres.append(canon)
    genres.sort()

    year_min, year_max = _extract_year_bounds(positive_text, y_bounds["min"], y_bounds["max"])

    language: str | None = None
    for code, cues in lang_map.items():
        if any(word_in_text(cue, pos_low) for cue in cues):
            language = code
            break

    directors: list[str] = []
    for pat in _DIRECTOR_PATTERNS:
        for m in re.finditer(pat, positive_text):
            name = m.group(1).strip(" ,.;:!?")
            if people_set is None or name.lower() in people_set:
                if name not in directors:
                    directors.append(name)

    actors: list[str] = []
    for pat in _ACTOR_PATTERNS:
        for m in re.finditer(pat, positive_text):
            name = m.group(1).strip(" ,.;:!?")
            if people_set is None or name.lower() in people_set:
                if name not in actors and name not in directors:
                    actors.append(name)

    if people_set and not actors and not directors:
        tokens = re.findall(r"[A-Za-zÀ-ÿ'’\-]+", positive_text)
        for n in (3, 2):
            for i in range(len(tokens) - n + 1):
                cand = " ".join(tokens[i : i + n])
                if cand.lower() in people_set and cand[0].isupper():
                    if cand not in actors:
                        actors.append(cand)

    ref_titles: list[str] = []
    if title_map is not None:
        for m in re.finditer(_SIMILARITY_CUES, positive_text):
            tail = positive_text[m.end() : m.end() + 90]
            words = re.findall(r"[A-Za-z0-9:'’\-]+", tail)
            for n in range(min(len(words), 8), 0, -1):
                cand = " ".join(words[:n]).strip(" ,.;:!?\"'")
                matched = title_map.get(cand.lower())
                if matched and matched not in ref_titles:
                    ref_titles.append(matched)
                    break
    else:
        for m in re.finditer(_SIMILARITY_PATTERN, positive_text):
            cand = m.group(1).strip(" ,.;:!?\"'")
            if cand and cand not in ref_titles:
                ref_titles.append(cand)

    return {
        "actors": sorted(actors),
        "directors": sorted(directors),
        "genres": genres,
        "year_min": year_min,
        "year_max": year_max,
        "language": language,
        "excluded_genres": excluded_genres,
        "excluded_terms": excluded_terms,
        "reference_titles": sorted(ref_titles),
    }


def _to_item_set(extraction: dict[str, Any]) -> set[tuple[str, str]]:
    items: set[tuple[str, str]] = set()
    for field in ("actors", "directors", "genres", "excluded_genres", "excluded_terms", "reference_titles"):
        for val in extraction.get(field) or []:
            items.add((field, str(val).strip().lower()))
    if extraction.get("year_min") is not None:
        items.add(("year_min", str(extraction["year_min"])))
    if extraction.get("year_max") is not None:
        items.add(("year_max", str(extraction["year_max"])))
    if extraction.get("language") is not None:
        items.add(("language", str(extraction["language"]).strip().lower()))
    return items


def evaluate_extractions(records: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute micro P/R/F1 and per-field exact-match metrics over bounded extractions."""
    tp = fp = fn = 0
    field_counts: dict[str, list[int]] = {}
    for rec, pred in zip(records, predictions, strict=True):
        gold_items = _to_item_set(rec["extraction"])
        pred_items = _to_item_set(pred)
        tp += len(gold_items & pred_items)
        fp += len(pred_items - gold_items)
        fn += len(gold_items - pred_items)
        for field, val in gold_items | pred_items:
            counts = field_counts.setdefault(field, [0, 0, 0])
            in_g = (field, val) in gold_items
            in_p = (field, val) in pred_items
            if in_g and in_p:
                counts[0] += 1
            elif in_p:
                counts[1] += 1
            else:
                counts[2] += 1

    def _prf(c_tp: int, c_fp: int, c_fn: int) -> dict[str, float]:
        p = c_tp / (c_tp + c_fp) if (c_tp + c_fp) else 1.0
        r = c_tp / (c_tp + c_fn) if (c_tp + c_fn) else 1.0
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "support": c_tp + c_fn}

    return {
        "overall_micro": _prf(tp, fp, fn),
        "per_field": {field: _prf(*counts) for field, counts in sorted(field_counts.items())},
        "total_records": len(records),
    }
