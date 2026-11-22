"""Build Deliverable 2 reviewed drift batch and unreviewed pending queue sample."""
from __future__ import annotations

import json
from typing import Any

from ml.validate import (
    REPO_ROOT,
    _shingles,
    jaccard_similarity,
    load_jsonl,
    load_yaml,
    normalize_text,
    validate_records,
)

def _empty_extraction() -> dict[str, Any]:
    return {
        "actors": [],
        "directors": [],
        "genres": [],
        "year_min": None,
        "year_max": None,
        "language": None,
        "excluded_genres": [],
        "excluded_terms": [],
        "reference_titles": [],
    }


# 30 drifted queries across 15 paraphrase groups (short slang + long multi-clause shifts).
_REVIEWED_SPECS: list[tuple[str, str, str, list[str], dict[str, Any], list[tuple[str, str]]]] = [
    (
        "d2_q0001", "d2_grp_001",
        "no jumpscares pls just creepy Korean mystery with Song Kang-ho from the 2000s like Memories of Murder",
        ["negation_content", "attribute_language_country", "person_actor", "attribute_era", "similarity_vibe"],
        {"actors": ["Song Kang-ho"], "language": "ko", "year_min": 2000, "year_max": 2009, "excluded_terms": ["jumpscares"], "reference_titles": ["Memories of Murder"]},
        [("excluded_term", "jumpscares"), ("language_country", "Korean"), ("actor", "Song Kang-ho"), ("era", "2000s"), ("reference_title", "Memories of Murder")],
    ),
    (
        "d2_q0002", "d2_grp_001",
        "skip cheap scares and give me moody 2000s South Korean detective cinema starring Song Kang-ho reminiscent of Memories of Murder",
        ["negation_content", "attribute_language_country", "person_actor", "attribute_era", "similarity_vibe"],
        {"actors": ["Song Kang-ho"], "language": "ko", "year_min": 2000, "year_max": 2009, "excluded_terms": ["cheap scares"], "reference_titles": ["Memories of Murder"]},
        [("excluded_term", "cheap scares"), ("language_country", "South Korean"), ("actor", "Song Kang-ho"), ("era", "2000s"), ("reference_title", "Memories of Murder")],
    ),
    (
        "d2_q0003", "d2_grp_002",
        "rainy neon Blade Runner cyberpunk vibes but zero horror and zero slashers for late night chilling",
        ["sensory_visual", "similarity_vibe", "negation_genre", "negation_content", "affective_tone"],
        {"excluded_genres": ["Horror"], "excluded_terms": ["slashers"], "reference_titles": ["Blade Runner"]},
        [("reference_title", "Blade Runner"), ("excluded_genre", "horror"), ("excluded_term", "slashers")],
    ),
    (
        "d2_q0004", "d2_grp_002",
        "late-night rain-soaked city lights similar to Blade Runner without horror or gore to zone out to",
        ["sensory_visual", "similarity_vibe", "negation_genre", "negation_content", "affective_tone"],
        {"excluded_genres": ["Horror"], "excluded_terms": ["gore"], "reference_titles": ["Blade Runner"]},
        [("reference_title", "Blade Runner"), ("excluded_genre", "horror"), ("excluded_term", "gore")],
    ),
    (
        "d2_q0005", "d2_grp_003",
        "chaotic high-adrenaline thriller by Denis Villeneuve starring Pedro Pascal, not romance",
        ["sensory_pacing", "attribute_genre", "person_director", "person_actor", "negation_genre"],
        {"directors": ["Denis Villeneuve"], "actors": ["Pedro Pascal"], "genres": ["Thriller"], "excluded_genres": ["Romance"]},
        [("genre", "thriller"), ("director", "Denis Villeneuve"), ("actor", "Pedro Pascal"), ("excluded_genre", "romance")],
    ),
    (
        "d2_q0006", "d2_grp_003",
        "breathless relentless crime thriller directed by Denis Villeneuve with Pedro Pascal and zero romance subplots",
        ["sensory_pacing", "attribute_genre", "person_director", "person_actor", "negation_genre"],
        {"directors": ["Denis Villeneuve"], "actors": ["Pedro Pascal"], "genres": ["Crime", "Thriller"], "excluded_genres": ["Romance"]},
        [("genre", "crime"), ("genre", "thriller"), ("director", "Denis Villeneuve"), ("actor", "Pedro Pascal"), ("excluded_genre", "romance")],
    ),
    (
        "d2_q0007", "d2_grp_004",
        "need unhinged folk terror like Midsommar with Florence Pugh where grief slowly breaks reality",
        ["similarity_vibe", "person_actor", "thematic_plot", "affective_tone"],
        {"actors": ["Florence Pugh"], "reference_titles": ["Midsommar"]},
        [("reference_title", "Midsommar"), ("actor", "Florence Pugh")],
    ),
    (
        "d2_q0008", "d2_grp_004",
        "daylight dread and cult isolation mood like Midsommar featuring Florence Pugh about trauma and belonging",
        ["similarity_vibe", "person_actor", "thematic_plot", "affective_tone"],
        {"actors": ["Florence Pugh"], "reference_titles": ["Midsommar"]},
        [("reference_title", "Midsommar"), ("actor", "Florence Pugh")],
    ),
    (
        "d2_q0009", "d2_grp_005",
        "chill quiet 2020s romance like Past Lives for Sunday night healing, not comedy",
        ["sensory_pacing", "attribute_era", "attribute_genre", "similarity_vibe", "affective_outcome", "negation_genre"],
        {"genres": ["Romance"], "year_min": 2020, "year_max": 2029, "excluded_genres": ["Comedy"], "reference_titles": ["Past Lives"]},
        [("era", "2020s"), ("genre", "romance"), ("reference_title", "Past Lives"), ("excluded_genre", "comedy")],
    ),
    (
        "d2_q0010", "d2_grp_005",
        "unhurried tender 2020s drama similar to Past Lives to feel emotionally grounded without comedy tropes",
        ["sensory_pacing", "attribute_era", "attribute_genre", "similarity_vibe", "affective_outcome", "negation_genre"],
        {"genres": ["Drama"], "year_min": 2020, "year_max": 2029, "excluded_genres": ["Comedy"], "reference_titles": ["Past Lives"]},
        [("era", "2020s"), ("genre", "drama"), ("reference_title", "Past Lives"), ("excluded_genre", "comedy")],
    ),
    (
        "d2_q0011", "d2_grp_006",
        "smart twisty whodunit like Knives Out with zero gore and sharp comedy energy",
        ["attribute_genre", "similarity_plot", "negation_content", "affective_tone"],
        {"genres": ["Comedy"], "excluded_terms": ["gore"], "reference_titles": ["Knives Out"]},
        [("reference_title", "Knives Out"), ("excluded_term", "gore"), ("genre", "comedy")],
    ),
    (
        "d2_q0012", "d2_grp_006",
        "clever mystery banter in the spirit of Knives Out, no bloodshed, witty screenplay",
        ["attribute_genre", "similarity_plot", "negation_content", "affective_tone"],
        {"genres": ["Mystery"], "excluded_terms": ["bloodshed"], "reference_titles": ["Knives Out"]},
        [("genre", "mystery"), ("reference_title", "Knives Out"), ("excluded_term", "bloodshed")],
    ),
    (
        "d2_q0013", "d2_grp_007",
        "black-and-white 1960s Akira Kurosawa samurai drama with no melodrama, high tension",
        ["sensory_visual", "attribute_era", "person_director", "attribute_genre", "negation_content", "affective_tone"],
        {"directors": ["Akira Kurosawa"], "genres": ["Drama"], "year_min": 1960, "year_max": 1969, "excluded_terms": ["melodrama"]},
        [("era", "1960s"), ("director", "Akira Kurosawa"), ("genre", "drama"), ("excluded_term", "melodrama")],
    ),
    (
        "d2_q0014", "d2_grp_007",
        "stark monochrome 1960s Japanese classic directed by Akira Kurosawa about honor without camp",
        ["sensory_visual", "attribute_era", "attribute_language_country", "person_director", "thematic_plot", "negation_content"],
        {"directors": ["Akira Kurosawa"], "language": "ja", "year_min": 1960, "year_max": 1969, "excluded_terms": ["camp"]},
        [("era", "1960s"), ("language_country", "Japanese"), ("director", "Akira Kurosawa"), ("excluded_term", "camp")],
    ),
    (
        "d2_q0015", "d2_grp_008",
        "Park Chan-wook Korean noir mystery post-2010, dark obsession vibes, no comedy",
        ["person_director", "attribute_language_country", "attribute_genre", "attribute_era", "affective_tone", "thematic_plot", "negation_genre"],
        {"directors": ["Park Chan-wook"], "language": "ko", "genres": ["Noir", "Mystery"], "year_min": 2010, "year_max": 2030, "excluded_genres": ["Comedy"]},
        [("director", "Park Chan-wook"), ("language_country", "Korean"), ("genre", "noir"), ("genre", "mystery"), ("era", "post-2010"), ("excluded_genre", "comedy")],
    ),
    (
        "d2_q0016", "d2_grp_008",
        "twisted 2010s Korean neo-noir by Park Chan-wook exploring jealousy without sitcom humor",
        ["person_director", "attribute_language_country", "attribute_genre", "attribute_era", "thematic_plot", "negation_content"],
        {"directors": ["Park Chan-wook"], "language": "ko", "genres": ["Noir"], "year_min": 2010, "year_max": 2019, "excluded_terms": ["sitcom humor"]},
        [("era", "2010s"), ("language_country", "Korean"), ("genre", "noir"), ("director", "Park Chan-wook"), ("excluded_term", "sitcom humor")],
    ),
    (
        "d2_q0017", "d2_grp_009",
        "mind-bending science fiction by Christopher Nolan starring Anne Hathaway with big orchestral tension",
        ["attribute_genre", "person_director", "person_actor", "sensory_visual", "affective_tone"],
        {"directors": ["Christopher Nolan"], "actors": ["Anne Hathaway"], "genres": ["Science Fiction"]},
        [("genre", "science fiction"), ("director", "Christopher Nolan"), ("actor", "Anne Hathaway")],
    ),
    (
        "d2_q0018", "d2_grp_009",
        "cerebral sci-fi directed by Christopher Nolan featuring Anne Hathaway about time and survival",
        ["attribute_genre", "person_director", "person_actor", "thematic_plot"],
        {"directors": ["Christopher Nolan"], "actors": ["Anne Hathaway"], "genres": ["Science Fiction"]},
        [("genre", "sci-fi"), ("director", "Christopher Nolan"), ("actor", "Anne Hathaway")],
    ),
    (
        "d2_q0019", "d2_grp_010",
        "fast whiplash tension like Whiplash, obsession and mentorship, zero fantasy",
        ["sensory_pacing", "similarity_plot", "thematic_plot", "negation_genre"],
        {"excluded_genres": ["Fantasy"], "reference_titles": ["Whiplash"]},
        [("reference_title", "Whiplash"), ("excluded_genre", "fantasy")],
    ),
    (
        "d2_q0020", "d2_grp_010",
        "breakneck drum-tempo drama reminiscent of Whiplash about perfectionism, not superhero action",
        ["sensory_pacing", "attribute_genre", "similarity_plot", "thematic_plot", "negation_genre"],
        {"genres": ["Drama"], "excluded_genres": ["Superhero", "Action"], "reference_titles": ["Whiplash"]},
        [("genre", "drama"), ("reference_title", "Whiplash"), ("excluded_genre", "superhero"), ("excluded_genre", "action")],
    ),
    (
        "d2_q0021", "d2_grp_011",
        "pastel cozy Greta Gerwig coming-of-age vibe starring Emma Stone without cynicism",
        ["sensory_visual", "person_director", "person_actor", "affective_tone", "negation_content", "thematic_plot"],
        {"directors": ["Greta Gerwig"], "actors": ["Emma Stone"], "excluded_terms": ["cynicism"]},
        [("director", "Greta Gerwig"), ("actor", "Emma Stone"), ("excluded_term", "cynicism")],
    ),
    (
        "d2_q0022", "d2_grp_011",
        "warm bittersweet drama by Greta Gerwig featuring Emma Stone about sisterhood, no gore",
        ["affective_tone", "attribute_genre", "person_director", "person_actor", "thematic_plot", "negation_content"],
        {"directors": ["Greta Gerwig"], "actors": ["Emma Stone"], "genres": ["Drama"], "excluded_terms": ["gore"]},
        [("genre", "drama"), ("director", "Greta Gerwig"), ("actor", "Emma Stone"), ("excluded_term", "gore")],
    ),
    (
        "d2_q0023", "d2_grp_012",
        "slow claustrophobic black-and-white madness like The Lighthouse starring Willem Dafoe",
        ["sensory_pacing", "sensory_visual", "similarity_vibe", "person_actor", "affective_tone"],
        {"actors": ["Willem Dafoe"], "reference_titles": ["The Lighthouse"]},
        [("reference_title", "The Lighthouse"), ("actor", "Willem Dafoe")],
    ),
    (
        "d2_q0024", "d2_grp_012",
        "stormy monochrome psychological descent similar to The Lighthouse with Willem Dafoe, no slapstick",
        ["sensory_visual", "similarity_vibe", "person_actor", "negation_content", "thematic_plot"],
        {"actors": ["Willem Dafoe"], "excluded_terms": ["slapstick"], "reference_titles": ["The Lighthouse"]},
        [("reference_title", "The Lighthouse"), ("actor", "Willem Dafoe"), ("excluded_term", "slapstick")],
    ),
    (
        "d2_q0025", "d2_grp_013",
        "no violence French coming-of-age indie drama to lift my spirits on a rough night",
        ["negation_content", "attribute_language_country", "attribute_genre", "thematic_plot", "affective_outcome"],
        {"language": "fr", "genres": ["Drama"], "excluded_terms": ["violence"]},
        [("excluded_term", "violence"), ("language_country", "French"), ("genre", "drama")],
    ),
    (
        "d2_q0026", "d2_grp_013",
        "gentle French family drama without profanity or cruelty to calm my nerves tonight",
        ["attribute_language_country", "attribute_genre", "negation_content", "affective_outcome"],
        {"language": "fr", "genres": ["Family", "Drama"], "excluded_terms": ["profanity", "cruelty"]},
        [("language_country", "French"), ("genre", "family"), ("genre", "drama"), ("excluded_term", "profanity"), ("excluded_term", "cruelty")],
    ),
    (
        "d2_q0027", "d2_grp_014",
        "synth-heavy neon night-drive crime vibe like Drive with Oscar Isaac, not comedy",
        ["sensory_visual", "attribute_genre", "similarity_vibe", "person_actor", "negation_genre"],
        {"actors": ["Oscar Isaac"], "genres": ["Crime"], "excluded_genres": ["Comedy"], "reference_titles": ["Drive"]},
        [("genre", "crime"), ("reference_title", "Drive"), ("actor", "Oscar Isaac"), ("excluded_genre", "comedy")],
    ),
    (
        "d2_q0028", "d2_grp_014",
        "hypnotic synthwave thriller mood akin to Drive featuring Oscar Isaac about loyalty",
        ["sensory_visual", "attribute_genre", "similarity_vibe", "person_actor", "thematic_plot"],
        {"actors": ["Oscar Isaac"], "genres": ["Thriller"], "reference_titles": ["Drive"]},
        [("genre", "thriller"), ("reference_title", "Drive"), ("actor", "Oscar Isaac")],
    ),
    (
        "d2_q0029", "d2_grp_015",
        "zero jumpscares pls existential alien first-contact sci-fi like Arrival with Ethan Hawke",
        ["negation_content", "thematic_plot", "attribute_genre", "similarity_plot", "person_actor"],
        {"actors": ["Ethan Hawke"], "genres": ["Science Fiction"], "excluded_terms": ["jumpscares"], "reference_titles": ["Arrival"]},
        [("excluded_term", "jumpscares"), ("genre", "sci-fi"), ("reference_title", "Arrival"), ("actor", "Ethan Hawke")],
    ),
    (
        "d2_q0030", "d2_grp_015",
        "quiet philosophical 2010s sci-fi reminiscent of Arrival starring Ethan Hawke, no horror",
        ["sensory_pacing", "thematic_plot", "attribute_era", "attribute_genre", "similarity_vibe", "person_actor", "negation_genre"],
        {"actors": ["Ethan Hawke"], "genres": ["Science Fiction"], "year_min": 2010, "year_max": 2019, "excluded_genres": ["Horror"], "reference_titles": ["Arrival"]},
        [("era", "2010s"), ("genre", "sci-fi"), ("reference_title", "Arrival"), ("actor", "Ethan Hawke"), ("excluded_genre", "horror")],
    ),
]


def _resolve_spans(text: str, span_specs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    lower_text = text.lower()
    for field, value in span_specs:
        idx = lower_text.find(value.lower())
        if idx < 0:
            raise ValueError(f"Span '{value}' not found in '{text}'")
        spans.append({"field": field, "value": value, "start": idx, "end": idx + len(value)})
    return spans


def build_records(taxonomy: dict[str, Any]) -> list[dict[str, Any]]:
    fine_to_broad: dict[str, str] = taxonomy["fine_to_broad"]
    compound_min = int(taxonomy["compound_rule"]["min_distinct_broad_categories"])
    records: list[dict[str, Any]] = []
    for qid, gid, text, fine_labels, ext_patch, span_specs in _REVIEWED_SPECS:
        broad = sorted({fine_to_broad[lbl] for lbl in fine_labels})
        ext = _empty_extraction()
        ext.update(ext_patch)
        records.append({
            "query_id": qid,
            "group_id": gid,
            "text": text,
            "fine_labels": fine_labels,
            "broad_labels": broad,
            "is_compound": len(broad) >= compound_min,
            "extraction": ext,
            "spans": _resolve_spans(text, span_specs),
            "provenance": {
                "source": "d2_live_drift_batch",
                "collection_batch": "d2_v1_reviewed",
                "is_synthetic": True,
                "annotation_method": "human_curated_drift_batch",
                "reviewed_by": "fasih-ur-rehman1",
                "review_status": "approved",
                "revision": 1,
            },
        })
    return records


def build_unreviewed_queue(taxonomy: dict[str, Any]) -> list[dict[str, Any]]:
    fine_to_broad: dict[str, str] = taxonomy["fine_to_broad"]
    raw = [
        ("d2_pending_01", "d2_pending_grp_01", "trippy midnight Japanese anime vibes no dialogue", ["sensory_visual", "attribute_language_country", "attribute_genre", "negation_content"], {"language": "ja", "genres": ["Anime"], "excluded_terms": ["dialogue"]}),
        ("d2_pending_02", "d2_pending_grp_02", "short punchy heist film under ninety minutes no gore", ["sensory_pacing", "thematic_plot", "negation_content"], {"excluded_terms": ["gore"]}),
        ("d2_pending_03", "d2_pending_grp_03", "comfort watch like Chef to boost mood fast", ["affective_outcome", "similarity_vibe"], {"reference_titles": ["Chef"]}),
        ("d2_pending_04", "d2_pending_grp_04", "90s Hong Kong neo-noir kinetic gunplay", ["attribute_era", "attribute_genre", "sensory_pacing"], {"genres": ["Noir"], "year_min": 1990, "year_max": 1999}),
        ("d2_pending_05", "d2_pending_grp_05", "bleak winter mystery with Willem Dafoe", ["sensory_visual", "attribute_genre", "person_actor"], {"genres": ["Mystery"], "actors": ["Willem Dafoe"]}),
    ]
    queue: list[dict[str, Any]] = []
    for qid, gid, text, fine, ext_patch in raw:
        broad = sorted({fine_to_broad[lbl] for lbl in fine})
        ext = _empty_extraction()
        ext.update(ext_patch)
        queue.append({
            "query_id": qid,
            "group_id": gid,
            "text": text,
            "fine_labels": fine,
            "broad_labels": broad,
            "is_compound": len(broad) >= 2,
            "extraction": ext,
            "spans": [],
            "provenance": {
                "source": "d2_live_drift_candidate",
                "collection_batch": "d2_pending_review",
                "is_synthetic": True,
                "annotation_method": "production_log_candidate_unreviewed",
                "reviewed_by": "unassigned",
                "review_status": "draft",
                "revision": 1,
            },
        })
    return queue


def main() -> int:
    tax = load_yaml(REPO_ROOT / "configs" / "taxonomy.yaml")
    schema_path = REPO_ROOT / "schemas" / "dataset_record.schema.json"
    records = build_records(tax)
    stats = validate_records(records, tax, schema_path, require_approved=True)

    # Verify zero overlap or near-duplicate leakage with frozen D1 splits.
    for split in ("train", "val", "course_regression"):
        existing = load_jsonl(REPO_ROOT / "data" / "processed" / f"{split}.jsonl")
        ex_ids = {r["query_id"] for r in existing}
        ex_grps = {r["group_id"] for r in existing}
        ex_norms = {normalize_text(r["text"]) for r in existing}
        ex_sh = [(r["query_id"], _shingles(r["text"])) for r in existing]
        for rec in records:
            assert rec["query_id"] not in ex_ids, f"ID overlap with {split}"
            assert rec["group_id"] not in ex_grps, f"Group overlap with {split}"
            assert normalize_text(rec["text"]) not in ex_norms, f"Exact overlap with {split}"
            sh = _shingles(rec["text"])
            for ex_qid, ex_s in ex_sh:
                sim = jaccard_similarity(sh, ex_s)
                assert sim < 0.85, f"Near duplicate {rec['query_id']} vs {split}:{ex_qid} ({sim:.3f})"

    out_dir = REPO_ROOT / "data" / "demo_batches"
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_path = out_dir / "d2_drift_batch_reviewed.jsonl"
    batch_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")

    queue_path = out_dir / "unreviewed_queue_sample.json"
    queue_path.write_text(json.dumps(build_unreviewed_queue(tax), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"batch_path": str(batch_path), "queue_path": str(queue_path), "stats": stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
