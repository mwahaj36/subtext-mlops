"""Tests for schema/provenance validation, leakage detection, splitting, and extraction."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.extract import evaluate_extractions, extract_constraints, load_lexicons  # noqa: E402
from ml.validate import (  # noqa: E402
    DataValidationError,
    ensure_valid_for_training,
    load_jsonl,
    load_yaml,
    sha256_file,
    validate_raw_dataset,
    validate_records,
    validate_splits,
)


def _sample_record(
    qid: str = "q_1",
    group_id: str = "g_1",
    text: str = "Korean thriller starring Song Kang-ho without gore",
    fine_labels: list[str] | None = None,
    broad_labels: list[str] | None = None,
    is_compound: bool = True,
) -> dict:
    return {
        "query_id": qid,
        "group_id": group_id,
        "text": text,
        "fine_labels": fine_labels or ["attribute_language_country", "attribute_genre", "person_actor", "negation_content"],
        "broad_labels": broad_labels or ["attribute", "negation", "person"],
        "is_compound": is_compound,
        "extraction": {
            "actors": ["Song Kang-ho"],
            "directors": [],
            "genres": ["Thriller"],
            "year_min": None,
            "year_max": None,
            "language": "ko",
            "excluded_genres": [],
            "excluded_terms": ["gore"],
            "reference_titles": [],
        },
        "spans": [
            {"field": "language_country", "value": "Korean", "start": 0, "end": 6},
            {"field": "genre", "value": "thriller", "start": 7, "end": 15},
            {"field": "actor", "value": "Song Kang-ho", "start": 25, "end": 37},
            {"field": "excluded_term", "value": "gore", "start": 46, "end": 50},
        ],
        "provenance": {
            "source": "course_curated_d1",
            "collection_batch": "d1_v1",
            "is_synthetic": True,
            "annotation_method": "rubric_curated_and_verified",
            "reviewed_by": "mwahaj36",
            "review_status": "approved",
            "revision": 1,
        },
    }


def _taxonomy() -> dict:
    return load_yaml(REPO_ROOT / "configs" / "taxonomy.yaml")


def _schema_path() -> Path:
    return REPO_ROOT / "schemas" / "dataset_record.schema.json"


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_real_dataset_and_splits_pass_validation() -> None:
    report = ensure_valid_for_training(check_frozen_hashes=True)
    assert report["status"] == "passed"
    leakage = report["splits"]["leakage_checks"]
    assert leakage == {
        "query_id_overlap": 0,
        "group_id_overlap": 0,
        "exact_normalized_text_overlap": 0,
        "near_duplicate_pairs": 0,
    }


def test_corruption_fixtures_fail_closed(tmp_path: Path) -> None:
    tax = _taxonomy()
    schema = _schema_path()

    # 1. Corrupt JSON line
    bad_json = tmp_path / "bad.jsonl"
    bad_json.write_text('{"query_id": "1"}\n{corrupt\n', encoding="utf-8")
    with pytest.raises(DataValidationError, match="Corrupt JSON"):
        validate_raw_dataset(bad_json, tax, schema)

    # 2. Missing required field / empty query text
    empty_rec = _sample_record(text="   ")
    with pytest.raises(DataValidationError, match="whitespace-only"):
        validate_records([empty_rec], tax, schema)

    # 3. Duplicate query_id
    r1 = _sample_record(qid="dup_1", group_id="g1")
    r2 = _sample_record(qid="dup_1", group_id="g2")
    with pytest.raises(DataValidationError, match="duplicate query_id"):
        validate_records([r1, r2], tax, schema)

    # 4. Unknown fine label
    unk = _sample_record(fine_labels=["nonexistent_intent"])
    with pytest.raises(DataValidationError, match="unknown fine_labels"):
        validate_records([unk], tax, schema)

    # 5. Span offset out of bounds & substring mismatch
    bad_span = _sample_record()
    bad_span["spans"][0]["end"] = 999
    with pytest.raises(DataValidationError, match="span out of bounds"):
        validate_records([bad_span], tax, schema)

    mismatch_span = _sample_record()
    mismatch_span["spans"][0]["value"] = "French"
    with pytest.raises(DataValidationError, match="span text"):
        validate_records([mismatch_span], tax, schema)

    # 6. Invalid year range & unsupported genre
    bad_year = _sample_record()
    bad_year["extraction"]["year_min"] = 2020
    bad_year["extraction"]["year_max"] = 2010
    with pytest.raises(DataValidationError, match="year_min 2020 > year_max 2010"):
        validate_records([bad_year], tax, schema)

    bad_genre = _sample_record()
    bad_genre["extraction"]["genres"] = ["UnsupportedGenre"]
    with pytest.raises(DataValidationError, match="unsupported genre"):
        validate_records([bad_genre], tax, schema)

    # 7. Unapproved provenance
    draft = _sample_record()
    draft["provenance"]["review_status"] = "draft"
    with pytest.raises(DataValidationError, match="unapproved review_status"):
        validate_records([draft], tax, schema, require_approved=True)

    # 8. Provenance SHA-256 mismatch
    raw_file = tmp_path / "raw.jsonl"
    _write_jsonl(raw_file, [_sample_record()])
    prov_file = tmp_path / "prov.json"
    prov_file.write_text(json.dumps({"sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(DataValidationError, match="SHA-256 mismatch"):
        validate_raw_dataset(raw_file, tax, schema, provenance_path=prov_file)


def _make_split_dir_from_real(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    tax = _taxonomy()
    cfg = copy.deepcopy(load_yaml(REPO_ROOT / "configs" / "train.yaml"))
    cfg["split"]["min_label_count_train"] = 1
    cfg["split"]["min_label_count_val"] = 1
    cfg["split"]["min_label_count_regression"] = 1

    splits_dir = tmp_path / "processed"
    splits_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split_name in ("train", "val", "course_regression"):
        src = REPO_ROOT / "data" / "processed" / f"{split_name}.jsonl"
        dst = splits_dir / f"{split_name}.jsonl"
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        hashes[split_name] = sha256_file(dst)

    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(json.dumps({"split_sha256": hashes}), encoding="utf-8")
    return splits_dir, manifest_path, tax, cfg


def test_leakage_fixtures_fail_closed(tmp_path: Path) -> None:
    schema = _schema_path()

    # 9. Shared group_id between train and val
    sdir, mpath, tax, cfg = _make_split_dir_from_real(tmp_path / "case_group")
    val_recs = load_jsonl(sdir / "val.jsonl")
    train_recs = load_jsonl(sdir / "train.jsonl")
    val_recs[0]["group_id"] = train_recs[0]["group_id"]
    _write_jsonl(sdir / "val.jsonl", val_recs)
    with pytest.raises(DataValidationError, match="group_id overlap"):
        validate_splits(sdir, tax, cfg, schema, split_manifest_path=mpath, check_frozen_hashes=False)

    # 10. Exact normalized duplicate text across splits with different group_ids
    sdir, mpath, tax, cfg = _make_split_dir_from_real(tmp_path / "case_exact")
    train_recs = load_jsonl(sdir / "train.jsonl")
    reg_recs = load_jsonl(sdir / "course_regression.jsonl")
    reg_recs[0]["text"] = train_recs[0]["text"].upper() + "!!"
    reg_recs[0]["spans"] = []
    _write_jsonl(sdir / "course_regression.jsonl", reg_recs)
    with pytest.raises(DataValidationError, match="Exact normalized text overlap"):
        validate_splits(sdir, tax, cfg, schema, split_manifest_path=mpath, check_frozen_hashes=False)

    # 11. Near-duplicate paraphrase across val and course_regression
    sdir, mpath, tax, cfg = _make_split_dir_from_real(tmp_path / "case_near")
    val_recs = load_jsonl(sdir / "val.jsonl")
    reg_recs = load_jsonl(sdir / "course_regression.jsonl")
    # Pick a long query and make a tiny punctuation/token tweak so Jaccard >= 0.85
    longest = max(val_recs, key=lambda r: len(r["text"].split()))
    reg_recs[0]["text"] = longest["text"] + " please"
    reg_recs[0]["spans"] = []
    _write_jsonl(sdir / "course_regression.jsonl", reg_recs)
    with pytest.raises(DataValidationError, match="Near-duplicate"):
        validate_splits(sdir, tax, cfg, schema, split_manifest_path=mpath, check_frozen_hashes=False)

    # 12. Tampered frozen split failing checksum check
    sdir, mpath, tax, cfg = _make_split_dir_from_real(tmp_path / "case_hash")
    reg_path = sdir / "course_regression.jsonl"
    reg_recs = load_jsonl(reg_path)
    reg_recs[0]["provenance"]["revision"] = 2
    _write_jsonl(reg_path, reg_recs)
    with pytest.raises(DataValidationError, match="Frozen split checksum mismatch"):
        validate_splits(sdir, tax, cfg, schema, split_manifest_path=mpath, check_frozen_hashes=True)


def test_training_preflight_guard_halts_before_fit(tmp_path: Path) -> None:
    cfg = copy.deepcopy(load_yaml(REPO_ROOT / "configs" / "train.yaml"))
    bad_raw = tmp_path / "bad_raw.jsonl"
    bad_raw.write_text('{"bad": true}\n', encoding="utf-8")
    prov = tmp_path / "prov.json"
    prov.write_text(json.dumps({"sha256": sha256_file(bad_raw)}), encoding="utf-8")

    # Point config at corrupted raw dataset inside tmp_path
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg["paths"]["raw_dataset"] = str(bad_raw)
    cfg["paths"]["provenance_manifest"] = str(prov)
    cfg["paths"]["taxonomy"] = str(REPO_ROOT / "configs" / "taxonomy.yaml")
    cfg["paths"]["dataset_schema"] = str(REPO_ROOT / "schemas" / "dataset_record.schema.json")
    cfg["paths"]["processed_dir"] = str(REPO_ROOT / "data" / "processed")
    cfg["paths"]["split_manifest"] = str(REPO_ROOT / "data" / "manifests" / "split_manifest.json")
    cfg["paths"]["validation_report"] = str(tmp_path / "report.json")

    import yaml
    cfg_path = cfg_dir / "train.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    fit_called = False

    def fake_fit() -> None:
        nonlocal fit_called
        fit_called = True

    with pytest.raises(DataValidationError):
        ensure_valid_for_training(train_cfg_path=cfg_path, fit_fn=fake_fit)
    assert fit_called is False


def test_extraction_baseline_handles_constraints_and_false_negation() -> None:
    tax = _taxonomy()
    people_set, title_map = load_lexicons()

    ext = extract_constraints(
        "Korean thriller directed by Park Chan-wook between 2002 and 2016 without gore",
        tax,
        people_set=people_set,
        title_map=title_map,
    )
    assert ext["language"] == "ko"
    assert ext["genres"] == ["Thriller"]
    assert ext["directors"] == ["Park Chan-wook"]
    assert ext["year_min"] == 2002
    assert ext["year_max"] == 2016
    assert ext["excluded_terms"] == ["gore"]

    false_neg = extract_constraints(
        "No idea what to watch tonight, maybe a cozy 90s comedy",
        tax,
        people_set=people_set,
        title_map=title_map,
    )
    assert false_neg["excluded_genres"] == []
    assert false_neg["excluded_terms"] == []
    assert false_neg["genres"] == ["Comedy"]
    assert (false_neg["year_min"], false_neg["year_max"]) == (1990, 1999)

    summary = evaluate_extractions([{"extraction": ext}], [ext])
    assert summary["overall_micro"]["f1"] == 1.0
