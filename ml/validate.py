"""Fail-closed schema, quality, provenance, and split-leakage validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import jsonschema
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


class DataValidationError(ValueError):
    """Raised when dataset records, provenance, or split boundaries fail checks."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _shingles(text: str) -> set[str]:
    norm = normalize_text(text)
    words = norm.split()
    grams = set(words)
    for i in range(max(0, len(words) - 2)):
        grams.add(" ".join(words[i : i + 3]))
    return grams


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DataValidationError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise DataValidationError(f"Invalid YAML object in {path}")
    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise DataValidationError(f"Missing dataset file: {path}")
    records: list[dict[str, Any]] = []
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"Unreadable dataset file {path}: {exc}") from exc
    for idx, line in enumerate(raw_lines, start=1):
        if not line.strip():
            raise DataValidationError(f"Empty/blank line at {path}:{idx}")
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataValidationError(f"Corrupt JSON at {path}:{idx}: {exc}") from exc
        if not isinstance(obj, dict):
            raise DataValidationError(f"Record at {path}:{idx} must be a JSON object")
        records.append(obj)
    if not records:
        raise DataValidationError(f"Dataset file {path} contains 0 records")
    return records


def _build_record_validator(schema_path: Path) -> jsonschema.Draft202012Validator:
    if not schema_path.exists():
        raise DataValidationError(f"Missing schema file: {schema_path}")
    schema_dir = schema_path.parent.resolve()
    raw_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    extraction_schema = json.loads((schema_dir / "extraction.schema.json").read_text(encoding="utf-8"))
    # Resolve local $ref without network calls via explicit store mapping.
    try:
        from referencing import Registry, Resource

        registry = Registry().with_resources(
            [
                ("extraction.schema.json", Resource.from_contents(extraction_schema)),
                (
                    (schema_dir / "extraction.schema.json").as_uri(),
                    Resource.from_contents(extraction_schema),
                ),
            ]
        )
        return jsonschema.Draft202012Validator(raw_schema, registry=registry)
    except ImportError:
        inline_schema = dict(raw_schema)
        inline_schema["properties"] = dict(raw_schema["properties"])
        inline_schema["properties"]["extraction"] = extraction_schema
        return jsonschema.Draft202012Validator(inline_schema)


def validate_records(
    records: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    schema_path: Path,
    *,
    require_approved: bool = True,
) -> dict[str, Any]:
    """Validate every record against schema, taxonomy, extraction bounds, spans, and provenance."""
    validator = _build_record_validator(schema_path)
    label_order: list[str] = taxonomy["label_order"]
    label_set = set(label_order)
    fine_to_broad: dict[str, str] = taxonomy["fine_to_broad"]
    canonical_genres = set(taxonomy["canonical_genres"])
    canonical_langs = set(taxonomy["canonical_languages"].keys())
    year_min_bound = int(taxonomy["year_bounds"]["min"])
    year_max_bound = int(taxonomy["year_bounds"]["max"])
    compound_min = int(taxonomy["compound_rule"]["min_distinct_broad_categories"])

    seen_ids: set[str] = set()
    label_counts: Counter[str] = Counter({lbl: 0 for lbl in label_order})
    multi_label_hist: Counter[int] = Counter()
    lengths: list[int] = []
    errors: list[str] = []

    for idx, rec in enumerate(records, start=1):
        qid = rec.get("query_id", f"<row {idx}>")
        try:
            validator.validate(rec)
        except jsonschema.ValidationError as exc:
            errors.append(f"{qid}: schema error ({exc.message})")
            continue

        if rec["query_id"] in seen_ids:
            errors.append(f"{qid}: duplicate query_id")
        seen_ids.add(rec["query_id"])

        text = rec["text"]
        if not text.strip():
            errors.append(f"{qid}: whitespace-only text")
        lengths.append(len(text))

        fine_labels = rec["fine_labels"]
        unknown = [lbl for lbl in fine_labels if lbl not in label_set]
        if unknown:
            errors.append(f"{qid}: unknown fine_labels {unknown}")
            continue
        expected_broad = sorted({fine_to_broad[lbl] for lbl in fine_labels})
        if sorted(rec["broad_labels"]) != expected_broad:
            errors.append(f"{qid}: broad_labels {rec['broad_labels']} != expected {expected_broad}")
        expected_compound = len(expected_broad) >= compound_min
        if rec["is_compound"] != expected_compound:
            errors.append(f"{qid}: is_compound {rec['is_compound']} != expected {expected_compound}")

        for lbl in fine_labels:
            label_counts[lbl] += 1
        multi_label_hist[len(fine_labels)] += 1

        ext = rec["extraction"]
        for g in ext["genres"] + ext["excluded_genres"]:
            if g not in canonical_genres:
                errors.append(f"{qid}: unsupported genre '{g}'")
        lang = ext["language"]
        if lang is not None and lang not in canonical_langs:
            errors.append(f"{qid}: unsupported language '{lang}'")
        ymin, ymax = ext["year_min"], ext["year_max"]
        if ymin is not None and not (year_min_bound <= ymin <= year_max_bound):
            errors.append(f"{qid}: year_min {ymin} out of bounds")
        if ymax is not None and not (year_min_bound <= ymax <= year_max_bound):
            errors.append(f"{qid}: year_max {ymax} out of bounds")
        if ymin is not None and ymax is not None and ymin > ymax:
            errors.append(f"{qid}: year_min {ymin} > year_max {ymax}")

        for sp in rec["spans"]:
            s, e, val = sp["start"], sp["end"], sp["value"]
            if s < 0 or e > len(text) or s >= e:
                errors.append(f"{qid}: span out of bounds [{s}:{e}] for len={len(text)}")
            elif text[s:e].lower() != val.lower():
                errors.append(f"{qid}: span text '{text[s:e]}' != '{val}'")

        prov = rec["provenance"]
        if require_approved and prov["review_status"] != "approved":
            errors.append(f"{qid}: unapproved review_status '{prov['review_status']}'")
        if not prov["reviewed_by"].strip() or not prov["source"].strip():
            errors.append(f"{qid}: incomplete provenance reviewer/source")

        if len(errors) >= 25:
            break

    if errors:
        raise DataValidationError("; ".join(errors))

    return {
        "total_records": len(records),
        "unique_groups": len({r["group_id"] for r in records}),
        "label_counts": dict(label_counts),
        "multi_label_histogram": {str(k): v for k, v in sorted(multi_label_hist.items())},
        "query_char_length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(sum(lengths) / len(lengths), 2),
        },
    }


def validate_raw_dataset(
    raw_path: Path,
    taxonomy: dict[str, Any],
    schema_path: Path,
    provenance_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and validate raw JSONL and optionally verify its SHA-256 against dataset provenance."""
    records = load_jsonl(raw_path)
    actual_sha = sha256_file(raw_path)
    if provenance_path is not None:
        if not provenance_path.exists():
            raise DataValidationError(f"Missing provenance manifest: {provenance_path}")
        prov_meta = json.loads(provenance_path.read_text(encoding="utf-8"))
        expected_sha = prov_meta.get("sha256")
        if not expected_sha or expected_sha != actual_sha:
            raise DataValidationError(
                f"Raw dataset SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
            )
    stats = validate_records(records, taxonomy, schema_path, require_approved=True)
    stats["raw_sha256"] = actual_sha
    return records, stats


def validate_splits(
    splits_dir: Path,
    taxonomy: dict[str, Any],
    train_cfg: dict[str, Any],
    schema_path: Path,
    split_manifest_path: Path | None = None,
    *,
    check_frozen_hashes: bool = False,
) -> dict[str, Any]:
    """Verify split schemas, minimum label support, frozen checksums, and zero leakage."""
    split_names = ("train", "val", "course_regression")
    split_records: dict[str, list[dict[str, Any]]] = {}
    split_stats: dict[str, Any] = {}
    actual_hashes: dict[str, str] = {}

    min_counts = {
        "train": int(train_cfg["split"]["min_label_count_train"]),
        "val": int(train_cfg["split"]["min_label_count_val"]),
        "course_regression": int(train_cfg["split"]["min_label_count_regression"]),
    }

    for name in split_names:
        path = splits_dir / f"{name}.jsonl"
        recs = load_jsonl(path)
        actual_hashes[name] = sha256_file(path)
        stats = validate_records(recs, taxonomy, schema_path, require_approved=True)
        for lbl, cnt in stats["label_counts"].items():
            if cnt < min_counts[name]:
                raise DataValidationError(
                    f"Split '{name}' label '{lbl}' count {cnt} < minimum {min_counts[name]}"
                )
        split_records[name] = recs
        split_stats[name] = stats

    if check_frozen_hashes:
        if split_manifest_path is None or not split_manifest_path.exists():
            raise DataValidationError(f"Missing split manifest for hash check: {split_manifest_path}")
        manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        expected_hashes = manifest.get("split_sha256", {})
        for name in split_names:
            if expected_hashes.get(name) != actual_hashes[name]:
                raise DataValidationError(
                    f"Frozen split checksum mismatch for '{name}': "
                    f"expected {expected_hashes.get(name)}, got {actual_hashes[name]}"
                )

    all_ids: dict[str, set[str]] = {
        name: {r["query_id"] for r in recs} for name, recs in split_records.items()
    }
    all_groups: dict[str, set[str]] = {
        name: {r["group_id"] for r in recs} for name, recs in split_records.items()
    }
    all_norms: dict[str, set[str]] = {
        name: {normalize_text(r["text"]) for r in recs} for name, recs in split_records.items()
    }
    shingles_by_split = {
        name: [(r["query_id"], r["group_id"], _shingles(r["text"])) for r in recs]
        for name, recs in split_records.items()
    }

    near_thresh = float(train_cfg["split"]["near_duplicate_threshold"])
    leakage_counts = {
        "query_id_overlap": 0,
        "group_id_overlap": 0,
        "exact_normalized_text_overlap": 0,
        "near_duplicate_pairs": 0,
    }

    pairs = [("train", "val"), ("train", "course_regression"), ("val", "course_regression")]
    for a, b in pairs:
        id_ov = all_ids[a] & all_ids[b]
        if id_ov:
            leakage_counts["query_id_overlap"] += len(id_ov)
            raise DataValidationError(f"query_id overlap between {a} and {b}: {sorted(id_ov)[:5]}")
        grp_ov = all_groups[a] & all_groups[b]
        if grp_ov:
            leakage_counts["group_id_overlap"] += len(grp_ov)
            raise DataValidationError(f"group_id overlap between {a} and {b}: {sorted(grp_ov)[:5]}")
        txt_ov = all_norms[a] & all_norms[b]
        if txt_ov:
            leakage_counts["exact_normalized_text_overlap"] += len(txt_ov)
            raise DataValidationError(
                f"Exact normalized text overlap between {a} and {b}: {sorted(txt_ov)[:3]}"
            )
        for qid_a, _, sh_a in shingles_by_split[a]:
            for qid_b, _, sh_b in shingles_by_split[b]:
                sim = jaccard_similarity(sh_a, sh_b)
                if sim >= near_thresh:
                    leakage_counts["near_duplicate_pairs"] += 1
                    raise DataValidationError(
                        f"Near-duplicate cross-split pair ({a}:{qid_a} vs {b}:{qid_b}, jaccard={sim:.3f})"
                    )

    return {
        "status": "passed",
        "split_sha256": actual_hashes,
        "leakage_checks": leakage_counts,
        "near_duplicate_threshold": near_thresh,
        "splits": split_stats,
    }


def ensure_valid_for_training(
    *,
    train_cfg_path: Path | None = None,
    check_frozen_hashes: bool = True,
    fit_fn: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Fail-closed training preflight guard; halts before invoking fit_fn if checks fail."""
    cfg_path = train_cfg_path or (REPO_ROOT / "configs" / "train.yaml")
    train_cfg = load_yaml(cfg_path)
    paths = train_cfg["paths"]
    root = cfg_path.resolve().parent.parent

    def _p(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (root / p)

    taxonomy = load_yaml(_p(paths["taxonomy"]))
    schema_path = _p(paths["dataset_schema"])
    _, raw_stats = validate_raw_dataset(
        _p(paths["raw_dataset"]),
        taxonomy,
        schema_path,
        provenance_path=_p(paths["provenance_manifest"]),
    )
    split_report = validate_splits(
        _p(paths["processed_dir"]),
        taxonomy,
        train_cfg,
        schema_path,
        split_manifest_path=_p(paths["split_manifest"]),
        check_frozen_hashes=check_frozen_hashes,
    )
    report = {
        "status": "passed",
        "raw": raw_stats,
        "splits": split_report,
    }
    report_path = _p(paths["validation_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if fit_fn is not None:
        return fit_fn()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate raw dataset and/or group-disjoint splits.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "train.yaml")
    parser.add_argument("--raw", type=Path, default=None)
    parser.add_argument("--provenance", type=Path, default=None)
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--check-frozen-hashes", action="store_true")
    parser.add_argument("--report-out", type=Path, default=None)
    args = parser.parse_args(argv)

    train_cfg = load_yaml(args.config)
    root = args.config.resolve().parent.parent
    paths = train_cfg["paths"]
    taxonomy = load_yaml(root / paths["taxonomy"])
    schema_path = root / paths["dataset_schema"]

    report: dict[str, Any] = {"status": "passed"}
    if args.raw is not None:
        _, raw_stats = validate_raw_dataset(args.raw, taxonomy, schema_path, args.provenance)
        report["raw"] = raw_stats
    if args.splits_dir is not None:
        split_report = validate_splits(
            args.splits_dir,
            taxonomy,
            train_cfg,
            schema_path,
            split_manifest_path=args.split_manifest,
            check_frozen_hashes=args.check_frozen_hashes,
        )
        report["splits"] = split_report

    out_path = args.report_out or (root / paths["validation_report"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Validation passed. Report written to {out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DataValidationError as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
