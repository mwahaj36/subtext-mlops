"""Deterministic group-disjoint multi-label stratified splitting."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ml.validate import (
    REPO_ROOT,
    load_yaml,
    normalize_text,
    sha256_file,
    validate_raw_dataset,
    validate_splits,
)


def _cluster_records(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Merge records sharing either group_id or exact normalized text into atomic clusters."""
    parent = list(range(len(records)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_group: dict[str, int] = {}
    by_text: dict[str, int] = {}
    for idx, rec in enumerate(records):
        g = rec["group_id"]
        if g in by_group:
            union(by_group[g], idx)
        else:
            by_group[g] = idx
        norm = normalize_text(rec["text"])
        if norm in by_text:
            union(by_text[norm], idx)
        else:
            by_text[norm] = idx

    clusters: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for idx, rec in enumerate(records):
        clusters[find(idx)].append(rec)
    return list(clusters.values())


def split_dataset(train_cfg_path: Path | None = None) -> dict[str, Any]:
    """Validate raw dataset, create group-disjoint splits, and persist manifests."""
    cfg_path = train_cfg_path or (REPO_ROOT / "configs" / "train.yaml")
    train_cfg = load_yaml(cfg_path)
    root = cfg_path.resolve().parent.parent
    paths = train_cfg["paths"]
    split_cfg = train_cfg["split"]

    taxonomy = load_yaml(root / paths["taxonomy"])
    schema_path = root / paths["dataset_schema"]
    raw_path = root / paths["raw_dataset"]
    prov_path = root / paths["provenance_manifest"]

    records, raw_stats = validate_raw_dataset(raw_path, taxonomy, schema_path, provenance_path=prov_path)

    label_order: list[str] = taxonomy["label_order"]
    clusters = _cluster_records(records)
    seed = int(train_cfg["seed"])
    rng = random.Random(seed)
    rng.shuffle(clusters)

    split_names = ("train", "val", "course_regression")
    ratios = {
        "train": float(split_cfg["train_ratio"]),
        "val": float(split_cfg["val_ratio"]),
        "course_regression": float(split_cfg["regression_ratio"]),
    }
    min_req = {
        "train": int(split_cfg["min_label_count_train"]),
        "val": int(split_cfg["min_label_count_val"]),
        "course_regression": int(split_cfg["min_label_count_regression"]),
    }

    cluster_meta = []
    total_labels: Counter[str] = Counter()
    for idx, cl in enumerate(clusters):
        lc: Counter[str] = Counter()
        for r in cl:
            for lbl in r["fine_labels"]:
                lc[lbl] += 1
                total_labels[lbl] += 1
        cluster_meta.append({"idx": idx, "records": cl, "size": len(cl), "labels": lc})

    total_n = len(records)
    target_sizes = {s: ratios[s] * total_n for s in split_names}
    target_labels = {
        s: {lbl: max(float(min_req[s]), ratios[s] * total_labels[lbl]) for lbl in label_order}
        for s in split_names
    }

    assigned: dict[str, list[dict[str, Any]]] = {s: [] for s in split_names}
    assigned_sizes = {s: 0 for s in split_names}
    assigned_labels = {s: Counter({lbl: 0 for lbl in label_order}) for s in split_names}
    remaining = list(cluster_meta)

    while remaining:
        # First satisfy any label still below required minimum, prioritizing rarest remaining labels.
        best_target: tuple[int, float, float, str, str] | None = None
        for s in ("val", "course_regression", "train"):
            for lbl in label_order:
                cur = assigned_labels[s][lbl]
                if cur < min_req[s]:
                    avail = sum(1 for c in remaining if c["labels"][lbl] > 0)
                    key = (0, float(avail), -(min_req[s] - cur), s, lbl)
                    if best_target is None or key < best_target:
                        best_target = key
                else:
                    deficit = target_labels[s][lbl] - cur
                    if deficit > 0.5:
                        avail = sum(1 for c in remaining if c["labels"][lbl] > 0)
                        key = (1, float(avail), -deficit, s, lbl)
                        if best_target is None or key < best_target:
                            best_target = key

        if best_target is not None:
            _, _, _, target_split, target_lbl = best_target
            candidates = [c for c in remaining if c["labels"][target_lbl] > 0]
            if not candidates:
                candidates = remaining
            chosen = max(
                candidates,
                key=lambda c: (
                    sum(
                        min(
                            c["labels"][lbl],
                            max(0.0, target_labels[target_split][lbl] - assigned_labels[target_split][lbl]),
                        )
                        for lbl in label_order
                    ),
                    target_sizes[target_split] - assigned_sizes[target_split] - c["size"],
                    -c["idx"],
                ),
            )
        else:
            target_split = max(
                split_names,
                key=lambda s: (target_sizes[s] - assigned_sizes[s]) / max(target_sizes[s], 1.0),
            )
            chosen = remaining[0]

        remaining.remove(chosen)
        assigned[target_split].extend(chosen["records"])
        assigned_sizes[target_split] += chosen["size"]
        assigned_labels[target_split].update(chosen["labels"])

    processed_dir = root / paths["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    split_hashes: dict[str, str] = {}
    split_summary: dict[str, Any] = {}

    for s in split_names:
        recs = sorted(assigned[s], key=lambda r: r["query_id"])
        out_path = processed_dir / f"{s}.jsonl"
        out_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs),
            encoding="utf-8",
        )
        split_hashes[s] = sha256_file(out_path)
        split_summary[s] = {
            "path": str(out_path.relative_to(root).as_posix()),
            "sha256": split_hashes[s],
            "num_records": len(recs),
            "num_groups": len({r["group_id"] for r in recs}),
            "label_counts": {lbl: int(assigned_labels[s][lbl]) for lbl in label_order},
        }

    manifest = {
        "seed": seed,
        "raw_dataset": paths["raw_dataset"],
        "raw_sha256": raw_stats["raw_sha256"],
        "total_records": total_n,
        "total_groups": len({r["group_id"] for r in records}),
        "split_sha256": split_hashes,
        "splits": split_summary,
    }
    manifest_path = root / paths["split_manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    validation_report = validate_splits(
        processed_dir,
        taxonomy,
        train_cfg,
        schema_path,
        split_manifest_path=manifest_path,
        check_frozen_hashes=True,
    )
    report_path = root / paths["validation_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"status": "passed", "raw": raw_stats, "splits": validation_report}, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Create group-disjoint multi-label dataset splits.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "train.yaml")
    args = parser.parse_args()
    manifest = split_dataset(args.config)
    counts = {k: v["num_records"] for k, v in manifest["splits"].items()}
    print(f"Group-disjoint splits generated and verified: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
