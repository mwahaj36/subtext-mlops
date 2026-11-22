"""Closed-loop challenger retraining, gate comparison, and gated MLflow registration."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import torch
import torch.nn as nn
from mlflow.models.signature import ModelSignature
from mlflow.tracking import MlflowClient
from mlflow.types.schema import ColSpec, Schema
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from ml.evaluate import (
    apply_temperature_scaling,
    configure_mlflow,
    encode_labels,
    evaluate_multilabel,
    fit_temperature_scaling,
    get_git_dvc_lineage,
    sigmoid,
    tune_thresholds,
)
from ml.extract import load_lexicons
from ml.gate import evaluate_gates
from ml.register import SubtextQueryUnderstandingPyFunc
from ml.train import MiniLMMultiLabelModel, QueryDataset, predict_logits, set_seed
from ml.train_baseline import _log_metrics_chunked
from ml.validate import (
    REPO_ROOT,
    DataValidationError,
    _shingles,
    ensure_valid_for_training,
    jaccard_similarity,
    load_jsonl,
    load_yaml,
    normalize_text,
    sha256_file,
    validate_records,
)

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _load_batch_records(batch_path: Path) -> list[dict[str, Any]]:
    if not batch_path.exists():
        raise DataValidationError(f"Missing batch file: {batch_path}")
    if batch_path.suffix.lower() == ".json":
        payload = json.loads(batch_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload or not all(isinstance(x, dict) for x in payload):
            raise DataValidationError(f"JSON batch {batch_path} must be a non-empty list of objects")
        return payload
    return load_jsonl(batch_path)


def validate_drift_batch(
    batch_path: Path,
    train_cfg_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fail-closed validation of reviewed drift batch: requires approved status and zero leakage."""
    cfg_path = train_cfg_path or (REPO_ROOT / "configs" / "train.yaml")
    train_cfg = load_yaml(cfg_path)
    root = cfg_path.resolve().parent.parent
    paths = train_cfg["paths"]
    taxonomy = load_yaml(root / paths["taxonomy"])
    schema_path = root / paths["dataset_schema"]

    records = _load_batch_records(batch_path)
    unapproved = [
        f"{r.get('query_id', '<row>')}:{r.get('provenance', {}).get('review_status')}"
        for r in records
        if not isinstance(r.get("provenance"), dict) or r["provenance"].get("review_status") != "approved"
    ]
    if unapproved:
        raise DataValidationError(
            f"Unreviewed or unapproved records blocked from retraining: {unapproved[:5]}"
        )

    stats = validate_records(records, taxonomy, schema_path, require_approved=True)
    processed_dir = root / paths["processed_dir"]
    near_thresh = float(train_cfg["split"]["near_duplicate_threshold"])

    for split_name in ("val", "course_regression", "train"):
        split_recs = load_jsonl(processed_dir / f"{split_name}.jsonl")
        ex_ids = {r["query_id"] for r in split_recs}
        ex_grps = {r["group_id"] for r in split_recs}
        ex_norms = {normalize_text(r["text"]) for r in split_recs}
        ex_shingles = [(r["query_id"], _shingles(r["text"])) for r in split_recs]

        for rec in records:
            qid = rec["query_id"]
            if qid in ex_ids:
                raise DataValidationError(f"query_id overlap between batch ({qid}) and {split_name}")
            if normalize_text(rec["text"]) in ex_norms:
                raise DataValidationError(f"Normalized text overlap between batch ({qid}) and {split_name}")
            if split_name in ("val", "course_regression"):
                if rec["group_id"] in ex_grps:
                    raise DataValidationError(f"group_id overlap between batch ({qid}) and {split_name}")
                rec_sh = _shingles(rec["text"])
                for ex_qid, ex_s in ex_shingles:
                    sim = jaccard_similarity(rec_sh, ex_s)
                    if sim >= near_thresh:
                        raise DataValidationError(
                            f"Near-duplicate leakage ({qid} vs {split_name}:{ex_qid}, jaccard={sim:.3f})"
                        )

    stats["batch_sha256"] = sha256_file(batch_path)
    return records, stats


def _train_combined_model(
    train_recs: list[dict[str, Any]],
    val_recs: list[dict[str, Any]],
    label_order: list[str],
    prim_cfg: dict[str, Any],
    seed: int,
    epochs: int,
) -> tuple[MiniLMMultiLabelModel, Any, torch.device, int, list[dict[str, float]]]:
    x_train = [r["text"] for r in train_recs]
    x_val = [r["text"] for r in val_recs]
    y_train = encode_labels(train_recs, label_order)
    y_val = encode_labels(val_recs, label_order)

    model_name = str(prim_cfg["model_name"])
    model_rev = str(prim_cfg["model_revision"])
    max_len = int(prim_cfg["max_length"])
    batch_size = int(prim_cfg["batch_size"])
    lr = float(prim_cfg["learning_rate"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_rev)
    backbone = AutoModel.from_pretrained(model_name, revision=model_rev)
    model = MiniLMMultiLabelModel(
        encoder=backbone,
        hidden_size=int(backbone.config.hidden_size),
        num_labels=len(label_order),
        dropout=float(prim_cfg["dropout"]),
    ).to(device)

    pos_counts = np.maximum(np.sum(y_train, axis=0), 1.0)
    neg_counts = np.maximum(len(y_train) - pos_counts, 1.0)
    pos_weights = np.clip(neg_counts / pos_counts, 1.0, float(prim_cfg["pos_weight_cap"]))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(
        [{"params": model.encoder.parameters(), "lr": lr}, {"params": model.head.parameters(), "lr": lr * 20.0}],
        weight_decay=float(prim_cfg["weight_decay"]),
    )

    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(QueryDataset(x_train, y_train), batch_size=batch_size, shuffle=True, generator=gen)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, len(loader) * epochs))

    best_val_f1, best_epoch = -1.0, 1
    best_state = copy.deepcopy(model.state_dict())
    epoch_history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n_batches = 0.0, 0
        for batch_texts, batch_y in loader:
            enc = tokenizer(list(batch_texts), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            optimizer.zero_grad()
            logits = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
            loss = criterion(logits, batch_y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())
            n_batches += 1

        val_logits_ep = predict_logits(model, tokenizer, x_val, max_len, batch_size, device)
        val_f1_ep = round(float(f1_score(y_val, (sigmoid(val_logits_ep) >= 0.50).astype(int), average="macro", zero_division=0)), 4)
        epoch_history.append({"epoch": epoch, "train_loss": round(running_loss / max(1, n_batches), 4), "val_raw_macro_f1": val_f1_ep})
        if val_f1_ep >= best_val_f1:
            best_val_f1, best_epoch, best_state = val_f1_ep, epoch, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, tokenizer, device, best_epoch, epoch_history


def retrain_challenger(
    batch_path: Path,
    train_cfg_path: Path | None = None,
    *,
    out_dir: Path | None = None,
    comparison_out_path: Path | None = None,
    epochs: int | None = None,
    local_mlflow: bool = False,
    register_challenger: bool = True,
) -> dict[str, Any]:
    """Retrain challenger on train + approved drift batch, gate against champion, and log/register."""
    t0 = time.perf_counter()
    cfg_path = train_cfg_path or (REPO_ROOT / "configs" / "train.yaml")
    ensure_valid_for_training(train_cfg_path=cfg_path, check_frozen_hashes=True)
    batch_recs, batch_stats = validate_drift_batch(batch_path, cfg_path)

    train_cfg = load_yaml(cfg_path)
    root = cfg_path.resolve().parent.parent
    paths = train_cfg["paths"]
    taxonomy = load_yaml(root / paths["taxonomy"])
    label_order: list[str] = taxonomy["label_order"]
    seed = int(train_cfg["seed"])
    set_seed(seed)

    prim_cfg, cal_cfg = train_cfg["primary"], train_cfg["calibration"]
    run_epochs = int(epochs or prim_cfg["epochs"])
    max_len, batch_size = int(prim_cfg["max_length"]), int(prim_cfg["batch_size"])

    processed_dir = root / paths["processed_dir"]
    train_recs = load_jsonl(processed_dir / "train.jsonl")
    val_recs = load_jsonl(processed_dir / "val.jsonl")
    reg_recs = load_jsonl(processed_dir / "course_regression.jsonl")
    combined_train = train_recs + batch_recs

    model, tokenizer, device, best_epoch, epoch_history = _train_combined_model(
        combined_train, val_recs, label_order, prim_cfg, seed, run_epochs
    )
    x_val = [r["text"] for r in val_recs]
    x_reg = [r["text"] for r in reg_recs]
    x_batch = [r["text"] for r in batch_recs]
    y_val = encode_labels(val_recs, label_order)

    val_logits = predict_logits(model, tokenizer, x_val, max_len, batch_size, device)
    reg_logits = predict_logits(model, tokenizer, x_reg, max_len, batch_size, device)
    batch_logits = predict_logits(model, tokenizer, x_batch, max_len, batch_size, device)

    temperatures = fit_temperature_scaling(val_logits, y_val, label_order)
    val_cal = apply_temperature_scaling(val_logits, temperatures, label_order)
    reg_cal = apply_temperature_scaling(reg_logits, temperatures, label_order)
    batch_cal = apply_temperature_scaling(batch_logits, temperatures, label_order)
    thresholds = tune_thresholds(
        val_cal, y_val, label_order,
        grid_min=float(cal_cfg["threshold_grid_min"]),
        grid_max=float(cal_cfg["threshold_grid_max"]),
        grid_step=float(cal_cfg["threshold_grid_step"]),
    )

    people_set, title_map = load_lexicons()
    val_eval = evaluate_multilabel(val_recs, val_cal, thresholds, label_order, taxonomy, raw_probs=sigmoid(val_logits), people_set=people_set, title_map=title_map)
    reg_eval = evaluate_multilabel(reg_recs, reg_cal, thresholds, label_order, taxonomy, raw_probs=sigmoid(reg_logits), people_set=people_set, title_map=title_map)
    batch_eval = evaluate_multilabel(batch_recs, batch_cal, thresholds, label_order, taxonomy, raw_probs=sigmoid(batch_logits), people_set=people_set, title_map=title_map)

    artifact_dir = out_dir or (root / "artifacts" / "challenger")
    encoder_dir = artifact_dir / "encoder"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(encoder_dir)
    model.encoder.save_pretrained(encoder_dir)
    torch.save(model.head.state_dict(), artifact_dir / "head.pt")

    champion_metrics_path = root / "artifacts" / "primary" / "metrics.json"
    champion_metrics = json.loads(champion_metrics_path.read_text(encoding="utf-8"))
    reg_summary_path = root / "artifacts" / "registry" / "registration_summary.json"
    champion_version = str(json.loads(reg_summary_path.read_text(encoding="utf-8"))["selected_model"]["registered_version"]) if reg_summary_path.exists() else "2"

    batch_sha = batch_stats["batch_sha256"]
    fp_src = f"{prim_cfg['model_type']}|{batch_sha}|{val_eval['aggregate']['macro_f1']}|{reg_eval['aggregate']['macro_f1']}"
    challenger_fp = hashlib.sha256(fp_src.encode("utf-8")).hexdigest()[:16]
    lineage = get_git_dvc_lineage(root)
    completed_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    runtime_s = round(time.perf_counter() - t0, 3)

    manifest = {
        "model_type": prim_cfg["model_type"],
        "model_name": str(prim_cfg["model_name"]),
        "model_revision": str(prim_cfg["model_revision"]),
        "hidden_size": int(model.encoder.config.hidden_size),
        "num_labels": len(label_order),
        "max_length": max_len,
        "dropout": float(prim_cfg["dropout"]),
        "best_epoch": best_epoch,
        "label_order": label_order,
        "temperatures": temperatures,
        "thresholds": thresholds,
        "bundle_fingerprint": challenger_fp,
        "drift_batch": batch_path.name,
        "drift_batch_sha256": batch_sha,
        "lineage": lineage,
    }
    manifest_path = artifact_dir / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    metrics_path = artifact_dir / "metrics.json"
    summary: dict[str, Any] = {
        "run_id": "",
        "run_name": "d2_challenger_minilm_multilabel",
        "run_role": "challenger",
        "completed_utc": completed_utc,
        "model_type": prim_cfg["model_type"],
        "runtime_seconds": runtime_s,
        "best_epoch": best_epoch,
        "lineage": lineage,
        "calibration": {"method": cal_cfg["method"], "temperatures": temperatures, "thresholds": thresholds},
        "drift_batch": {"path": str(batch_path), "sha256": batch_sha, "records": len(batch_recs)},
        "val": val_eval,
        "course_regression": reg_eval,
        "batch_eval": batch_eval,
    }
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    gate_res = evaluate_gates(
        champion_metrics_path, metrics_path, root / "configs" / "gates.yaml", root / paths["validation_report"]
    )

    tracking_uri = configure_mlflow(train_cfg, local_only=local_mlflow)
    challenger_version = "not_registered"
    with mlflow.start_run(run_name="d2_challenger_minilm_multilabel") as run:
        run_id = run.info.run_id
        summary["run_id"] = run_id
        metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        mlflow.set_tags({
            "run_role": "challenger",
            "stage": "deliverable_2",
            "gate_passed": str(gate_res["passed"]).lower(),
            "champion_run_id": champion_metrics.get("run_id", "unknown"),
            "bundle_fingerprint": challenger_fp,
            "drift_batch_sha256": batch_sha,
        })
        mlflow.log_params({"epochs": run_epochs, "best_epoch": best_epoch, "train_records": len(train_recs), "batch_records": len(batch_recs), "combined_train": len(combined_train)})
        flat = {"runtime_seconds": runtime_s, "val_macro_f1": float(val_eval["aggregate"]["macro_f1"]), "reg_macro_f1": float(reg_eval["aggregate"]["macro_f1"]), "reg_ece": float(reg_eval["aggregate"]["ece"]), "batch_macro_f1": float(batch_eval["aggregate"]["macro_f1"]), "extraction_micro_f1": float(reg_eval["aggregate"]["extraction_micro_f1"])}
        _log_metrics_chunked(flat)

        reg_name = str(train_cfg["selection"]["registered_model_name"])
        if register_challenger and gate_res["passed"] and not tracking_uri.startswith("file:"):
            client = MlflowClient()
            existing = client.search_model_versions(f"name='{reg_name}'")
            matched = next((mv for mv in existing if (getattr(mv, "tags", {}) or {}).get("bundle_fingerprint") == challenger_fp or f"fingerprint={challenger_fp}" in (getattr(mv, "description", "") or "")), None)
            if matched:
                challenger_version = str(matched.version)
            else:
                sig = ModelSignature(inputs=Schema([ColSpec("string", "text")]), outputs=Schema([ColSpec("string", "prediction_json")]))
                mlflow.pyfunc.log_model(artifact_path="subtext_query_understanding", python_model=SubtextQueryUnderstandingPyFunc(), artifacts={"bundle_dir": str(artifact_dir), "taxonomy": str(root / "configs" / "taxonomy.yaml"), "routing": str(root / "configs" / "routing.yaml")}, signature=sig, registered_model_name=reg_name)
                versions = client.search_model_versions(f"name='{reg_name}'")
                created = next((v for v in versions if v.run_id == run_id), versions[0])
                challenger_version = str(created.version)
                try:
                    client.update_model_version(name=reg_name, version=challenger_version, description=f"Deliverable 2 challenger candidate, val_macro_f1={val_eval['aggregate']['macro_f1']}, reg_macro_f1={reg_eval['aggregate']['macro_f1']}, fingerprint={challenger_fp}")
                    client.set_model_version_tag(reg_name, challenger_version, "bundle_fingerprint", challenger_fp)
                    client.set_model_version_tag(reg_name, challenger_version, "model_role", "challenger")
                except Exception:
                    pass

        manifest["registered_version"] = challenger_version
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        mlflow.log_artifacts(str(artifact_dir), artifact_path="challenger_bundle")

    cand_summary = dict(gate_res["candidate_summary"])
    cand_summary["run_id"] = summary["run_id"]
    resolved_batch = batch_path.resolve()
    batch_label = resolved_batch.relative_to(root).as_posix() if resolved_batch.is_relative_to(root) else batch_path.as_posix()
    comparison = {
        "completed_utc": completed_utc,
        "tracking_uri": tracking_uri,
        "drift_batch": batch_label,
        "drift_batch_sha256": batch_sha,
        "batch_records": len(batch_recs),
        "train_records_total": len(combined_train),
        "champion_version": champion_version,
        "champion_run_id": champion_metrics.get("run_id", "unknown"),
        "challenger_version": challenger_version,
        "challenger_run_id": summary["run_id"],
        "challenger_fingerprint": challenger_fp,
        "gate_passed": gate_res["passed"],
        "decision": "promote_candidate" if gate_res["passed"] else "reject",
        "gate_checks": gate_res["checks"],
        "champion_summary": gate_res["champion_summary"],
        "challenger_summary": cand_summary,
        "batch_eval_aggregate": batch_eval["aggregate"],
    }
    comp_path = comparison_out_path or (root / "evidence" / "d2" / "retrain_comparison.json")
    comp_path.parent.mkdir(parents=True, exist_ok=True)
    comp_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="Closed-loop challenger retraining on reviewed drift batch.")
    parser.add_argument("--batch", type=Path, default=REPO_ROOT / "data" / "demo_batches" / "d2_drift_batch_reviewed.jsonl")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "train.yaml")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--comparison-out", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--local-mlflow", action="store_true")
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    res = retrain_challenger(
        args.batch,
        args.config,
        out_dir=args.out_dir,
        comparison_out_path=args.comparison_out,
        epochs=args.epochs,
        local_mlflow=args.local_mlflow,
        register_challenger=not args.no_register,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
