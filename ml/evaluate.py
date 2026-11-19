"""Evaluation, temperature-scaling calibration, threshold tuning, and MLflow lineage helpers."""
from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score

from ml.extract import evaluate_extractions, extract_constraints, load_lexicons
from ml.validate import REPO_ROOT, load_yaml, sha256_file

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"


def sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def logit(p: np.ndarray) -> np.ndarray:
    p_clip = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p_clip / (1.0 - p_clip))


def encode_labels(records: list[dict[str, Any]], label_order: list[str]) -> np.ndarray:
    idx_map = {lbl: i for i, lbl in enumerate(label_order)}
    y = np.zeros((len(records), len(label_order)), dtype=np.int32)
    for row_i, rec in enumerate(records):
        for lbl in rec["fine_labels"]:
            y[row_i, idx_map[lbl]] = 1
    return y


def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error averaged across all binary label decisions."""
    flat_y = y_true.reshape(-1).astype(np.float64)
    flat_p = probs.reshape(-1).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(flat_y)
    for b_idx in range(n_bins):
        lo, hi = bins[b_idx], bins[b_idx + 1]
        if b_idx == n_bins - 1:
            mask = (flat_p >= lo) & (flat_p <= hi)
        else:
            mask = (flat_p >= lo) & (flat_p < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            acc = float(np.mean(flat_y[mask]))
            conf = float(np.mean(flat_p[mask]))
            ece += (cnt / n_total) * abs(acc - conf)
    return round(float(ece), 4)


def compute_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    return round(float(np.mean((probs - y_true) ** 2)), 4)


def fit_temperature_scaling(val_logits: np.ndarray, val_y: np.ndarray, label_order: list[str]) -> dict[str, float]:
    """Fit per-label temperature T_j > 0 minimizing binary cross-entropy on validation logits."""
    temps: dict[str, float] = {}
    for j, lbl in enumerate(label_order):
        z = val_logits[:, j].astype(np.float64)
        y = val_y[:, j].astype(np.float64)

        def nll(temp: float) -> float:
            p = np.clip(sigmoid(z / temp), 1e-7, 1.0 - 1e-7)
            return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

        res = minimize_scalar(nll, bounds=(0.25, 4.0), method="bounded")
        temps[lbl] = round(float(res.x if res.success else 1.0), 4)
    return temps


def apply_temperature_scaling(logits: np.ndarray, temps: dict[str, float], label_order: list[str]) -> np.ndarray:
    t_vec = np.array([temps.get(lbl, 1.0) for lbl in label_order], dtype=np.float64)
    return sigmoid(logits / t_vec[None, :])


def tune_thresholds(
    val_probs: np.ndarray,
    val_y: np.ndarray,
    label_order: list[str],
    grid_min: float = 0.20,
    grid_max: float = 0.80,
    grid_step: float = 0.02,
) -> dict[str, float]:
    """Tune per-label decision thresholds on validation set to maximize per-label F1."""
    grid = np.arange(grid_min, grid_max + 1e-9, grid_step)
    thresholds: dict[str, float] = {}
    for j, lbl in enumerate(label_order):
        y_col = val_y[:, j]
        p_col = val_probs[:, j]
        best_th = 0.50
        best_score = -1.0
        for th in grid:
            pred = (p_col >= th).astype(np.int32)
            score = float(f1_score(y_col, pred, zero_division=0))
            # Tie-break closer to 0.50
            if score > best_score + 1e-9 or (
                abs(score - best_score) <= 1e-9 and abs(th - 0.50) < abs(best_th - 0.50)
            ):
                best_score = score
                best_th = float(th)
        thresholds[lbl] = round(best_th, 2)
    return thresholds


def apply_thresholds(probs: np.ndarray, thresholds: dict[str, float], label_order: list[str]) -> np.ndarray:
    th_vec = np.array([thresholds.get(lbl, 0.50) for lbl in label_order], dtype=np.float64)
    preds = (probs >= th_vec[None, :]).astype(np.int32)
    # Ensure at least the top-scoring label is predicted if all fall below threshold
    empty_rows = np.where(np.sum(preds, axis=1) == 0)[0]
    for r_idx in empty_rows:
        top_j = int(np.argmax(probs[r_idx]))
        preds[r_idx, top_j] = 1
    return preds


def evaluate_multilabel(
    records: list[dict[str, Any]],
    probs: np.ndarray,
    thresholds: dict[str, float],
    label_order: list[str],
    taxonomy: dict[str, Any],
    *,
    raw_probs: np.ndarray | None = None,
    people_set: set[str] | None = None,
    title_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compute aggregate, per-label, slice, calibration, extraction, and error-summary metrics."""
    y_true = encode_labels(records, label_order)
    y_pred = apply_thresholds(probs, thresholds, label_order)

    macro_f1 = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)
    micro_f1 = round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 4)
    samples_f1 = round(float(f1_score(y_true, y_pred, average="samples", zero_division=0)), 4)
    macro_p = round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4)
    macro_r = round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4)
    micro_p = round(float(precision_score(y_true, y_pred, average="micro", zero_division=0)), 4)
    micro_r = round(float(recall_score(y_true, y_pred, average="micro", zero_division=0)), 4)
    subset_acc = round(float(np.mean(np.all(y_true == y_pred, axis=1))), 4)
    h_loss = round(float(hamming_loss(y_true, y_pred)), 4)
    brier = compute_brier(y_true, probs)
    ece = compute_ece(y_true, probs)

    raw_brier = compute_brier(y_true, raw_probs) if raw_probs is not None else brier
    raw_ece = compute_ece(y_true, raw_probs) if raw_probs is not None else ece

    per_label: dict[str, Any] = {}
    for j, lbl in enumerate(label_order):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        per_label[lbl] = {
            "f1": round(float(f1_score(yt, yp, zero_division=0)), 4),
            "precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
            "recall": round(float(recall_score(yt, yp, zero_division=0)), 4),
            "support": int(np.sum(yt)),
            "predicted_positives": int(np.sum(yp)),
            "threshold": thresholds.get(lbl, 0.50),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    # Rare labels slice (5 lowest-support labels in this evaluation partition)
    rare_labels = sorted(label_order, key=lambda lbl: (per_label[lbl]["support"], lbl))[:5]
    rare_f1 = round(float(np.mean([per_label[lbl]["f1"] for lbl in rare_labels])), 4)

    def _slice_stats(indices: list[int]) -> dict[str, Any]:
        if not indices:
            return {"count": 0, "micro_f1": 0.0, "subset_accuracy": 0.0}
        st = y_true[indices]
        sp = y_pred[indices]
        return {
            "count": len(indices),
            "micro_f1": round(float(f1_score(st, sp, average="micro", zero_division=0)), 4),
            "subset_accuracy": round(float(np.mean(np.all(st == sp, axis=1))), 4),
        }

    single_idx = [i for i, r in enumerate(records) if len(r["fine_labels"]) == 1]
    multi_idx = [i for i, r in enumerate(records) if len(r["fine_labels"]) >= 2]
    compound_idx = [i for i, r in enumerate(records) if r["is_compound"]]
    neg_idx = [
        i
        for i, r in enumerate(records)
        if "negation_genre" in r["fine_labels"] or "negation_content" in r["fine_labels"]
    ]
    person_idx = [
        i
        for i, r in enumerate(records)
        if "person_actor" in r["fine_labels"] or "person_director" in r["fine_labels"]
    ]

    if people_set is None or title_map is None:
        people_set, title_map = load_lexicons()
    ext_preds = [
        extract_constraints(r["text"], taxonomy, people_set=people_set, title_map=title_map)
        for r in records
    ]
    ext_eval = evaluate_extractions(records, ext_preds)

    errors: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        if not np.all(y_true[i] == y_pred[i]) and len(errors) < 15:
            pred_lbls = [label_order[j] for j in range(len(label_order)) if y_pred[i, j] == 1]
            top_scores = {
                label_order[j]: round(float(probs[i, j]), 3)
                for j in np.argsort(-probs[i])[:5]
            }
            errors.append(
                {
                    "query_id": rec["query_id"],
                    "group_id": rec["group_id"],
                    "text": rec["text"],
                    "gold_labels": rec["fine_labels"],
                    "predicted_labels": pred_lbls,
                    "top_scores": top_scores,
                }
            )

    return {
        "num_records": len(records),
        "aggregate": {
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "samples_f1": samples_f1,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "subset_accuracy": subset_acc,
            "hamming_loss": h_loss,
            "brier_score": brier,
            "ece": ece,
            "uncalibrated_brier_score": raw_brier,
            "uncalibrated_ece": raw_ece,
            "extraction_micro_f1": ext_eval["overall_micro"]["f1"],
            "extraction_micro_precision": ext_eval["overall_micro"]["precision"],
            "extraction_micro_recall": ext_eval["overall_micro"]["recall"],
        },
        "slices": {
            "single_intent": _slice_stats(single_idx),
            "multi_intent": _slice_stats(multi_idx),
            "compound_broad": _slice_stats(compound_idx),
            "negation": _slice_stats(neg_idx),
            "person": _slice_stats(person_idx),
            "rare_labels": {"labels": rare_labels, "macro_f1": rare_f1},
        },
        "per_label": per_label,
        "extraction": ext_eval,
        "error_examples": errors,
    }


def select_route(
    predicted_fine_labels: list[str],
    extraction: dict[str, Any],
    routing_cfg: dict[str, Any],
) -> tuple[str, str]:
    """Apply versioned deterministic routing contract from predicted labels and constraints."""
    lbl_set = set(predicted_fine_labels)
    if extraction.get("reference_titles") or (lbl_set & {"similarity_plot", "similarity_vibe"}):
        return "similarity_anchor", "Reference title or similarity intent matched"
    hard_constraints = (
        bool(extraction.get("actors"))
        or bool(extraction.get("directors"))
        or bool(extraction.get("genres"))
        or extraction.get("year_min") is not None
        or extraction.get("year_max") is not None
        or extraction.get("language") is not None
        or bool(extraction.get("excluded_genres"))
        or bool(extraction.get("excluded_terms"))
    )
    if hard_constraints or (
        lbl_set
        & {
            "person_actor",
            "person_director",
            "attribute_genre",
            "attribute_era",
            "attribute_language_country",
            "negation_genre",
            "negation_content",
        }
    ):
        return "structured_filter", "Explicit entity, metadata, or negation constraints matched"
    if lbl_set & {"affective_outcome", "affective_tone", "sensory_visual", "sensory_pacing", "thematic_plot"}:
        return "hybrid_concept", "Affective, sensory, or thematic intent matched for hybrid concept scoring"
    return "dense_semantic", "Default open-ended semantic vector retrieval"


def get_git_dvc_lineage(root: Path | None = None) -> dict[str, str]:
    repo = root or REPO_ROOT
    try:
        git_sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        )
    except Exception:
        git_sha = "unknown"

    dvc_pointer = repo / "data" / "raw" / "d1_queries.jsonl.dvc"
    raw_dvc_md5 = "unknown"
    if dvc_pointer.exists():
        dvc_data = load_yaml(dvc_pointer)
        outs = dvc_data.get("outs") or []
        if outs and isinstance(outs[0], dict):
            raw_dvc_md5 = str(outs[0].get("md5", "unknown"))

    tax = load_yaml(repo / "configs" / "taxonomy.yaml")
    routing = load_yaml(repo / "configs" / "routing.yaml")
    raw_file = repo / "data" / "raw" / "d1_queries.jsonl"
    split_manifest = repo / "data" / "manifests" / "split_manifest.json"
    split_hashes: dict[str, str] = {}
    if split_manifest.exists():
        split_hashes = json.loads(split_manifest.read_text(encoding="utf-8")).get("split_sha256", {})

    return {
        "git_commit": git_sha,
        "raw_dvc_md5": raw_dvc_md5,
        "raw_sha256": sha256_file(raw_file) if raw_file.exists() else "missing",
        "train_sha256": split_hashes.get("train", "missing"),
        "val_sha256": split_hashes.get("val", "missing"),
        "course_regression_sha256": split_hashes.get("course_regression", "missing"),
        "taxonomy_version": str(tax.get("version", "1.0.0")),
        "route_policy_version": str(routing.get("version", "1.0.0")),
        "python_version": platform.python_version(),
        "platform": f"{platform.system()}-{platform.release()}-{platform.machine()}",
    }


def configure_mlflow(train_cfg: dict[str, Any], *, local_only: bool = False) -> str:
    """Configure MLflow tracking URI (DagsHub remote by default when credentials exist, else local mlruns)."""
    import mlflow

    load_dotenv(REPO_ROOT / ".env")
    uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    user = os.getenv("MLFLOW_TRACKING_USERNAME", "").strip()
    pw = os.getenv("MLFLOW_TRACKING_PASSWORD", "").strip()

    if local_only or not (uri and user and pw):
        local_dir = (REPO_ROOT / "mlruns").resolve()
        local_uri = local_dir.as_uri()
        mlflow.set_tracking_uri(local_uri)
        tracking_target = local_uri
    else:
        mlflow.set_tracking_uri(uri)
        tracking_target = uri

    exp_name = train_cfg["selection"]["mlflow_experiment"]
    mlflow.set_experiment(exp_name)
    return tracking_target
