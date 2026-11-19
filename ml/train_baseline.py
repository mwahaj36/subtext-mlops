"""Classical multi-label baseline (TF-IDF + One-Vs-Rest Logistic Regression) logged to MLflow."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion

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
from ml.validate import REPO_ROOT, ensure_valid_for_training, load_jsonl, load_yaml

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"


def _log_metrics_chunked(metrics: dict[str, float], chunk_size: int = 40) -> None:
    items = list(metrics.items())
    for i in range(0, len(items), chunk_size):
        mlflow.log_metrics(dict(items[i : i + chunk_size]))


def train_and_evaluate_baseline(
    train_cfg_path: Path | None = None,
    *,
    out_dir: Path | None = None,
    local_mlflow: bool = False,
) -> dict[str, Any]:
    """Validate dataset/splits, train TF-IDF+OvR LogReg baseline, calibrate on val, log to MLflow."""
    t0 = time.perf_counter()
    cfg_path = train_cfg_path or (REPO_ROOT / "configs" / "train.yaml")
    ensure_valid_for_training(train_cfg_path=cfg_path, check_frozen_hashes=True)

    train_cfg = load_yaml(cfg_path)
    root = cfg_path.resolve().parent.parent
    paths = train_cfg["paths"]
    taxonomy = load_yaml(root / paths["taxonomy"])
    label_order: list[str] = taxonomy["label_order"]
    seed = int(train_cfg["seed"])
    base_cfg = train_cfg["baseline"]
    cal_cfg = train_cfg["calibration"]

    processed_dir = root / paths["processed_dir"]
    train_recs = load_jsonl(processed_dir / "train.jsonl")
    val_recs = load_jsonl(processed_dir / "val.jsonl")
    reg_recs = load_jsonl(processed_dir / "course_regression.jsonl")

    x_train_text = [r["text"] for r in train_recs]
    x_val_text = [r["text"] for r in val_recs]
    x_reg_text = [r["text"] for r in reg_recs]

    y_train = encode_labels(train_recs, label_order)
    y_val = encode_labels(val_recs, label_order)

    w_min, w_max = int(base_cfg["word_ngram_range"][0]), int(base_cfg["word_ngram_range"][1])
    c_min, c_max = int(base_cfg["char_ngram_range"][0]), int(base_cfg["char_ngram_range"][1])
    max_feat = int(base_cfg["max_features"])

    vectorizer = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(w_min, w_max),
                    max_features=max_feat // 2,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(c_min, c_max),
                    max_features=max_feat // 2,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
        ]
    )
    clf = OneVsRestClassifier(
        LogisticRegression(
            C=float(base_cfg["logreg_c"]),
            max_iter=int(base_cfg["max_iter"]),
            class_weight="balanced",
            random_state=seed,
            solver="lbfgs",
        )
    )

    x_train = vectorizer.fit_transform(x_train_text)
    clf.fit(x_train, y_train)

    x_val = vectorizer.transform(x_val_text)
    x_reg = vectorizer.transform(x_reg_text)

    val_logits = np.asarray(clf.decision_function(x_val), dtype=np.float64)
    reg_logits = np.asarray(clf.decision_function(x_reg), dtype=np.float64)

    val_raw_probs = sigmoid(val_logits)
    reg_raw_probs = sigmoid(reg_logits)

    temperatures = fit_temperature_scaling(val_logits, y_val, label_order)
    val_cal_probs = apply_temperature_scaling(val_logits, temperatures, label_order)
    reg_cal_probs = apply_temperature_scaling(reg_logits, temperatures, label_order)

    thresholds = tune_thresholds(
        val_cal_probs,
        y_val,
        label_order,
        grid_min=float(cal_cfg["threshold_grid_min"]),
        grid_max=float(cal_cfg["threshold_grid_max"]),
        grid_step=float(cal_cfg["threshold_grid_step"]),
    )

    people_set, title_map = load_lexicons()
    val_eval = evaluate_multilabel(
        val_recs,
        val_cal_probs,
        thresholds,
        label_order,
        taxonomy,
        raw_probs=val_raw_probs,
        people_set=people_set,
        title_map=title_map,
    )
    reg_eval = evaluate_multilabel(
        reg_recs,
        reg_cal_probs,
        thresholds,
        label_order,
        taxonomy,
        raw_probs=reg_raw_probs,
        people_set=people_set,
        title_map=title_map,
    )

    runtime_s = round(time.perf_counter() - t0, 3)
    lineage = get_git_dvc_lineage(root)
    artifact_dir = out_dir or (root / "artifacts" / "baseline")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = artifact_dir / "baseline_bundle.joblib"
    joblib.dump(
        {
            "model_type": base_cfg["model_type"],
            "vectorizer": vectorizer,
            "classifier": clf,
            "label_order": label_order,
            "temperatures": temperatures,
            "thresholds": thresholds,
            "lineage": lineage,
        },
        bundle_path,
    )

    tracking_uri = configure_mlflow(train_cfg, local_only=local_mlflow)
    completed_utc = dt.datetime.now(dt.timezone.utc).isoformat()

    with mlflow.start_run(run_name="d1_classical_baseline_tfidf_ovr_logreg") as run:
        run_id = run.info.run_id
        mlflow.set_tags(
            {
                "run_role": "baseline",
                "stage": "deliverable_1",
                "model_family": "tfidf_ovr_logreg",
                "git_commit": lineage["git_commit"],
                "raw_dvc_md5": lineage["raw_dvc_md5"],
                "raw_sha256": lineage["raw_sha256"],
                "taxonomy_version": lineage["taxonomy_version"],
                "route_policy_version": lineage["route_policy_version"],
                "completed_utc": completed_utc,
            }
        )
        mlflow.log_params(
            {
                "model_type": base_cfg["model_type"],
                "seed": seed,
                "word_ngram_range": f"{w_min},{w_max}",
                "char_ngram_range": f"{c_min},{c_max}",
                "max_features": max_feat,
                "logreg_c": base_cfg["logreg_c"],
                "max_iter": base_cfg["max_iter"],
                "calibration_method": cal_cfg["method"],
                "train_records": len(train_recs),
                "val_records": len(val_recs),
                "regression_records": len(reg_recs),
            }
        )

        flat_metrics: dict[str, float] = {"runtime_seconds": runtime_s}
        for prefix, ev in (("val", val_eval), ("reg", reg_eval)):
            for k, v in ev["aggregate"].items():
                flat_metrics[f"{prefix}_{k}"] = float(v)
            for s_name, s_obj in ev["slices"].items():
                if "micro_f1" in s_obj:
                    flat_metrics[f"{prefix}_slice_{s_name}_micro_f1"] = float(s_obj["micro_f1"])
                    flat_metrics[f"{prefix}_slice_{s_name}_count"] = float(s_obj["count"])
                elif "macro_f1" in s_obj:
                    flat_metrics[f"{prefix}_slice_{s_name}_macro_f1"] = float(s_obj["macro_f1"])
            for lbl, l_obj in ev["per_label"].items():
                flat_metrics[f"{prefix}_f1_{lbl}"] = float(l_obj["f1"])
        _log_metrics_chunked(flat_metrics)

        summary = {
            "run_id": run_id,
            "run_name": "d1_classical_baseline_tfidf_ovr_logreg",
            "run_role": "baseline",
            "completed_utc": completed_utc,
            "tracking_uri": tracking_uri,
            "model_type": base_cfg["model_type"],
            "runtime_seconds": runtime_s,
            "lineage": lineage,
            "hyperparameters": {
                "seed": seed,
                "word_ngram_range": [w_min, w_max],
                "char_ngram_range": [c_min, c_max],
                "max_features": max_feat,
                "logreg_c": float(base_cfg["logreg_c"]),
                "max_iter": int(base_cfg["max_iter"]),
            },
            "calibration": {
                "method": cal_cfg["method"],
                "temperatures": temperatures,
                "thresholds": thresholds,
            },
            "val": val_eval,
            "course_regression": reg_eval,
        }
        metrics_path = artifact_dir / "metrics.json"
        metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
        val_report_path = root / paths["validation_report"]
        if val_report_path.exists():
            mlflow.log_artifact(str(val_report_path), artifact_path="validation")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and log the classical multi-label baseline.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "train.yaml")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--local-mlflow", action="store_true")
    args = parser.parse_args()

    res = train_and_evaluate_baseline(args.config, out_dir=args.out_dir, local_mlflow=args.local_mlflow)
    print(
        json.dumps(
            {
                "run_id": res["run_id"],
                "model_type": res["model_type"],
                "val_macro_f1": res["val"]["aggregate"]["macro_f1"],
                "val_micro_f1": res["val"]["aggregate"]["micro_f1"],
                "reg_macro_f1": res["course_regression"]["aggregate"]["macro_f1"],
                "reg_micro_f1": res["course_regression"]["aggregate"]["micro_f1"],
                "extraction_micro_f1": res["course_regression"]["aggregate"]["extraction_micro_f1"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
