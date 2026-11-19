"""Candidate comparison, MLflow Model Registry packaging, and fresh-process reload verification."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import joblib
import jsonschema
import mlflow
import mlflow.pyfunc
import numpy as np
import torch
from mlflow.models.signature import ModelSignature
from mlflow.tracking import MlflowClient
from mlflow.types.schema import ColSpec, Schema
from transformers import AutoModel, AutoTokenizer

from ml.evaluate import (
    apply_temperature_scaling,
    apply_thresholds,
    configure_mlflow,
    select_route,
)
from ml.extract import extract_constraints, load_lexicons
from ml.train import MiniLMMultiLabelModel
from ml.validate import REPO_ROOT, load_yaml

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SubtextQueryUnderstandingPyFunc(mlflow.pyfunc.PythonModel):
    """Self-contained MLflow PyFunc bundle for multi-label intent prediction, extraction, and routing."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.taxonomy = load_yaml(Path(context.artifacts["taxonomy"]))
        self.routing = load_yaml(Path(context.artifacts["routing"]))
        self.label_order: list[str] = self.taxonomy["label_order"]
        self.fine_to_broad: dict[str, str] = self.taxonomy["fine_to_broad"]
        self.compound_min = int(self.taxonomy["compound_rule"]["min_distinct_broad_categories"])
        self.people_set, self.title_map = load_lexicons()

        bundle_dir = Path(context.artifacts["bundle_dir"])
        manifest_file = bundle_dir / "bundle_manifest.json"
        if manifest_file.exists():
            self.manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.model_type = self.manifest["model_type"]
            self.temperatures = self.manifest["temperatures"]
            self.thresholds = self.manifest["thresholds"]
            self.max_length = int(self.manifest["max_length"])
            enc_dir = bundle_dir / "encoder"
            self.tokenizer = AutoTokenizer.from_pretrained(enc_dir)
            backbone = AutoModel.from_pretrained(enc_dir)
            self.model = MiniLMMultiLabelModel(
                encoder=backbone,
                hidden_size=int(self.manifest["hidden_size"]),
                num_labels=len(self.label_order),
                dropout=float(self.manifest.get("dropout", 0.1)),
            )
            state = torch.load(bundle_dir / "head.pt", map_location="cpu", weights_only=True)
            self.model.head.load_state_dict(state)
            self.model.eval()
        else:
            raw_b = joblib.load(bundle_dir / "baseline_bundle.joblib")
            self.manifest = raw_b
            self.model_type = raw_b["model_type"]
            self.temperatures = raw_b["temperatures"]
            self.thresholds = raw_b["thresholds"]
            self.vectorizer = raw_b["vectorizer"]
            self.classifier = raw_b["classifier"]

    def _extract_texts(self, model_input: Any) -> list[str]:
        if hasattr(model_input, "columns"):
            if "text" in model_input.columns:
                return [str(x) for x in model_input["text"].tolist()]
            return [str(x) for x in model_input.iloc[:, 0].tolist()]
        if isinstance(model_input, list):
            return [str(x["text"] if isinstance(x, dict) else x) for x in model_input]
        if isinstance(model_input, dict):
            val = model_input.get("text", [])
            return [str(val)] if isinstance(val, str) else [str(x) for x in val]
        return [str(model_input)]

    def predict(self, context, model_input):
        texts = self._extract_texts(model_input)
        if self.model_type == "minilm_multilabel":
            with torch.no_grad():
                enc = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                logits = self.model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                ).cpu().numpy().astype(np.float64)
        else:
            x_vec = self.vectorizer.transform(texts)
            logits = np.asarray(self.classifier.decision_function(x_vec), dtype=np.float64)

        probs = apply_temperature_scaling(logits, self.temperatures, self.label_order)
        preds = apply_thresholds(probs, self.thresholds, self.label_order)

        outputs: list[str] = []
        for idx, text in enumerate(texts):
            fine_preds = [
                self.label_order[j] for j in range(len(self.label_order)) if preds[idx, j] == 1
            ]
            broad_preds = sorted({self.fine_to_broad[lbl] for lbl in fine_preds})
            is_compound = len(broad_preds) >= self.compound_min
            scores = {
                self.label_order[j]: round(float(probs[idx, j]), 4)
                for j in range(len(self.label_order))
            }
            extraction = extract_constraints(
                text, self.taxonomy, people_set=self.people_set, title_map=self.title_map
            )
            route, reason = select_route(fine_preds, extraction, self.routing)
            payload = {
                "request_id": f"req_{uuid.uuid4().hex[:12]}",
                "schema_version": "1.0.0",
                "taxonomy_version": str(self.taxonomy["version"]),
                "route_policy_version": str(self.routing["version"]),
                "model_id": self.model_type,
                "model_version": str(self.manifest.get("lineage", {}).get("git_commit", "1"))[:12],
                "predicted_fine_labels": fine_preds,
                "predicted_broad_labels": broad_preds,
                "is_compound": is_compound,
                "label_scores": scores,
                "applied_thresholds": {lbl: float(self.thresholds[lbl]) for lbl in self.label_order},
                "extraction": extraction,
                "selected_route": route,
                "route_reason": reason,
                "llm_fallback": {
                    "called": False,
                    "status": "not_called",
                    "reason": "D1 offline small-model execution",
                },
            }
            outputs.append(json.dumps(payload))
        return outputs


def _validate_prediction_payload(payload: dict[str, Any], root: Path) -> None:
    pred_schema = json.loads((root / "schemas" / "prediction_contract.schema.json").read_text(encoding="utf-8"))
    ext_schema = json.loads((root / "schemas" / "extraction.schema.json").read_text(encoding="utf-8"))
    inline = dict(pred_schema)
    inline["properties"] = dict(pred_schema["properties"])
    inline["properties"]["extraction"] = ext_schema
    jsonschema.Draft202012Validator(inline).validate(payload)


def verify_reload_in_current_process(model_uri: str, root: Path | None = None) -> list[dict[str, Any]]:
    """Load a saved or registered MLflow pyfunc model and verify contract-compliant predictions."""
    repo = root or REPO_ROOT
    loaded = mlflow.pyfunc.load_model(model_uri)
    sample_queries = [
        "Korean crime thriller starring Song Kang-ho with no gore",
        "Lonely late-night Tokyo neon atmosphere like Lost in Translation",
        "90s animated fantasy directed by Hayao Miyazaki to soothe my mood",
    ]
    raw_outs = loaded.predict([{"text": q} for q in sample_queries])
    parsed: list[dict[str, Any]] = []
    for q, item in zip(sample_queries, raw_outs, strict=True):
        obj = json.loads(item) if isinstance(item, str) else item
        _validate_prediction_payload(obj, repo)
        obj["input_query"] = q
        parsed.append(obj)
    return parsed


def compare_and_register(
    train_cfg_path: Path | None = None,
    *,
    local_mlflow: bool = False,
) -> dict[str, Any]:
    """Compare baseline and primary runs by validation rule, register winner idempotently, and test reload."""
    cfg_path = train_cfg_path or (REPO_ROOT / "configs" / "train.yaml")
    train_cfg = load_yaml(cfg_path)
    root = cfg_path.resolve().parent.parent

    base_metrics_path = root / "artifacts" / "baseline" / "metrics.json"
    prim_metrics_path = root / "artifacts" / "primary" / "metrics.json"
    if not base_metrics_path.exists() or not prim_metrics_path.exists():
        raise FileNotFoundError("Both baseline and primary metrics.json must exist before registration.")

    base_m = json.loads(base_metrics_path.read_text(encoding="utf-8"))
    prim_m = json.loads(prim_metrics_path.read_text(encoding="utf-8"))

    # Verify baseline ran before primary
    if base_m["completed_utc"] > prim_m["completed_utc"]:
        raise RuntimeError(
            f"Baseline timestamp ({base_m['completed_utc']}) must precede primary ({prim_m['completed_utc']})."
        )

    primary_metric = train_cfg["selection"]["primary_metric"]  # val_macro_f1
    tie_breaker = train_cfg["selection"]["tie_breaker"]  # val_micro_f1
    reg_name = train_cfg["selection"]["registered_model_name"]

    b_key = (
        float(base_m["val"]["aggregate"]["macro_f1"]),
        float(base_m["val"]["aggregate"]["micro_f1"]),
    )
    p_key = (
        float(prim_m["val"]["aggregate"]["macro_f1"]),
        float(prim_m["val"]["aggregate"]["micro_f1"]),
    )

    if p_key >= b_key:
        selected = prim_m
        selected_dir = root / "artifacts" / "primary"
    else:
        selected = base_m
        selected_dir = root / "artifacts" / "baseline"

    fingerprint_src = "|".join(
        [
            selected["model_type"],
            selected["lineage"]["raw_sha256"],
            str(selected["val"]["aggregate"]["macro_f1"]),
            str(selected["course_regression"]["aggregate"]["macro_f1"]),
        ]
    )
    fingerprint = hashlib.sha256(fingerprint_src.encode("utf-8")).hexdigest()[:16]

    reg_dir = root / "artifacts" / "registry"
    local_pyfunc_dir = reg_dir / "pyfunc_model"
    if local_pyfunc_dir.exists():
        shutil.rmtree(local_pyfunc_dir)
    reg_dir.mkdir(parents=True, exist_ok=True)

    signature = ModelSignature(
        inputs=Schema([ColSpec("string", "text")]),
        outputs=Schema([ColSpec("string", "prediction_json")]),
    )
    artifacts_map = {
        "bundle_dir": str(selected_dir),
        "taxonomy": str(root / "configs" / "taxonomy.yaml"),
        "routing": str(root / "configs" / "routing.yaml"),
    }

    mlflow.pyfunc.save_model(
        path=str(local_pyfunc_dir),
        python_model=SubtextQueryUnderstandingPyFunc(),
        artifacts=artifacts_map,
        signature=signature,
    )

    tracking_uri = configure_mlflow(train_cfg, local_only=local_mlflow)
    client = MlflowClient()
    registered_version: str = "local_only"
    registered_model_uri: str = str(local_pyfunc_dir)
    reused_existing = False

    if not tracking_uri.startswith("file:"):
        try:
            existing_versions = client.search_model_versions(f"name='{reg_name}'")
        except Exception:
            existing_versions = []

        for mv in existing_versions:
            mv_tags = getattr(mv, "tags", {}) or {}
            mv_desc = getattr(mv, "description", "") or ""
            if mv_tags.get("bundle_fingerprint") == fingerprint or f"fingerprint={fingerprint}" in mv_desc:
                registered_version = str(mv.version)
                registered_model_uri = f"models:/{reg_name}/{registered_version}"
                reused_existing = True
                break

        if not reused_existing:
            with mlflow.start_run(run_name=f"d1_register_{selected['model_type']}") as reg_run:
                mlflow.set_tags(
                    {
                        "run_role": "registration",
                        "stage": "deliverable_1",
                        "selected_run_id": selected["run_id"],
                        "selected_model_type": selected["model_type"],
                        "bundle_fingerprint": fingerprint,
                        "git_commit": selected["lineage"]["git_commit"],
                        "raw_dvc_md5": selected["lineage"]["raw_dvc_md5"],
                    }
                )
                model_info = mlflow.pyfunc.log_model(
                    artifact_path="subtext_query_understanding",
                    python_model=SubtextQueryUnderstandingPyFunc(),
                    artifacts=artifacts_map,
                    signature=signature,
                    registered_model_name=reg_name,
                )
                # Retrieve the created version
                versions = client.search_model_versions(f"name='{reg_name}'")
                matching = [v for v in versions if v.run_id == reg_run.info.run_id]
                latest_v = matching[0].version if matching else (versions[0].version if versions else "1")
                registered_version = str(latest_v)
                registered_model_uri = f"models:/{reg_name}/{registered_version}"
                try:
                    client.update_model_version(
                        name=reg_name,
                        version=registered_version,
                        description=(
                            f"Deliverable 1 selected model ({selected['model_type']}), "
                            f"val_macro_f1={selected['val']['aggregate']['macro_f1']}, "
                            f"reg_macro_f1={selected['course_regression']['aggregate']['macro_f1']}, "
                            f"fingerprint={fingerprint}"
                        ),
                    )
                except Exception:
                    pass
                _ = model_info

    # Verify reload in a fresh Python subprocess
    cmd = [
        sys.executable,
        "-m",
        "ml.register",
        "--verify-only",
        "--model-uri",
        str(local_pyfunc_dir),
    ]
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=True)
    fresh_predictions = json.loads(proc.stdout)

    summary = {
        "registered_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "tracking_uri": tracking_uri,
        "selection_rule": {
            "primary_metric": primary_metric,
            "tie_breaker": tie_breaker,
        },
        "comparison": {
            "baseline": {
                "run_id": base_m["run_id"],
                "model_type": base_m["model_type"],
                "completed_utc": base_m["completed_utc"],
                "val_macro_f1": base_m["val"]["aggregate"]["macro_f1"],
                "val_micro_f1": base_m["val"]["aggregate"]["micro_f1"],
                "val_subset_accuracy": base_m["val"]["aggregate"]["subset_accuracy"],
                "val_ece": base_m["val"]["aggregate"]["ece"],
                "val_uncalibrated_ece": base_m["val"]["aggregate"]["uncalibrated_ece"],
                "reg_macro_f1": base_m["course_regression"]["aggregate"]["macro_f1"],
                "reg_micro_f1": base_m["course_regression"]["aggregate"]["micro_f1"],
                "reg_subset_accuracy": base_m["course_regression"]["aggregate"]["subset_accuracy"],
                "reg_ece": base_m["course_regression"]["aggregate"]["ece"],
                "extraction_micro_f1": base_m["course_regression"]["aggregate"]["extraction_micro_f1"],
            },
            "primary": {
                "run_id": prim_m["run_id"],
                "model_type": prim_m["model_type"],
                "completed_utc": prim_m["completed_utc"],
                "val_macro_f1": prim_m["val"]["aggregate"]["macro_f1"],
                "val_micro_f1": prim_m["val"]["aggregate"]["micro_f1"],
                "val_subset_accuracy": prim_m["val"]["aggregate"]["subset_accuracy"],
                "val_ece": prim_m["val"]["aggregate"]["ece"],
                "val_uncalibrated_ece": prim_m["val"]["aggregate"]["uncalibrated_ece"],
                "reg_macro_f1": prim_m["course_regression"]["aggregate"]["macro_f1"],
                "reg_micro_f1": prim_m["course_regression"]["aggregate"]["micro_f1"],
                "reg_subset_accuracy": prim_m["course_regression"]["aggregate"]["subset_accuracy"],
                "reg_ece": prim_m["course_regression"]["aggregate"]["ece"],
                "extraction_micro_f1": prim_m["course_regression"]["aggregate"]["extraction_micro_f1"],
            },
        },
        "selected_model": {
            "model_type": selected["model_type"],
            "source_run_id": selected["run_id"],
            "registered_model_name": reg_name,
            "registered_version": registered_version,
            "registered_model_uri": registered_model_uri,
            "local_pyfunc_path": "artifacts/registry/pyfunc_model",
            "bundle_fingerprint": fingerprint,
            "reused_existing_version": reused_existing,
            "label_order": selected["calibration"]["thresholds"] and list(selected["calibration"]["thresholds"].keys()),
            "thresholds": selected["calibration"]["thresholds"],
            "temperatures": selected["calibration"]["temperatures"],
        },
        "fresh_process_reload_verified": True,
        "fresh_process_sample_predictions": fresh_predictions,
    }
    summary_path = reg_dir / "registration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare candidate runs, register winner, and verify reload.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "train.yaml")
    parser.add_argument("--local-mlflow", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--model-uri", type=str, default=str(REPO_ROOT / "artifacts" / "registry" / "pyfunc_model"))
    args = parser.parse_args()

    if args.verify_only:
        preds = verify_reload_in_current_process(args.model_uri)
        print(json.dumps(preds, indent=2))
        return 0

    summary = compare_and_register(args.config, local_mlflow=args.local_mlflow)
    print(
        json.dumps(
            {
                "selected_model": summary["selected_model"]["model_type"],
                "registered_model_name": summary["selected_model"]["registered_model_name"],
                "registered_version": summary["selected_model"]["registered_version"],
                "registered_model_uri": summary["selected_model"]["registered_model_uri"],
                "reused_existing_version": summary["selected_model"]["reused_existing_version"],
                "fresh_process_reload_verified": summary["fresh_process_reload_verified"],
                "comparison": summary["comparison"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
