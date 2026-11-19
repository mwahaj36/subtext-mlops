"""Primary MiniLM fine-grained multi-label encoder training, calibration, and MLflow logging."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
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
from ml.train_baseline import _log_metrics_chunked
from ml.validate import REPO_ROOT, ensure_valid_for_training, load_jsonl, load_yaml

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class QueryDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray) -> None:
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        return self.texts[idx], self.labels[idx]


class MiniLMMultiLabelModel(nn.Module):
    """Mean-pooled MiniLM encoder with LayerNorm and a supervised linear multi-label head."""

    def __init__(self, encoder: nn.Module, hidden_size: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    @staticmethod
    def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(out.last_hidden_state, attention_mask)
        return self.head(pooled)


@torch.no_grad()
def predict_logits(
    model: MiniLMMultiLabelModel,
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    all_logits: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        )
        all_logits.append(logits.cpu().numpy())
    return np.vstack(all_logits).astype(np.float64)


def train_and_evaluate_primary(
    train_cfg_path: Path | None = None,
    *,
    out_dir: Path | None = None,
    local_mlflow: bool = False,
) -> dict[str, Any]:
    """Validate dataset/splits, fine-tune MiniLM multi-label model, calibrate, and log to MLflow."""
    t0 = time.perf_counter()
    cfg_path = train_cfg_path or (REPO_ROOT / "configs" / "train.yaml")
    ensure_valid_for_training(train_cfg_path=cfg_path, check_frozen_hashes=True)

    train_cfg = load_yaml(cfg_path)
    root = cfg_path.resolve().parent.parent
    paths = train_cfg["paths"]
    taxonomy = load_yaml(root / paths["taxonomy"])
    label_order: list[str] = taxonomy["label_order"]
    num_labels = len(label_order)

    seed = int(train_cfg["seed"])
    set_seed(seed)

    prim_cfg = train_cfg["primary"]
    cal_cfg = train_cfg["calibration"]
    model_name = str(prim_cfg["model_name"])
    model_rev = str(prim_cfg["model_revision"])
    max_len = int(prim_cfg["max_length"])
    batch_size = int(prim_cfg["batch_size"])
    epochs = int(prim_cfg["epochs"])
    lr = float(prim_cfg["learning_rate"])
    weight_decay = float(prim_cfg["weight_decay"])
    pos_weight_cap = float(prim_cfg["pos_weight_cap"])
    dropout = float(prim_cfg["dropout"])

    processed_dir = root / paths["processed_dir"]
    train_recs = load_jsonl(processed_dir / "train.jsonl")
    val_recs = load_jsonl(processed_dir / "val.jsonl")
    reg_recs = load_jsonl(processed_dir / "course_regression.jsonl")

    x_train_text = [r["text"] for r in train_recs]
    x_val_text = [r["text"] for r in val_recs]
    x_reg_text = [r["text"] for r in reg_recs]

    y_train = encode_labels(train_recs, label_order)
    y_val = encode_labels(val_recs, label_order)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_rev)
    backbone = AutoModel.from_pretrained(model_name, revision=model_rev)
    hidden_size = int(backbone.config.hidden_size)

    model = MiniLMMultiLabelModel(
        encoder=backbone,
        hidden_size=hidden_size,
        num_labels=num_labels,
        dropout=dropout,
    ).to(device)

    pos_counts = np.maximum(np.sum(y_train, axis=0), 1.0)
    neg_counts = np.maximum(len(y_train) - pos_counts, 1.0)
    pos_weights = np.clip(neg_counts / pos_counts, 1.0, pos_weight_cap)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weights, dtype=torch.float32, device=device)
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": lr},
            {"params": model.head.parameters(), "lr": lr * 20.0},
        ],
        weight_decay=weight_decay,
    )

    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(
        QueryDataset(x_train_text, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=gen,
    )
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    best_val_f1 = -1.0
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    epoch_history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for batch_texts, batch_y in train_loader:
            enc = tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            optimizer.zero_grad()
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
            loss = criterion(logits, batch_y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())
            n_batches += 1

        avg_loss = round(running_loss / max(1, n_batches), 4)
        val_logits_ep = predict_logits(model, tokenizer, x_val_text, max_len, batch_size, device)
        val_preds_ep = (sigmoid(val_logits_ep) >= 0.50).astype(np.int32)
        val_f1_ep = round(float(f1_score(y_val, val_preds_ep, average="macro", zero_division=0)), 4)
        epoch_history.append({"epoch": epoch, "train_loss": avg_loss, "val_raw_macro_f1": val_f1_ep})

        if val_f1_ep >= best_val_f1:
            best_val_f1 = val_f1_ep
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    val_logits = predict_logits(model, tokenizer, x_val_text, max_len, batch_size, device)
    reg_logits = predict_logits(model, tokenizer, x_reg_text, max_len, batch_size, device)

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
    artifact_dir = out_dir or (root / "artifacts" / "primary")
    encoder_dir = artifact_dir / "encoder"
    encoder_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(encoder_dir)
    model.encoder.save_pretrained(encoder_dir)
    head_path = artifact_dir / "head.pt"
    torch.save(model.head.state_dict(), head_path)

    baseline_metrics_path = root / "artifacts" / "baseline" / "metrics.json"
    baseline_run_id = "none"
    if baseline_metrics_path.exists():
        baseline_run_id = json.loads(
            baseline_metrics_path.read_text(encoding="utf-8")
        ).get("run_id", "none")

    bundle_manifest = {
        "model_type": prim_cfg["model_type"],
        "model_name": model_name,
        "model_revision": model_rev,
        "hidden_size": hidden_size,
        "num_labels": num_labels,
        "max_length": max_len,
        "dropout": dropout,
        "best_epoch": best_epoch,
        "label_order": label_order,
        "temperatures": temperatures,
        "thresholds": thresholds,
        "lineage": lineage,
    }
    manifest_path = artifact_dir / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(bundle_manifest, indent=2) + "\n", encoding="utf-8")

    tracking_uri = configure_mlflow(train_cfg, local_only=local_mlflow)
    completed_utc = dt.datetime.now(dt.timezone.utc).isoformat()

    with mlflow.start_run(run_name="d1_primary_minilm_multilabel") as run:
        run_id = run.info.run_id
        mlflow.set_tags(
            {
                "run_role": "primary",
                "stage": "deliverable_1",
                "model_family": "minilm_multilabel",
                "pretrained_model": model_name,
                "pretrained_revision": model_rev,
                "preceding_baseline_run_id": baseline_run_id,
                "git_commit": lineage["git_commit"],
                "raw_dvc_md5": lineage["raw_dvc_md5"],
                "raw_sha256": lineage["raw_sha256"],
                "taxonomy_version": lineage["taxonomy_version"],
                "route_policy_version": lineage["route_policy_version"],
                "device": str(device),
                "completed_utc": completed_utc,
            }
        )
        mlflow.log_params(
            {
                "model_type": prim_cfg["model_type"],
                "model_name": model_name,
                "model_revision": model_rev,
                "seed": seed,
                "max_length": max_len,
                "batch_size": batch_size,
                "epochs": epochs,
                "best_epoch": best_epoch,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "pos_weight_cap": pos_weight_cap,
                "dropout": dropout,
                "calibration_method": cal_cfg["method"],
                "train_records": len(train_recs),
                "val_records": len(val_recs),
                "regression_records": len(reg_recs),
            }
        )

        for ep_item in epoch_history:
            ep = int(ep_item["epoch"])
            mlflow.log_metrics(
                {
                    "epoch_train_loss": float(ep_item["train_loss"]),
                    "epoch_val_raw_macro_f1": float(ep_item["val_raw_macro_f1"]),
                },
                step=ep,
            )

        flat_metrics: dict[str, float] = {
            "runtime_seconds": runtime_s,
            "best_epoch": float(best_epoch),
        }
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
            "run_name": "d1_primary_minilm_multilabel",
            "run_role": "primary",
            "preceding_baseline_run_id": baseline_run_id,
            "completed_utc": completed_utc,
            "tracking_uri": tracking_uri,
            "model_type": prim_cfg["model_type"],
            "model_name": model_name,
            "model_revision": model_rev,
            "device": str(device),
            "runtime_seconds": runtime_s,
            "best_epoch": best_epoch,
            "epoch_history": epoch_history,
            "lineage": lineage,
            "hyperparameters": {
                "seed": seed,
                "max_length": max_len,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "pos_weight_cap": pos_weight_cap,
                "dropout": dropout,
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
        mlflow.log_artifact(str(manifest_path), artifact_path="bundle")
        val_report_path = root / paths["validation_report"]
        if val_report_path.exists():
            mlflow.log_artifact(str(val_report_path), artifact_path="validation")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and log the primary MiniLM multi-label model.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "train.yaml")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--local-mlflow", action="store_true")
    args = parser.parse_args()

    res = train_and_evaluate_primary(args.config, out_dir=args.out_dir, local_mlflow=args.local_mlflow)
    print(
        json.dumps(
            {
                "run_id": res["run_id"],
                "preceding_baseline_run_id": res["preceding_baseline_run_id"],
                "model_type": res["model_type"],
                "best_epoch": res["best_epoch"],
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
