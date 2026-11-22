"""Unit tests for closed-loop drift batch validation and gate-controlled retraining."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.retrain import retrain_challenger, validate_drift_batch
from ml.validate import REPO_ROOT, DataValidationError, load_jsonl


def test_reviewed_drift_batch_passes_validation() -> None:
    recs, stats = validate_drift_batch(REPO_ROOT / "data" / "demo_batches" / "d2_drift_batch_reviewed.jsonl")
    assert len(recs) == 30
    assert stats["unique_groups"] == 15
    assert all(r["provenance"]["review_status"] == "approved" for r in recs)


def test_unreviewed_queue_sample_is_blocked() -> None:
    queue_path = REPO_ROOT / "data" / "demo_batches" / "unreviewed_queue_sample.json"
    with pytest.raises(DataValidationError, match="Unreviewed or unapproved"):
        validate_drift_batch(queue_path)


def test_leakage_against_frozen_val_is_blocked(tmp_path: Path) -> None:
    val_first = load_jsonl(REPO_ROOT / "data" / "processed" / "val.jsonl")[0]
    leaky = dict(val_first)
    leaky["query_id"] = "d2_leaky_999"
    leaky_path = tmp_path / "leaky.jsonl"
    leaky_path.write_text(json.dumps(leaky) + "\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="overlap"):
        validate_drift_batch(leaky_path)


def test_retrain_blocks_registration_when_gate_fails(tmp_path: Path, monkeypatch) -> None:
    import ml.retrain as retrain_mod

    class _FakeEnc:
        class _Cfg:
            hidden_size = 384
        config = _Cfg()
        def save_pretrained(self, d: Path) -> None:
            (d / "config.json").write_text("{}", encoding="utf-8")

    class _FakeTok:
        def save_pretrained(self, d: Path) -> None:
            (d / "tokenizer.json").write_text("{}", encoding="utf-8")

    class _FakeModel:
        encoder = _FakeEnc()
        class _Head:
            def state_dict(self):
                return {}
        head = _Head()

    monkeypatch.setattr(retrain_mod, "_train_combined_model", lambda *a, **k: (_FakeModel(), _FakeTok(), "cpu", 1, [{"epoch": 1, "train_loss": 0.2, "val_raw_macro_f1": 0.7}]))
    import numpy as np
    monkeypatch.setattr(retrain_mod, "predict_logits", lambda model, tok, texts, *a: np.zeros((len(texts), 14), dtype=np.float64))
    monkeypatch.setattr(retrain_mod, "fit_temperature_scaling", lambda logits, y, lbls: {lbl: 1.0 for lbl in lbls})
    monkeypatch.setattr(retrain_mod, "tune_thresholds", lambda probs, y, lbls, **k: {lbl: 0.5 for lbl in lbls})

    def _bad_eval(recs, *a, **k):
        return {
            "aggregate": {"macro_f1": 0.45, "micro_f1": 0.48, "subset_accuracy": 0.1, "ece": 0.15, "uncalibrated_ece": 0.2, "brier": 0.2, "extraction_micro_f1": 0.99},
            "slices": {"negation": {"micro_f1": 0.4, "count": 5}, "compound_broad": {"micro_f1": 0.4, "count": 5}},
            "per_label": {},
        }

    monkeypatch.setattr(retrain_mod, "evaluate_multilabel", _bad_eval)
    monkeypatch.setattr(retrain_mod.torch, "save", lambda obj, path: Path(path).write_bytes(b"pt"))

    def _local_mlflow(_cfg, local_only: bool = False) -> str:
        import mlflow

        uri = (tmp_path / "mlruns").as_uri()
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("test-retrain-unit")
        return uri

    monkeypatch.setattr(retrain_mod, "configure_mlflow", _local_mlflow)

    comp = retrain_challenger(
        REPO_ROOT / "data" / "demo_batches" / "d2_drift_batch_reviewed.jsonl",
        out_dir=tmp_path / "challenger",
        comparison_out_path=tmp_path / "comp.json",
        epochs=1,
        local_mlflow=True,
        register_challenger=True,
    )
    assert comp["gate_passed"] is False
    assert comp["decision"] == "reject"
    assert comp["challenger_version"] == "not_registered"
    assert any(c["name"] == "reg_macro_f1_floor" and not c["passed"] for c in comp["gate_checks"])


def test_retrain_promotes_candidate_when_gate_passes(tmp_path: Path, monkeypatch) -> None:
    import numpy as np
    import ml.retrain as retrain_mod

    class _FakeEnc:
        class _Cfg:
            hidden_size = 384
        config = _Cfg()
        def save_pretrained(self, d: Path) -> None:
            (d / "config.json").write_text("{}", encoding="utf-8")

    class _FakeTok:
        def save_pretrained(self, d: Path) -> None:
            (d / "tokenizer.json").write_text("{}", encoding="utf-8")

    class _FakeModel:
        encoder = _FakeEnc()
        class _Head:
            def state_dict(self):
                return {}
        head = _Head()

    monkeypatch.setattr(retrain_mod, "_train_combined_model", lambda *a, **k: (_FakeModel(), _FakeTok(), "cpu", 2, [{"epoch": 1, "train_loss": 0.1, "val_raw_macro_f1": 0.88}]))
    monkeypatch.setattr(retrain_mod, "predict_logits", lambda model, tok, texts, *a: np.zeros((len(texts), 14), dtype=np.float64))
    monkeypatch.setattr(retrain_mod, "fit_temperature_scaling", lambda logits, y, lbls: {lbl: 1.0 for lbl in lbls})
    monkeypatch.setattr(retrain_mod, "tune_thresholds", lambda probs, y, lbls, **k: {lbl: 0.5 for lbl in lbls})

    def _good_eval(recs, *a, **k):
        return {
            "aggregate": {"macro_f1": 0.89, "micro_f1": 0.90, "subset_accuracy": 0.5, "ece": 0.02, "uncalibrated_ece": 0.04, "brier": 0.03, "extraction_micro_f1": 0.99},
            "slices": {"negation": {"micro_f1": 0.72, "count": 5}, "compound_broad": {"micro_f1": 0.82, "count": 5}},
            "per_label": {},
        }

    monkeypatch.setattr(retrain_mod, "evaluate_multilabel", _good_eval)
    monkeypatch.setattr(retrain_mod.torch, "save", lambda obj, path: Path(path).write_bytes(b"pt"))

    def _local_mlflow(_cfg, local_only: bool = False) -> str:
        import mlflow

        uri = (tmp_path / "mlruns").as_uri()
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("test-retrain-unit")
        return uri

    monkeypatch.setattr(retrain_mod, "configure_mlflow", _local_mlflow)

    comp = retrain_challenger(
        REPO_ROOT / "data" / "demo_batches" / "d2_drift_batch_reviewed.jsonl",
        out_dir=tmp_path / "challenger",
        comparison_out_path=tmp_path / "comp_pass.json",
        epochs=1,
        local_mlflow=True,
        register_challenger=False,
    )
    assert comp["gate_passed"] is True
    assert comp["decision"] == "promote_candidate"
    assert len(comp["gate_checks"]) == 8
    assert all(c["passed"] for c in comp["gate_checks"])
    assert (tmp_path / "comp_pass.json").exists()
