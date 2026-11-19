"""Tests for multi-label evaluation, temperature calibration, threshold tuning, routing, and pyfunc reload."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.evaluate import (  # noqa: E402
    apply_temperature_scaling,
    apply_thresholds,
    compute_brier,
    compute_ece,
    fit_temperature_scaling,
    select_route,
    sigmoid,
    tune_thresholds,
)
from ml.register import verify_reload_in_current_process  # noqa: E402
from ml.validate import load_yaml  # noqa: E402


def test_temperature_scaling_and_threshold_tuning_shapes_and_bounds() -> None:
    rng = np.random.default_rng(42)
    labels = ["l1", "l2", "l3"]
    y_val = rng.integers(0, 2, size=(40, 3))
    # Overconfident logits scaled by 3x
    raw_logits = (y_val * 2 - 1) * 3.0 + rng.normal(0, 1.2, size=(40, 3))
    raw_probs = sigmoid(raw_logits)

    temps = fit_temperature_scaling(raw_logits, y_val, labels)
    assert set(temps.keys()) == set(labels)
    assert all(0.25 <= t <= 4.0 for t in temps.values())

    cal_probs = apply_temperature_scaling(raw_logits, temps, labels)
    assert cal_probs.shape == (40, 3)
    assert compute_brier(y_val, cal_probs) <= compute_brier(y_val, raw_probs) + 0.05
    assert 0.0 <= compute_ece(y_val, cal_probs) <= 1.0

    thresholds = tune_thresholds(cal_probs, y_val, labels)
    assert set(thresholds.keys()) == set(labels)
    assert all(0.20 <= th <= 0.80 for th in thresholds.values())

    preds = apply_thresholds(np.full((5, 3), 0.01), thresholds, labels)
    # Fallback rule guarantees at least 1 label per query even when all scores are below threshold
    assert np.all(np.sum(preds, axis=1) >= 1)


def test_deterministic_route_selection() -> None:
    routing_cfg = load_yaml(REPO_ROOT / "configs" / "routing.yaml")
    r1, _ = select_route(["similarity_vibe"], {"reference_titles": ["Drive"]}, routing_cfg)
    assert r1 == "similarity_anchor"

    r2, _ = select_route(["attribute_genre"], {"genres": ["Thriller"], "reference_titles": []}, routing_cfg)
    assert r2 == "structured_filter"

    r3, _ = select_route(["affective_outcome"], {"genres": [], "reference_titles": []}, routing_cfg)
    assert r3 == "hybrid_concept"

    r4, _ = select_route([], {"genres": [], "reference_titles": []}, routing_cfg)
    assert r4 == "dense_semantic"


def test_registered_bundle_artifacts_and_reload() -> None:
    reg_summary_path = REPO_ROOT / "artifacts" / "registry" / "registration_summary.json"
    assert reg_summary_path.exists(), "Run pipeline before running registry reload test"
    reg_summary = json.loads(reg_summary_path.read_text(encoding="utf-8"))

    comp = reg_summary["comparison"]
    assert comp["baseline"]["completed_utc"] <= comp["primary"]["completed_utc"]
    assert reg_summary["fresh_process_reload_verified"] is True

    pyfunc_dir = REPO_ROOT / "artifacts" / "registry" / "pyfunc_model"
    preds = verify_reload_in_current_process(str(pyfunc_dir), REPO_ROOT)
    assert len(preds) == 3
    assert preds[0]["extraction"]["language"] == "ko"
    assert "gore" in preds[0]["extraction"]["excluded_terms"]
    assert preds[1]["selected_route"] == "similarity_anchor"
