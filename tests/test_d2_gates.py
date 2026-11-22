"""Tests for fail-closed CI evaluation gates."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.gate import evaluate_gates, main as gate_main  # noqa: E402


def _champion_metrics() -> dict:
    return json.loads((REPO_ROOT / "artifacts" / "primary" / "metrics.json").read_text(encoding="utf-8"))


def _validation_report() -> dict:
    return json.loads((REPO_ROOT / "data" / "manifests" / "validation_report.json").read_text(encoding="utf-8"))


def test_real_champion_passes_all_gates(tmp_path: Path) -> None:
    out_file = tmp_path / "gate_report.json"
    code = gate_main(
        [
            "--champion",
            str(REPO_ROOT / "artifacts" / "primary" / "metrics.json"),
            "--candidate",
            str(REPO_ROOT / "artifacts" / "primary" / "metrics.json"),
            "--config",
            str(REPO_ROOT / "configs" / "gates.yaml"),
            "--validation-report",
            str(REPO_ROOT / "data" / "manifests" / "validation_report.json"),
            "--out",
            str(out_file),
        ]
    )
    assert code == 0
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved["passed"] is True
    assert all(chk["passed"] for chk in saved["checks"])


@pytest.mark.parametrize(
    ("mutator", "expected_failed_check"),
    [
        (lambda m: m["course_regression"]["aggregate"].__setitem__("macro_f1", 0.72), "reg_macro_f1_floor"),
        (lambda m: m["course_regression"]["slices"]["negation"].__setitem__("micro_f1", 0.50), "negation_slice_f1_delta"),
        (lambda m: m["course_regression"]["slices"]["compound_broad"].__setitem__("micro_f1", 0.65), "compound_slice_f1_delta"),
        (lambda m: m["course_regression"]["aggregate"].__setitem__("ece", 0.14), "reg_calibration_ece_cap"),
        (lambda m: m["course_regression"]["aggregate"].__setitem__("extraction_micro_f1", 0.80), "extraction_micro_f1_floor"),
    ],
)
def test_degraded_candidate_fails_closed(tmp_path: Path, mutator, expected_failed_check: str) -> None:
    cand = copy.deepcopy(_champion_metrics())
    mutator(cand)
    cand_path = tmp_path / "cand_metrics.json"
    cand_path.write_text(json.dumps(cand), encoding="utf-8")

    code = gate_main(
        [
            "--champion",
            str(REPO_ROOT / "artifacts" / "primary" / "metrics.json"),
            "--candidate",
            str(cand_path),
            "--config",
            str(REPO_ROOT / "configs" / "gates.yaml"),
            "--validation-report",
            str(REPO_ROOT / "data" / "manifests" / "validation_report.json"),
        ]
    )
    assert code == 1

    rep = evaluate_gates(
        REPO_ROOT / "artifacts" / "primary" / "metrics.json",
        cand_path,
        REPO_ROOT / "configs" / "gates.yaml",
        REPO_ROOT / "data" / "manifests" / "validation_report.json",
    )
    assert rep["passed"] is False
    failed = {chk["name"] for chk in rep["checks"] if not chk["passed"]}
    assert expected_failed_check in failed


def test_missing_or_corrupt_metrics_and_leakage_fail_closed(tmp_path: Path) -> None:
    missing = evaluate_gates(
        REPO_ROOT / "artifacts" / "primary" / "metrics.json",
        tmp_path / "does_not_exist.json",
        REPO_ROOT / "configs" / "gates.yaml",
        REPO_ROOT / "data" / "manifests" / "validation_report.json",
    )
    assert missing["passed"] is False
    assert any(c["name"] == "metrics_contract" and not c["passed"] for c in missing["checks"])

    bad_val = copy.deepcopy(_validation_report())
    bad_val["splits"]["leakage_checks"]["group_id_overlap"] = 2
    bad_val_path = tmp_path / "bad_val.json"
    bad_val_path.write_text(json.dumps(bad_val), encoding="utf-8")
    leaked = evaluate_gates(
        REPO_ROOT / "artifacts" / "primary" / "metrics.json",
        REPO_ROOT / "artifacts" / "primary" / "metrics.json",
        REPO_ROOT / "configs" / "gates.yaml",
        bad_val_path,
    )
    assert leaked["passed"] is False
    assert any(c["name"] == "dataset_validation" and not c["passed"] for c in leaked["checks"])
