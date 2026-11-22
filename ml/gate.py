"""Fail-closed CI evaluation gate comparing candidate vs. champion metrics."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON object in {label}: {path}")
    return data


def _finite_float(obj: dict[str, Any], *keys: str) -> float:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(".".join(keys))
        cur = cur[key]
    val = float(cur)
    if not math.isfinite(val):
        raise ValueError(f"Non-finite value for {'.'.join(keys)}: {val}")
    return val


def _check_validation(report_path: Path | None) -> dict[str, Any]:
    if report_path is None:
        return {"name": "dataset_validation", "passed": False, "observed": "missing", "threshold": "passed", "detail": "Validation report path not provided"}
    try:
        rep = _load_json(report_path, "validation report")
        top_ok = rep.get("status") == "passed"
        splits_ok = rep.get("splits", {}).get("status") == "passed"
        leakage = rep.get("splits", {}).get("leakage_checks", {})
        expected_keys = ("query_id_overlap", "group_id_overlap", "exact_normalized_text_overlap", "near_duplicate_pairs")
        has_all_leak_keys = all(k in leakage for k in expected_keys)
        leak_total = sum(int(leakage[k]) for k in expected_keys) if has_all_leak_keys else -1
        ok = bool(top_ok and splits_ok and has_all_leak_keys and leak_total == 0)
        return {
            "name": "dataset_validation",
            "passed": ok,
            "observed": f"status={rep.get('status')}, splits={rep.get('splits', {}).get('status')}, leakage={leak_total}",
            "threshold": "status=passed, splits=passed, leakage=0",
            "detail": "All dataset schema and split-leakage checks passed" if ok else "Validation or leakage gate failed",
        }
    except Exception as exc:
        return {"name": "dataset_validation", "passed": False, "observed": str(exc), "threshold": "passed", "detail": "Validation report unreadable or invalid"}


def _build_markdown(report: dict[str, Any]) -> str:
    status_lbl = "PASSED" if report["passed"] else "FAILED"
    lines = [
        f"## Evaluation Gate Report (`{status_lbl}` — policy v{report['gate_version']})",
        "",
        "| Check | Result | Observed | Requirement |",
        "|---|---|---|---|",
    ]
    for chk in report["checks"]:
        icon = "PASS" if chk["passed"] else "FAIL"
        lines.append(f"| `{chk['name']}` | **{icon}** | `{chk['observed']}` | `{chk['threshold']}` |")
    return "\n".join(lines)


def evaluate_gates(
    champion_path: Path,
    candidate_path: Path,
    config_path: Path,
    validation_report_path: Path | None = None,
) -> dict[str, Any]:
    """Run all absolute, slice, calibration, and relative regression checks fail-closed."""
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid gate config: {config_path}")
    except Exception as exc:
        report = {
            "gate_version": "unknown",
            "passed": False,
            "checks": [{"name": "config_load", "passed": False, "observed": str(exc), "threshold": "valid yaml", "detail": "Failed loading gate config"}],
            "champion_summary": {},
            "candidate_summary": {},
        }
        report["markdown_summary"] = _build_markdown(report)
        return report

    checks: list[dict[str, Any]] = []
    if cfg.get("require_validation_passed", True):
        checks.append(_check_validation(validation_report_path))

    champ_sum: dict[str, Any] = {}
    cand_sum: dict[str, Any] = {}
    try:
        champ = _load_json(champion_path, "champion metrics")
        cand = _load_json(candidate_path, "candidate metrics")

        c_val_f1 = _finite_float(cand, "val", "aggregate", "macro_f1")
        c_reg_f1 = _finite_float(cand, "course_regression", "aggregate", "macro_f1")
        c_ext_f1 = _finite_float(cand, "course_regression", "aggregate", "extraction_micro_f1")
        c_reg_ece = _finite_float(cand, "course_regression", "aggregate", "ece")
        c_neg_f1 = _finite_float(cand, "course_regression", "slices", "negation", "micro_f1")
        c_comp_f1 = _finite_float(cand, "course_regression", "slices", "compound_broad", "micro_f1")

        b_reg_f1 = _finite_float(champ, "course_regression", "aggregate", "macro_f1")
        b_neg_f1 = _finite_float(champ, "course_regression", "slices", "negation", "micro_f1")
        b_comp_f1 = _finite_float(champ, "course_regression", "slices", "compound_broad", "micro_f1")

        cand_sum = {
            "model_type": cand.get("model_type"),
            "run_id": cand.get("run_id"),
            "val_macro_f1": round(c_val_f1, 4),
            "reg_macro_f1": round(c_reg_f1, 4),
            "extraction_micro_f1": round(c_ext_f1, 4),
            "reg_ece": round(c_reg_ece, 4),
            "reg_negation_micro_f1": round(c_neg_f1, 4),
            "reg_compound_micro_f1": round(c_comp_f1, 4),
        }
        champ_sum = {
            "model_type": champ.get("model_type"),
            "run_id": champ.get("run_id"),
            "reg_macro_f1": round(b_reg_f1, 4),
            "reg_negation_micro_f1": round(b_neg_f1, 4),
            "reg_compound_micro_f1": round(b_comp_f1, 4),
        }

        min_val = float(cfg["min_val_macro_f1"])
        min_reg = float(cfg["min_reg_macro_f1"])
        min_ext = float(cfg["min_extraction_micro_f1"])
        max_ece = float(cfg["max_reg_ece"])
        max_reg_drop = float(cfg["max_regression_macro_f1_drop"])
        max_neg_drop = float(cfg["max_negation_slice_f1_drop"])
        max_comp_drop = float(cfg["max_compound_slice_f1_drop"])

        checks.extend([
            {
                "name": "val_macro_f1_floor",
                "passed": c_val_f1 >= min_val,
                "observed": f"{c_val_f1:.4f}",
                "threshold": f">= {min_val:.4f}",
                "detail": "Candidate validation macro-F1 floor",
            },
            {
                "name": "reg_macro_f1_floor",
                "passed": c_reg_f1 >= min_reg,
                "observed": f"{c_reg_f1:.4f}",
                "threshold": f">= {min_reg:.4f}",
                "detail": "Candidate course regression macro-F1 floor",
            },
            {
                "name": "extraction_micro_f1_floor",
                "passed": c_ext_f1 >= min_ext,
                "observed": f"{c_ext_f1:.4f}",
                "threshold": f">= {min_ext:.4f}",
                "detail": "Candidate bounded constraint extraction micro-F1 floor",
            },
            {
                "name": "reg_calibration_ece_cap",
                "passed": c_reg_ece <= max_ece,
                "observed": f"{c_reg_ece:.4f}",
                "threshold": f"<= {max_ece:.4f}",
                "detail": "Candidate course regression ECE calibration bound",
            },
            {
                "name": "champion_reg_macro_f1_delta",
                "passed": (b_reg_f1 - c_reg_f1) <= max_reg_drop,
                "observed": f"cand={c_reg_f1:.4f} vs champ={b_reg_f1:.4f} (drop={b_reg_f1 - c_reg_f1:+.4f})",
                "threshold": f"drop <= {max_reg_drop:.4f}",
                "detail": "Regression macro-F1 delta vs. pinned champion",
            },
            {
                "name": "negation_slice_f1_delta",
                "passed": (b_neg_f1 - c_neg_f1) <= max_neg_drop,
                "observed": f"cand={c_neg_f1:.4f} vs champ={b_neg_f1:.4f} (drop={b_neg_f1 - c_neg_f1:+.4f})",
                "threshold": f"drop <= {max_neg_drop:.4f}",
                "detail": "Negation slice micro-F1 regression tolerance",
            },
            {
                "name": "compound_slice_f1_delta",
                "passed": (b_comp_f1 - c_comp_f1) <= max_comp_drop,
                "observed": f"cand={c_comp_f1:.4f} vs champ={b_comp_f1:.4f} (drop={b_comp_f1 - c_comp_f1:+.4f})",
                "threshold": f"drop <= {max_comp_drop:.4f}",
                "detail": "Compound-query slice micro-F1 regression tolerance",
            },
        ])
    except Exception as exc:
        checks.append(
            {
                "name": "metrics_contract",
                "passed": False,
                "observed": f"{type(exc).__name__}: {exc}",
                "threshold": "valid champion & candidate metrics",
                "detail": "Missing, unreadable, or malformed metrics artifacts",
            }
        )

    passed = bool(checks) and all(c["passed"] for c in checks)
    report = {
        "gate_version": str(cfg.get("version", "1.0.0")),
        "passed": passed,
        "checks": checks,
        "champion_summary": champ_sum,
        "candidate_summary": cand_sum,
    }
    report["markdown_summary"] = _build_markdown(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed evaluation gate check.")
    parser.add_argument("--champion", type=Path, default=REPO_ROOT / "artifacts" / "primary" / "metrics.json")
    parser.add_argument("--candidate", type=Path, default=REPO_ROOT / "artifacts" / "primary" / "metrics.json")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "gates.yaml")
    parser.add_argument("--validation-report", type=Path, default=REPO_ROOT / "data" / "manifests" / "validation_report.json")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report = evaluate_gates(
        args.champion,
        args.candidate,
        args.config,
        validation_report_path=args.validation_report,
    )
    print(report["markdown_summary"])
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
