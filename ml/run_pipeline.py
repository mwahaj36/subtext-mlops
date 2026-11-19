"""One-command Deliverable 1 reproduction entrypoint (`python -m ml.run_pipeline`)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from ml.validate import REPO_ROOT, sha256_file

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _ensure_raw_data(root: Path) -> str:
    """Pull raw data via DVC when credentials exist; otherwise rebuild and verify exact SHA-256."""
    load_dotenv(root / ".env")
    raw_path = root / "data" / "raw" / "d1_queries.jsonl"
    prov_path = root / "data" / "manifests" / "dataset_provenance.json"
    expected_sha = json.loads(prov_path.read_text(encoding="utf-8"))["sha256"]

    user = os.getenv("MLFLOW_TRACKING_USERNAME", "").strip()
    pw = os.getenv("MLFLOW_TRACKING_PASSWORD", "").strip()
    if user and pw:
        subprocess.run(["dvc", "remote", "modify", "--local", "origin", "auth", "basic"], cwd=root, check=True)
        subprocess.run(["dvc", "remote", "modify", "--local", "origin", "user", user], cwd=root, check=True)
        subprocess.run(["dvc", "remote", "modify", "--local", "origin", "password", pw], cwd=root, check=True)
        if not raw_path.exists():
            subprocess.run(["dvc", "pull", "data/raw/d1_queries.jsonl.dvc"], cwd=root, check=True)
            source_mode = "dvc_pull_dagshub"
        else:
            source_mode = "present_verified_dvc"
    elif raw_path.exists():
        source_mode = "present_local"
    else:
        subprocess.run([sys.executable, "-m", "ml.build_d1_dataset"], cwd=root, check=True)
        source_mode = "deterministic_offline_rebuild"

    actual_sha = sha256_file(raw_path)
    if actual_sha != expected_sha:
        raise RuntimeError(f"Raw dataset SHA-256 mismatch: expected {expected_sha}, got {actual_sha}")
    return source_mode


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute full D1 pipeline and print reproduction summary.")
    parser.add_argument("--force", action="store_true", help="Force re-execution of all DVC stages.")
    args = parser.parse_args()

    source_mode = _ensure_raw_data(REPO_ROOT)
    cmd = ["dvc", "repro"]
    if args.force:
        cmd.append("--force")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    reg_summary = json.loads(
        (REPO_ROOT / "artifacts" / "registry" / "registration_summary.json").read_text(encoding="utf-8")
    )
    split_manifest = json.loads(
        (REPO_ROOT / "data" / "manifests" / "split_manifest.json").read_text(encoding="utf-8")
    )

    comp = reg_summary["comparison"]
    sel = reg_summary["selected_model"]
    report = {
        "data_source_mode": source_mode,
        "raw_sha256": split_manifest["raw_sha256"],
        "split_counts": {k: v["num_records"] for k, v in split_manifest["splits"].items()},
        "baseline_tfidf_ovr_logreg": comp["baseline"],
        "primary_minilm_multilabel": comp["primary"],
        "selected_model": {
            "model_type": sel["model_type"],
            "registered_model_name": sel["registered_model_name"],
            "registered_version": sel["registered_version"],
            "registered_model_uri": sel["registered_model_uri"],
            "bundle_fingerprint": sel["bundle_fingerprint"],
        },
        "fresh_process_reload_verified": reg_summary["fresh_process_reload_verified"],
    }
    print("=== SUBTEXT DELIVERABLE 1 REPRODUCTION SUMMARY ===")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
