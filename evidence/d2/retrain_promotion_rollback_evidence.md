# Closed-Loop Retraining, Promotion & Rollback Evidence (Deliverable 2)

## 1. DVC-Tracked Reviewed Drift Batch & Unreviewed Queue Block
- **Reviewed Drift Batch (`data/demo_batches/d2_drift_batch_reviewed.jsonl.dvc`):**
  - `30` colloquial/long multi-clause queries across `15` paraphrase groups (`d2_q0001..d2_q0030`), built deterministically by [`ml/build_d2_drift_batch.py`](../../ml/build_d2_drift_batch.py) and pushed to DagsHub DVC remote.
  - DVC MD5: `22799024651fdab6b8608d9688dd6856` | Size: `32590` bytes
  - SHA-256: `89f2ca9b91382ccfbf8cd462dd7a81d7c9a6d8dd57bc24a0aa2e14556a39cb46`
- **Fail-Closed Provenance & Leakage Enforcement (`ml/retrain.py::validate_drift_batch`):**
  - [`data/demo_batches/unreviewed_queue_sample.json`](../../data/demo_batches/unreviewed_queue_sample.json) holds `5` unreviewed (`review_status: "draft"`) records. Verified by `tests/test_d2_retrain.py::test_unreviewed_queue_sample_is_blocked` — unapproved batches raise `DataValidationError` and cannot enter retraining.
  - Leakage against frozen `val.jsonl` (`58`) and `course_regression.jsonl` (`66`) fails closed (`tests/test_d2_retrain.py::test_leakage_against_frozen_val_is_blocked`).

---

## 2. Champion (v2) vs. Challenger (v3) Comparison in DagsHub MLflow
- **Command:** `python -m ml.retrain --batch data/demo_batches/d2_drift_batch_reviewed.jsonl --epochs 12`
- **Comparison Summary Artifact:** [`evidence/d2/retrain_comparison.json`](retrain_comparison.json)
- **MLflow Experiment & Registry:**
  - Tracking UI: `https://dagshub.com/mwahaj36/subtext-mlops.mlflow`
  - **Champion (D1 `v2`):** Run `01b02fc1496548f8b5bd4beb36e6cb99`, Registered Model `subtext-intent-multilabel` v2, fingerprint `4fb663cb67dcc1f0`
  - **Challenger (D2 `v3`):** Run `18deb0ca31c643c6b5a1a049dfd604b4` (`d2_challenger_minilm_multilabel`), Registered Model `subtext-intent-multilabel` **v3**, fingerprint `46ee6c3817f7b280`

| Gate Check (`configs/gates.yaml`) | Champion (`v2`) | Challenger (`v3`) | Requirement | Result |
|---|---|---|---|---|
| `dataset_validation` | `passed, leakage=0` | `passed, leakage=0` | `passed, leakage=0` | **PASS** |
| `val_macro_f1_floor` | `0.8770` | **`0.8978`** | `>= 0.8000` | **PASS** |
| `reg_macro_f1_floor` | `0.8183` | **`0.8594`** | `>= 0.7800` | **PASS** |
| `extraction_micro_f1_floor` | `0.9872` | **`0.9872`** | `>= 0.9500` | **PASS** |
| `reg_calibration_ece_cap` | `0.0299` | `0.0338` | `<= 0.0800` | **PASS** |
| `champion_reg_macro_f1_delta` | baseline (`0.8183`) | `0.8594` (`drop=-0.0411`) | `drop <= 0.0200` | **PASS** |
| `negation_slice_f1_delta` | baseline (`0.7080`) | `0.7748` (`drop=-0.0668`) | `drop <= 0.0300` | **PASS** |
| `compound_slice_f1_delta` | baseline (`0.8012`) | `0.8389` (`drop=-0.0377`) | `drop <= 0.0300` | **PASS** |

- Decision recorded in `retrain_comparison.json`: **`"gate_passed": true`**, **`"decision": "promote_candidate"`**.
- Automated retraining trigger workflow: [`.github/workflows/retrain.yml`](../../.github/workflows/retrain.yml) (`workflow_dispatch` + `repository_dispatch` `drift-alert`).

---

## 3. Digest-Pinned Promotion & Complete-Bundle Rollback Proof
- **Release Workflow:** [`.github/workflows/release.yml`](../../.github/workflows/release.yml) runs `ml.gate` fail-closed before building/pushing digest-pinned ECR images, and supports instant digest-only EC2 `rollback` without rebuilding.
- **Promotion Proof (`:d2-v2` — Challenger v3 + `/metrics`):**
  - Triggered `release.yml` (`action=promote`, `bundle_subdir=artifacts/challenger`, `image_tag=d2-v2`, Run [`45085318415`](https://github.com/mwahaj36/subtext-mlops/actions/runs/45085318415)).
  - Pushed digest `904966651145.dkr.ecr.us-east-1.amazonaws.com/subtext-mlops-serve@sha256:838afa22ad784906aedaed84a348cc572142c4cb29a7e1aaf092a6dcfa2ee13a` and recreated container on EC2 (`model_version: "3"`, `bundle_fingerprint: "46ee6c3817f7b280"`, `/metrics` active).
  - *Note:* Run `45085318415`'s trailing curl verification exited `3` due to a trailing CRLF in EC2 `.env` `API_DOMAIN`; stripped `\r` on EC2 and added `tr -d '\r'` in `release.yml` so subsequent runs exit `0`.
- **Instant Complete-Bundle Rollback Proof (`45s` revert to `:d2-v1` Champion v2):**
  - Triggered `release.yml` (`action=rollback`, `image_digest=sha256:19753266eba1199622f5e37a86c068c88a1aa9de2b6d05a99198c22ce246a4e5`, Run [`45085918888`](https://github.com/mwahaj36/subtext-mlops/actions/runs/45085918888) — **PASS** in `45s`).
  - Live `GET https://54-210-225-89.sslip.io/version` reverted cleanly to Champion v2:
    ```json
    {
      "model_name": "subtext-intent-multilabel",
      "model_id": "minilm_multilabel",
      "model_version": "2",
      "bundle_fingerprint": "4fb663cb67dcc1f0",
      "image_digest": "904966651145.dkr.ecr.us-east-1.amazonaws.com/subtext-mlops-serve@sha256:19753266eba1199622f5e37a86c068c88a1aa9de2b6d05a99198c22ce246a4e5",
      "model_loaded": true
    }
    ```
- Following PR #13 merge, `release.yml` (`promote`, `artifacts/challenger`, tag `d2-v3`) promotes Challenger v3 + case-normalized LLM validation live on EC2.
