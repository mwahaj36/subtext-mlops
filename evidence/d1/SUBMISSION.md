# Deliverable 1 — Submission Package & Audit (`subtext-mlops`)

## 1. Ready-to-Paste Assignment Submission Message

Copy and paste the block below into the assignment submission box (and supply the course DagsHub read token in the private instructor/TA comment if requested for authenticated `dvc pull`):

```text
Project: Subtext MLOps — Fine-Grained Multi-Label Query Understanding & Constraint Extraction
Team Members:
- Muhammad Wahaj (23L-0560) — Stage 1 (Deliverable 1 & Shared Infrastructure)
- Tayyab Khalid (23L-0501) — Stage 2 (Deliverable 2 Part A)
- Fasih-ur-Rehman (23L-0526) — Stage 3 (Deliverable 2 Part B)

1. GitHub Repository & Deliverable 1 Release:
- Repository: https://github.com/mwahaj36/subtext-mlops
- Deliverable 1 Tag / Release: https://github.com/mwahaj36/subtext-mlops/releases/tag/deliverable-1 (tag: deliverable-1)
- Provenance & Inherited vs. New Course Work Inventory: PROVENANCE.md and CONTRIBUTIONS.md

2. DVC Configuration, Pipeline Stages, Lockfile & Data Access:
- DVC Remote Configuration: .dvc/config (single DagsHub DVC remote: https://dagshub.com/mwahaj36/subtext-mlops.dvc)
- Raw Dataset Pointer & Provenance: data/raw/d1_queries.jsonl.dvc (MD5: d115780c1b7ed690fcf529d8afce9aea, SHA-256: 93b8888f69318cc801aee52b295dbb6c63447d9b1eb9dfa889574844c83462f3, 455 reviewed queries across 219 group_ids) and data/manifests/dataset_provenance.json
- Pipeline Definition & Lockfile: dvc.yaml and dvc.lock (4 stages: split_and_validate -> train_baseline -> train_primary -> register_model)
- Group-Disjoint Split & Validation Manifests: data/manifests/split_manifest.json (331 train / 58 val / 66 course_regression; zero group, exact, or near-duplicate overlap) and data/manifests/validation_report.json
- Grader DVC Pull Access:
    dvc remote modify --local origin auth basic
    dvc remote modify --local origin user <dagshub_username>
    dvc remote modify --local origin password <course_dagshub_token_in_private_comment>
    dvc pull

3. DagsHub & MLflow Experiment Tracking and Model Registry:
- DagsHub Project: https://dagshub.com/mwahaj36/subtext-mlops
- MLflow Tracking URI: https://dagshub.com/mwahaj36/subtext-mlops.mlflow
- MLflow Experiment (subtext-intent-d1, ID 0): https://dagshub.com/mwahaj36/subtext-mlops.mlflow/#/experiments/0
- Classical Baseline Run (logged FIRST — TF-IDF Word+Char + OvR Logistic Regression):
    Run ID: 8a2bd18153c94978b98b644e08e12b39
    URL: https://dagshub.com/mwahaj36/subtext-mlops.mlflow/#/experiments/0/runs/8a2bd18153c94978b98b644e08e12b39
    Metrics: Val Macro-F1 = 0.8090, Val Micro-F1 = 0.8091, Val ECE = 0.0180 (raw 0.1170), Course Regression Macro-F1 = 0.7827, Regression Micro-F1 = 0.7750, Extraction Micro-F1 = 0.9872
- Primary Model Run (logged SECOND — Fine-Tuned MiniLM-L6-v2 + LayerNorm + Linear Multi-Label Head, best epoch 5):
    Run ID: 01b02fc1496548f8b5bd4beb36e6cb99
    URL: https://dagshub.com/mwahaj36/subtext-mlops.mlflow/#/experiments/0/runs/01b02fc1496548f8b5bd4beb36e6cb99
    Metrics: Val Macro-F1 = 0.8770, Val Micro-F1 = 0.8727, Val ECE = 0.0287, Course Regression Macro-F1 = 0.8183, Regression Micro-F1 = 0.7896, Extraction Micro-F1 = 0.9872
- Registered Model Bundle (Selected by predeclared val_macro_f1 rule):
    Registered Model Name: subtext-intent-multilabel
    Registered Version: 2 (models:/subtext-intent-multilabel/2, fingerprint: 4fb663cb67dcc1f0)
    Registration Run URL: https://dagshub.com/mwahaj36/subtext-mlops.mlflow/#/experiments/0/runs/f6d50f4c3b9a4fa8ad80c5f58a03b445
    Models UI: https://dagshub.com/mwahaj36/subtext-mlops.mlflow/#/models/subtext-intent-multilabel

4. One-Command Clean-Clone Reproduction:
- Command:
    git clone https://github.com/mwahaj36/subtext-mlops.git
    cd subtext-mlops
    docker compose run --rm train
- Works both with .env (pulls from DagsHub DVC and logs to DagsHub MLflow) and in zero-credential clean-clone mode (verifies pinned SHA-256 dataset and logs to local ./mlruns, reproducing exact Val Macro-F1 = 0.8770 and Regression Macro-F1 = 0.8183).
- Evidence logs attached from evidence/d1/: mlflow_runs_and_registry.json, dvc_pull_verification.log, validation_and_negative_checks.log, and clean_clone_docker_repro.log.
```

---

## 2. Deliverable 1 Requirement Audit

| Deliverable 1 Requirement | Status | Verified Evidence & Location |
|---|---|---|
| **Isolated course repo & honest provenance** | **PASSED** | [`PROVENANCE.md`](../../PROVENANCE.md), [`CONTRIBUTIONS.md`](../../CONTRIBUTIONS.md), [`README.md`](../../README.md); seeded from `Subtext` commit `e32bb6dc366145a11d16c984acc9b0d24eb4bd4f` with zero production remotes/secrets. |
| **Bounded fine-grained multi-label taxonomy & contracts** | **PASSED** | `14` fine labels across `7` broad categories + derived `is_compound` rule in [`configs/taxonomy.yaml`](../../configs/taxonomy.yaml), [`configs/routing.yaml`](../../configs/routing.yaml), and JSON Schemas in [`schemas/`](../../schemas/). |
| **DVC-tracked raw/processed data, single remote & lockfile** | **PASSED** | [`.dvc/config`](../../.dvc/config) (`https://dagshub.com/mwahaj36/subtext-mlops.dvc`), [`data/raw/d1_queries.jsonl.dvc`](../../data/raw/d1_queries.jsonl.dvc), [`dvc.yaml`](../../dvc.yaml), [`dvc.lock`](../../dvc.lock), and clean-clone pull proof in [`evidence/d1/dvc_pull_verification.log`](dvc_pull_verification.log) (`22 files fetched and 25 files added`). |
| **Fail-closed schema, quality & split-leakage checks before fitting** | **PASSED** | [`ml/validate.py`](../../ml/validate.py), [`ml/split.py`](../../ml/split.py), [`data/manifests/validation_report.json`](../../data/manifests/validation_report.json), and `12` intentional corruption/leakage failure fixtures + pre-fit halt test in [`tests/test_ml_validation.py`](../../tests/test_ml_validation.py) (logged in [`evidence/d1/validation_and_negative_checks.log`](validation_and_negative_checks.log)). |
| **Classical baseline logged BEFORE primary MiniLM model** | **PASSED** | Baseline run `8a2bd18153c94978b98b644e08e12b39` (`2026-11-19T18:50:53Z`, Val Macro-F1 `0.8090`, Reg Macro-F1 `0.7827`) preceded primary MiniLM run `01b02fc1496548f8b5bd4beb36e6cb99` (`2026-11-19T18:53:13Z`, Val Macro-F1 `0.8770`, Reg Macro-F1 `0.8183`). Full metrics in [`artifacts/baseline/metrics.json`](../../artifacts/baseline/metrics.json) and [`artifacts/primary/metrics.json`](../../artifacts/primary/metrics.json). |
| **Validation calibration & threshold tuning** | **PASSED** | Per-label temperature scaling (`T_j in [0.25, 4.0]`) and threshold grid search (`[0.20, 0.80]`) fit strictly on `val` in [`ml/evaluate.py`](../../ml/evaluate.py), with raw vs. calibrated Brier/ECE reported on both `val` and `course_regression`. |
| **Selected model registered in MLflow & fresh-process reload verified** | **PASSED** | Registered on DagsHub MLflow Model Registry as `subtext-intent-multilabel` version `2` (`models:/subtext-intent-multilabel/2`, run `f6d50f4c3b9a4fa8ad80c5f58a03b445`) and verified in a fresh subprocess ([`artifacts/registry/registration_summary.json`](../../artifacts/registry/registration_summary.json), [`evidence/d1/mlflow_runs_and_registry.json`](mlflow_runs_and_registry.json)). |
| **One-command clean-clone reproduction (`docker compose run --rm train`)** | **PASSED** | Verified in a fresh `git clone` inside Linux Docker with no `.env` and no `.dvc/cache`, reproducing identical metrics (`Val Macro-F1 = 0.8770`, `Reg Macro-F1 = 0.8183`); full log in [`evidence/d1/clean_clone_docker_repro.log`](clean_clone_docker_repro.log). |
| **Stage 1 → Stage 2 & Stage 3 handoff committed** | **PASSED** | [`docs/HANDOFF_D1_TO_D2.md`](../../docs/HANDOFF_D1_TO_D2.md). |

---

## 3. Submission Checklist for Wahaj

### A. Paste These Links
- **GitHub Repo:** `https://github.com/mwahaj36/subtext-mlops`
- **Deliverable 1 Release / Tag:** `https://github.com/mwahaj36/subtext-mlops/releases/tag/deliverable-1`
- **DagsHub Project:** `https://dagshub.com/mwahaj36/subtext-mlops`
- **MLflow Experiment UI:** `https://dagshub.com/mwahaj36/subtext-mlops.mlflow/#/experiments/0`
- **Classical Baseline Run UI:** `https://dagshub.com/mwahaj36/subtext-mlops.mlflow/#/experiments/0/runs/8a2bd18153c94978b98b644e08e12b39`
- **Primary MiniLM Run UI:** `https://dagshub.com/mwahaj36/subtext-mlops.mlflow/#/experiments/0/runs/01b02fc1496548f8b5bd4beb36e6cb99`
- **Model Registry UI (`subtext-intent-multilabel` v2):** `https://dagshub.com/mwahaj36/subtext-mlops.mlflow/#/models/subtext-intent-multilabel`

### B. Attach These Evidence Files (from `evidence/d1/`)
1. `evidence/d1/SUBMISSION.md`
2. `evidence/d1/mlflow_runs_and_registry.json` (live DagsHub MLflow API export of experiment, runs, parameters, metrics, tags, and registered model version `2`)
3. `evidence/d1/dvc_pull_verification.log` (clean-clone `dvc pull` verification log from DagsHub DVC remote)
4. `evidence/d1/validation_and_negative_checks.log` (automated schema/corruption/leakage/pre-fit halt test log)
5. `evidence/d1/clean_clone_docker_repro.log` (clean-clone `docker compose run --rm train` execution log)

### C. Remaining Manual Actions for Wahaj (External Portal / Optional Screenshots)
1. **Classroom Submission:** Paste the ready-to-paste message from Section 1 and attach the files in Section 3.B.
2. **Optional Browser Screenshots:** If your submission portal specifically requests `.png` screenshots of the MLflow UI in addition to exported logs, open the 3 MLflow UI links in Section 3.A (`experiments/0`, primary run `01b02fc1496548f8b5bd4beb36e6cb99`, and `models/subtext-intent-multilabel`) and attach browser screenshots.
3. **Grader Token Note:** Include the course DagsHub token in a **private comment** to the instructor/TA on the assignment portal so they can run authenticated `dvc pull` directly (never post tokens publicly).
4. **Stage 2 Cloud/Provider Activation (Before Tayyab Deploys D2 Step 3):** When Tayyab is ready to deploy Stage 2 to AWS/Vercel, complete the external AWS IAM/ECR/S3, Vercel `subtext-mlops-course`, and capped LLM API key setup listed in [`docs/HANDOFF_D1_TO_D2.md`](../../docs/HANDOFF_D1_TO_D2.md) Section 3.
