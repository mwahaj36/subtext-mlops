# Contributions Ledger

Authorship and PR records follow actual verified work performed by each team member under their own GitHub account.

## Team Members

| Member | Roll Number | GitHub Account | Scope |
|---|---|---|---|
| Muhammad Wahaj | 23L-0560 | `mwahaj36` | Isolated repository & provenance, DagsHub MLflow & single DVC remote, 14-label fine-grained taxonomy & JSON schemas, curated D1 dataset & provenance, fail-closed schema/leakage validation & group-disjoint splitting, classical baseline + MiniLM multi-label training & calibration, MLflow Model Registry bundle & fresh-process reload, one-command Docker reproduction, and `evidence/d1/SUBMISSION.md` |
| Tayyab Khalid | 23L-0501 | `TayyabKhalid25` | Pinned serving API (`/health`, `/ready`, `/version`), selective LLM fallback & contract validation, UI prediction diagnostics, and cloud deployment |
| Fasih-ur-Rehman | 23L-0526 | `fasih-ur-rehman1` | Evaluation-gated CI/CD, live monitoring & alerting, DVC-versioned drift batch, automated retraining, champion/challenger promotion, and rollback |

## Recorded Deliverables & Activity

| Date | Deliverable | Member | Work Performed | Verification / Artifact |
|---|---|---|---|---|
| 2026-11-19 | Deliverable 1 | Muhammad Wahaj (`mwahaj36`) | Seeded isolated course repository from base Subtext commit `e32bb6dc366145a11d16c984acc9b0d24eb4bd4f` with `PROVENANCE.md`, course `.gitignore`, and `.env.example`; configured collaborators and `main` branch protection | Initial commit `631d278`; [`PROVENANCE.md`](PROVENANCE.md) |
| 2026-11-19 | Deliverable 1 | Muhammad Wahaj (`mwahaj36`) | Configured DagsHub DVC remote (`.dvc/config`), bounded 14-label intent taxonomy and contracts (`configs/`, `schemas/`), curated D1 dataset (`455` queries / `219` groups) pushed to DagsHub DVC, deterministic extraction baseline (`ml/extract.py`), fail-closed validation (`ml/validate.py`), group-disjoint stratified splits (`ml/split.py`), and corruption/leakage unit tests (`tests/test_ml_validation.py`) | Merged PR [#1](https://github.com/mwahaj36/subtext-mlops/pull/1) (`e1da451`) |
| 2026-11-19 | Deliverable 1 | Muhammad Wahaj (`mwahaj36`) | Implemented temperature-scaling calibration and threshold tuning (`ml/evaluate.py`), classical TF-IDF+OvR Logistic Regression baseline logged before fine-tuned MiniLM multi-label encoder (`ml/train_baseline.py`, `ml/train.py`), DagsHub MLflow Model Registry packaging and fresh-process reload verification (`ml/register.py`, `subtext-intent-multilabel` v2), real 4-stage DVC pipeline (`dvc.yaml`, `dvc.lock`) pushed to DagsHub DVC, and one-command Docker reproduction (`Dockerfile.train`, `compose.yaml`, `ml/run_pipeline.py`) | Merged PR [#2](https://github.com/mwahaj36/subtext-mlops/pull/2) (`4cf65a8`); DagsHub MLflow runs `8a2bd18153c94978b98b644e08e12b39`, `01b02fc1496548f8b5bd4beb36e6cb99`, `f6d50f4c3b9a4fa8ad80c5f58a03b445` |
| 2026-11-20 | Deliverable 1 | Muhammad Wahaj (`mwahaj36`) | Verified clean-clone `dvc pull` and zero-cache one-command `docker compose run --rm train` reproduction; compiled D1 evidence bundle (`evidence/d1/SUBMISSION.md`) | PR [#3](https://github.com/mwahaj36/subtext-mlops/pull/3); tag `deliverable-1` |
