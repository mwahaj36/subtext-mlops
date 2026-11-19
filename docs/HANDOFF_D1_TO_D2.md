# Stage 1 to Stage 2 & Stage 3 Handoff (`deliverable-1` → Deliverable 2)

This document records the frozen Deliverable 1 model release, versioned contracts, dataset/corpus checksums, shared infrastructure inventory, teardown procedures, and file-ownership boundaries for **Stage 2 (Tayyab Khalid, `23L-0501`)** and **Stage 3 (Fasih-ur-Rehman, `23L-0526`)**.

---

## 1. Pinned Deliverable 1 Registered Model Release

Stage 2 serving (`backend/`) and Stage 3 evaluation gates (`.github/workflows/`, `ml/`) must load this exact pinned release rather than an unpinned floating alias or inference-time training run.

| Field | Pinned Value |
|---|---|
| **Registered Model Name** | `subtext-intent-multilabel` |
| **Pinned Registered Version** | `2` |
| **MLflow Registry URI** | `models:/subtext-intent-multilabel/2` |
| **DVC-Tracked Local PyFunc Bundle** | `artifacts/registry/pyfunc_model` (fetched via `dvc pull`) |
| **Selected Architecture** | `minilm_multilabel` (`sentence-transformers/all-MiniLM-L6-v2` fine-tuned encoder + `LayerNorm` + linear multi-label classification head) |
| **Pinned Pretrained Base Checkpoint** | `sentence-transformers/all-MiniLM-L6-v2` @ HF commit `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| **Bundle Fingerprint** | `4fb663cb67dcc1f0` |
| **Primary Training MLflow Run ID** | `01b02fc1496548f8b5bd4beb36e6cb99` |
| **Preceding Classical Baseline Run ID** | `8a2bd18153c94978b98b644e08e12b39` (`tfidf_ovr_logreg`, registered as version `1` for baseline comparison) |
| **Registration MLflow Run ID** | `f6d50f4c3b9a4fa8ad80c5f58a03b445` |
| **Validation / Regression Metrics** | Validation Macro-F1: `0.8770` (vs. baseline `0.8090`), Validation Micro-F1: `0.8727` (vs. `0.8091`), Course Regression Macro-F1: `0.8183` (vs. `0.7827`), Course Regression Micro-F1: `0.7896` (vs. `0.7750`), Extraction Micro-F1: `0.9872` |

### Canonical Label Order, Calibrated Temperatures, and Decision Thresholds

Defined in `configs/taxonomy.yaml` (`version: "1.0.0"`) and saved in `artifacts/primary/bundle_manifest.json` and `artifacts/registry/registration_summary.json`:

| Index | Fine Label | Broad Category | Temperature (`T_j`) | Decision Threshold (`th_j`) |
|---|---|---|---|---|
| 0 | `affective_outcome` | `affective` | `1.0031` | `0.60` |
| 1 | `affective_tone` | `affective` | `1.2177` | `0.36` |
| 2 | `sensory_visual` | `sensory` | `1.0302` | `0.22` |
| 3 | `sensory_pacing` | `sensory` | `0.8923` | `0.38` |
| 4 | `person_actor` | `person` | `0.2500` | `0.50` |
| 5 | `person_director` | `person` | `0.2500` | `0.50` |
| 6 | `similarity_plot` | `similarity` | `1.2340` | `0.50` |
| 7 | `similarity_vibe` | `similarity` | `0.6809` | `0.78` |
| 8 | `attribute_genre` | `attribute` | `1.0140` | `0.50` |
| 9 | `attribute_era` | `attribute` | `0.2500` | `0.50` |
| 10 | `attribute_language_country` | `attribute` | `0.2500` | `0.50` |
| 11 | `negation_genre` | `negation` | `0.7405` | `0.56` |
| 12 | `negation_content` | `negation` | `0.3942` | `0.24` |
| 13 | `thematic_plot` | `thematic` | `1.8540` | `0.46` |

- **Compound Query Rule:** `is_compound = len(set(predicted_broad_labels)) >= 2`.
- **Fallback Rule:** When all 14 calibrated probabilities fall below their per-label threshold, the top-scoring fine label is emitted (`ml/evaluate.py:apply_thresholds`), and in Deliverable 2 selective LLM escalation may be triggered when confidence margin is below `max_confidence_margin_for_escalation` (`0.15` in `configs/routing.yaml`).

---

## 2. Frozen Schemas, Dataset Partitions, and Movie Corpus Snapshot

### Schemas and Config Contracts

- **Taxonomy & Controlled Vocabularies:** `configs/taxonomy.yaml` (`version: "1.0.0"`)
- **Training & Split Parameters:** `configs/train.yaml` (`seed: 42`)
- **Routing Policy Contract:** `configs/routing.yaml` (`version: "1.0.0"`, routes: `similarity_anchor`, `structured_filter`, `hybrid_concept`, `dense_semantic`, plus `llm_fallback` policy parameters for Stage 2)
- **Dataset Record Schema:** `schemas/dataset_record.schema.json`
- **Bounded Extraction Schema:** `schemas/extraction.schema.json` (`actors`, `directors`, `genres`, `year_min`, `year_max`, `language`, `excluded_genres`, `excluded_terms`, `reference_titles`)
- **Serving Prediction Response Schema:** `schemas/prediction_contract.schema.json` (`schema_version: "1.0.0"`)

### Frozen D1 Dataset and Group-Disjoint Split Checksums

Managed via DVC (`data/raw/d1_queries.jsonl.dvc`, `dvc.yaml`, `dvc.lock`) and verified by `ml/validate.py`:

| Artifact | Path | Records / Groups | SHA-256 / DVC MD5 |
|---|---|---|---|
| **Raw D1 Dataset** | `data/raw/d1_queries.jsonl` | `455` records / `219` groups | SHA-256: `93b8888f69318cc801aee52b295dbb6c63447d9b1eb9dfa889574844c83462f3`<br>DVC MD5: `d115780c1b7ed690fcf529d8afce9aea` |
| **Train Partition** | `data/processed/train.jsonl` | `331` records / `158` groups | SHA-256: `4adf8455a2190b54b79313ba3b45dc0f2a569e7bbb35a8611f8a21b23df6c29c` |
| **Validation Partition** | `data/processed/val.jsonl` | `58` records / `29` groups | SHA-256: `a55c6b889b79de151db0e55f48a148231bb1d0bd7b61cca11d3d6cc2e57f9342` |
| **Course Regression Benchmark** | `data/processed/course_regression.jsonl` | `66` records / `32` groups | SHA-256: `d94d739435f347f9adbff2563a8282ee7993834d3a216a3708abfbe2d4f3a996` |
| **Provenance Manifest** | `data/manifests/dataset_provenance.json` | — | Tracked in Git |
| **Split Manifest** | `data/manifests/split_manifest.json` | — | Tracked in Git (`cache: false`) |
| **Validation Report** | `data/manifests/validation_report.json` | — | Tracked in Git (`cache: false`) |

### Approved Movie Corpus & Lexicon Snapshot (Zero Live AstraDB Dependency)

The repository includes frozen read-only movie, people, genre, and concept-index snapshots inside `backend/` so Stage 2 and Stage 3 never connect to the live Subtext production AstraDB or Neon database:

- `backend/galaxy_points.json` (`98,524` movie IDs, titles, and 3D coordinates)
- `backend/people_index.json` (`334,960` canonical actor/director names)
- `backend/genre_centroids.json` (precomputed genre centroids)
- `backend/concept_index.npz` & `backend/corpus_mean.npy` (offline centered TMDB keyword concept vectors)
- `backend/linear_projection.npz` (projection matrix)

---

## 3. Non-Secret Shared Infrastructure Inventory (Deliverable 1 + Deliverable 2)

| Component | Course Resource / Endpoint | Notes & Isolation Guarantees |
|---|---|---|
| **GitHub Repository** | `https://github.com/mwahaj36/subtext-mlops` | Public repo; `main` branch protected (PR required, status checks `backend` and `frontend` required, force-push and branch deletion disabled). Collaborators: `mwahaj36` (admin), `fasih-ur-rehman1` (write), `TayyabKhalid25` (write). |
| **DagsHub Project** | `https://dagshub.com/mwahaj36/subtext-mlops` | Connected exclusively to `mwahaj36/subtext-mlops` (live `Subtext` repo excluded). |
| **MLflow Tracking & Registry** | `https://dagshub.com/mwahaj36/subtext-mlops.mlflow` | Experiment: `subtext-intent-d1` (ID `0`). Registered Model: `subtext-intent-multilabel` (active D1 version: `2`). |
| **Single DVC Remote** | `https://dagshub.com/mwahaj36/subtext-mlops.dvc` | Configured as default remote `origin` in `.dvc/config` (single remote only; replaces the proposal's S3 DVC remote to avoid redundant storage and simplify team/grader access). |
| **GitHub Actions Secrets Configured** | `MLFLOW_TRACKING_URI`<br>`MLFLOW_TRACKING_USERNAME`<br>`MLFLOW_TRACKING_PASSWORD` | Repository secrets configured on `mwahaj36/subtext-mlops`. Additional Stage 2/3 deployment secrets listed below are added when cloud serving/LLM endpoints are enabled. |
| **AWS Course Deployment Target (D2 Step 3)** | Region: `us-east-1`<br>Budget Alert: `subtext-mlops-course-cap` (\$10/month alert thresholds at 50%/80%/100%)<br>IAM User: `subtext-mlops-course-ci` (scoped strictly to course ECR/S3/EC2 tags)<br>ECR Repo: `subtext-mlops-serve`<br>S3 Bucket: `subtext-mlops-course-mwahaj36`<br>EC2 Instance Tag: `Project=subtext-mlops-course` | No hourly EC2 compute is left running during Stage 1 (**\$0** idle burn). Stage 2 builds `Dockerfile.serve`, pushes digest-pinned images to `subtext-mlops-serve`, and deploys to a single tagged EC2 instance with HTTPS reverse proxy. |
| **Separate Course Frontend Target (D2 Step 3)** | Vercel project: `subtext-mlops-course` (Root directory: `frontend`, `NEXT_PUBLIC_API_URL` pointing only to the course EC2 HTTPS API) | Completely separate from live `sbtxt.vercel.app`. |
| **Isolated Course Demo Database (D2 Steps 2–3)** | `postgres:16-alpine` container in Stage 2's EC2 Docker Compose (`DATABASE_URL=postgresql://subtext_demo:...@db:5432/subtext_course`) | Zero connection to live production Neon database; mass sync and email sending disabled by default. |
| **Selective LLM Fallback Provider (D2 Step 1)** | OpenAI-compatible endpoint configured via `LLM_PROVIDER_BASE_URL`, `LLM_MODEL_ID` (e.g., pinned `qwen/qwen3-4b`), and `LLM_API_KEY` (course-only key with hard \$5 spend cap; pricing reference date recorded at activation) | Stage 2 stores versioned LLM response snapshots for deterministic regression gates while testing live connectivity separately. |
| **Observability & Alert Target (D2 Step 5)** | Prometheus + Grafana services in EC2 Docker Compose (`monitoring/`) + course-only Discord/Slack incoming webhook (`ALERT_WEBHOOK_URL`) | Raw free-text queries must never be used as Prometheus labels. |

### Required GitHub Actions Secret Names Across Stages

- **Provisioned in Stage 1:**
  - `MLFLOW_TRACKING_URI` (`https://dagshub.com/mwahaj36/subtext-mlops.mlflow`)
  - `MLFLOW_TRACKING_USERNAME` (`mwahaj36`)
  - `MLFLOW_TRACKING_PASSWORD` (course DagsHub token)
- **To be set when Stage 2 / Stage 3 activate cloud deployment & LLM fallback:**
  - `AWS_REGION` (`us-east-1`), `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (for `subtext-mlops-course-ci`)
  - `EC2_HOST`, `EC2_SSH_KEY` (for Stage 2/3 deployment & rollback workflow)
  - `LLM_PROVIDER_BASE_URL`, `LLM_MODEL_ID`, `LLM_API_KEY`
  - `ALERT_WEBHOOK_URL`

### Private-Channel Access Sharing Steps for Tayyab (Stage 2) and Fasih (Stage 3)

1. Each team member authenticates with their **own** GitHub account (`TayyabKhalid25` for Stage 2, `fasih-ur-rehman1` for Stage 3) via `gh auth login`. Never share GitHub personal access tokens.
2. Receive the course `.env` values (`MLFLOW_TRACKING_PASSWORD` / DagsHub token and D2 provider keys) over the team's private encrypted channel and place them in local gitignored `.env`.
3. Configure local DVC authentication (stored strictly in gitignored `.dvc/config.local`):
   ```bash
   dvc remote modify --local origin auth basic
   dvc remote modify --local origin user mwahaj36
   dvc remote modify --local origin password <course_dagshub_token>
   dvc pull
   ```

### Exact Teardown Procedure (Confirming Live Subtext Is Untouched)

At course completion, tear down **only** course-tagged resources (never touch `sbtxt.vercel.app`, `mwahaj36-subtext.hf.space`, production Neon, or production AstraDB):

1. **AWS EC2 / ECR / S3 / IAM (if launched in Stage 2):**
   ```bash
   # Terminate only EC2 instances tagged Project=subtext-mlops-course
   aws ec2 describe-instances \
     --filters "Name=tag:Project,Values=subtext-mlops-course" "Name=instance-state-name,Values=running,stopped" \
     --query "Reservations[].Instances[].InstanceId" --output text \
     | xargs -r aws ec2 terminate-instances --instance-ids

   # Delete course ECR repository and course S3 bucket
   aws ecr delete-repository --repository-name subtext-mlops-serve --force --region us-east-1
   aws s3 rb s3://subtext-mlops-course-mwahaj36 --force --region us-east-1

   # Delete course IAM user access keys, inline policies, and user
   aws iam delete-user --user-name subtext-mlops-course-ci
   ```
2. **Vercel Course Frontend:** Delete project `subtext-mlops-course` in Vercel settings (leave `sbtxt.vercel.app` intact).
3. **DagsHub & LLM Token:** Revoke token `subtext-mlops-course` at `https://dagshub.com/user/settings/tokens` and revoke the course LLM API key.

---

## 4. Strict File-Ownership Boundaries for Stage 2 and Stage 3

To keep `deliverable-1`, `evidence/d1/`, and frozen D1 splits/models reproducible throughout Deliverable 2:

| Stage & Owner | Branch Prefix | Allowed Paths | Forbidden Paths (Must Remain Untouched) |
|---|---|---|---|
| **Stage 2:** Tayyab Khalid (`23L-0501`, `TayyabKhalid25`) | `tayyab/d2-*` | `backend/`, `frontend/`, `prompts/`, `schemas/` (additive D2 serving fields only), `configs/routing.yaml`, `deploy/`, `Dockerfile.serve`, `compose.yaml` (additive `serve` service), `tests/` (serving/fallback/UI integration tests), `evidence/d2/serving_and_deploy_evidence.md`, `docs/HANDOFF_D2A_TO_D2B.md`, Tayyab's row in `CONTRIBUTIONS.md` | `evidence/d1/`, `PROVENANCE.md`, `docs/HANDOFF_D1_TO_D2.md`, D1 stages in `dvc.yaml`/`dvc.lock`, `data/raw/`, `data/processed/`, `data/manifests/` D1 splits, `artifacts/` D1 bundles |
| **Stage 3:** Fasih-ur-Rehman (`23L-0526`, `fasih-ur-rehman1`) | `fasih/d2-*` | `.github/workflows/`, `ml/evaluate.py` (additive gate CLI) & `ml/retrain.py`, `configs/gates.yaml`, `configs/monitoring.yaml`, `monitoring/`, `data/demo_batches/` (+ additive DVC tracking strictly for demo batches), `tests/` (CI gate/monitoring/retrain/rollback tests), `evidence/d2/` (gate/monitoring/retrain/rollback evidence + `SUBMISSION.md`), Fasih's row in `CONTRIBUTIONS.md` | `evidence/d1/`, `PROVENANCE.md`, `docs/HANDOFF_D1_TO_D2.md`, `docs/HANDOFF_D2A_TO_D2B.md`, `evidence/d2/serving_and_deploy_evidence.md`, breaking changes to Stage 2's live serving contract |
