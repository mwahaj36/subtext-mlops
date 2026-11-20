# Subtext MLOps — Multi-Label Query Understanding & Retrieval Lifecycle

This repository hosts the isolated MLOps course extension of **Subtext**, a movie discovery application built around dense/hybrid semantic retrieval, taste-weighted re-ranking, and constraint filtering.

> **Scope & Provenance:** The underlying Subtext application predates this course as an independent side project (`Subtext` base commit `e32bb6dc366145a11d16c984acc9b0d24eb4bd4f`). See [`PROVENANCE.md`](PROVENANCE.md) for the exact inherited-vs-new work inventory and [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md) for verified team contributions. This course project adds a trained **fine-grained multi-label intent classification and bounded entity/attribute/exclusion extraction** pipeline with full DVC, MLflow, validation, and container reproduction lifecycle to improve the retrieval stage relevant to RAG (no generative chatbot is claimed or included).

---

## 1. ML Task & Versioned Prediction Contract (Deliverable 1)

Given a natural-language movie discovery query, the pipeline predicts:

1. **Fine-Grained Multi-Label Intents (`14` labels across `7` broad categories + derived `is_compound` flag, defined in [`configs/taxonomy.yaml`](configs/taxonomy.yaml)):**
   - **Affective:** `affective_outcome` (desired emotional impact on viewer), `affective_tone` (film mood/tone)
   - **Sensory:** `sensory_visual` (visual texture/lighting/setting), `sensory_pacing` (rhythm/tempo/score atmosphere)
   - **Person:** `person_actor`, `person_director`
   - **Similarity:** `similarity_plot` (story/premise similarity), `similarity_vibe` (mood/aesthetic similarity)
   - **Attribute:** `attribute_genre`, `attribute_era` (decade/year/range), `attribute_language_country`
   - **Negation:** `negation_genre` (excluded genre), `negation_content` (excluded trope/theme/element)
   - **Thematic:** `thematic_plot` (explicit plot situation or theme)
   - **Compound Rule:** `is_compound = len(set(broad_labels)) >= 2` (preserves constituent labels instead of hiding them inside an opaque compound class).
2. **Bounded Constraint Extraction ([`schemas/extraction.schema.json`](schemas/extraction.schema.json), [`ml/extract.py`](ml/extract.py)):**
   - Extracts and normalizes `actors`, `directors`, `genres`, `year_min`, `year_max`, `language`, `excluded_genres`, `excluded_terms`, and `reference_titles` grounded against repository indices (`backend/people_index.json`, `backend/galaxy_points.json`), while ignoring false-negation discourse filler (e.g., `"no idea what to watch"`).
3. **Versioned Route Selection ([`configs/routing.yaml`](configs/routing.yaml), [`schemas/prediction_contract.schema.json`](schemas/prediction_contract.schema.json)):**
   - Selects `similarity_anchor`, `structured_filter`, `hybrid_concept`, or `dense_semantic` with an explicit observable `route_reason`. Selective LLM fallback (`llm_fallback`) is defined in the schema and routing config and deferred to Deliverable 2 serving integration.

---

## 2. Dataset, Leakage Controls & Single DVC Remote

- **Dataset (`data/raw/d1_queries.jsonl`):** `455` reviewed multi-label queries across `219` paraphrase/scenario `group_id` families (`63`–`210` records and `29`–`98` groups per fine label; maximum cross-group token/3-gram Jaccard similarity `0.4444 < 0.85`).
- **Provenance Disclosure ([`data/manifests/dataset_provenance.json`](data/manifests/dataset_provenance.json)):** Course-curated synthetic starter dataset (`is_synthetic: true`, `synthetic_share: 1.0`) grounded in approved repository people (`334,960` names) and movie title (`98,524` films) indices, reviewed by `mwahaj36`. Contains zero production user logs and zero private research confirmation queries.
- **Group-Disjoint Splits ([`ml/split.py`](ml/split.py), [`data/manifests/split_manifest.json`](data/manifests/split_manifest.json)):**
  - `train`: `331` records (`158` groups)
  - `val`: `58` records (`29` groups) — used exclusively for model selection, temperature-scaling calibration, and per-label threshold tuning
  - `course_regression`: `66` records (`32` groups) — evaluated once per candidate to establish the course regression benchmark
- **Fail-Closed Automated Validation ([`ml/validate.py`](ml/validate.py), [`tests/test_ml_validation.py`](tests/test_ml_validation.py)):**
  - Runs automatically before model fitting (`ensure_valid_for_training`) and halts on missing/corrupt files, JSON schema violations, duplicate IDs, unknown labels, broad/compound inconsistencies, unsupported genres/languages, invalid year ranges, character span offset/substring mismatches, unapproved provenance, checksum tampering, `group_id` overlap, exact normalized duplicate overlap, cross-split near-duplicate leakage (`Jaccard >= 0.85`), or insufficient per-label split support.
- **Single DVC Remote ([`.dvc/config`](.dvc/config)):**
  - Uses DagsHub DVC storage (`https://dagshub.com/mwahaj36/subtext-mlops.dvc`) as the single remote `origin`. *Note vs. proposal:* DagsHub DVC storage is used in place of a separate S3 DVC prefix so experiment tracking, model registry, and DVC artifacts live under one course project without redundant remotes.

---

## 3. Deliverable 1 Experimental Results & Registered Model

All runs are tracked in DagsHub MLflow at `https://dagshub.com/mwahaj36/subtext-mlops.mlflow` (Experiment: `subtext-intent-d1`, ID `0`). The classical baseline was trained and logged **before** the primary MiniLM encoder.

| Model & Role | MLflow Run ID | Completed (UTC) | Val Macro-F1 | Val Micro-F1 | Val Subset Acc | Val ECE (Raw → Cal) | Regression Macro-F1 | Regression Micro-F1 | Regression ECE | Extraction Micro-F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Classical Baseline** (`tfidf_ovr_logreg`: word+char TF-IDF + OvR LogReg) | `8a2bd18153c94978b98b644e08e12b39` | `2026-11-19T18:50:53Z` | `0.8090` | `0.8091` | `0.3276` | `0.1170` → `0.0180` | `0.7827` | `0.7750` | `0.0232` | `0.9872` |
| **Primary Model** (`minilm_multilabel`: fine-tuned `all-MiniLM-L6-v2` + `LayerNorm` + linear head, best epoch `5`) | `01b02fc1496548f8b5bd4beb36e6cb99` | `2026-11-19T18:53:13Z` | **`0.8770`** | **`0.8727`** | **`0.4483`** | `0.0282` → `0.0287` | **`0.8183`** | **`0.7896`** | `0.0299` | `0.9872` |

- **Candidate Selection & Registration ([`ml/register.py`](ml/register.py), [`artifacts/registry/registration_summary.json`](artifacts/registry/registration_summary.json)):**
  - Under the predeclared selection rule (`val_macro_f1`, tie-breaker `val_micro_f1`), `minilm_multilabel` (`0.8770` vs. `0.8090`) was selected and registered in the DagsHub MLflow Model Registry as **`subtext-intent-multilabel` version `2`** (`models:/subtext-intent-multilabel/2`, registration run `f6d50f4c3b9a4fa8ad80c5f58a03b445`, bundle fingerprint `4fb663cb67dcc1f0`).
  - Fresh-process reload verification (`python -m ml.register --verify-only`) confirmed the registered `mlflow.pyfunc` bundle loads cleanly and outputs schema-valid prediction contracts.

---

## 4. One-Command Clean-Clone Reproduction

### Prerequisites & Grader Data/Tracking Access

- **Docker & Docker Compose** (or Python `3.11+` for local execution).
- **Hardware & Runtime:** Reference runs executed on CPU (`torch 2.15.0+cpu`, 16 GB+ system RAM recommended, ~2.5 GB disk for container image and Hugging Face checkpoint cache). Full pipeline runtime (`split_and_validate` → `train_baseline` → `train_primary` → `register_model`) is **~3–5 minutes on CPU**. Floating-point tolerance across CPU/GPU architectures is `±0.015` Macro-F1.
- **DagsHub DVC / MLflow Credentials:**
  - Copy `.env.example` to `.env` and populate `MLFLOW_TRACKING_URI=https://dagshub.com/mwahaj36/subtext-mlops.mlflow`, `MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD` (supplied to graders in the submission note), **or** run completely offline without `.env` (in credential-free clean-clone mode, `ml/run_pipeline.py` deterministically materializes and verifies the SHA-256-pinned raw dataset `93b8888f69318cc801aee52b295dbb6c63447d9b1eb9dfa889574844c83462f3` and logs to local `./mlruns`).
  - To pull prebuilt DVC artifacts directly from DagsHub DVC storage:
    ```bash
    dvc remote modify --local origin auth basic
    dvc remote modify --local origin user <dagshub_username>
    dvc remote modify --local origin password <dagshub_token>
    dvc pull
    ```

### One-Command Docker Reproduction (Works With or Without `.env`)

From a fresh clone of this repository:

```bash
docker compose run --rm train
```

This single command:
1. Pulls `data/raw/d1_queries.jsonl` from DagsHub DVC when credentials are provided (or deterministically builds and verifies SHA-256 `93b8888f69318cc801aee52b295dbb6c63447d9b1eb9dfa889574844c83462f3` in zero-credential mode).
2. Executes `dvc repro --force` across all 4 stages (`split_and_validate` → `train_baseline` → `train_primary` → `register_model`).
3. Logs runs to DagsHub MLflow (when `.env` is present) or local `./mlruns`, registers/packages the winning `mlflow.pyfunc` bundle idempotently, verifies fresh-process reload, and prints the JSON reproduction summary.

### Local Virtualenv Reproduction & Test Suite

```bash
python -m venv .venv
# Windows: .venv\Scripts\pip install -r requirements-ml.txt
# Linux/macOS: .venv/bin/pip install -r requirements-ml.txt

python -m ml.run_pipeline --force
pytest tests/test_ml_validation.py tests/test_ml_training.py -v
pytest backend/tests
ruff check ml/ tests/ backend/
```

---

## 5. Known Limitations & Evidence Bundle

- **Synthetic Starter Distribution:** Deliverable 1 uses a rubric-curated dataset (`455` queries across `219` groups) grounded in repository movie/people lexicons; live user traffic exhibits noisier spelling, colloquialisms, and out-of-lexicon entities that motivate selective LLM fallback and drift monitoring.
- **Deliverable 1 Evidence Bundle:** See [`evidence/d1/SUBMISSION.md`](evidence/d1/SUBMISSION.md).
