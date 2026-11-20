# Provenance and Scope Inventory

This repository (`subtext-mlops`) is an isolated MLOps course extension seeded from a pre-existing independent side project (**Subtext**). It is kept operationally and scientifically separate from the live Subtext product and private research artifacts.

## Base Snapshot

- **Origin repository:** `Subtext` (independent side project by Muhammad Wahaj, created prior to this course; not an FYP or prior course project)
- **Exported source commit:** `e32bb6dc366145a11d16c984acc9b0d24eb4bd4f`
- **Isolation rules:** Independent Git history and remotes (`origin` only; no production `hf` remote), separate database/corpus, separate tracking/storage, and separate course deployment.

## Inherited vs. New Work Inventory

### 1. Inherited from Base Subtext (`e32bb6dc366145a11d16c984acc9b0d24eb4bd4f`)

- `backend/`: FastAPI application (`main.py`), dense/hybrid vector search and quantization engine (`search*.py`, `engine/hnsw.py`, `engine/quantization.py`, `concepts.py`, `taste.py`), Letterboxd sync/auth/DB utilities (`auth.py`, `database.py`, `sync*.py`, `movie.py`), precomputed indices (`concept_index.npz`, `corpus_mean.npy`, `linear_projection.npz`, `galaxy_points.json`, `genre_centroids.json`, `people_index.json`), and existing unit tests (`backend/tests/`).
- `frontend/`: Next.js user interface (search, profile, onboarding, auth, galaxy visualizer, shared components).
- Base build and lint configs: `Dockerfile`, `backend/Dockerfile`, `pyproject.toml`, `pyrightconfig.json`, `requirements.txt`, `.github/workflows/ci.yml`, `TMDB_DATA_REFERENCE.md`, `GMAIL_OAUTH_SETUP.md`, `start_subtext.bat`.

### 2. Deliberately Excluded from Base Snapshot

- `.git/` history, `.env` secrets, and Hugging Face deployment remote (`hf`).
- `.github/workflows/daily_sync.yml` (scheduled production mass-sync workflow).
- Private research drafts, production database exports, and live query logs.
- `.gitattributes` LFS pointer setup (tracked binaries copied directly so clean clones work without production LFS remotes).

### 3. New Course Work (Added in `subtext-mlops`)

- **Core ML Pipeline & Experiment Lifecycle:** Course configuration, bounded fine-grained multi-label intent taxonomy and entity/attribute/exclusion schemas (`configs/`, `schemas/`), reviewed and DVC-versioned training/validation/regression datasets (`data/`, `dvc.yaml`, `dvc.lock`), automated schema and split-leakage validation (`ml/validate.py`, `ml/split.py`), classical multi-label baseline + trained MiniLM encoder + calibration/threshold tuning (`ml/train_baseline.py`, `ml/train.py`, `ml/evaluate.py`), MLflow tracking and model registration (`ml/register.py`), one-command reproduction (`Dockerfile.train`, `compose.yaml`), and submission evidence (`evidence/d1/`).
- **Serving, Diagnostics, Monitoring & CI/CD:** Pinned model inference endpoints (`/health`, `/ready`, `/version`) and validation-tuned selective LLM fallback (`backend/`, `prompts/`), search UI prediction diagnostics (`frontend/`), separate cloud deployment (`Dockerfile.serve`, `deploy/`), evaluation-gated CI/CD (`.github/workflows/`, `configs/gates.yaml`), live observability (`monitoring/`, `configs/monitoring.yaml`), closed-loop drift retraining (`ml/retrain.py`, `data/demo_batches/`), and champion/challenger promotion & rollback (`evidence/d2/`).
