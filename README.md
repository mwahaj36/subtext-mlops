# Subtext MLOps — Multi-Label Query Understanding & Retrieval Lifecycle

This repository hosts the isolated MLOps course extension of **Subtext**, a movie discovery application built around dense/hybrid semantic retrieval, taste-weighted re-ranking, and constraint filtering.

> **Scope & Provenance:** The underlying Subtext application predates this course as an independent side project. See [`PROVENANCE.md`](PROVENANCE.md) for the exact inherited-vs-new inventory. This extension adds a trained **fine-grained multi-label intent classification and entity/attribute/exclusion extraction** pipeline with selective LLM fallback to improve the retrieval stage relevant to RAG (no generative RAG chatbot is claimed or included).

## Project Relay & Milestones

- **Deliverable 1 (Stage 1 — Muhammad Wahaj):** Versioned raw/processed data via DVC, automated schema/quality and group-disjoint split-leakage validation, classical multi-label baseline logged before MiniLM multi-label training, calibration/threshold tuning, MLflow experiment tracking and model registration on DagsHub, and one-command clean-clone reproduction.
- **Deliverable 2 Part A (Stage 2 — Tayyab Khalid):** Pinned serving API (`/health`, `/ready`, `/version`) with validation-tuned selective LLM fallback, concise prediction diagnostics in the isolated search UI, and separate cloud deployment.
- **Deliverable 2 Part B (Stage 3 — Fasih-ur-Rehman):** Evaluation-gated CI/CD blocking degraded models, live Prometheus/Grafana/Evidently latency and drift monitoring, DVC-versioned reviewed drift batch, automated retraining, champion/challenger promotion, and complete-bundle rollback.

See [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md) for verified team contributions.

## Operational Isolation

- Connects only to course-designated DagsHub MLflow tracking, DVC storage, isolated database/corpus snapshots, and a dedicated demo deployment.
- Does not share production credentials, user sessions, query logs, or live Hugging Face / Vercel deployment targets with the original Subtext product.

## Quick Verification (Base Application Tests)

Existing inherited unit tests run completely offline without external credentials:

```bash
python -m pip install -r backend/requirements.txt pytest ruff
ruff check backend/
pytest backend/tests
```

Frontend production build check:

```bash
cd frontend
npm ci
npm run build
```

Reproducible D1 training, DVC pull, MLflow tracking, and D2 serving/monitoring commands will be documented here as each milestone lands.
