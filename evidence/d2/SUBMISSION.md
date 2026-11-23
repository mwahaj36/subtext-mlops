# Deliverable 2 — Final Submission Package & Verification (`subtext-mlops`)

## 1. Ready-to-Paste Submission Summary

```text
Project: Subtext MLOps — Production Serving, Eval-Gated CI/CD, Observability & Closed-Loop Retraining
Repository: https://github.com/mwahaj36/subtext-mlops
Release Tag: https://github.com/mwahaj36/subtext-mlops/releases/tag/deliverable-2 (D1 baseline: deliverable-1)

Team Members:
- Muhammad Wahaj (23L-0560) — mwahaj36
- Tayyab Khalid (23L-0501) — TayyabKhalid25
- Fasih-ur-Rehman (23L-0526) — fasih-ur-rehman1

1. Live Deployment URLs (Isolated Course Stack):
- Live Frontend UI: https://subtext-mlops-course.vercel.app/search
- Live Serving API: https://54-210-225-89.sslip.io
  - Health / Readiness: https://54-210-225-89.sslip.io/health , https://54-210-225-89.sslip.io/ready
  - Live Model Version & Digest: https://54-210-225-89.sslip.io/version
  - Live Prometheus Metrics: https://54-210-225-89.sslip.io/metrics

2. Eval-Gated CI/CD Proofs:
- CI Workflow: .github/workflows/ci.yml (backend, frontend, ml-gates)
- Passing Gate Run (PR #9): https://github.com/mwahaj36/subtext-mlops/actions/runs/45082948425 (all 8 checks PASS)
- Blocked Regression Proof (PR #10, closed unmerged): https://github.com/mwahaj36/subtext-mlops/actions/runs/45083118624 (reg_macro_f1_floor 0.8183 < 0.9900 -> exit 1)

3. Live Observability & Drift Monitoring:
- Prometheus Exporter: backend/metrics.py -> GET /metrics
- Prometheus & Alerting Configs: monitoring/prometheus.yml, monitoring/alert_rules.yml, monitoring/grafana/dashboards/subtext_serving.json
- Drift Detector & Real Alert Report: monitoring/drift_monitor.py -> evidence/d2/drift_report.json, evidence/d2/drift_report.html (query_length_psi = 8.2962 >= 0.25 on 30-query DVC batch -> alert_triggered: true, retrain_recommended: true)

4. Closed-Loop Retraining, Promotion & Rollback:
- DVC Reviewed Drift Batch: data/demo_batches/d2_drift_batch_reviewed.jsonl.dvc (MD5: 22799024651fdab6b8608d9688dd6856) + unreviewed queue block test (data/demo_batches/unreviewed_queue_sample.json)
- Automated Retrain & Release Workflows: .github/workflows/retrain.yml, .github/workflows/release.yml
- DagsHub MLflow Registry: https://dagshub.com/mwahaj36/subtext-mlops.mlflow/#/models/subtext-intent-multilabel
  - Champion v2 (D1): Run 01b02fc1496548f8b5bd4beb36e6cb99 (Val Macro-F1 = 0.8770, Reg Macro-F1 = 0.8183)
  - Challenger v3 (D2): Run 18deb0ca31c643c6b5a1a049dfd604b4 (Val Macro-F1 = 0.8978, Reg Macro-F1 = 0.8594; all 8 gates PASSED)
- Instant Rollback Verification: https://github.com/mwahaj36/subtext-mlops/actions/runs/45085918888 (45s switch to Champion digest sha256:19753266eba1199622f5e37a86c068c88a1aa9de2b6d05a99198c22ce246a4e5)

5. Evidence Files:
- evidence/d2/serving_and_deploy_evidence.md
- evidence/d2/ci_gates_evidence.md
- evidence/d2/monitoring_and_drift_evidence.md
- evidence/d2/retrain_promotion_rollback_evidence.md
- evidence/d2/retrain_comparison.json
- evidence/d2/drift_report.json & evidence/d2/drift_report.html
- Demo Video Link: <INSERT_5_MIN_DEMO_VIDEO_LINK_AFTER_RECORDING>
```

---

## 2. Deliverable 2 Requirement Mapping

| Deliverable 2 Requirement | Implemented Artifacts & Live Evidence |
|---|---|
| **Serving API & UI + Explanations** | Containerized FastAPI (`Dockerfile.serve`, `backend/main.py`, `backend/query_understanding.py`, `backend/course_catalog.py`) on EC2 HTTPS + Caddy (`https://54-210-225-89.sslip.io`) & isolated Vercel UI (`https://subtext-mlops-course.vercel.app/search`) with `QueryDiagnosticsPanel.js` showing intent confidences, route reason, constraints, LLM status, retrieval modes, and fallback handling; see [`serving_and_deploy_evidence.md`](serving_and_deploy_evidence.md). |
| **Eval-Gated CI/CD** | `configs/gates.yaml`, `ml/gate.py`, `tests/test_d2_gates.py`, `.github/workflows/ci.yml` (`ml-gates` job). Verified passing on PR [`#9`](https://github.com/mwahaj36/subtext-mlops/pull/9) and blocking failing threshold on closed PR [`#10`](https://github.com/mwahaj36/subtext-mlops/pull/10); see [`ci_gates_evidence.md`](ci_gates_evidence.md). |
| **Live Observability** | `backend/metrics.py` (`GET /metrics`), `configs/monitoring.yaml`, `monitoring/prometheus.yml`, `monitoring/alert_rules.yml`, `monitoring/grafana/dashboards/subtext_serving.json`, `monitoring/drift_monitor.py`, and generated [`drift_report.json`](drift_report.json)/[`drift_report.html`](drift_report.html); see [`monitoring_and_drift_evidence.md`](monitoring_and_drift_evidence.md). |
| **Closed-Loop Retraining & Safe Promotion/Rollback** | `ml/build_d2_drift_batch.py`, DVC batch (`d2_drift_batch_reviewed.jsonl.dvc`), unreviewed queue block (`unreviewed_queue_sample.json`), `ml/retrain.py`, `.github/workflows/retrain.yml`, `.github/workflows/release.yml`, MLflow Champion v2 vs Challenger v3 comparison ([`retrain_comparison.json`](retrain_comparison.json)), and live EC2 promote/rollback verification; see [`retrain_promotion_rollback_evidence.md`](retrain_promotion_rollback_evidence.md). |
