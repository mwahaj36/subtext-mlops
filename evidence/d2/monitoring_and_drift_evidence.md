# Live Observability & Drift Monitoring Evidence (Deliverable 2)

## 1. Prometheus Serving Telemetry (`GET /metrics`)
- **Implementation:** `backend/metrics.py`, wired into `/predict`, `/search`, `/ready`, and `/metrics` in `backend/main.py`, merged via PR [`#11`](https://github.com/mwahaj36/subtext-mlops/pull/11) (`03d07a0c7d164d034d277824f5e37a86c068c88a`).
- **Privacy Guard:** Tested in `tests/test_d2_monitoring.py::test_prometheus_metrics_endpoint_and_no_raw_query_leak` — raw query strings are never exported in Prometheus label values.
- **Exported Metric Families:**
  - `subtext_requests_total{endpoint,status}`
  - `subtext_intent_predictions_total{label}`
  - `subtext_compound_queries_total{is_compound}`
  - `subtext_route_selected_total{route}`
  - `subtext_llm_fallback_total{status}`
  - `subtext_request_latency_ms` (`component="small_model" | "llm" | "retrieval" | "total"`)
  - `subtext_query_length_chars`
  - `subtext_confidence_margin`
- **Live EC2 Scrape Excerpt (`GET https://54-210-225-89.sslip.io/metrics`):**

```text
# HELP subtext_requests_total Total HTTP requests handled by Subtext serving API
# TYPE subtext_requests_total counter
subtext_requests_total{endpoint="/predict",status="200"} 2
# HELP subtext_intent_predictions_total Predicted fine-grained intent label counts
# TYPE subtext_intent_predictions_total counter
subtext_intent_predictions_total{label="affective_outcome"} 1
subtext_intent_predictions_total{label="affective_tone"} 1
subtext_intent_predictions_total{label="attribute_genre"} 2
subtext_intent_predictions_total{label="attribute_language_country"} 1
subtext_intent_predictions_total{label="negation_content"} 1
subtext_intent_predictions_total{label="person_actor"} 1
# HELP subtext_route_selected_total Selected query-understanding retrieval route counts
# TYPE subtext_route_selected_total counter
subtext_route_selected_total{route="structured_filter"} 2
# HELP subtext_llm_fallback_total Selective LLM fallback status counts
# TYPE subtext_llm_fallback_total counter
subtext_llm_fallback_total{status="not_called"} 2
# HELP subtext_request_latency_ms Serving component latency in ms
# TYPE subtext_request_latency_ms histogram
subtext_request_latency_ms_sum{component="small_model"} 702.2652
subtext_request_latency_ms_count{component="small_model"} 2
```

---

## 2. Prometheus & Grafana Stack Configuration
- **Prometheus Scrape Config:** [`monitoring/prometheus.yml`](../../monitoring/prometheus.yml) (15s scrape interval against `/metrics`).
- **Alert Rules:** [`monitoring/alert_rules.yml`](../../monitoring/alert_rules.yml) (`SubtextHigh5xxErrorRate`, `SubtextHighP95Latency`, `SubtextLLMFallbackFailureRate`, `SubtextServiceNotReady`).
- **Grafana Dashboard:** [`monitoring/grafana/dashboards/subtext_serving.json`](../../monitoring/grafana/dashboards/subtext_serving.json) (RPS, 5xx error rate, P95 latency breakdown, LLM fallback rate, route share, and intent prediction mix).

---

## 3. Input & Prediction Drift Monitor (`monitoring/drift_monitor.py`)
- **Reference Distribution (`configs/monitoring.yaml`):** Grounded in D1 training split (`n=331`, `reference_version: d1-train-v1`), with length buckets (`<45`: `0.2538`, `45-80`: `0.7281`, `>80`: `0.0181`) and 14 fine-label training proportions.
- **Anti-Noise Guards:** Minimum sample guard (`min_sample_size: 20`) and alert cooldown (`cooldown_seconds: 900`).
- **Real Drift Detection Run on `data/demo_batches/d2_drift_batch_reviewed.jsonl` (`n=30`):**
  - Artifacts: [`evidence/d2/drift_report.json`](drift_report.json) and [`evidence/d2/drift_report.html`](drift_report.html)
  - Observed outputs:
    - `sample_size`: `30` (`>= 20`)
    - `query_length_psi`: **`8.2962`** (threshold `>= 0.25` -> **ALERT TRIGGERED**)
    - `label_prevalence_l1`: `0.181` (threshold `0.35`)
    - `alert_triggered`: `true`
    - `alerts`: `["query_length_psi=8.2962 >= 0.25"]`
    - `retrain_recommended`: **`true`**
