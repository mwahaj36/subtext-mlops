"""Tests for Prometheus /metrics exposition, label cardinality bounds, and drift monitor gating."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
for p in (REPO_ROOT, BACKEND_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import main  # noqa: E402
from monitoring.drift_monitor import evaluate_drift, render_html_report  # noqa: E402
from query_understanding import QueryUnderstandingService  # noqa: E402


@pytest.fixture(scope="module")
def loaded_service() -> QueryUnderstandingService:
    svc = QueryUnderstandingService(REPO_ROOT)
    assert svc.load() is True
    return svc


def test_prometheus_metrics_endpoint_and_no_raw_query_leak(
    monkeypatch: pytest.MonkeyPatch, loaded_service: QueryUnderstandingService
) -> None:
    monkeypatch.setattr(main, "qu_service", loaded_service)
    monkeypatch.setattr(main.search_runtime, "collection", None)
    client = TestClient(main.app)

    secret_phrase = "SecretUniqueQueryTextNeverInMetrics"
    resp = client.post("/predict", json={"query": f"Korean thriller with {secret_phrase}"})
    assert resp.status_code == 200

    search_resp = client.get(
        "/search",
        params={"q": "Korean crime thriller starring Song Kang-ho with no gore", "k": 3},
    )
    assert search_resp.status_code == 200

    m = client.get("/metrics")
    assert m.status_code == 200
    assert "text/plain" in m.headers["content-type"]
    body = m.text
    assert "subtext_requests_total" in body
    assert "subtext_intent_predictions_total" in body
    assert "subtext_route_selected_total" in body
    assert "subtext_llm_fallback_total" in body
    assert "subtext_request_latency_ms_bucket" in body
    assert secret_phrase not in body


def test_drift_monitor_sample_guard_alert_webhook_and_cooldown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = REPO_ROOT / "configs" / "monitoring.yaml"
    state_path = tmp_path / "alert_state.json"

    small_batch = tmp_path / "small.jsonl"
    small_rows = [{"query_length": 240, "predicted_fine_labels": ["thematic_plot"], "llm_fallback": {"called": True, "status": "succeeded"}, "latency_ms": 2100} for _ in range(5)]
    small_batch.write_text("\n".join(json.dumps(r) for r in small_rows) + "\n", encoding="utf-8")

    small_rep = evaluate_drift(small_batch, cfg_path, state_path=state_path, now_epoch=1000.0, webhook_url="")
    assert small_rep["alert_triggered"] is False
    assert small_rep["should_notify"] is False
    assert "below_min_sample_size" in str(small_rep["suppressed_reason"])

    drifted_batch = tmp_path / "drifted.jsonl"
    drifted_rows = [{"query_length": 260, "predicted_fine_labels": ["thematic_plot", "negation_content"], "llm_fallback": {"called": True, "status": "timeout"}, "latency_ms": 1800} for _ in range(25)]
    drifted_batch.write_text("\n".join(json.dumps(r) for r in drifted_rows) + "\n", encoding="utf-8")

    posted: list[dict[str, Any]] = []

    class _FakeResp:
        def raise_for_status(self) -> None:
            return None

    def _fake_post(url: str, json: dict[str, Any], timeout: float) -> _FakeResp:
        posted.append({"url": url, "payload": json, "timeout": timeout})
        return _FakeResp()

    monkeypatch.setattr("monitoring.drift_monitor.httpx.post", _fake_post)

    first_rep = evaluate_drift(
        drifted_batch,
        cfg_path,
        state_path=state_path,
        now_epoch=2000.0,
        webhook_url="https://hooks.example.invalid/alert",
    )
    assert first_rep["alert_triggered"] is True
    assert first_rep["should_notify"] is True
    assert first_rep["webhook_dispatched"] is True
    assert len(posted) == 1

    html = render_html_report(first_rep)
    assert "Subtext Serving Drift Report" in html

    dedup_rep = evaluate_drift(
        drifted_batch,
        cfg_path,
        state_path=state_path,
        now_epoch=2100.0,
        webhook_url="https://hooks.example.invalid/alert",
    )
    assert dedup_rep["alert_triggered"] is True
    assert dedup_rep["should_notify"] is False
    assert "cooldown_dedup_active" in str(dedup_rep["suppressed_reason"])
    assert len(posted) == 1
