"""Batch drift monitor comparing serving predictions against versioned D1 reference distributions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def compute_psi(expected_props: list[float], actual_counts: list[int]) -> float:
    total = max(sum(actual_counts), 1)
    psi = 0.0
    for exp_p, obs_cnt in zip(expected_props, actual_counts):
        p = max(float(exp_p), 1e-4)
        q = max(obs_cnt / total, 1e-4)
        psi += (q - p) * math.log(q / p)
    return round(float(psi), 4)


def _bin_lengths(lengths: list[int], edges: list[int]) -> list[int]:
    counts = [0] * (len(edges) - 1)
    for val in lengths:
        placed = False
        for i in range(len(edges) - 1):
            if edges[i] <= val < edges[i + 1]:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    return counts


def evaluate_drift(
    batch_path: Path,
    monitoring_cfg_path: Path,
    *,
    state_path: Path | None = None,
    now_epoch: float | None = None,
    webhook_url: str | None = None,
) -> dict[str, Any]:
    """Evaluate prediction log batch drift with sample-size guard, cooldown, and deduplication."""
    with monitoring_cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    records = [
        json.loads(line)
        for line in batch_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    n_samples = len(records)
    th = cfg["thresholds"]
    min_samples = int(th["min_sample_size"])
    cooldown = float(th["cooldown_seconds"])
    now_ts = float(now_epoch if now_epoch is not None else time.time())

    lengths = [int(r.get("query_length", len(str(r.get("text", ""))))) for r in records]
    edges = [int(x) for x in cfg["reference_distribution"]["query_length_edges"]]
    exp_len_props = [float(x) for x in cfg["reference_distribution"]["query_length_proportions"]]
    obs_len_counts = _bin_lengths(lengths, edges) if n_samples else [0] * len(exp_len_props)
    psi = compute_psi(exp_len_props, obs_len_counts) if n_samples else 0.0

    ref_lbl = {k: float(v) for k, v in cfg["reference_distribution"]["label_prevalence"].items()}
    lbl_counts: Counter[str] = Counter()
    llm_escalations = 0
    llm_failures = 0
    latencies: list[float] = []
    for rec in records:
        for lbl in rec.get("predicted_fine_labels", rec.get("fine_labels", [])):
            lbl_counts[str(lbl)] += 1
        llm = rec.get("llm_fallback", {})
        if llm.get("called"):
            llm_escalations += 1
            if llm.get("status") in {"timeout", "invalid_output", "error"}:
                llm_failures += 1
        if "latency_ms" in rec:
            latencies.append(float(rec["latency_ms"]))

    obs_lbl_prev = {k: round(lbl_counts[k] / max(n_samples, 1), 4) for k in ref_lbl}
    label_l1 = round(sum(abs(obs_lbl_prev[k] - ref_lbl[k]) for k in ref_lbl) / max(len(ref_lbl), 1), 4)
    esc_rate = round(llm_escalations / max(n_samples, 1), 4)
    err_rate = round(llm_failures / max(llm_escalations, 1), 4) if llm_escalations else 0.0
    p95_lat = round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 2) if latencies else 0.0

    raw_alerts: list[str] = []
    if n_samples >= min_samples:
        if psi >= float(th["psi_alert_threshold"]):
            raw_alerts.append(f"query_length_psi={psi} >= {th['psi_alert_threshold']}")
        elif psi >= float(th["psi_warning_threshold"]):
            raw_alerts.append(f"query_length_psi_warning={psi} >= {th['psi_warning_threshold']}")
        if label_l1 >= float(th["label_l1_drift_alert"]):
            raw_alerts.append(f"label_prevalence_l1={label_l1} >= {th['label_l1_drift_alert']}")
        if esc_rate >= float(th["llm_escalation_rate_alert"]):
            raw_alerts.append(f"llm_escalation_rate={esc_rate} >= {th['llm_escalation_rate_alert']}")
        if err_rate >= float(th["llm_error_rate_alert"]):
            raw_alerts.append(f"llm_error_rate={err_rate} >= {th['llm_error_rate_alert']}")
        if p95_lat >= float(th["p95_latency_ms_alert"]):
            raw_alerts.append(f"p95_latency_ms={p95_lat} >= {th['p95_latency_ms_alert']}")

    alert_key = hashlib.sha256("|".join(sorted(raw_alerts)).encode("utf-8")).hexdigest()[:12] if raw_alerts else ""
    suppressed_reason = None
    should_notify = False

    if n_samples < min_samples:
        suppressed_reason = f"below_min_sample_size ({n_samples} < {min_samples})"
    elif raw_alerts:
        prev_state = {}
        if state_path and state_path.exists():
            prev_state = json.loads(state_path.read_text(encoding="utf-8"))
        prev_ts = float(prev_state.get("last_alert_epoch", 0.0))
        prev_key = str(prev_state.get("last_alert_key", ""))
        if prev_key == alert_key and (now_ts - prev_ts) < cooldown:
            suppressed_reason = f"cooldown_dedup_active ({int(now_ts - prev_ts)}s < {int(cooldown)}s)"
        else:
            should_notify = True
            if state_path:
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(
                    json.dumps({"last_alert_epoch": now_ts, "last_alert_key": alert_key, "alerts": raw_alerts}, indent=2),
                    encoding="utf-8",
                )

    webhook_dispatched = False
    target_hook = webhook_url if webhook_url is not None else os.getenv("ALERT_WEBHOOK_URL", "").strip()
    if should_notify and target_hook:
        payload = {
            "text": f"[Subtext MLOps Drift Alert] ref={cfg['reference_version']} samples={n_samples} alerts={raw_alerts}",
            "alerts": raw_alerts,
            "reference_version": cfg["reference_version"],
            "sample_size": n_samples,
        }
        try:
            httpx.post(target_hook, json=payload, timeout=5.0).raise_for_status()
            webhook_dispatched = True
        except Exception:
            webhook_dispatched = False

    return {
        "monitoring_version": str(cfg["version"]),
        "reference_version": str(cfg["reference_version"]),
        "sample_size": n_samples,
        "min_sample_size": min_samples,
        "metrics": {
            "query_length_psi": psi,
            "label_prevalence_l1": label_l1,
            "llm_escalation_rate": esc_rate,
            "llm_error_rate": err_rate,
            "p95_latency_ms": p95_lat,
        },
        "alert_triggered": bool(raw_alerts) and (n_samples >= min_samples),
        "alerts": raw_alerts,
        "should_notify": should_notify,
        "suppressed_reason": suppressed_reason,
        "webhook_dispatched": webhook_dispatched,
        "retrain_recommended": psi >= float(th["psi_alert_threshold"]) or label_l1 >= float(th["label_l1_drift_alert"]) or esc_rate >= float(th["llm_escalation_rate_alert"]),
    }


def render_html_report(report: dict[str, Any]) -> str:
    m = report["metrics"]
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in m.items())
    alert_items = "".join(f"<li>{a}</li>" for a in report["alerts"]) or "<li>None</li>"
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>Subtext Drift Report</title>"
        "<style>body{font-family:sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;}"
        "table{border-collapse:collapse;width:100%;}td,th{border:1px solid #ddd;padding:8px;text-align:left;}"
        ".alert{background:#fff3cd;border:1px solid #ffeeba;padding:12px;margin:1rem 0;}</style></head><body>"
        f"<h1>Subtext Serving Drift Report ({report['reference_version']})</h1>"
        f"<p>Samples: <b>{report['sample_size']}</b> (min {report['min_sample_size']}) | "
        f"Alert Triggered: <b>{report['alert_triggered']}</b> | Notify: <b>{report['should_notify']}</b></p>"
        f"<div class='alert'><b>Active Alerts:</b><ul>{alert_items}</ul></div>"
        f"<table><tr><th>Metric</th><th>Observed</th></tr>{rows}</table></body></html>"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate prediction batch drift against D1 reference.")
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "monitoring.yaml")
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--html-out", type=Path, default=None)
    args = parser.parse_args(argv)

    rep = evaluate_drift(args.batch, args.config, state_path=args.state)
    print(json.dumps(rep, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    if args.html_out is not None:
        args.html_out.parent.mkdir(parents=True, exist_ok=True)
        args.html_out.write_text(render_html_report(rep), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
