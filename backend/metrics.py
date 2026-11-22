"""Thread-safe Prometheus 0.0.4 metrics registry with strictly bounded labels."""
from __future__ import annotations

import math
import threading
from collections import defaultdict
from typing import Iterable

LATENCY_BUCKETS = (5.0, 20.0, 50.0, 150.0, 400.0, 1000.0, 2500.0, math.inf)
LEN_BUCKETS = (35.0, 55.0, 75.0, 100.0, 200.0, 500.0, math.inf)
MARGIN_BUCKETS = (0.05, 0.10, 0.15, 0.25, 0.40, 1.0, math.inf)


def _esc(val: str) -> str:
    return val.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "")


def _fmt_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{_esc(v)}"' for k, v in labels)
    return f"{{{inner}}}"


class _ServingMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], int] = defaultdict(int)
        self._hist_buckets: dict[str, dict[tuple[tuple[str, str], ...], list[int]]] = defaultdict(dict)
        self._hist_sums: dict[str, dict[tuple[tuple[str, str], ...], float]] = defaultdict(lambda: defaultdict(float))
        self._hist_counts: dict[str, dict[tuple[tuple[str, str], ...], int]] = defaultdict(lambda: defaultdict(int))

    def _inc(self, name: str, labels: tuple[tuple[str, str], ...] = (), amount: int = 1) -> None:
        self._counters[(name, labels)] += amount

    def _observe(
        self,
        name: str,
        buckets: tuple[float, ...],
        value: float,
        labels: tuple[tuple[str, str], ...] = (),
    ) -> None:
        if labels not in self._hist_buckets[name]:
            self._hist_buckets[name][labels] = [0] * len(buckets)
        arr = self._hist_buckets[name][labels]
        for idx, bound in enumerate(buckets):
            if value <= bound:
                arr[idx] += 1
        self._hist_sums[name][labels] += float(value)
        self._hist_counts[name][labels] += 1

    def record_request(self, endpoint: str, status: int, total_ms: float) -> None:
        with self._lock:
            self._inc("subtext_requests_total", (("endpoint", endpoint), ("status", str(status))))
            self._observe("subtext_request_latency_ms", LATENCY_BUCKETS, total_ms, (("component", "total"),))

    def record_prediction(
        self,
        *,
        query_len: int,
        prediction: dict,
        model_ms: float,
        llm_ms: float = 0.0,
        retrieval_ms: float | None = None,
    ) -> None:
        scores = prediction.get("label_scores", {})
        thresholds = prediction.get("applied_thresholds", {})
        margins = [
            abs(float(scores[lbl]) - float(thresholds.get(lbl, 0.5)))
            for lbl in scores
        ]
        min_margin = min(margins) if margins else 0.0
        llm_status = str(prediction.get("llm_fallback", {}).get("status", "not_called"))
        route = str(prediction.get("selected_route", "dense_semantic"))
        is_comp = "true" if prediction.get("is_compound") else "false"

        with self._lock:
            self._observe("subtext_request_latency_ms", LATENCY_BUCKETS, model_ms, (("component", "small_model"),))
            if llm_ms > 0.0:
                self._observe("subtext_request_latency_ms", LATENCY_BUCKETS, llm_ms, (("component", "llm_fallback"),))
            if retrieval_ms is not None:
                self._observe("subtext_request_latency_ms", LATENCY_BUCKETS, retrieval_ms, (("component", "retrieval"),))
            for lbl in prediction.get("predicted_fine_labels", []):
                self._inc("subtext_intent_predictions_total", (("label", str(lbl)),))
            self._inc("subtext_compound_queries_total", (("is_compound", is_comp),))
            self._inc("subtext_route_selected_total", (("route", route),))
            self._inc("subtext_llm_fallback_total", (("status", llm_status),))
            self._observe("subtext_query_length_chars", LEN_BUCKETS, float(query_len))
            self._observe("subtext_confidence_margin", MARGIN_BUCKETS, float(min_margin))

    def _emit_hist(
        self,
        lines: list[str],
        name: str,
        help_text: str,
        buckets: Iterable[float],
    ) -> None:
        b_list = list(buckets)
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} histogram")
        for labels, counts in sorted(self._hist_buckets.get(name, {}).items()):
            for idx, bound in enumerate(b_list):
                le = "+Inf" if math.isinf(bound) else str(int(bound) if bound >= 1 and bound.is_integer() else bound)
                l_with_le = labels + (("le", le),)
                lines.append(f"{name}_bucket{_fmt_labels(l_with_le)} {counts[idx]}")
            lines.append(f"{name}_sum{_fmt_labels(labels)} {self._hist_sums[name][labels]:.4f}")
            lines.append(f"{name}_count{_fmt_labels(labels)} {self._hist_counts[name][labels]}")

    def render_prometheus(self) -> str:
        with self._lock:
            lines: list[str] = []
            by_metric: dict[str, list[tuple[tuple[tuple[str, str], ...], int]]] = defaultdict(list)
            for (name, labels), count in sorted(self._counters.items()):
                by_metric[name].append((labels, count))

            meta = {
                "subtext_requests_total": "Total HTTP requests handled by Subtext serving API",
                "subtext_intent_predictions_total": "Predicted fine-grained intent label counts",
                "subtext_compound_queries_total": "Compound vs single-category query counts",
                "subtext_route_selected_total": "Selected query-understanding retrieval route counts",
                "subtext_llm_fallback_total": "Selective LLM fallback status counts",
            }
            for name, help_text in meta.items():
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} counter")
                for labels, count in by_metric.get(name, []):
                    lines.append(f"{name}{_fmt_labels(labels)} {count}")

            self._emit_hist(lines, "subtext_request_latency_ms", "Serving component latency in ms", LATENCY_BUCKETS)
            self._emit_hist(lines, "subtext_query_length_chars", "Input query length distribution in chars", LEN_BUCKETS)
            self._emit_hist(lines, "subtext_confidence_margin", "Minimum distance to calibrated thresholds", MARGIN_BUCKETS)
            return "\n".join(lines) + "\n"


metrics_registry = _ServingMetrics()
