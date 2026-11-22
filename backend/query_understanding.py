"""Pinned Deliverable 1 query-understanding service with validation-tuned selective LLM fallback."""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
import jsonschema
import numpy as np
import torch
import torch.nn as nn
import yaml
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.evaluate import apply_temperature_scaling, apply_thresholds, select_route  # noqa: E402
from ml.extract import extract_constraints, load_lexicons  # noqa: E402
from backend.search_rules import _NEG_SKIP  # noqa: E402

logger = logging.getLogger("subtext.query_understanding")

_UNSUPPORTED_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "runtime",
        re.compile(
            r"\b(?:(?:under|less\s+than|over|more\s+than|around|max)\s+\d+\s*(?:min|mins|minutes|hr|hrs|hours)|"
            r"\d+\s*(?:min|mins|minutes|hr|hrs|hours)\s+(?:or\s+less|long|runtime)|short\s+runtime)\b",
            re.IGNORECASE,
        ),
        "Runtime limits are outside the bounded v1.0 extraction contract and were not applied.",
    ),
    (
        "streaming_provider",
        re.compile(
            r"\bon\s+(?:netflix|hulu|prime(?:\s+video)?|amazon\s+prime|disney\+?|hbo(?:\s+max)?|max|apple\s+tv\+?|peacock|paramount\+?)\b",
            re.IGNORECASE,
        ),
        "Streaming provider filters are outside the bounded v1.0 extraction contract and were not applied.",
    ),
    (
        "age_rating",
        re.compile(
            r"\b(?:rated\s+(?:g|pg|pg-?13|r|nc-?17)|pg-?13|nc-?17|r-rated|tv-?ma|tv-?14)\b",
            re.IGNORECASE,
        ),
        "MPAA/TV age rating filters are outside the bounded v1.0 extraction contract and were not applied.",
    ),
    (
        "budget",
        re.compile(
            r"\b(?:low[- ]budget|micro[- ]?budget|high[- ]budget|big[- ]budget|budget\s+(?:under|over))\b",
            re.IGNORECASE,
        ),
        "Production budget filters are outside the bounded v1.0 extraction contract and were not applied.",
    ),
)

_NEGATION_CUE_RE = re.compile(r"\b(?:not|no|without|except|avoid|minus)\b", re.IGNORECASE)


def normalize_query_key(text: str) -> str:
    """Normalize query text for deterministic versioned cache and snapshot lookups."""
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    return cleaned.rstrip(".!?")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def detect_unsupported_constraints(query: str) -> list[str]:
    """Return user-visible notices when a query asks for out-of-contract filters."""
    return [msg for _, pattern, msg in _UNSUPPORTED_RULES if pattern.search(query)]


class _EncoderWithHead(nn.Module):
    """Mean-pooled MiniLM encoder + calibrated classification head matching training."""

    def __init__(self, encoder: nn.Module, hidden_size: int, num_labels: int, dropout: float) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(out.size()).float()
        pooled = torch.sum(out * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        return self.head(pooled)


class QueryUnderstandingService:
    """Loads pinned model release once and serves contract-validated predictions with selective LLM fallback."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or REPO_ROOT
        self.ready = False
        self.load_error: str | None = "Service not initialized"
        self.taxonomy: dict[str, Any] = {}
        self.routing: dict[str, Any] = {}
        self.prompt_cfg: dict[str, Any] = {}
        self.snapshots: dict[str, Any] = {}
        self.manifest: dict[str, Any] = {}
        self.label_order: list[str] = []
        self.label_set: set[str] = set()
        self.fine_to_broad: dict[str, str] = {}
        self.canonical_genres: set[str] = set()
        self.canonical_languages: set[str] = set()
        self.compound_min = 2
        self.temperatures: dict[str, float] = {}
        self.thresholds: dict[str, float] = {}
        self.registered_model_name = "subtext-intent-multilabel"
        self.registered_version = "2"
        self.bundle_fingerprint = "unknown"
        self.model_id = "minilm_multilabel"
        self.max_length = 128
        self.tokenizer: Any = None
        self.model: _EncoderWithHead | None = None
        self.people_set: set[str] = set()
        self.title_map: dict[str, str] = {}
        self._prediction_validator: jsonschema.Draft202012Validator | None = None
        self._extraction_validator: jsonschema.Draft202012Validator | None = None
        self._llm_cache: dict[tuple[str, str, str, str], tuple[list[str], dict[str, Any]]] = {}
        self.llm_stats: dict[str, Any] = {
            "escalations_total": 0,
            "calls_total": 0,
            "cache_hits": 0,
            "by_status": defaultdict(int),
            "by_reason": defaultdict(int),
        }

    def load(
        self,
        bundle_dir: Path | None = None,
        registry_summary_path: Path | None = None,
    ) -> bool:
        """Load configs, schemas, lexicons, and pinned model artifacts once (fail closed on missing weights)."""
        self.ready = False
        try:
            self.taxonomy = _load_yaml(self.root / "configs" / "taxonomy.yaml")
            self.routing = _load_yaml(self.root / "configs" / "routing.yaml")
            self.prompt_cfg = _load_yaml(self.root / "prompts" / "intent_fallback_v1.yaml")
            snap_raw = json.loads((self.root / "prompts" / "llm_snapshots_v1.json").read_text(encoding="utf-8"))
            self.snapshots = snap_raw.get("entries", {})

            self.label_order = list(self.taxonomy["label_order"])
            self.label_set = set(self.label_order)
            self.fine_to_broad = dict(self.taxonomy["fine_to_broad"])
            self.canonical_genres = set(self.taxonomy.get("canonical_genres", []))
            self.canonical_languages = set(self.taxonomy.get("canonical_languages", {}).keys())
            self.compound_min = int(self.taxonomy["compound_rule"]["min_distinct_broad_categories"])

            pred_schema = json.loads((self.root / "schemas" / "prediction_contract.schema.json").read_text(encoding="utf-8"))
            ext_schema = json.loads((self.root / "schemas" / "extraction.schema.json").read_text(encoding="utf-8"))
            inline = dict(pred_schema)
            inline["properties"] = dict(pred_schema["properties"])
            inline["properties"]["extraction"] = ext_schema
            self._prediction_validator = jsonschema.Draft202012Validator(inline)
            self._extraction_validator = jsonschema.Draft202012Validator(ext_schema)

            self.people_set, self.title_map = load_lexicons(
                self.root / "backend" / "people_index.json",
                self.root / "backend" / "galaxy_points.json",
            )

            reg_path = registry_summary_path or (self.root / "artifacts" / "registry" / "registration_summary.json")
            if reg_path.exists():
                reg = json.loads(reg_path.read_text(encoding="utf-8")).get("selected_model", {})
                self.registered_model_name = str(reg.get("registered_model_name", self.registered_model_name))
                self.registered_version = str(reg.get("registered_version", self.registered_version))
                self.bundle_fingerprint = str(reg.get("bundle_fingerprint", self.bundle_fingerprint))

            env_bundle = os.getenv("MODEL_BUNDLE_DIR", "").strip()
            resolved_bundle = bundle_dir or (Path(env_bundle) if env_bundle else (self.root / "artifacts" / "primary"))
            manifest_path = resolved_bundle / "bundle_manifest.json"
            enc_dir = resolved_bundle / "encoder"
            head_path = resolved_bundle / "head.pt"

            if not manifest_path.exists() or not enc_dir.exists() or not head_path.exists():
                raise FileNotFoundError(
                    f"Missing pinned bundle artifacts in {resolved_bundle} "
                    "(requires bundle_manifest.json, encoder/, and head.pt)"
                )

            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.model_id = str(self.manifest["model_type"])
            self.temperatures = {str(k): float(v) for k, v in self.manifest["temperatures"].items()}
            self.thresholds = {str(k): float(v) for k, v in self.manifest["thresholds"].items()}
            self.max_length = int(self.manifest["max_length"])

            self.tokenizer = AutoTokenizer.from_pretrained(str(enc_dir), local_files_only=True)
            backbone = AutoModel.from_pretrained(str(enc_dir), local_files_only=True)
            head_model = _EncoderWithHead(
                encoder=backbone,
                hidden_size=int(self.manifest["hidden_size"]),
                num_labels=len(self.label_order),
                dropout=float(self.manifest.get("dropout", 0.1)),
            )
            state = torch.load(head_path, map_location="cpu", weights_only=True)
            head_model.head.load_state_dict(state)
            head_model.eval()
            self.model = head_model
            self.load_error = None
            self.ready = True
            return True
        except Exception as exc:
            self.ready = False
            self.model = None
            self.tokenizer = None
            self.load_error = f"{type(exc).__name__}: {exc}"
            logger.warning("QueryUnderstandingService failed closed: %s", self.load_error)
            return False

    def version_info(self) -> dict[str, Any]:
        return {
            "model_name": self.registered_model_name,
            "model_id": self.model_id,
            "model_version": self.registered_version,
            "bundle_fingerprint": self.bundle_fingerprint,
            "taxonomy_version": str(self.taxonomy.get("version", "1.0.0")),
            "route_policy_version": str(self.routing.get("version", "1.0.0")),
            "prompt_version": str(self.prompt_cfg.get("prompt_version", "1.0.0")),
            "schema_version": "1.0.0",
            "image_digest": os.getenv("IMAGE_DIGEST", "local-dev"),
            "model_loaded": self.ready,
        }

    def _validate_query(self, query: str) -> str:
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Query cannot be empty or whitespace.")
        if len(cleaned) > 500:
            raise ValueError("Query exceeds maximum length of 500 characters.")
        return cleaned

    def _should_escalate(
        self,
        probs_row: np.ndarray,
        raw_above_count: int,
        fine_preds: list[str],
        is_compound: bool,
        extraction: dict[str, Any],
        query: str,
    ) -> tuple[bool, str]:
        policy = self.routing.get("llm_fallback", {})
        if policy.get("escalate_on_empty_predictions", True) and raw_above_count == 0:
            return True, "empty_thresholded_predictions"

        max_margin = float(policy.get("max_confidence_margin_for_escalation", 0.15))
        margins = [abs(float(probs_row[j]) - float(self.thresholds.get(lbl, 0.5))) for j, lbl in enumerate(self.label_order)]
        min_margin = min(margins) if margins else 0.0
        if min_margin < max_margin:
            return True, f"low_confidence_margin ({min_margin:.3f})"

        if policy.get("escalate_on_unresolved_compound_negation", True):
            q_low = query.lower()
            has_real_negation = bool(_NEGATION_CUE_RE.search(q_low)) and not any(skip in q_low for skip in _NEG_SKIP)
            has_neg_signal = ("negation_genre" in fine_preds) or ("negation_content" in fine_preds) or is_compound
            no_exclusions = not extraction.get("excluded_genres") and not extraction.get("excluded_terms")
            if has_real_negation and has_neg_signal and no_exclusions:
                return True, "unresolved_compound_negation"

        return False, "high_confidence_small_model"

    def _validate_llm_json(self, raw_text: str) -> tuple[list[str], dict[str, Any]]:
        if self._extraction_validator is None:
            raise RuntimeError("Extraction validator unavailable")
        stripped = raw_text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE | re.DOTALL).strip()
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError("LLM response is not a JSON object")

        labels = parsed.get("predicted_fine_labels")
        if not isinstance(labels, list) or not labels:
            raise ValueError("LLM predicted_fine_labels must be a non-empty list")
        cleaned_labels: list[str] = []
        for lbl in labels:
            if not isinstance(lbl, str) or lbl not in self.label_set:
                raise ValueError(f"Unknown fine label in LLM output: {lbl!r}")
            if lbl not in cleaned_labels:
                cleaned_labels.append(lbl)

        extraction = parsed.get("extraction")
        if not isinstance(extraction, dict):
            raise ValueError("LLM extraction must be an object")
        self._extraction_validator.validate(extraction)

        for g in list(extraction.get("genres", [])) + list(extraction.get("excluded_genres", [])):
            if g not in self.canonical_genres:
                raise ValueError(f"Unsupported genre in LLM extraction: {g!r}")
        lang = extraction.get("language")
        if lang is not None and lang not in self.canonical_languages:
            raise ValueError(f"Unsupported language in LLM extraction: {lang!r}")
        y_min, y_max = extraction.get("year_min"), extraction.get("year_max")
        if y_min is not None and y_max is not None and y_min > y_max:
            raise ValueError("LLM extraction year_min exceeds year_max")

        return cleaned_labels, extraction

    def _invoke_live_llm(
        self,
        query: str,
        candidate_labels: list[str],
        candidate_extraction: dict[str, Any],
    ) -> tuple[str, dict[str, Any] | None, str]:
        base_url = os.getenv("LLM_PROVIDER_BASE_URL", "").strip().rstrip("/")
        api_key = os.getenv("LLM_API_KEY", "").strip()
        model_name = os.getenv("LLM_MODEL_ID", "").strip()
        policy = self.routing.get("llm_fallback", {})
        timeout_sec = min(float(policy.get("timeout_seconds", 3.5)), 5.0)
        max_retries = min(int(policy.get("max_retries", 1)), 1)
        max_tokens = min(int(policy.get("max_tokens", 256)), 256)

        user_prompt = str(self.prompt_cfg["user_template"]).format(
            query=query,
            candidate_labels=json.dumps(candidate_labels),
            candidate_extraction=json.dumps(candidate_extraction),
        )
        req_body = {
            "model": model_name,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": str(self.prompt_cfg["system_prompt"])},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        for attempt in range(max_retries + 1):
            try:
                with httpx.Client(timeout=timeout_sec) as client:
                    resp = client.post(f"{base_url}/chat/completions", json=req_body, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return "ok", content, ""
            except httpx.TimeoutException as exc:
                if attempt < max_retries:
                    continue
                return "timeout", None, str(exc)
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500 and attempt < max_retries:
                    continue
                return "error", None, str(exc)
        return "error", None, "exhausted retries"

    def _run_llm_fallback(
        self,
        query: str,
        small_fine: list[str],
        small_extraction: dict[str, Any],
        reason: str,
    ) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
        policy = self.routing.get("llm_fallback", {})
        if not policy.get("enabled", True):
            return small_fine, small_extraction, {"called": False, "status": "not_called", "reason": f"{reason}; fallback_disabled"}

        self.llm_stats["escalations_total"] += 1
        self.llm_stats["by_reason"][reason] += 1

        cache_key = (
            f"{self.registered_model_name}:{self.registered_version}",
            str(self.prompt_cfg.get("prompt_version", "1.0.0")),
            str(self.routing.get("version", "1.0.0")),
            normalize_query_key(query),
        )
        if cache_key in self._llm_cache:
            self.llm_stats["cache_hits"] += 1
            self.llm_stats["by_status"]["succeeded"] += 1
            cached_labels, cached_ext = self._llm_cache[cache_key]
            return cached_labels, cached_ext, {"called": False, "status": "succeeded", "reason": f"{reason}; cache_hit"}

        has_live_creds = all(os.getenv(k, "").strip() for k in ("LLM_PROVIDER_BASE_URL", "LLM_API_KEY", "LLM_MODEL_ID"))
        if has_live_creds:
            self.llm_stats["calls_total"] += 1
            status, content, detail = self._invoke_live_llm(query, small_fine, small_extraction)
            if status == "ok" and isinstance(content, str):
                try:
                    val_labels, val_ext = self._validate_llm_json(content)
                    self._llm_cache[cache_key] = (val_labels, val_ext)
                    self.llm_stats["by_status"]["succeeded"] += 1
                    return val_labels, val_ext, {"called": True, "status": "succeeded", "reason": reason}
                except Exception as exc:
                    self.llm_stats["by_status"]["invalid_output"] += 1
                    logger.warning("LLM output failed schema/label validation: %s", exc)
                    return small_fine, small_extraction, {"called": True, "status": "invalid_output", "reason": f"{reason}; {exc}"}
            self.llm_stats["by_status"][status] += 1
            logger.warning("LLM fallback failed (%s): %s", status, detail)
            return small_fine, small_extraction, {"called": True, "status": status, "reason": f"{reason}; {detail}"}

        if policy.get("use_snapshots_when_unconfigured", True):
            snap = self.snapshots.get(normalize_query_key(query))
            if snap is not None:
                try:
                    val_labels, val_ext = self._validate_llm_json(json.dumps(snap))
                    self._llm_cache[cache_key] = (val_labels, val_ext)
                    self.llm_stats["calls_total"] += 1
                    self.llm_stats["by_status"]["succeeded"] += 1
                    return val_labels, val_ext, {"called": True, "status": "succeeded", "reason": f"{reason}; versioned_snapshot"}
                except Exception as exc:
                    self.llm_stats["by_status"]["invalid_output"] += 1
                    return small_fine, small_extraction, {"called": True, "status": "invalid_output", "reason": f"{reason}; {exc}"}

        self.llm_stats["by_status"]["not_called"] += 1
        return small_fine, small_extraction, {"called": False, "status": "not_called", "reason": f"{reason}; llm_provider_unconfigured"}

    def predict(self, query: str, *, allow_llm_fallback: bool = True) -> dict[str, Any]:
        """Run pinned small-model inference, selective LLM fallback, and contract validation."""
        cleaned_query = self._validate_query(query)
        if not self.ready or self.model is None or self.tokenizer is None or self._prediction_validator is None:
            raise RuntimeError(f"Model bundle unavailable: {self.load_error}")

        with torch.no_grad():
            enc = self.tokenizer(
                [cleaned_query],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            logits = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).cpu().numpy().astype(np.float64)

        probs = apply_temperature_scaling(logits, self.temperatures, self.label_order)
        probs_row = probs[0]
        th_vec = np.array([self.thresholds.get(lbl, 0.5) for lbl in self.label_order], dtype=np.float64)
        raw_above_count = int(np.sum(probs_row >= th_vec))
        preds = apply_thresholds(probs, self.thresholds, self.label_order)

        small_fine = [self.label_order[j] for j in range(len(self.label_order)) if preds[0, j] == 1]
        small_broad = sorted({self.fine_to_broad[lbl] for lbl in small_fine})
        small_compound = len(small_broad) >= self.compound_min
        extraction = extract_constraints(cleaned_query, self.taxonomy, people_set=self.people_set, title_map=self.title_map)

        escalate, esc_reason = self._should_escalate(probs_row, raw_above_count, small_fine, small_compound, extraction, cleaned_query)
        if allow_llm_fallback and escalate:
            final_fine, final_ext, llm_meta = self._run_llm_fallback(cleaned_query, small_fine, extraction, esc_reason)
        else:
            final_fine, final_ext = small_fine, extraction
            llm_meta = {"called": False, "status": "not_called", "reason": esc_reason if not allow_llm_fallback else "high_confidence_small_model"}

        final_broad = sorted({self.fine_to_broad[lbl] for lbl in final_fine})
        route, route_reason = select_route(final_fine, final_ext, self.routing)
        payload: dict[str, Any] = {
            "request_id": f"req_{uuid.uuid4().hex[:12]}",
            "schema_version": "1.0.0",
            "taxonomy_version": str(self.taxonomy["version"]),
            "route_policy_version": str(self.routing["version"]),
            "model_id": self.model_id,
            "model_version": self.registered_version,
            "predicted_fine_labels": final_fine,
            "predicted_broad_labels": final_broad,
            "is_compound": len(final_broad) >= self.compound_min,
            "label_scores": {self.label_order[j]: round(float(probs_row[j]), 4) for j in range(len(self.label_order))},
            "applied_thresholds": {lbl: float(self.thresholds[lbl]) for lbl in self.label_order},
            "extraction": final_ext,
            "selected_route": route,
            "route_reason": route_reason,
            "llm_fallback": llm_meta,
        }
        self._prediction_validator.validate(payload)
        return payload
