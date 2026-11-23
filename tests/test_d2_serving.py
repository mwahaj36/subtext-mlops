"""Unit and integration tests for Deliverable 2 Part A serving API, contract validation, and LLM fallback."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
import jsonschema
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
for p in (REPO_ROOT, BACKEND_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import main  # noqa: E402
from query_understanding import QueryUnderstandingService  # noqa: E402


def _load_contract_validator() -> jsonschema.Draft202012Validator:
    pred_schema = json.loads((REPO_ROOT / "schemas" / "prediction_contract.schema.json").read_text(encoding="utf-8"))
    ext_schema = json.loads((REPO_ROOT / "schemas" / "extraction.schema.json").read_text(encoding="utf-8"))
    inline = dict(pred_schema)
    inline["properties"] = dict(pred_schema["properties"])
    inline["properties"]["extraction"] = ext_schema
    return jsonschema.Draft202012Validator(inline)


@pytest.fixture(scope="module")
def loaded_service() -> QueryUnderstandingService:
    svc = QueryUnderstandingService(REPO_ROOT)
    assert svc.load() is True, f"Failed to load pinned D1 bundle: {svc.load_error}"
    return svc


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, loaded_service: QueryUnderstandingService) -> TestClient:
    loaded_service._llm_cache.clear()
    monkeypatch.setattr(main, "qu_service", loaded_service)
    monkeypatch.setattr(main.search_runtime, "collection", None)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL_ID", raising=False)
    return TestClient(main.app)


def test_health_ready_and_version_endpoints(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}

    ready = client.get("/ready")
    assert ready.status_code == 200
    r_body = ready.json()
    assert r_body["status"] == "ready"
    assert r_body["model_loaded"] is True

    ver = client.get("/version")
    assert ver.status_code == 200
    v_body = ver.json()
    for field in (
        "model_name",
        "model_id",
        "model_version",
        "bundle_fingerprint",
        "taxonomy_version",
        "route_policy_version",
        "prompt_version",
        "schema_version",
        "image_digest",
        "model_loaded",
    ):
        assert field in v_body


def test_predict_contract_and_input_validation(client: TestClient) -> None:
    validator = _load_contract_validator()
    resp = client.post(
        "/predict",
        json={"query": "Korean crime thriller starring Song Kang-ho with no gore"},
    )
    assert resp.status_code == 200
    body = resp.json()
    validator.validate(body)
    assert body["extraction"]["language"] == "ko"
    assert "Song Kang-ho" in body["extraction"]["actors"]
    assert "gore" in body["extraction"]["excluded_terms"]
    assert body["selected_route"] == "structured_filter"
    assert body["llm_fallback"]["called"] is False
    assert body["llm_fallback"]["status"] == "not_called"

    assert client.post("/predict", json={"query": ""}).status_code == 422
    assert client.post("/predict", json={"query": "   "}).status_code == 422
    assert client.post("/predict", json={"query": "x" * 501}).status_code == 422


def test_missing_artifacts_fail_closed_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = QueryUnderstandingService(REPO_ROOT)
    assert broken.load(bundle_dir=tmp_path / "missing_bundle") is False
    monkeypatch.setattr(main, "qu_service", broken)
    c = TestClient(main.app)

    assert c.get("/health").status_code == 200
    assert c.get("/ready").status_code == 503
    assert c.get("/version").json()["model_loaded"] is False
    assert c.post("/predict", json={"query": "slow burn thriller"}).status_code == 503
    assert c.get("/search?q=slow+burn+thriller&include_diagnostics=true").status_code == 503


def test_selective_llm_snapshot_fallback_and_versioned_cache(
    client: TestClient, loaded_service: QueryUnderstandingService
) -> None:
    validator = _load_contract_validator()
    query = "something mind-bending but not scary"

    first = client.post("/predict", json={"query": query})
    assert first.status_code == 200
    body1 = first.json()
    validator.validate(body1)
    assert body1["llm_fallback"]["called"] is True
    assert body1["llm_fallback"]["status"] == "succeeded"
    assert "Horror" in body1["extraction"]["excluded_genres"]

    hits_before = loaded_service.llm_stats["cache_hits"]
    second = client.post("/predict", json={"query": query})
    assert second.status_code == 200
    body2 = second.json()
    validator.validate(body2)
    assert body2["llm_fallback"]["called"] is False
    assert body2["llm_fallback"]["status"] == "succeeded"
    assert "cache_hit" in body2["llm_fallback"]["reason"]
    assert loaded_service.llm_stats["cache_hits"] == hits_before + 1


@pytest.mark.parametrize(
    ("mode", "raw_payload", "expected_status"),
    [
        ("timeout", None, "timeout"),
        ("content", "not valid json {", "invalid_output"),
        (
            "content",
            json.dumps(
                {
                    "predicted_fine_labels": ["invented_unknown_label"],
                    "extraction": {
                        "actors": [],
                        "directors": [],
                        "genres": [],
                        "year_min": None,
                        "year_max": None,
                        "language": None,
                        "excluded_genres": [],
                        "excluded_terms": [],
                        "reference_titles": [],
                    },
                }
            ),
            "invalid_output",
        ),
    ],
)
def test_llm_failure_modes_fall_back_safely_to_small_model(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    raw_payload: str | None,
    expected_status: str,
) -> None:
    validator = _load_contract_validator()
    monkeypatch.setenv("LLM_PROVIDER_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL_ID", "test-model")

    class _FakeResp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": raw_payload}}]}

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> _FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def post(self, *args: Any, **kwargs: Any) -> _FakeResp:
            if mode == "timeout":
                raise httpx.ReadTimeout("simulated timeout")
            return _FakeResp()

    monkeypatch.setattr("query_understanding.httpx.Client", _FakeClient)

    resp = client.post("/predict", json={"query": "comforting autumn watch"})
    assert resp.status_code == 200
    body = resp.json()
    validator.validate(body)
    assert body["llm_fallback"]["called"] is True
    assert body["llm_fallback"]["status"] == expected_status
    assert len(body["predicted_fine_labels"]) >= 1


def test_llm_normalizes_lowercase_genres_and_aliases(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _load_contract_validator()
    monkeypatch.setenv("LLM_PROVIDER_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL_ID", "test-model")

    payload = json.dumps(
        {
            "predicted_fine_labels": ["affective_outcome", "attribute_genre", "negation_genre"],
            "extraction": {
                "actors": [],
                "directors": [],
                "genres": ["sci-fi", "COMEDY"],
                "year_min": None,
                "year_max": None,
                "language": "EN",
                "excluded_genres": ["horror", "Horror"],
                "excluded_terms": [],
                "reference_titles": [],
            },
        }
    )

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": payload}}]}

    class _Client:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

        def post(self, *a: Any, **k: Any) -> _Resp:
            return _Resp()

    monkeypatch.setattr("query_understanding.httpx.Client", _Client)
    resp = client.post("/predict", json={"query": "upbeat sci-fi comedy not scary"})
    assert resp.status_code == 200
    body = resp.json()
    validator.validate(body)
    assert body["llm_fallback"]["status"] == "succeeded"
    assert body["extraction"]["genres"] == ["Science Fiction", "Comedy"]
    assert body["extraction"]["excluded_genres"] == ["Horror"]
    assert body["extraction"]["language"] == "en"


def test_search_with_diagnostics_unsupported_constraints_and_retrieval_modes(
    client: TestClient,
) -> None:
    resp = client.get(
        "/search",
        params={
            "q": "Korean thriller under 90 minutes on Netflix rated R",
            "include_diagnostics": "true",
            "retrieval_mode": "auto",
            "k": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["retrieval_mode"] == "auto"
    assert len(data["results"]) >= 1
    assert data["applied_filters"]["language"] == "ko"
    notices = " ".join(data["unsupported_notices"]).lower()
    assert "runtime" in notices and "streaming" in notices and "rating" in notices
    ext = data["query_understanding"]["extraction"]
    assert set(ext.keys()) == {
        "actors",
        "directors",
        "genres",
        "year_min",
        "year_max",
        "language",
        "excluded_genres",
        "excluded_terms",
        "reference_titles",
    }

    for mode in ("dense_only", "static_hybrid"):
        r_mode = client.get(
            "/search",
            params={
                "q": "Korean crime thriller starring Song Kang-ho with no gore",
                "include_diagnostics": "true",
                "retrieval_mode": mode,
                "k": 5,
            },
        )
        assert r_mode.status_code == 200
        assert r_mode.json()["retrieval_mode"] == mode
        assert len(r_mode.json()["results"]) >= 1

    assert client.get("/search?q=   ").status_code == 422
