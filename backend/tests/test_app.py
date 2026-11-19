import os
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

import main
from sync_store import normalize_lb_uri


def test_api_starts_and_public_routes_remain_available():
    response = TestClient(main.app).get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "Subtext is online"
    paths = main.app.openapi()["paths"]
    assert "/search" in paths
    assert "/api/v1/constellation/points" in paths
    assert not any(path.startswith("/research") for path in paths)


@pytest.mark.parametrize("uri,expected", [
    ("https://letterboxd.com/user/film/arrival/", "https://letterboxd.com/film/arrival/"),
    ("https://letterboxd.com/film/arrival", "https://letterboxd.com/film/arrival/"),
    (None, None),
])
def test_sync_uri_normalization(uri, expected):
    assert normalize_lb_uri(uri) == expected


def test_source_files_stay_below_1000_lines():
    root = Path(__file__).resolve().parents[2]
    excluded = {".git", "node_modules", ".next", "venv", ".venv", "__pycache__"}
    large = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in excluded]
        for name in files:
            path = Path(folder) / name
            if path.suffix in {".py", ".js", ".jsx", ".ts", ".tsx"}:
                if len(path.read_text(encoding="utf-8", errors="ignore").splitlines()) >= 1000:
                    large.append(str(path.relative_to(root)))
    assert not large, f"Source files must have fewer than 1000 lines: {large}"
