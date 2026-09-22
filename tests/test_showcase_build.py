from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from navlab.showcase import builder


_RESULTS = Path(__file__).resolve().parents[1] / "docs" / "results"


def _inline_catalog(html: str) -> dict:
    match = re.search(r'<script type="application/json" id="navlab-data">(.*?)</script>', html, flags=re.DOTALL)
    assert match, "showcase catalog script is missing"
    return json.loads(match.group(1))


def test_build_showcase_embeds_catalog_and_copies_verified_archives(tmp_path: Path) -> None:
    result = builder.build_showcase(_RESULTS, tmp_path / "showcase")
    output = Path(result["output"])
    html = (output / "index.html").read_text(encoding="utf-8")

    assert result["cases"] == 175
    assert result["maps"] == 13
    assert "<!-- NAVLAB_" not in html
    assert "fetch(" not in html
    catalog = _inline_catalog(html)
    assert catalog["schema_version"] == 1
    assert len(catalog["cases"]) == 175
    assert "</script" not in json.dumps(catalog, ensure_ascii=False).lower()

    manifest = json.loads((output / "build-manifest.json").read_text(encoding="utf-8"))
    assert manifest["exporter_version"] == builder.EXPORTER_VERSION
    assert "source_results_dir" not in manifest
    assert manifest["source_results"]["archives"] == ["heldout-raw-records.zip", "classical-raw-records.zip"]
    assert set(manifest["web_resource_sha256"]) == {"index.html", "style.css", "app.js"}
    assert hashlib.sha256((output / "index.html").read_bytes()).hexdigest() == manifest["generated_index_sha256"]
    for archive in manifest["source_archives"]:
        copied = output / "raw" / archive["name"]
        assert copied.is_file()
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == archive["sha256"]
        assert (output / "raw" / f"{archive['name']}.sha256").is_file()


def test_build_showcase_reports_missing_results_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="results directory does not exist"):
        builder.build_showcase(tmp_path / "missing", tmp_path / "output")


def test_script_safe_json_cannot_end_data_element() -> None:
    encoded = builder._script_safe_json({"value": "</script>\u2028\u2029"})
    assert "</script" not in encoded.lower()
    assert "\\u003c/script>" in encoded
    assert "\\u2028" in encoded
    assert "\\u2029" in encoded
