"""Standalone HTML showcase exporter.

The exporter only packages recorded result data.  It does not import optional
RL dependencies, train a policy, or execute a navigation rollout.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import shutil
from pathlib import Path
from typing import Any


EXPORTER_VERSION = "showcase-builder-v1"
_WEB_PACKAGE = "navlab.showcase.web"
_ARCHIVE_NAMES = ("heldout-raw-records.zip", "classical-raw-records.zip")


def _read_web_resource(name: str) -> str:
    """Read a packaged static resource, including when installed from a wheel."""
    try:
        return importlib.resources.files(_WEB_PACKAGE).joinpath(name).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise RuntimeError(f"packaged showcase web resource is missing: {name}") from exc


def _script_safe_json(value: dict[str, Any]) -> str:
    """Encode catalog JSON without allowing a value to close its script element."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _render_html(catalog: dict[str, Any]) -> str:
    template = _read_web_resource("index.html")
    style = _read_web_resource("style.css")
    script = _read_web_resource("app.js")
    markers = ("<!-- NAVLAB_STYLE -->", "<!-- NAVLAB_DATA -->", "<!-- NAVLAB_SCRIPT -->")
    if any(template.count(marker) != 1 for marker in markers):
        raise RuntimeError("showcase HTML template must contain each NAVLAB marker exactly once")
    data = '<script type="application/json" id="navlab-data">' + _script_safe_json(catalog) + "</script>"
    return template.replace(markers[0], style).replace(markers[1], data).replace(markers[2], script)


def _web_resource_hashes() -> dict[str, str]:
    """Record the exact static source used for a generated presentation."""
    return {
        name: hashlib.sha256(_read_web_resource(name).encode("utf-8")).hexdigest()
        for name in ("index.html", "style.css", "app.js")
    }


def _archive_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"name": path.name, "sha256": digest, "bytes": path.stat().st_size}


def _copy_archives(results_dir: Path, raw_dir: Path) -> list[dict[str, Any]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for name in _ARCHIVE_NAMES:
        source = results_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"required showcase source archive is missing: {source}")
        sidecar = results_dir / f"{name}.sha256"
        if not sidecar.is_file():
            raise FileNotFoundError(f"required showcase archive checksum is missing: {sidecar}")
        record = _archive_record(source)
        expected = sidecar.read_text(encoding="utf-8").strip().split(maxsplit=1)[0]
        if expected != record["sha256"]:
            raise ValueError(f"archive checksum does not match sidecar: {source}")
        shutil.copyfile(source, raw_dir / name)
        shutil.copyfile(sidecar, raw_dir / sidecar.name)
        records.append(record)
    return records


def build_showcase(results_dir: Path | str, output_dir: Path | str) -> dict[str, Any]:
    """Write an offline, self-contained showcase and its downloadable archives.

    ``results_dir`` must explicitly contain the frozen result archives.  The
    generated HTML embeds the entire validated catalog and all CSS/JS, so it
    opens from ``file://`` without network access or a repository checkout.
    """
    from navlab.showcase.dataset import build_catalog

    results = Path(results_dir)
    output = Path(output_dir)
    if not results.is_dir():
        raise FileNotFoundError(f"showcase results directory does not exist: {results}")
    catalog = build_catalog(results)
    if catalog.get("schema_version") != 1:
        raise ValueError("showcase catalog must use schema_version 1")
    output.mkdir(parents=True, exist_ok=True)
    archive_records = _copy_archives(results, output / "raw")
    html = _render_html(catalog)
    generated = output / "index.html"
    generated.write_text(html, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "exporter_version": EXPORTER_VERSION,
        "catalog_schema_version": catalog["schema_version"],
        "source_results": {"archives": [record["name"] for record in archive_records]},
        "source_archives": archive_records,
        "web_resource_sha256": _web_resource_hashes(),
        "generated_index_sha256": hashlib.sha256(generated.read_bytes()).hexdigest(),
        "output": {"index_html": "index.html", "raw_dir": "raw"},
        "counts": {"cases": len(catalog.get("cases", [])), "maps": len(catalog.get("maps", {}))},
    }
    (output / "build-manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"output": str(output), **manifest["counts"], "archives": archive_records, "bytes": (output / "index.html").stat().st_size}
