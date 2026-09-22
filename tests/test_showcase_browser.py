"""Real offline interaction checks against Chrome, not HTML-string assertions."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api", reason="install .[ui-test] for browser interaction checks")
if shutil.which("google-chrome") is None:
    pytest.skip("google-chrome is required for offline showcase interaction checks", allow_module_level=True)

from navlab.showcase.builder import build_showcase


_RESULTS = Path(__file__).resolve().parents[1] / "docs" / "results"


@pytest.fixture(scope="module")
def showcase_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("showcase-browser") / "offline-showcase"
    build_showcase(_RESULTS, output)
    return output


def _set_range(page, selector: str, value: float) -> None:
    page.locator(selector).evaluate(
        "(node, value) => { node.value = String(value); node.dispatchEvent(new Event('input', {bubbles: true})); }",
        value,
    )


def test_offline_showcase_case_replay_comparison_and_downloads(showcase_dir: Path, tmp_path: Path) -> None:
    errors: list[str] = []
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900}, accept_downloads=True)
        page.set_default_timeout(120_000)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto((showcase_dir / "index.html").as_uri(), wait_until="load", timeout=120_000)
        page.locator("#case-select").wait_for(state="visible")

        # Initial example has actual recorded frames and a post-step timeline.
        assert page.locator("#case-select").input_value() == "end-to-end-detour-astar"
        assert page.locator("#timeline").is_disabled() is False
        assert float(page.locator("#timeline").get_attribute("max") or "0") > 0
        assert "t = 0.00 s · 樣本 0/200" in page.locator("#case-readout").inner_text()
        assert "實際速度\n0.00 m/s" in page.locator("#live-metrics").inner_text()
        _set_range(page, "#timeline", 0.01)  # before the first post-step sample at 0.05 s
        assert "樣本 0/200" in page.locator("#case-readout").inner_text()
        assert "實際速度\n0.00 m/s" in page.locator("#live-metrics").inner_text()

        # Case switching includes a planning failure with no invented playback.
        page.select_option("#case-group", "end_to_end")
        page.select_option("#case-select", "end-to-end-narrow-astar")
        assert page.locator("#timeline").is_disabled() is True
        assert "無可行路徑" in page.locator("#case-status").inner_text()

        page.select_option("#case-select", "end-to-end-detour-astar")
        _set_range(page, "#timeline", 2.5)
        assert page.locator("#timeline-output").inner_text() == "2.50 s"
        page.locator("#play-button").click()
        page.wait_for_timeout(350)
        assert float((page.locator("#timeline").input_value())) > 2.5
        page.locator("#play-button").click()  # pause
        paused = page.locator("#timeline").input_value()
        page.wait_for_timeout(120)
        assert page.locator("#timeline").input_value() == paused
        page.locator("#restart-button").click()
        assert page.locator("#timeline-output").inner_text() == "0.00 s"

        # Seeking to the recorded tail then changing case cancels any old RAF.
        _set_range(page, "#timeline", float(page.locator("#timeline").get_attribute("max") or "0"))
        page.locator("#play-button").click()
        page.select_option("#case-select", "end-to-end-narrow-astar")
        page.wait_for_timeout(150)
        assert page.locator("#timeline").is_disabled() is True
        assert page.locator("#timeline-output").inner_text() == "0.00 s"
        page.select_option("#case-select", "end-to-end-detour-astar")

        # Original text is downloadable without a server or repo-relative fetch.
        page.locator(".case-details summary").click()
        catalog = json.loads(page.locator("#navlab-data").text_content() or "{}")
        selected = next(case for case in catalog["cases"] if case["id"] == "end-to-end-detour-astar")
        with page.expect_download() as download_info:
            page.locator("#download-trace-csv").click()
        download = download_info.value
        destination = tmp_path / download.suggested_filename
        download.save_as(destination)
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == selected["source"]["sha256"]["trace_csv"]

        # Same-route comparison, seek, playback, and all eight policy summaries.
        page.locator('[data-mode="compare"]').click()
        page.locator("#route-select").wait_for(state="visible")
        page.select_option("#route-select", "6")
        page.select_option("#compare-a", "pure_pursuit")
        page.select_option("#compare-b", "ppo_seed0")
        assert "共用" in page.locator("#compare-status").inner_text()
        assert page.locator("#policy-summary tbody tr").count() == 8
        _set_range(page, "#compare-timeline", 1.0)
        page.locator("#compare-play").click()
        page.wait_for_timeout(300)
        assert float(page.locator("#compare-timeline").input_value()) > 1.0
        page.locator("#compare-play").click()

        page.screenshot(path=str(tmp_path / "offline-showcase.png"), full_page=True)
        browser.close()
    assert errors == []
