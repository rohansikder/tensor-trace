"""Exercise the real browser, Rust API, streaming, and cache comparison.

Start the release server with downloaded weights and a built web console first.
Install scripts/requirements-browser.txt and a Playwright Chromium browser, or
set BROWSER_EXECUTABLE to a system Chromium executable.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

from playwright.sync_api import Error as PlaywrightError, sync_playwright, expect


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8787")
    parser.add_argument("--unavailable-url", help="Optional second server started with a missing model")
    parser.add_argument("--output", type=Path, default=Path("build/browser-verification.json"))
    parser.add_argument("--screenshots", type=Path, default=Path("build/screenshots"))
    args = parser.parse_args()
    args.screenshots.mkdir(parents=True, exist_ok=True)
    report = {
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "checks": [],
        "pageErrors": [],
        "consoleErrors": [],
        "desktopViewport": {"width": 1487, "height": 1058},
        "mobileViewport": {"width": 390, "height": 844},
    }

    def passed(name: str) -> None:
        report["checks"].append({"name": name, "status": "passed"})
        print(f"PASS {name}", flush=True)

    def no_overflow(page) -> None:
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Horizontal page overflow"

    def wait_for_capacity(api) -> None:
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            response = api.post("/api/generate", data={"prompt": "Once", "maxTokens": 1, "trace": False})
            if response.status == 200:
                return
            assert response.status == 429, response.text()
            time.sleep(0.05)
        raise AssertionError("Cancelled work did not release the runtime within 12 seconds")

    try:
        with sync_playwright() as p:
            executable = os.environ.get("BROWSER_EXECUTABLE") or shutil.which("chromium")
            browser = p.chromium.launch(executable_path=executable, args=["--no-sandbox"])
            report["browserVersion"] = browser.version
            context = browser.new_context(viewport=report["desktopViewport"], base_url=args.base_url)
            page = context.new_page()
            page.set_default_timeout(30000)
            page.on("pageerror", lambda error: report["pageErrors"].append(str(error)))
            page.on("console", lambda event: report["consoleErrors"].append(event.text) if event.type == "error" else None)
            api = context.request
            startup_deadline = time.monotonic() + 30
            while True:
                try:
                    health = api.get("/api/health", timeout=1500).json()
                    assert health["status"] == "ready", health
                    break
                except PlaywrightError:
                    if time.monotonic() >= startup_deadline:
                        raise
                    time.sleep(0.25)
            page.goto(args.base_url, wait_until="networkidle")
            expect(page.get_by_role("button", name="Step through", exact=True)).to_be_enabled()
            expect(page.locator(".prediction-empty")).to_contain_text("No inference has run yet")
            passed("Model readiness and honest empty state")

            page.get_by_role("button", name="Step through", exact=True).click()
            expect(page.locator(".generated-token-picker .output-token")).to_have_count(1)
            for index in range(6):
                stage = page.locator(".path-step").nth(index)
                stage.click()
                expect(stage).to_have_attribute("aria-current", "step")
                assert page.locator(".inspection-grid canvas").count() >= (1 if index in (2, 3) else 0)
            page.locator(".path-step").nth(3).click()
            page.screenshot(path=str(args.screenshots / "walkthrough.png"), full_page=True)
            no_overflow(page)
            passed("Real single-token trace across all six visual stages")

            page.get_by_label("Attention layer", exact=True).select_option("1")
            expect(page.get_by_text("Captured layer 2, head 1.", exact=False)).to_be_visible()
            page.get_by_label("Attention head", exact=True).select_option("2")
            expect(page.get_by_text("Captured layer 2, head 3.", exact=False)).to_be_visible()
            passed("Layer and head selectors recapture actual attention")

            page.get_by_role("button", name="About this engine").click()
            dialog = page.get_by_role("dialog")
            expect(dialog).to_be_visible()
            page.get_by_role("link", name="Read the implementation").focus()
            page.keyboard.press("Tab")
            expect(page.get_by_role("button", name="Close model details")).to_be_focused()
            page.keyboard.press("Escape")
            expect(dialog).to_have_count(0)
            expect(page.get_by_role("button", name="About this engine")).to_be_focused()
            passed("Model details keyboard navigation and Escape dismissal")

            page.get_by_role("button", name="Generation settings", exact=True).click()
            page.get_by_label("Run length", exact=False).select_option("4")
            page.get_by_role("button", name="Run", exact=True).click()
            expect(page.get_by_role("button", name="Run", exact=True)).to_be_visible()
            expect(page.locator(".generated-token-picker .output-token")).to_have_count(4)
            assert page.locator(".decoded-output").inner_text()
            page.get_by_label("Temperature", exact=True).fill("0.7")
            expect(page.locator(".generated-token-picker")).to_have_count(0)
            passed("Streaming generation, decoded output, and settings invalidation")

            page.get_by_label("Run length", exact=False).select_option("64")
            page.get_by_label("KV cache enabled", exact=True).uncheck()
            page.get_by_role("button", name="Run", exact=True).click()
            page.get_by_role("button", name="Stop", exact=True).click()
            expect(page.get_by_text("Stopped. The tokens already received", exact=False)).to_be_visible()
            wait_for_capacity(api)
            passed("Generation cancellation releases the runtime")

            payload = {"prompt": "Once upon a time", "maxTokens": 8, "temperature": 0.7, "topP": 0.9, "seed": 42, "trace": False}
            first = api.post("/api/generate", data=payload)
            second = api.post("/api/generate", data=payload)
            assert first.ok and second.ok
            assert first.json()["generatedTokens"] == second.json()["generatedTokens"]
            assert api.post("/api/generate", data={"maxTokens": 65}).status == 400
            assert api.get("/api/").status == 404
            assert api.get("/api/unknown").status == 404
            assert api.post("/api/generate", data=b"{", headers={"Content-Type": "application/json"}).status == 400
            passed("Seeded repeatability and structured API input errors")

            page.get_by_role("button", name="Compare", exact=True).click()
            expect(page.get_by_text("No measurements yet", exact=True)).to_be_visible()
            with page.expect_response(lambda response: response.url.endswith("/api/benchmark/stream"), timeout=120000) as response_info:
                page.get_by_role("button", name="Run comparison", exact=True).click()
            expect(page.get_by_text("Generated token IDs match exactly", exact=True)).to_be_visible(timeout=120000)
            response = response_info.value
            frames = response.text().split("\n\n")
            benchmark = next(json.loads(line[6:]) for frame in frames if "event: done" in frame for line in frame.splitlines() if line.startswith("data: "))
            assert all(point["tokensMatch"] for point in benchmark["points"])
            assert len(benchmark["points"]) == 3
            report["benchmark"] = benchmark
            page.screenshot(path=str(args.screenshots / "compare.png"), full_page=True)
            no_overflow(page)
            passed("Measured cache comparison, live chart, and exact token equality")

            page.get_by_label("Repeats", exact=False).select_option("2")
            expect(page.get_by_text("No measurements yet", exact=True)).to_be_visible()
            page.get_by_role("button", name="Run comparison", exact=True).click()
            page.get_by_role("button", name="Cancel", exact=True).click()
            expect(page.get_by_text("Benchmark cancelled.", exact=False)).to_be_visible()
            wait_for_capacity(api)
            passed("Workload changes invalidate stale results; benchmark cancellation releases CPU")

            page.set_viewport_size(report["mobileViewport"])
            no_overflow(page)
            page.screenshot(path=str(args.screenshots / "compare-mobile.png"), full_page=True)
            page.get_by_role("button", name="Walkthrough", exact=True).click()
            reset = page.get_by_role("button", name="Reset", exact=True)
            if reset.is_enabled():
                reset.click()
            page.get_by_role("button", name="Step through", exact=True).click()
            expect(page.locator(".generated-token-picker .output-token")).to_have_count(1)
            page.locator(".path-step").nth(3).click()
            no_overflow(page)
            page.screenshot(path=str(args.screenshots / "walkthrough-mobile.png"), full_page=True)
            passed("390px mobile walkthrough and comparison without horizontal overflow")
            if args.unavailable_url:
                unavailable = browser.new_context(viewport=report["desktopViewport"], base_url=args.unavailable_url)
                assert unavailable.request.get("/api/health").json()["status"] == "missing_model"
                response = unavailable.request.get("/api/model")
                assert response.status == 503
                assert "scripts/fetch_model.py" in response.json()["error"]
                missing = unavailable.new_page()
                missing.goto(args.unavailable_url, wait_until="networkidle")
                expect(missing.get_by_text("Runtime offline", exact=True)).to_be_visible()
                expect(missing.get_by_role("button", name="Step through", exact=True)).to_be_disabled()
                unavailable.close()
                passed("Missing weights produce actionable API errors and a disabled unavailable UI")
            assert not report["pageErrors"], report["pageErrors"]
            assert not report["consoleErrors"], report["consoleErrors"]
            passed("No browser exceptions or console errors")
            browser.close()
            report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
