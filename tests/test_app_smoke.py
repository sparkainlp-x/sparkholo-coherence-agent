"""Streamlit AppTest smoke tests: the app loads, the button runs, the tamper demo is refused, and an injected
artifact latches before the AI is consulted.

Runs offline: NEBIUS_API_KEY is removed so no network call is made and no key is needed.
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)


def _run():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def test_app_loads_and_explanation_verifies():
    at = _run()
    assert any("OFFLINE template" in c.value for c in at.caption)
    at.button[0].click().run()
    assert not at.exception
    assert any("All claims independently verified" in s.value for s in at.success)
    assert not any("REFUSED" in e.value for e in at.error)


def test_tamper_checkbox_is_refused():
    at = _run()
    at.checkbox[0].check().run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert any("Explanation REFUSED" in e.value for e in at.error)
    assert not any("All claims independently verified" in s.value for s in at.success)


def _verified_texts(at):
    return [s.value for s in at.success if "All claims independently verified" in s.value]


@pytest.mark.parametrize("artifact", ["spike", "drift", "dropout"])
def test_injected_artifact_latches_and_ai_not_consulted(artifact, monkeypatch):
    from sparkholo import nemotron

    def boom(*a, **k):
        raise AssertionError("explainer must not be called on a latched run")

    monkeypatch.setattr(nemotron, "offline_explain", boom)
    monkeypatch.setattr(nemotron, "explain_with_raw", boom)
    at = _run()
    at.selectbox[0].set_value(artifact).run()
    assert not at.exception
    assert any("LATCHED: AI not consulted" in e.value for e in at.error)
    at.button[0].click().run()
    assert not at.exception
    assert any(e.value.startswith("LATCHED: AI not consulted") for e in at.error)
    assert not _verified_texts(at)
    assert not any("REFUSED" in e.value for e in at.error)


def test_clean_run_shows_pipeline_tags_and_evidence_record():
    at = _run()
    assert any("Residual-Latch" in m.value and "Archive" in m.value for m in at.markdown)
    assert any("[TARGET]" in m.value and "[SYNTHETIC]" in m.value for m in at.markdown)
    assert any("Residual-Latch clean" in s.value for s in at.success)
    at.button[0].click().run()
    assert not at.exception
    assert any("manifest SHA-256" in c.value for c in at.caption)
