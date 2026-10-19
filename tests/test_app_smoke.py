"""Streamlit AppTest smoke tests: the app loads, the button runs, and the tamper demo is refused.

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
    assert not at.success
