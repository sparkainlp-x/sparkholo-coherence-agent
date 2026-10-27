"""Explainer failures (missing key, rate limit, rejected key, timeout) fail closed with a clear, key-free message."""
from pathlib import Path

import openai
import pytest

try:  # openai>=3 ships on httpx2; older releases use httpx
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx

from sparkholo import nemotron
from sparkholo.pipeline import RunConfig, run

REQ = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
FAKE_KEY = "nb-test-0123456789abcdef"


def _status_error(cls, code):
    resp = httpx.Response(code, request=REQ, json={"error": {"message": f"bad key {FAKE_KEY}"}})
    return cls(f"Error code: {code} - key {FAKE_KEY}", response=resp, body=None)


@pytest.mark.parametrize("exc, needle", [
    (_status_error(openai.RateLimitError, 429), "rate limit"),
    (_status_error(openai.AuthenticationError, 401), "key was rejected"),
    (_status_error(openai.PermissionDeniedError, 403), "key was rejected"),
    (_status_error(openai.NotFoundError, 404), "NEMOTRON_MODEL"),
    (_status_error(openai.InternalServerError, 503), "HTTP 503"),
    (openai.APITimeoutError(request=REQ), "did not answer in time"),
    (openai.APIConnectionError(request=REQ), "Could not reach"),
])
def test_api_errors_fail_closed_without_key(exc, needle, monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", FAKE_KEY)

    def raiser(_m):
        raise exc

    res = run(RunConfig(), explainer=raiser, version="test")
    assert res.status == "EXPLAINER_ERROR" and res.explanation is None
    assert needle in res.error
    assert FAKE_KEY not in res.error and FAKE_KEY not in str(res.manifest)


def test_generic_error_redacts_key(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", FAKE_KEY)
    msg = nemotron.describe_error(RuntimeError(f"boom {FAKE_KEY}"))
    assert FAKE_KEY not in msg and "[redacted]" in msg


def test_live_mode_without_key_fails_closed(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    res = run(RunConfig(), live=True, version="test")
    assert res.status == "EXPLAINER_ERROR" and "NEBIUS_API_KEY is not set" in res.error


def test_app_shows_rate_limit_message_and_does_not_crash(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("NEBIUS_API_KEY", FAKE_KEY)

    def limited(*a, **k):
        raise _status_error(openai.RateLimitError, 429)

    monkeypatch.setattr(nemotron, "explain_with_raw", limited)
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=60)
    at.run()
    assert any("Nemotron via Nebius Token Factory" in c.value for c in at.caption)
    at.button[0].click().run()
    assert not at.exception
    assert any("rate limit" in e.value for e in at.error)
    assert not any("All claims independently verified" in s.value for s in at.success)
    assert not any(FAKE_KEY in e.value for e in at.error)


def test_app_live_mode_with_mocked_api_verifies(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("NEBIUS_API_KEY", FAKE_KEY)

    def mocked(m, timeout=60.0):
        exp = nemotron.offline_explain(m)
        return exp, "<think>mocked</think>" + str(exp)

    monkeypatch.setattr(nemotron, "explain_with_raw", mocked)
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=60)
    at.run()
    at.button[0].click().run()
    assert not at.exception
    assert any("All claims independently verified" in s.value for s in at.success)


def test_missing_key_message_is_plain_and_key_free(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    res = run(RunConfig(), live=True, version="test")
    assert "offline template" in res.error and "MissingKeyError" not in res.error


def test_tamper_demo_appends_to_the_explainers_own_output():
    """The tamper demo must modify what the explainer returned, not replace it with the offline template."""
    marker = {"narrative": "Custom explainer text.", "claims": [{"type": "dominant_band", "session": "baseline",
                                                                  "band": "alpha"}]}
    res = run(RunConfig(tamper=True), explainer=lambda m: (marker, "RAW-FROM-EXPLAINER"), version="test")
    assert res.status == "REFUSED"
    assert res.explanation["narrative"].startswith("Custom explainer text.")
    assert "RAW-FROM-EXPLAINER" in res.ai_raw_output and "tamper demo" in res.ai_raw_output
