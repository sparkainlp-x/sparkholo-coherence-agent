import math

import numpy as np
import pytest

from sparkholo.nemotron import _extract_json, offline_explain
from sparkholo.signals import band_powers, session_metrics, synth_session
from sparkholo.verify import check_claim, verify

FS = 256


@pytest.fixture(scope="module")
def metrics():
    _, b = synth_session(seed=42)
    _, f = synth_session(seed=43, alpha_gain=1.5)
    return {"baseline": session_metrics(b, FS), "followup": session_metrics(f, FS)}


def test_deterministic():
    assert np.array_equal(synth_session(seed=7)[1], synth_session(seed=7)[1])


def test_relative_power_sums_to_one():
    _, d = synth_session()
    _, rel = band_powers(d[0], FS)
    assert math.isclose(sum(rel.values()), 1.0, rel_tol=1e-9)


def test_alpha_dominant_by_construction(metrics):
    assert metrics["baseline"]["dominant_band"] == "alpha"


def test_bad_signal_rejected():
    with pytest.raises(ValueError):
        band_powers(np.array([1.0, float("nan")] * 300), FS)


def test_offline_explanation_verifies(metrics):
    assert verify(offline_explain(metrics), metrics).ok


def test_tampered_claim_refused(metrics):
    v = verify(offline_explain(metrics, tamper=True), metrics)
    assert not v.ok and any("claim failed" in p for p in v.problems)


@pytest.mark.parametrize("word", ["diagnose", "treatment", "heals", "therapy", "regeneration", "clinical"])
def test_forbidden_wording(metrics, word):
    exp = offline_explain(metrics)
    exp["narrative"] += f" This shows {word}."
    assert not verify(exp, metrics).ok


def test_unbacked_number_refused(metrics):
    exp = offline_explain(metrics)
    exp["narrative"] += " Coherence rose 47%."
    assert not verify(exp, metrics).ok


@pytest.mark.parametrize("exp", [None, {}, {"narrative": "x"}, {"narrative": "", "claims": []}, {"narrative": "x", "claims": "bad"}])
def test_malformed_refused(metrics, exp):
    assert not verify(exp, metrics).ok


@pytest.mark.parametrize("claim", [
    {"type": "nope"}, {"type": "relative_power", "session": "x", "band": "alpha", "value": 0.3},
    {"type": "relative_power", "session": "baseline", "band": "alpha", "value": float("nan")},
    {"type": "relative_power", "session": "baseline", "band": "alpha", "value": True},
    {"type": "change", "band": "alpha"}, "string",
])
def test_bad_claims_fail(metrics, claim):
    assert not check_claim(claim, metrics).ok


def test_extract_json_strips_reasoning():
    assert _extract_json('<think>{"a":1}</think> {"narrative":"n","claims":[]}')["narrative"] == "n"


def test_extract_json_fails_closed():
    with pytest.raises(ValueError):
        _extract_json("no json here")
