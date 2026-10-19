"""Fail-closed pipeline (latched => AI never called), evidence tags, and the hashed evidence record."""
import copy
import json
from unittest import mock

import pytest

from sparkholo import evidence, nemotron, pipeline
from sparkholo.evidence import (SCHEMA, build_manifest, fmt, tagged, untagged_numbers, verify_manifest)
from sparkholo.pipeline import RunConfig, prepare, run

V = "0.2.0+test"


def offline(m):
    exp = nemotron.offline_explain(m)
    return exp, json.dumps(exp)


@pytest.mark.parametrize("artifact", ["spike", "drift", "dropout"])
def test_latched_run_never_calls_explainer(artifact):
    fake = mock.Mock(side_effect=AssertionError("must not be called"))
    res = run(RunConfig(artifact=artifact), explainer=fake, version=V)
    fake.assert_not_called()
    assert res.status == "LATCHED" and not res.ai_consulted and res.ai_raw_output is None
    assert res.manifest["verdict"]["ai_consulted"] is False


@pytest.mark.parametrize("artifact", ["spike", "drift", "dropout"])
def test_latched_live_mode_never_calls_nemotron(artifact, monkeypatch):
    live = mock.Mock(side_effect=AssertionError("must not be called"))
    monkeypatch.setattr(nemotron, "explain_with_raw", live)
    res = run(RunConfig(artifact=artifact), live=True, version=V)
    live.assert_not_called()
    assert res.status == "LATCHED"


def test_clean_run_calls_explainer_once_and_verifies():
    fake = mock.Mock(side_effect=offline)
    res = run(RunConfig(), explainer=fake, version=V)
    fake.assert_called_once()
    assert res.status == "VERIFIED" and res.ai_consulted and res.verdict.ok
    assert json.loads(res.ai_raw_output) == res.explanation


def test_live_mode_uses_nemotron(monkeypatch):
    monkeypatch.setattr(nemotron, "explain_with_raw", mock.Mock(side_effect=offline))
    res = run(RunConfig(), live=True, version=V)
    assert res.status == "VERIFIED" and res.explainer_mode == "nemotron-live"


def test_tamper_is_refused():
    res = run(RunConfig(tamper=True), version=V)
    assert res.status == "REFUSED" and not res.verdict.ok
    assert res.manifest["verdict"]["ok"] is False


def test_explainer_error_fails_closed():
    res = run(RunConfig(), explainer=mock.Mock(side_effect=RuntimeError("down")), version=V)
    assert res.status == "EXPLAINER_ERROR" and res.explanation is None and "down" in res.error
    assert verify_manifest(res.manifest) == []


@pytest.mark.parametrize("cfg", [RunConfig(), RunConfig(tamper=True), RunConfig(artifact="drift"),
                                 RunConfig(artifact="dropout")])
def test_every_number_in_record_is_tagged(cfg):
    m = run(cfg, version=V).manifest
    assert untagged_numbers(m) == []
    assert m["config"]["latch"]["tau"] == {"value": 0.5, "tag": "TARGET"}
    assert m["config"]["verifier"]["rel_power_tolerance"]["tag"] == "TARGET"
    assert m["latch"]["tau"]["tag"] == "TARGET"


def test_metrics_are_tagged_synthetic():
    m = run(RunConfig(), version=V).manifest
    leaf = m["metrics"]["baseline"]["relative_power"]["alpha"]
    assert leaf["tag"] == "SYNTHETIC" and isinstance(leaf["value"], float)


def test_manifest_hash_is_deterministic():
    a = run(RunConfig(), version=V).manifest
    b = run(RunConfig(), version=V).manifest
    assert a == b and a["manifest_sha256"] == b["manifest_sha256"]
    assert run(RunConfig(seed=7), version=V).manifest["manifest_sha256"] != a["manifest_sha256"]
    assert run(RunConfig(), version="0.2.0+other").manifest["manifest_sha256"] != a["manifest_sha256"]


def test_manifest_contents():
    m = run(RunConfig(), version=V).manifest
    assert m["schema"] == SCHEMA and m["pipeline"] == ["Observer", "Calibrator", "Residual-Latch", "Verifier",
                                                       "Archive"]
    assert m["inputs"]["seed"] == 42 and m["code_version"] == V
    assert set(m["sha256"]) == set(evidence.SECTIONS)
    assert m["verdict"]["status"] == "VERIFIED" and m["ai_raw_output"]
    json.dumps(m, allow_nan=False)


def test_verify_manifest_detects_tampering():
    m = run(RunConfig(), version=V).manifest
    assert verify_manifest(m) == []
    t = copy.deepcopy(m)
    t["verdict"]["status"] = "REFUSED"
    assert any("verdict" in p for p in verify_manifest(t))
    t = copy.deepcopy(m)
    t["metrics"]["followup"]["alpha_coherence"]["value"] = 0.99
    assert verify_manifest(t)
    t = copy.deepcopy(m)
    t["boundary"] = "edited"
    assert verify_manifest(t) == ["manifest_sha256 does not match"]
    t = copy.deepcopy(m)
    del t["latch"]
    assert any("missing section" in p for p in verify_manifest(t))
    t = copy.deepcopy(m)
    t["schema"] = "x"
    assert verify_manifest(t)
    assert verify_manifest([]) == ["manifest is not an object"]
    assert "missing section hashes" in verify_manifest({"schema": SCHEMA})
    t = copy.deepcopy(m)
    t["extra"] = float("nan")
    assert any("canonical" in p for p in verify_manifest(t))


def test_verify_manifest_round_trips_through_json():
    m = run(RunConfig(artifact="spike"), version=V).manifest
    assert verify_manifest(json.loads(json.dumps(m))) == []


def test_build_manifest_requires_all_sections():
    with pytest.raises(ValueError):
        build_manifest(config={})


def test_tag_helpers():
    assert fmt(0.31234, "SYNTHETIC") == "0.3123 [SYNTHETIC]"
    assert fmt(32, "TARGET") == "32 [TARGET]"
    assert fmt(None, "UNRUN") == "n/a [UNRUN]"
    with pytest.raises(ValueError):
        fmt(1.0, "GUESS")
    with pytest.raises(ValueError):
        tagged(1, "GUESS")
    assert tagged(70, "TARGET", "dBA") == {"value": 70, "tag": "TARGET", "unit": "dBA"}
    assert untagged_numbers({"a": [1, {"value": 2, "tag": "SYNTHETIC"}], "channel": 3, "b": True}) == [".a[0]"]


def test_code_version(monkeypatch):
    monkeypatch.setenv("SPARKHOLO_GIT_SHA", "abcdef1234567890")
    assert evidence.code_version().endswith("+abcdef123456")
    monkeypatch.delenv("SPARKHOLO_GIT_SHA")
    monkeypatch.delenv("RENDER_GIT_COMMIT", raising=False)
    monkeypatch.setattr(evidence.subprocess, "run", mock.Mock(side_effect=OSError))
    assert evidence.code_version().endswith("+unrecorded")


def test_prepare_metrics_match_unlatched_signals():
    p = prepare(RunConfig())
    assert set(p.metrics) == {"baseline", "followup"}
    assert pipeline.FS == 256
