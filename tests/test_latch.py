"""OES-32 residual latch: clean passes, artifacts / drift / NaN / wrong shape latch, formula is as specified."""
import numpy as np
import pytest

from sparkholo.latch import BLOCK, TAU, calibrate, evaluate, run_latch, score
from sparkholo.pipeline import ARTIFACTS, FS, RunConfig, inject_artifact, prepare
from sparkholo.signals import synth_session


@pytest.fixture(scope="module")
def sessions():
    _, b = synth_session(seed=42)
    _, f = synth_session(seed=43, alpha_gain=1.5)
    return b, f


def test_defaults_are_spec_values():
    assert BLOCK == 32 and TAU == 0.50


def test_score_formula():
    r = np.array([0.0, -1.0, 0.5, 0.5])
    expected = 0.45 * 1.0 + 0.35 * np.sqrt((0 + 1 + 0.25 + 0.25) / 4) + 0.20 * (2.0 / 4)
    assert score(r) == pytest.approx(expected)


def test_clean_default_run_passes(sessions):
    b, f = sessions
    res = run_latch(b, f, FS)
    assert not res.latched and res.reason == "clean" and not res.fired
    assert res.max_S < TAU
    assert res.n_blocks == 2 * b.shape[0] * (b.shape[1] // BLOCK)


@pytest.mark.parametrize("kind", ["spike", "drift"])
def test_injected_artifact_latches(sessions, kind):
    b, f = sessions
    res = run_latch(b, inject_artifact(f, kind), FS)
    assert res.latched and res.reason == "residual"
    assert res.fired and all(h.session == "followup" and h.S >= TAU for h in res.fired)


def test_drift_fires_on_drifted_channel_only(sessions):
    b, f = sessions
    res = run_latch(b, inject_artifact(f, "drift"), FS)
    assert {h.channel for h in res.fired} == {1}
    assert min(h.block for h in res.fired) >= f.shape[1] // BLOCK // 2


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_latches_and_names_block(sessions, bad):
    b, f = sessions
    f = f.copy()
    f[3, 5 * BLOCK + 1] = bad
    res = run_latch(b, f, FS)
    assert res.latched and res.reason == "non_finite"
    assert [(h.channel, h.block) for h in res.fired] == [(3, 5)]


def test_dropout_artifact_latches(sessions):
    b, f = sessions
    res = run_latch(b, inject_artifact(f, "dropout"), FS)
    assert res.latched and res.reason == "non_finite" and res.fired


@pytest.mark.parametrize("make", [
    lambda f: f[:, :-5],            # not a multiple of 32
    lambda f: f[:-1],               # wrong channel count
    lambda f: f[0],                 # 1-D
    lambda f: f[:, :10],            # shorter than one block
    lambda f: [["x"] * 64] * 4,     # not numeric
])
def test_wrong_shape_latches(sessions, make):
    b, f = sessions
    res = run_latch(b, make(f), FS)
    assert res.latched and res.reason == "wrong_shape"


@pytest.mark.parametrize("make", [
    lambda b: np.zeros_like(b),                        # flat: scale 0
    lambda b: np.where(np.arange(b.shape[1]) == 7, np.nan, b),
    lambda b: b[:, :-3],
])
def test_bad_baseline_fails_closed(sessions, make):
    b, f = sessions
    res = run_latch(make(b), f, FS)
    assert res.latched and res.reason in {"calibration_failed", "wrong_shape"}


def test_overflow_latches(sessions):
    b, f = sessions
    cal = calibrate(b)
    res = evaluate(np.full_like(f, 1e308), cal, FS, "followup")
    assert res.latched and res.reason == "non_finite"


def test_baseline_is_inside_its_own_envelope(sessions):
    b, _ = sessions
    cal = calibrate(b)
    res = evaluate(b, cal, FS, "baseline")
    assert not res.latched and res.max_S < TAU


def test_lower_tau_is_stricter(sessions):
    b, f = sessions
    assert run_latch(b, f, FS, tau=0.05).latched


def test_prepare_skips_metrics_when_latched():
    assert prepare(RunConfig()).metrics is not None
    assert prepare(RunConfig(artifact="spike")).metrics is None


def test_unknown_artifact_rejected():
    with pytest.raises(ValueError):
        prepare(RunConfig(artifact="bogus"))
    with pytest.raises(ValueError):
        inject_artifact(np.zeros((2, 64)), "bogus")
    assert set(ARTIFACTS) == {"none", "spike", "drift", "dropout"}
