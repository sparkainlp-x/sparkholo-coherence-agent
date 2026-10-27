"""Observer -> Calibrator -> Residual-Latch -> Verifier -> Archive.

* Observer: seeded SYNTHETIC baseline and follow-up sessions (optional demo artifact injected into the follow-up).
* Calibrator: per-channel OES-32 block template and scale from the baseline (see ``latch.calibrate``).
* Residual-Latch: fail-closed stop bit. If it latches, the AI is NOT consulted.
* Verifier: every AI claim is recomputed; the narrative is shown only if all claims verify.
* Archive: a hashed JSON evidence record of the run.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from . import nemotron
from .evidence import build_manifest, code_version, tag_tree, tagged
from .latch import BLOCK, MARGIN, TAU, LatchResult, run_latch
from .signals import session_metrics, synth_session
from .verify import COH_TOL, REL_TOL, verify

FS = 256
ARTIFACTS = {
    "none": "No artifact",
    "spike": "Electrode pop (spike burst, channel 3, t = 12 s)",
    "drift": "Baseline drift (ramp on channel 2, second half)",
    "dropout": "Dropout (NaN samples, channel 1, t = 20 s)",
}
Explainer = Callable[[dict], "tuple[dict, str]"]


@dataclass(frozen=True)
class RunConfig:
    seed: int = 42
    noise: float = 0.5
    shared: float = 0.6
    alpha_gain: float = 1.5
    theta_gain: float = 1.0
    artifact: str = "none"
    tamper: bool = False
    tau: float = TAU


@dataclass
class Prepared:
    config: RunConfig
    t: np.ndarray
    baseline: np.ndarray
    followup: np.ndarray
    latch: LatchResult
    metrics: dict | None


@dataclass
class RunResult:
    prepared: Prepared
    status: str  # LATCHED | VERIFIED | REFUSED | EXPLAINER_ERROR
    ai_consulted: bool
    explainer_mode: str
    explanation: dict | None = None
    ai_raw_output: str | None = None
    verdict: object = None
    error: str | None = None
    manifest: dict = field(default_factory=dict)


def inject_artifact(x: np.ndarray, kind: str, fs: int = FS) -> np.ndarray:
    """Return a copy of ``x`` with a demo artifact. Amplitudes are relative to the channel's own SD."""
    if kind not in ARTIFACTS:
        raise ValueError(f"unknown artifact {kind!r}")
    y = np.array(x, dtype=float, copy=True)
    if kind == "spike":
        c, i = 2 % y.shape[0], 12 * fs
        y[c, i:i + 8] += 15.0 * y[c].std() * np.array([1, -1, 1, -1, 1, -1, 1, -1])
    elif kind == "drift":
        c, h = 1 % y.shape[0], y.shape[1] // 2
        y[c, h:] += np.linspace(0.0, 8.0 * y[c].std(), y.shape[1] - h)
    elif kind == "dropout":
        i = 20 * fs
        y[0, i:i + fs // 2] = np.nan
    return y


def prepare(cfg: RunConfig) -> Prepared:
    """Observer + Calibrator + Residual-Latch. Metrics are computed only when the latch is clean."""
    if cfg.artifact not in ARTIFACTS:
        raise ValueError(f"unknown artifact {cfg.artifact!r}")
    t, base = synth_session(seed=cfg.seed, noise=cfg.noise, shared=cfg.shared)
    _, foll = synth_session(seed=cfg.seed + 1, noise=cfg.noise, shared=cfg.shared,
                            alpha_gain=cfg.alpha_gain, theta_gain=cfg.theta_gain)
    foll = inject_artifact(foll, cfg.artifact)
    latch = run_latch(base, foll, FS, tau=cfg.tau)
    metrics = None if latch.latched else {"baseline": session_metrics(base, FS), "followup": session_metrics(foll, FS)}
    return Prepared(cfg, t, base, foll, latch, metrics)


def _default_explainer(live: bool) -> tuple[Explainer, str]:
    if live:
        return nemotron.explain_with_raw, "nemotron-live"

    def offline(m: dict) -> tuple[dict, str]:
        exp = nemotron.offline_explain(m)
        return exp, json.dumps(exp, sort_keys=True)

    return offline, "offline-template"


def explain_and_verify(prep: Prepared, *, live: bool = False, explainer: Explainer | None = None,
                       version: str | None = None) -> RunResult:
    """Verifier + Archive. A latched run never calls the explainer (fail-closed)."""
    cfg = prep.config
    fn, mode = (explainer, "custom") if explainer is not None else _default_explainer(live)
    if cfg.tamper:
        mode += "+tamper-demo"
    if prep.latch.latched:
        res = RunResult(prep, "LATCHED", False, mode)
    else:
        try:
            exp, raw = fn(prep.metrics)
            if cfg.tamper:  # demo: one false claim is appended to whatever the explainer returned
                exp = nemotron.inject_false_claim(exp)
                raw = raw + "\n[tamper demo: false claim appended after the AI step]\n" + json.dumps(exp, sort_keys=True)
        except Exception as exc:  # noqa: BLE001 - fail closed: nothing is displayed
            res = RunResult(prep, "EXPLAINER_ERROR", True, mode, error=nemotron.describe_error(exc))
        else:
            verdict = verify(exp, prep.metrics)
            res = RunResult(prep, "VERIFIED" if verdict.ok else "REFUSED", True, mode, exp, raw, verdict)
    res.manifest = archive(res, version)
    return res


def run(cfg: RunConfig, **kwargs) -> RunResult:
    return explain_and_verify(prepare(cfg), **kwargs)


def archive(res: RunResult, version: str | None = None) -> dict:
    cfg = res.prepared.config
    lat = res.prepared.latch.to_dict()
    config = {
        "latch": {"block_samples": tagged(BLOCK, "TARGET"), "tau": tagged(cfg.tau, "TARGET"),
                  "margin": tagged(MARGIN, "TARGET"), "weights": {"peak": tagged(0.45, "TARGET"),
                  "rms": tagged(0.35, "TARGET"), "mean_abs": tagged(0.20, "TARGET")},
                  "reference": "baseline per-channel 32-sample block template"},
        "verifier": {"rel_power_tolerance": tagged(REL_TOL, "TARGET"),
                     "coherence_tolerance": tagged(COH_TOL, "TARGET")},
        "sampling_rate": tagged(FS, "SYNTHETIC", "Hz"),
    }
    inputs = {"seed": cfg.seed, **tag_tree({k: v for k, v in asdict(cfg).items() if k in
                                            ("noise", "shared", "alpha_gain", "theta_gain")}, "SYNTHETIC"),
              "artifact": cfg.artifact, "tamper_demo": cfg.tamper, "explainer_mode": res.explainer_mode}
    v = res.verdict
    verdict = {"status": res.status, "ai_consulted": res.ai_consulted, "error": res.error,
               "ok": None if v is None else v.ok,
               "problems": [] if v is None else list(v.problems),
               "claims": [] if v is None else [{"claim_json": json.dumps(r.claim, sort_keys=True, default=str),
                                                "ok": r.ok, "reason": r.reason} for r in v.results]}
    return build_manifest(config=config, inputs=inputs, code_version=version or code_version(),
                          metrics=tag_tree(res.prepared.metrics, "SYNTHETIC") if res.prepared.metrics else None,
                          latch=tag_tree(lat, "SYNTHETIC") | {"tau": tagged(lat["tau"], "TARGET")},
                          ai_raw_output=res.ai_raw_output, verdict=verdict)
