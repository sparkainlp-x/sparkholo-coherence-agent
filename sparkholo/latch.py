"""OES-32 residual latch: Spark AI NLP's OES residual contract applied BEFORE any AI call.

Pipeline position: Observer -> Calibrator -> **Residual-Latch** -> Verifier -> Archive.

* Each channel is split into 32-sample blocks (OES-32).
* Calibrator (documented choice): the reference ``x_ref`` is the baseline session's per-channel *block template*
  (mean of all baseline blocks, sample by sample). The normalisation scale is ``MARGIN`` (TARGET 3.0) times the
  largest baseline block residual ``max|x - x_ref|`` of that channel, so the baseline sits well inside the envelope.
* ``R = max|x - x_ref|`` (raw units). On the normalised residual ``r = (x - x_ref) / scale``:
  ``S = 0.45*Peak + 0.35*RMS + 0.20*MeanAbs``; a block latches when ``S >= tau`` (TARGET 0.50).
* FAIL-CLOSED: NaN, inf, a wrong shape, a failed calibration or ANY latched block latches the run.
  A latched run never consults the AI.

The latch is a data-integrity stop bit only. It is not a measure of any person, brain state or health.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

BLOCK = 32
TAU = 0.50       # TARGET: latch threshold
MARGIN = 3.0     # TARGET: calibration margin over the largest baseline block residual
WEIGHTS = (0.45, 0.35, 0.20)


class LatchInputError(ValueError):
    """Input that cannot be evaluated (wrong shape or failed calibration)."""


@dataclass(frozen=True)
class Calibration:
    template: np.ndarray  # (channels, 32)
    scale: np.ndarray     # (channels,)
    n_blocks: int


@dataclass
class BlockHit:
    session: str
    channel: int
    block: int
    t_start_s: float
    R: float | None
    S: float | None
    reason: str


@dataclass
class LatchResult:
    latched: bool
    reason: str  # clean | residual | non_finite | wrong_shape | calibration_failed
    detail: str
    tau: float
    fired: list[BlockHit] = field(default_factory=list)
    max_R: float | None = None
    max_S: float | None = None
    n_blocks: int = 0

    def to_dict(self) -> dict:
        return {**asdict(self), "fired": [asdict(h) for h in self.fired]}


def score(r: np.ndarray) -> np.ndarray:
    """S over the last axis of a normalised residual array."""
    a = np.abs(r)
    return WEIGHTS[0] * a.max(-1) + WEIGHTS[1] * np.sqrt((r * r).mean(-1)) + WEIGHTS[2] * a.mean(-1)


def _as_blocks(data, channels: int | None = None) -> np.ndarray:
    try:
        x = np.asarray(data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise LatchInputError(f"not numeric: {exc}") from exc
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < BLOCK:
        raise LatchInputError(f"expected (channels, samples >= {BLOCK}), got {x.shape}")
    if channels is not None and x.shape[0] != channels:
        raise LatchInputError(f"expected {channels} channels, got {x.shape[0]}")
    if x.shape[1] % BLOCK:
        raise LatchInputError(f"samples ({x.shape[1]}) must be a multiple of {BLOCK}")
    return x.reshape(x.shape[0], -1, BLOCK)


def calibrate(baseline, margin: float = MARGIN) -> Calibration:
    b = _as_blocks(baseline)
    if not np.all(np.isfinite(b)):
        raise LatchInputError("baseline contains NaN or inf")
    template = b.mean(axis=1)
    peak = np.abs(b - template[:, None, :]).max(axis=(1, 2))
    scale = margin * peak
    if not (np.all(np.isfinite(scale)) and np.all(scale > 0)):
        raise LatchInputError("calibration scale must be finite and > 0 (flat baseline?)")
    return Calibration(template, scale, b.shape[1])


def evaluate(data, cal: Calibration, fs: int, session: str, tau: float = TAU) -> LatchResult:
    try:
        x = _as_blocks(data, channels=cal.template.shape[0])
    except LatchInputError as exc:
        return LatchResult(True, "wrong_shape", f"{session}: {exc}", tau)
    n = x.shape[1]
    t0 = lambda k: round(k * BLOCK / fs, 4)  # noqa: E731
    bad = ~np.isfinite(x).all(axis=-1)
    if bad.any():
        hits = [BlockHit(session, int(c), int(k), t0(k), None, None, "non_finite")
                for c, k in zip(*np.nonzero(bad), strict=True)]
        return LatchResult(True, "non_finite", f"{session}: {len(hits)} block(s) contain NaN or inf", tau, hits,
                           n_blocks=n * x.shape[0])
    resid = x - cal.template[:, None, :]
    R = np.abs(resid).max(-1)
    with np.errstate(over="ignore", invalid="ignore"):
        S = score(resid / cal.scale[:, None, None])
    if not (np.all(np.isfinite(S)) and np.all(np.isfinite(R))):
        hits = [BlockHit(session, int(c), int(k), t0(k), None, None, "non_finite")
                for c, k in zip(*np.nonzero(~(np.isfinite(S) & np.isfinite(R))), strict=True)]
        return LatchResult(True, "non_finite", f"{session}: residual overflowed", tau, hits, n_blocks=S.size)
    fired = [BlockHit(session, int(c), int(k), t0(k), round(float(R[c, k]), 6), round(float(S[c, k]), 6), "residual")
             for c, k in zip(*np.nonzero(S >= tau), strict=True)]
    latched = bool(fired)
    return LatchResult(latched, "residual" if latched else "clean",
                       f"{session}: {len(fired)} of {S.size} blocks with S >= tau" if latched
                       else f"{session}: all {S.size} blocks below tau",
                       tau, fired, round(float(R.max()), 6), round(float(S.max()), 6), int(S.size))


def run_latch(baseline, followup, fs: int, tau: float = TAU, margin: float = MARGIN) -> LatchResult:
    """Calibrate on the baseline, then evaluate both sessions. Any failure latches (fail-closed)."""
    try:
        cal = calibrate(baseline, margin)
    except LatchInputError as exc:
        return LatchResult(True, "calibration_failed", f"baseline: {exc}", tau)
    results = [evaluate(baseline, cal, fs, "baseline", tau), evaluate(followup, cal, fs, "followup", tau)]
    latched = [r for r in results if r.latched]
    fired = [h for r in results for h in r.fired]
    maxes = [r for r in results if r.max_S is not None]
    return LatchResult(
        bool(latched),
        latched[0].reason if latched else "clean",
        "; ".join(r.detail for r in results),
        tau,
        fired,
        max((r.max_R for r in maxes), default=None),
        max((r.max_S for r in maxes), default=None),
        sum(r.n_blocks for r in results),
    )
