"""Synthetic multi-channel signals and band metrics. SYNTHETIC data only; not EEG from any person."""
from __future__ import annotations

import numpy as np
from scipy.signal import coherence as _coherence
from scipy.signal import welch

BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 / 1 compatibility


def synth_session(
    seed: int = 42,
    fs: int = 256,
    seconds: float = 30.0,
    channels: int = 4,
    alpha_gain: float = 1.0,
    theta_gain: float = 1.0,
    beta_gain: float = 1.0,
    noise: float = 0.5,
    shared: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (t, data[channels, samples]) of SYNTHETIC sines + noise.

    `shared` (0..1) mixes a common source into every channel so inter-channel
    coherence is controllable. Deterministic for a given seed.
    """
    if not (0.0 <= shared <= 1.0):
        raise ValueError("shared must be in [0, 1]")
    if fs <= 0 or seconds <= 0 or channels < 2:
        raise ValueError("fs, seconds must be > 0 and channels >= 2")
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * seconds)) / fs

    def source() -> np.ndarray:
        ph = rng.uniform(0, 2 * np.pi, 4)
        return (
            alpha_gain * np.sin(2 * np.pi * 10 * t + ph[0])
            + theta_gain * 0.7 * np.sin(2 * np.pi * 6 * t + ph[1])
            + beta_gain * 0.4 * np.sin(2 * np.pi * 20 * t + ph[2])
            + 0.3 * np.sin(2 * np.pi * 2 * t + ph[3])
        )

    common = source()
    data = np.array(
        [shared * common + (1 - shared) * source() + noise * rng.standard_normal(t.size) for _ in range(channels)]
    )
    return t, data


def band_powers(x: np.ndarray, fs: int) -> tuple[dict[str, float], dict[str, float]]:
    """Absolute (Welch PSD integrated) and relative band power for one channel."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < fs or not np.all(np.isfinite(x)):
        raise ValueError("need a finite 1-D signal of at least 1 s")
    f, p = welch(x, fs=fs, nperseg=min(x.size, fs * 2))
    absolute = {}
    for name, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        absolute[name] = float(_trapz(p[m], f[m]))
    total = sum(absolute.values())
    if total <= 0:
        raise ValueError("zero total power")
    relative = {k: v / total for k, v in absolute.items()}
    return absolute, relative


def mean_band_coherence(data: np.ndarray, fs: int, band: str) -> float:
    """Mean magnitude-squared coherence across all channel pairs inside one band."""
    lo, hi = BANDS[band]
    vals = []
    for i in range(data.shape[0]):
        for j in range(i + 1, data.shape[0]):
            f, c = _coherence(data[i], data[j], fs=fs, nperseg=fs * 2)
            m = (f >= lo) & (f < hi)
            vals.append(float(np.mean(c[m])))
    return float(np.mean(vals))


def session_metrics(data: np.ndarray, fs: int) -> dict:
    """Channel-averaged relative band power, dominant band, alpha coherence."""
    rels = [band_powers(ch, fs)[1] for ch in data]
    rel = {b: float(np.mean([r[b] for r in rels])) for b in BANDS}
    return {
        "relative_power": rel,
        "dominant_band": max(rel, key=rel.get),
        "alpha_coherence": mean_band_coherence(data, fs, "alpha"),
    }
