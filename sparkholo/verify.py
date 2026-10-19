"""Fail-closed verification of AI-generated explanations against computed metrics.

The model must return structured claims. Every claim is recomputed here. The
narrative is shown ONLY if every claim verifies, every number in the narrative
is backed by a verified claim, and no forbidden clinical wording appears.
Anything unknown, missing or malformed is a FAIL.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .signals import BANDS

REL_TOL = 0.02   # absolute tolerance on relative power (fraction 0..1)
COH_TOL = 0.03   # absolute tolerance on coherence (0..1)
SESSIONS = ("baseline", "followup")

FORBIDDEN = re.compile(
    r"\b(diagnos\w*|treat\w*|cure[sd]?|curing|heal(s|ed|ing)?|therap\w*|clinical\w*|"
    r"patient\w*|disease\w*|disorder\w*|regenerat\w*|improv\w*\s+(brain|health|cognition)|"
    r"proven|guarantee\w*|consciousness)\b",
    re.IGNORECASE,
)
NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(%)?")


@dataclass
class ClaimResult:
    claim: dict
    ok: bool
    reason: str
    computed: object = None


@dataclass
class Verdict:
    ok: bool
    results: list[ClaimResult] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _num(v) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError("value must be a finite number")
    return float(v)


def check_claim(claim: dict, metrics: dict) -> ClaimResult:
    try:
        if not isinstance(claim, dict):
            return ClaimResult(claim, False, "claim is not an object")
        kind = claim.get("type")
        if kind == "relative_power":
            s, b = claim["session"], claim["band"]
            if s not in SESSIONS or b not in BANDS:
                return ClaimResult(claim, False, "unknown session or band")
            got = metrics[s]["relative_power"][b]
            ok = abs(got - _num(claim["value"])) <= REL_TOL
            return ClaimResult(claim, ok, "match" if ok else f"computed {got:.3f}", got)
        if kind == "dominant_band":
            s = claim["session"]
            if s not in SESSIONS:
                return ClaimResult(claim, False, "unknown session")
            got = metrics[s]["dominant_band"]
            ok = claim.get("band") == got
            return ClaimResult(claim, ok, "match" if ok else f"computed {got}", got)
        if kind == "change":
            b, d = claim["band"], claim["direction"]
            if b not in BANDS or d not in ("increase", "decrease"):
                return ClaimResult(claim, False, "unknown band or direction")
            delta = metrics["followup"]["relative_power"][b] - metrics["baseline"]["relative_power"][b]
            if abs(delta) <= REL_TOL:
                return ClaimResult(claim, False, f"change {delta:+.3f} is within tolerance; no direction can be claimed", delta)
            ok = (delta > 0) == (d == "increase")
            return ClaimResult(claim, ok, "match" if ok else f"computed {delta:+.3f}", delta)
        if kind == "alpha_coherence":
            s = claim["session"]
            if s not in SESSIONS:
                return ClaimResult(claim, False, "unknown session")
            got = metrics[s]["alpha_coherence"]
            ok = abs(got - _num(claim["value"])) <= COH_TOL
            return ClaimResult(claim, ok, "match" if ok else f"computed {got:.3f}", got)
        return ClaimResult(claim, False, f"unknown claim type {kind!r}")
    except (KeyError, TypeError, ValueError) as exc:
        return ClaimResult(claim, False, f"malformed claim: {exc}")


def verify(explanation: dict, metrics: dict) -> Verdict:
    v = Verdict(ok=False)
    if not isinstance(explanation, dict):
        v.problems.append("explanation is not an object")
        return v
    narrative = explanation.get("narrative")
    claims = explanation.get("claims")
    if not isinstance(narrative, str) or not narrative.strip():
        v.problems.append("missing narrative")
    if not isinstance(claims, list) or not claims:
        v.problems.append("no structured claims: unverifiable narratives are refused")
        claims = []
    v.results = [check_claim(c, metrics) for c in claims]
    for r in v.results:
        if not r.ok:
            v.problems.append(f"claim failed: {r.reason}")
    if isinstance(narrative, str):
        hit = FORBIDDEN.search(narrative)
        if hit:
            v.problems.append(f"forbidden clinical/overclaim wording: {hit.group(0)!r}")
        backed = []
        for r in v.results:
            if r.ok and isinstance(r.claim, dict) and isinstance(r.claim.get("value"), (int, float)):
                backed.append(float(r.claim["value"]))
        edges = {x for lo_hi in BANDS.values() for x in lo_hi}
        for m in NUMBER.finditer(narrative):
            n = float(m.group(1))
            if m.group(2):
                n /= 100.0
            if n in edges or any(abs(n - b) <= 0.005 + 1e-9 for b in backed):
                continue
            v.problems.append(f"number {m.group(0).strip()!r} in narrative is not backed by a verified claim")
    v.ok = not v.problems
    return v
