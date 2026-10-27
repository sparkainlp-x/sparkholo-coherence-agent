"""Minimal NVIDIA Nemotron client via Nebius Token Factory (OpenAI-compatible API).

Env vars (never hard-code the key):
  NEBIUS_API_KEY   required for live mode
  NEBIUS_BASE_URL  default below (Nebius Token Factory, OpenAI-compatible)
  NEMOTRON_MODEL   model id; confirm it is listed by GET {NEBIUS_BASE_URL}models
"""
from __future__ import annotations

import json
import os
import re

# Token Factory OpenAI-compatible endpoint. The Token Factory model catalog lists Nemotron 3 Super in the
# us-central1 region, and Nebius's Nemotron quickstart uses this regional endpoint. The global endpoint
# https://api.tokenfactory.nebius.com/v1/ serves EU-region models; set NEBIUS_BASE_URL to override.
DEFAULT_BASE_URL = "https://api.tokenfactory.us-central1.nebius.com/v1/"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"  # from Nebius docs; confirm against /v1/models

SYSTEM = (
    "You explain SYNTHETIC research signals. Never give medical, diagnostic or therapeutic "
    "statements. Return ONLY a JSON object: {\"narrative\": str, \"claims\": [...]}. "
    "Every number in the narrative must also appear as a claim. Allowed claims: "
    "{\"type\":\"relative_power\",\"session\":\"baseline|followup\",\"band\":B,\"value\":0..1}, "
    "{\"type\":\"dominant_band\",\"session\":S,\"band\":B}, "
    "{\"type\":\"change\",\"band\":B,\"direction\":\"increase|decrease\"}, "
    "{\"type\":\"alpha_coherence\",\"session\":S,\"value\":0..1}. "
    "Bands: delta, theta, alpha, beta, gamma. Write percentages as e.g. 31.2%."
)


class MissingKeyError(RuntimeError):
    """Live mode was requested but NEBIUS_API_KEY is not set."""


def _extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)  # reasoning traces
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model did not return JSON")
    return json.loads(text[start : end + 1])


def explain_with_raw(metrics: dict, timeout: float = 60.0) -> tuple[dict, str]:
    """Ask Nemotron for a structured explanation; return (parsed, raw text). Raises on any failure (fail-closed)."""
    from openai import OpenAI

    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        raise MissingKeyError("NEBIUS_API_KEY is not set")
    client = OpenAI(
        base_url=os.environ.get("NEBIUS_BASE_URL") or DEFAULT_BASE_URL, api_key=key, timeout=timeout
    )
    resp = client.chat.completions.create(
        model=os.environ.get("NEMOTRON_MODEL") or DEFAULT_MODEL,
        temperature=0,
        max_tokens=800,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Metrics (SYNTHETIC):\n" + json.dumps(metrics, indent=2)},
        ],
    )
    raw = resp.choices[0].message.content or ""
    return _extract_json(raw), raw


def describe_error(exc: BaseException) -> str:
    """Plain-language, key-free message for an explainer failure (shown in the UI and the evidence record).

    API error bodies are not echoed, because some providers include part of the key in them.
    """
    if isinstance(exc, MissingKeyError):
        return ("NEBIUS_API_KEY is not set, so Nemotron was not called. Set it as an environment variable "
                "to use Nemotron; without it the app runs the labelled offline template instead.")
    try:
        import openai
    except ImportError:  # pragma: no cover - openai is a hard requirement
        openai = None
    if openai is not None:
        if isinstance(exc, openai.RateLimitError):
            return ("Nebius Token Factory rate limit reached (HTTP 429). Wait a minute and try again; "
                    "nothing was displayed.")
        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return f"Nebius API key was rejected (HTTP {exc.status_code}). Check NEBIUS_API_KEY."
        if isinstance(exc, openai.NotFoundError):
            return "Model or endpoint not found (HTTP 404). Check NEMOTRON_MODEL and NEBIUS_BASE_URL."
        if isinstance(exc, openai.APITimeoutError):
            return "Nebius Token Factory did not answer in time. Try again."
        if isinstance(exc, openai.APIConnectionError):
            return "Could not reach Nebius Token Factory. Check the network and NEBIUS_BASE_URL."
        if isinstance(exc, openai.APIStatusError):
            return f"Nebius Token Factory returned HTTP {exc.status_code}. Try again later."
    msg = f"{type(exc).__name__}: {exc}"
    key = os.environ.get("NEBIUS_API_KEY")
    if key and len(key) >= 8:
        msg = msg.replace(key, "[redacted]")
    return msg[:300]


def explain(metrics: dict, timeout: float = 60.0) -> dict:
    """Ask Nemotron for a structured explanation. Raises on any failure (fail-closed)."""
    return explain_with_raw(metrics, timeout)[0]


def inject_false_claim(exp: dict) -> dict:
    """Return a copy of ``exp`` (from Nemotron or the offline template) with one false claim appended.

    Used only by the tamper demo, so the verifier can be seen refusing it. The 90% theta claim is far from any
    value the synthetic signals produce.
    """
    out = {"narrative": str(exp.get("narrative", "")) + " Theta power reaches 90.0% at follow-up.",
           "claims": list(exp.get("claims", [])) + [
               {"type": "relative_power", "session": "followup", "band": "theta", "value": 0.9}]}
    return out


def offline_explain(metrics: dict, tamper: bool = False) -> dict:
    """Deterministic template used when no key is set. Labelled OFFLINE in the UI.

    tamper=True injects one false claim so the demo can show the verifier blocking it.
    """
    b, f = metrics["baseline"], metrics["followup"]
    a0, a1 = b["relative_power"]["alpha"], f["relative_power"]["alpha"]
    claims = [
        {"type": "dominant_band", "session": "baseline", "band": b["dominant_band"]},
        {"type": "relative_power", "session": "baseline", "band": "alpha", "value": round(a0, 3)},
        {"type": "relative_power", "session": "followup", "band": "alpha", "value": round(a1, 3)},
        {"type": "alpha_coherence", "session": "followup", "value": round(f["alpha_coherence"], 3)},
    ]
    if abs(a1 - a0) > 0.02:
        claims.append({"type": "change", "band": "alpha", "direction": "increase" if a1 > a0 else "decrease"})
    narrative = (
        f"In this synthetic comparison the baseline is dominated by the {b['dominant_band']} band. "
        f"Relative alpha power is {a0*100:.1f}% at baseline and {a1*100:.1f}% at follow-up, "
        f"with follow-up alpha coherence of {f['alpha_coherence']:.3f}."
    )
    if tamper:
        claims.append({"type": "relative_power", "session": "followup", "band": "theta", "value": 0.9})
        narrative += " Theta power reaches 90.0% at follow-up."
    return {"narrative": narrative, "claims": claims}
