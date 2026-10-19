"""Minimal NVIDIA Nemotron client via Nebius Token Factory (OpenAI-compatible API).

Env vars (never hard-code the key):
  NEBIUS_API_KEY   required for live mode
  NEBIUS_BASE_URL  default below; confirm in your Token Factory dashboard
  NEMOTRON_MODEL   exact model id as listed in Token Factory
"""
from __future__ import annotations

import json
import os
import re

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"   # FOUNDER: confirm
DEFAULT_MODEL = "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5"      # FOUNDER: confirm id

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


def _extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)  # reasoning traces
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model did not return JSON")
    return json.loads(text[start : end + 1])


def explain(metrics: dict, timeout: float = 60.0) -> dict:
    """Ask Nemotron for a structured explanation. Raises on any failure (fail-closed)."""
    from openai import OpenAI

    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        raise RuntimeError("NEBIUS_API_KEY is not set")
    client = OpenAI(
        base_url=os.environ.get("NEBIUS_BASE_URL", DEFAULT_BASE_URL), api_key=key, timeout=timeout
    )
    resp = client.chat.completions.create(
        model=os.environ.get("NEMOTRON_MODEL", DEFAULT_MODEL),
        temperature=0,
        max_tokens=800,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Metrics (SYNTHETIC):\n" + json.dumps(metrics, indent=2)},
        ],
    )
    return _extract_json(resp.choices[0].message.content or "")


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
