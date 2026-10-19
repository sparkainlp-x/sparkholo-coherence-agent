"""Evidence tags and the per-run evidence record (Archive stage).

Every number shown in the UI or written to a report carries one tag:

* SYNTHETIC: computed from seeded synthetic signals (all metrics, residuals and generator inputs here);
* REPORTED: measured on named real data with provenance (none in this app);
* TARGET: a configured threshold or tolerance (tau, margin, verifier tolerances);
* UNRUN: not run.

The manifest stores SHA-256 hashes of each section plus one hash over the whole record. There is no wall-clock
timestamp, so the same inputs and code version give the same manifest hash.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from . import __version__

TAGS = ("SYNTHETIC", "REPORTED", "TARGET", "UNRUN")
SCHEMA = "sparkholo-evidence/1"
PIPELINE = ["Observer", "Calibrator", "Residual-Latch", "Verifier", "Archive"]
BOUNDARY = ("SYNTHETIC data only. Non-diagnostic research prototype. Not a medical device. "
            "No health, therapeutic or clinical claim.")
SECTIONS = ("config", "inputs", "code_version", "metrics", "latch", "ai_raw_output", "verdict")
IDENTIFIER_KEYS = frozenset({"channel", "block", "seed"})  # identifiers, not measurements


def tagged(value: Any, tag: str, unit: str | None = None) -> dict:
    if tag not in TAGS:
        raise ValueError(f"unknown evidence tag {tag!r}")
    out = {"value": value, "tag": tag}
    if unit:
        out["unit"] = unit
    return out


def fmt(value: float | int | None, tag: str, digits: int = 4) -> str:
    """Display string with its tag, e.g. ``0.3120 [SYNTHETIC]``."""
    if tag not in TAGS:
        raise ValueError(f"unknown evidence tag {tag!r}")
    if value is None:
        return f"n/a [{tag}]"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value} [{tag}]"
    return f"{value:.{digits}f} [{tag}]"


def tag_tree(obj: Any, tag: str) -> Any:
    """Wrap every numeric leaf of ``obj`` (except identifier keys) as ``{"value", "tag"}``."""
    if isinstance(obj, dict):
        return {k: (v if k in IDENTIFIER_KEYS else tag_tree(v, tag)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [tag_tree(v, tag) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float)):
        return tagged(round(obj, 6) if isinstance(obj, float) else obj, tag)
    return obj


def untagged_numbers(obj: Any, path: str = "") -> list[str]:
    """Paths of numeric leaves that are not inside a tagged ``{"value", "tag"}`` object."""
    if isinstance(obj, dict):
        if set(obj) >= {"value", "tag"} and obj["tag"] in TAGS:
            return []
        return [p for k, v in obj.items() if k not in IDENTIFIER_KEYS for p in untagged_numbers(v, f"{path}.{k}")]
    if isinstance(obj, list):
        return [p for i, v in enumerate(obj) for p in untagged_numbers(v, f"{path}[{i}]")]
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return [path]
    return []


def code_version() -> str:
    """Package version plus the git commit when known (env SPARKHOLO_GIT_SHA / RENDER_GIT_COMMIT, or git)."""
    sha = os.environ.get("SPARKHOLO_GIT_SHA") or os.environ.get("RENDER_GIT_COMMIT")
    if not sha:
        try:
            sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, capture_output=True,
                                 text=True, timeout=2, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            sha = ""
    return f"{__version__}+{sha[:12]}" if sha else f"{__version__}+unrecorded"


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha256(obj: Any) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def build_manifest(**sections: Any) -> dict:
    missing = set(SECTIONS) - set(sections)
    if missing:
        raise ValueError(f"missing manifest sections: {sorted(missing)}")
    body = {"schema": SCHEMA, "boundary": BOUNDARY, "pipeline": PIPELINE, **{k: sections[k] for k in SECTIONS}}
    body["sha256"] = {k: sha256(sections[k]) for k in SECTIONS}
    body["manifest_sha256"] = sha256(body)
    return body


def verify_manifest(manifest: Any) -> list[str]:
    """Return the list of problems; an empty list means the record is intact."""
    if not isinstance(manifest, dict):
        return ["manifest is not an object"]
    problems = []
    if manifest.get("schema") != SCHEMA:
        problems.append(f"unknown schema {manifest.get('schema')!r}")
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict):
        return problems + ["missing section hashes"]
    for k in SECTIONS:
        if k not in manifest:
            problems.append(f"missing section {k!r}")
        elif hashes.get(k) != sha256(manifest[k]):
            problems.append(f"section {k!r} does not match its SHA-256")
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    try:
        if manifest.get("manifest_sha256") != sha256(body):
            problems.append("manifest_sha256 does not match")
    except (TypeError, ValueError) as exc:
        problems.append(f"manifest is not canonical JSON: {exc}")
    return problems
