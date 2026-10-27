# SparkHolo Coherence Agent

Accountable AI explanations for **synthetic** holographic and neural-coherence research signals. A Spark AI NLP **OES-32 residual latch** checks the signals *before* any AI is consulted; NVIDIA Nemotron (via Nebius Token Factory) then explains the metrics, an independent verifier re-computes every claim before anything is displayed, and every run is archived as a hashed evidence record.

**Pipeline:** `Observer → Calibrator → Residual-Latch → Verifier → Archive`

> **Boundaries:** synthetic data only · non-diagnostic research prototype · not a medical device · no health, therapeutic or clinical claims.

## How it works
1. **Observer.** `sparkholo/signals.py` generates seeded synthetic multi-channel signals for a baseline and a follow-up session (seed 42 by default). A sidebar control can inject a demo artifact into the follow-up: electrode-pop spike, baseline drift, or NaN dropout.
2. **Calibrator.** `sparkholo/latch.py` splits each channel into 32-sample blocks. The reference `x_ref` is the baseline session's per-channel **block template** (the sample-by-sample mean of all baseline blocks). The normalisation scale is `margin × max|x − x_ref|` over the baseline (margin 3.0, TARGET).
3. **Residual-Latch** (Spark AI NLP's OES residual contract, OES-32). For every block: `R = max|x − x_ref|`, and on the normalised residual `S = 0.45·Peak + 0.35·RMS + 0.20·MeanAbs`. A block latches when `S ≥ τ` (τ = 0.50, TARGET). **Fail-closed:** NaN, inf, a wrong shape, a failed calibration or any latched block stops the run; Nemotron is **never called** and the UI shows **LATCHED: AI not consulted** with the blocks that fired.
4. **Verifier.** Only on a clean latch: Welch relative band power (delta…gamma), dominant band and mean alpha coherence are computed; `sparkholo/nemotron.py` sends them to Nemotron, which must reply with JSON `{"narrative": ..., "claims": [...]}`; `sparkholo/verify.py` re-computes each claim (tolerance 0.02 on relative power, 0.03 on coherence, both TARGET), checks that every number in the narrative is backed by a verified claim and blocks clinical wording. **Any failure withholds the narrative.**
5. **Archive.** `sparkholo/evidence.py` writes a JSON evidence record per run (download button in the app): config, seed, code version (package version + git sha), tagged metrics, latch result, raw AI output, verdict, a SHA-256 per section and one over the whole record. `verify_manifest()` re-checks every hash. No wall-clock time is stored, so identical inputs give an identical hash.

**Evidence tags.** Every number in the UI and in the record carries a tag: `SYNTHETIC` (computed from seeded synthetic signals: all metrics, R and S), `TARGET` (configured thresholds and tolerances: τ, margin, block size, weights, verifier tolerances), `REPORTED` (measured on real data: none in this app), `UNRUN` (not run).

**What the latch means.** It is a data-integrity stop bit: the follow-up left the envelope calibrated on the baseline. It is not a measure of any person or brain state. With the default controls the run is clean (max S 0.29, SYNTHETIC); the injected spike, drift and dropout latch. Extreme settings can also latch clean synthetic data, which is the intended fail-closed behaviour; whether they do depends on the seed (checked: noise 0 latches at seed 1 but not at seed 42, max S 0.495; both follow-up gains at 3.0 latch at seed 42, max S 0.525, but not at seed 1 or 21).

## Setup
```bash
git clone https://github.com/sparkainlp-x/sparkholo-coherence-agent.git && cd sparkholo-coherence-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
export NEBIUS_API_KEY=...            # from Nebius Token Factory; never commit it
# optional overrides (defaults shown):
# export NEMOTRON_MODEL=nvidia/nemotron-3-super-120b-a12b
# export NEBIUS_BASE_URL=https://api.tokenfactory.us-central1.nebius.com/v1/
#   (the region where the Token Factory catalog lists Nemotron 3 Super; EU models use https://api.tokenfactory.nebius.com/v1/)
streamlit run app.py
```
Without `NEBIUS_API_KEY`, the app runs a clearly labelled **offline template** explainer (sidebar: "OFFLINE template (no API key set)") so it still demos end to end. The template's output goes through the same verifier; no Nemotron call is made and none is claimed. Live Nemotron output is only produced when the key is set.

The **tamper demo** appends one false claim (theta 90%) to whatever the explainer returned (Nemotron output when the key is set, the template otherwise) and shows the verifier refusing it.

## Nebius Token Factory and Nemotron
- Token Factory exposes an OpenAI-compatible API, so the client is the standard `openai` package with `base_url` set.
- Default model `nvidia/nemotron-3-super-120b-a12b` (confirm it appears in `GET /v1/models` for your key); both the model id and base URL can be overridden with environment variables.
- Reasoning traces (`<think>…</think>`) are stripped. Non-JSON output is refused.

## Tests
```bash
pip install pytest && python -m pytest -q   # plain `pytest -q` also works (pytest.ini sets the path)
```
91 tests, offline, no API key: signal determinism and band maths; verifier refusal of tampered, unbacked, malformed and clinically worded explanations; the residual latch (clean passes, spike and drift latch, NaN/inf/wrong shape/bad baseline latch, the S formula); the pipeline (a latched run never calls the explainer, checked with a mock, in offline and live mode); evidence tags on every number in the record; deterministic manifest hashes and tamper detection by `verify_manifest`; explainer failures (missing key, HTTP 429 rate limit, rejected key, timeout) fail closed with a clear message that never contains the key; and Streamlit `AppTest` smoke tests (app loads, a rate-limited live call shows a message instead of crashing, button verifies, tamper checkbox yields **Explanation REFUSED**, each injected artifact yields **LATCHED: AI not consulted**). GitHub Actions runs them on Python 3.10–3.13.

## Deploy (Render)
`render.yaml` is a Render Blueprint for one free web service. In Render choose **New > Blueprint**, pick this repo, and enter `NEBIUS_API_KEY` when prompted (leave `NEMOTRON_MODEL` and `NEBIUS_BASE_URL` empty to use the defaults). The same settings for a manual **New > Web Service**:

| Setting | Value |
|---|---|
| Runtime | Python 3 (`PYTHON_VERSION=3.13.5`) |
| Build command | `pip install -r requirements-render.txt` (exact versions; `requirements.txt` keeps ranges for CI) |
| Start command | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false` |
| Health check path | `/_stcore/health` |
| Secret env var | `NEBIUS_API_KEY` |
| Optional env vars | `NEMOTRON_MODEL`, `NEBIUS_BASE_URL` |

Without the key the app runs the labelled offline template. If Token Factory rate-limits or rejects a call, the app shows a plain message and withholds the explanation (fail-closed). Free Render services sleep after about 15 minutes idle, so the first visit can take a minute.

## Credits
The residual-latch stage implements Spark AI NLP's OES residual contract (OES-32 blocks, Peak/RMS/MeanAbs score, fail-closed latch).

## Licence
Apache-2.0. See `LICENSE`. © 2026 Jean-François Brisson / Spark AI NLP.
