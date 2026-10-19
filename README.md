# ✨ SparkHolo Coherence Agent

Accountable AI explanations for **synthetic** holographic and neural-coherence research signals. NVIDIA Nemotron (via Nebius Token Factory) explains the metrics, and an independent verifier re-computes every claim before anything is displayed.

> **Boundaries:** synthetic data only · non-diagnostic research prototype · not a medical device · no health, therapeutic or clinical claims.

## How it works
1. `sparkholo/signals.py` generates seeded synthetic multi-channel signals for a baseline and a follow-up session.
2. It computes Welch relative band power (delta, theta, alpha, beta, gamma), the dominant band and mean alpha coherence.
3. `sparkholo/nemotron.py` sends the metrics to Nemotron, which must reply with JSON: `{"narrative": ..., "claims": [...]}`.
4. `sparkholo/verify.py` re-computes each claim (tolerance 0.02 on relative power, 0.03 on coherence). It checks that every number in the narrative is backed by a verified claim and blocks clinical wording. **Any failure withholds the narrative** (fail-closed).
5. `app.py` (Streamlit) shows the signals, metrics, the claim-by-claim verification table, and the narrative only if it passed.

## Setup
```bash
git clone https://github.com/sparkainlp-x/sparkholo-coherence-agent.git && cd sparkholo-coherence-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
export NEBIUS_API_KEY=...            # from Nebius Token Factory; never commit it
export NEMOTRON_MODEL=...            # exact Nemotron model id from Token Factory
# optional: export NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1/
streamlit run app.py
```
Without `NEBIUS_API_KEY`, the app runs a clearly labelled **offline template** explainer so it still demos end to end.

## Nebius Token Factory and Nemotron
- Token Factory exposes an OpenAI-compatible API, so the client is the standard `openai` package with `base_url` set.
- The model id and base URL are configurable through environment variables. Check both against your Token Factory dashboard.
- Reasoning traces (`<think>…</think>`) are stripped. Non-JSON output is refused.

## Tests
```bash
pip install pytest && pytest -q
```
28 tests: 26 cover determinism, band maths, refusal of tampered, unbacked, malformed and clinically worded explanations, and JSON parsing; 2 Streamlit `AppTest` smoke tests load `app.py` offline, click **Generate and verify explanation**, and check that the tamper checkbox yields **Explanation REFUSED**. GitHub Actions runs them on Python 3.10–3.13. No API key is needed for the tests.

## Deploy (Render)
`render.yaml` is included. Create a Render Web Service from the repo and set `NEBIUS_API_KEY` and `NEMOTRON_MODEL` as secret environment variables. Start command:
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## Licence
Apache-2.0. See `LICENSE`. © 2026 Jean-François Brisson / Spark AI NLP.
