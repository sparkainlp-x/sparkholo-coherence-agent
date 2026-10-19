# ✨ SparkHolo Coherence Agent: Accountable AI for Disability-Led Holographic Research

Built for the Nebius × NVIDIA Global AI Hackathon by Jean-François Brisson / Spark AI NLP (Fredericton, NB, Canada).

Research AI is only useful when every sentence it writes can be checked. SparkHolo Coherence Agent is an interactive research prototype that generates **synthetic** multi-channel coherence signals, runs Spark AI NLP's **OES-32 residual latch** before any AI is consulted, asks **NVIDIA Nemotron** to explain the metrics, and then **independently re-computes every claim** before a single word reaches the screen. If the data fails the latch, the AI is never called; if any number doesn't match, the explanation is withheld. Pipeline: Observer → Calibrator → Residual-Latch → Verifier → Archive.

**What's inside**
🌊 **Synthetic signal engine.** Deterministic, seeded multi-channel signals (delta, theta, alpha, beta and gamma components plus noise) with a controllable shared source, so inter-channel coherence can be explored safely. No human data is used anywhere.
📊 **Frequency-band analysis.** Welch power spectra, relative band power, dominant band and mean alpha-band coherence, compared between a baseline and a follow-up session.
🔒 **Residual-Latch (OES-32).** Spark AI NLP's OES residual contract runs before the AI step: each channel is cut into 32-sample blocks and compared with a baseline block template (R = max|x − x_ref|; S = 0.45·Peak + 0.35·RMS + 0.20·MeanAbs; τ = 0.50, a TARGET setting). NaN, inf, a wrong shape or any block over τ latches the run fail-closed: Nemotron is not consulted, and the UI names the blocks that fired. A sidebar control injects a spike, drift or dropout to show it.
🧠 **NVIDIA Nemotron via Nebius Token Factory.** An open NVIDIA model (Nemotron 3 Super) is called through Token Factory's OpenAI-compatible API and must return a narrative plus a structured list of claims.
🛡️ **Verification and safeguards layer.** Every claim is re-computed from the raw metrics within a stated tolerance. Every number in the narrative must be backed by a verified claim. Clinical or overclaiming wording ("diagnose", "treatment", "heals", "therapy", "regeneration") is blocked. Unknown, malformed or missing output is refused. Failures are shown openly, claim by claim.
🧾 **Evidence record.** Every run produces a downloadable JSON manifest: SHA-256 hashes of the config, seed, code version, metrics, latch result, raw AI output and verdict, with a verify function that detects any edit. Every number carries an evidence tag (SYNTHETIC, REPORTED, TARGET or UNRUN).
🔍 **Tamper demo.** One toggle injects a false AI claim so anyone can watch the verifier catch it.
♿ **Disability-led, open design.** Built by a disabled founder, with plain-language warnings, keyboard-friendly Streamlit controls and an Apache-2.0 licence.
🚀 **Deployable.** Streamlit on Render, with the API key held only in an environment variable.

**Boundaries.** All data is synthetic. This is a non-diagnostic research prototype, not a medical device, and it makes no health, therapeutic or clinical claims. No results on people are reported.

**Our goal:** to make sure future research into holographic interfaces, led by and for disabled people, starts from AI explanations that are numerically verified and openly auditable, never from unchecked narratives.

Thank you to NVIDIA for the open Nemotron models, to Nebius for Token Factory, and to the organizers for creating space for accountable AI. 🙏

#NVIDIA #Nemotron #Nebius #TokenFactory #AccountableAI #OpenScience #DisabilityLed #SparkAINLP
