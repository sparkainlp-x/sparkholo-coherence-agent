# Submission write-up (draft, short form)

**Title:** SparkHolo Coherence Agent: an AI explainer that is checked before it is shown

**One line:** A research prototype where NVIDIA Nemotron (via Nebius Token Factory) explains synthetic multi-channel signals, and every number it states is independently re-computed before any text reaches the screen.

**Problem.** Language models explain data fluently, and fluent is not the same as correct. In disability-led research on holographic and neural-coherence interfaces, an unchecked sentence can mislead the people the research is for.

**What we built.**
1. **Observer.** Seeded synthetic signals (baseline and follow-up sessions). No human data.
2. **Residual-Latch (OES-32).** Before any AI call, each 32-sample block is compared with the baseline block template. Score S = 0.45*Peak + 0.35*RMS + 0.20*MeanAbs on normalised residuals; a block at or above tau = 0.50 (TARGET) latches the run. NaN/inf/wrong shape also latch. Fail-closed: Nemotron is not called.
3. **Nemotron via Token Factory.** When `NEBIUS_API_KEY` is set, Nemotron must return a narrative plus structured claims (JSON).
4. **Verifier.** Re-computes each claim from the raw metrics (tolerance 0.02 relative power, 0.03 coherence, TARGET), requires every number in the narrative to be backed, blocks clinical or overclaiming words. Any failure withholds the narrative.
5. **Archive.** Downloadable JSON evidence record with SHA-256 per section and for the whole record; every number is tagged SYNTHETIC, TARGET, REPORTED or UNRUN.

**What it shows (default settings, seed 42, all SYNTHETIC).** Clean latch (max S 0.286, 1920 blocks). Spike, drift and dropout each latch. Baseline alpha relative power 52.7%, follow-up 68.7%, follow-up alpha coherence 0.304. The tamper toggle appends a false claim (theta 90%); the verifier computes 0.089 and refuses the narrative.

**Honest limits.**
- Synthetic data only. Non-diagnostic. Not a medical device. No health, therapeutic or clinical claims.
- The latch is a data-integrity stop bit, not a measure of any person or brain state.
- Thresholds (tau, margin, block size, weights, tolerances) are TARGET settings, not validated values.
- Without an API key the app uses a labelled offline template; Nemotron output exists only when the key is set. UNRUN in this draft: live Nemotron call from the deployed app (needs the key and the deployment).

**Why Nemotron / Nebius.** An open model behind an OpenAI-compatible endpoint made the "explainer" swappable; the verifier is what makes it accountable.

**Open source.** Apache-2.0, 91 offline tests, CI on Python 3.10-3.13. Repo: https://github.com/sparkainlp-x/sparkholo-coherence-agent

**Built by** Jean-Francois Brisson, Spark AI NLP (Fredericton, NB, Canada), disability-led.
