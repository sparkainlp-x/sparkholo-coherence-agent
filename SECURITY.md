# Security policy

SparkHolo Coherence Agent is a research prototype that uses **synthetic data only**. It is not a medical device and
must not be used with data from real people.

## API keys

The only secret is `NEBIUS_API_KEY`, read from an environment variable (for example a Render secret). It must never
be committed, logged or shown in the UI. The repository contains no key, CI runs offline without one, and `.env`
files and `.streamlit/secrets.toml` are git-ignored. If a key is ever exposed, revoke it in the Nebius Token Factory
dashboard immediately.

## What counts as a security issue

- any way to leak the API key;
- any way to make the app display an AI narrative that has **not** passed verification (a bypass of the
  fail-closed verifier in `sparkholo/verify.py`);
- any way for model output to inject content or code into the page.

## Supported versions

Only the `main` branch receives fixes.

## Reporting a vulnerability

Please **do not open a public issue**. Contact the maintainer, Jean-François Brisson, privately via the contact
details on the GitHub profile (https://github.com/sparkainlp-x) or the website (https://sparkainlpx.xyz). Include
steps to reproduce and the impact you expect. You should get an acknowledgement within 7 days.
