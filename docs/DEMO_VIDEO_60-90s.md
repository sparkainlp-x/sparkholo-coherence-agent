# Demo video: 60-90 s script and shot list (draft)

Record with `NEBIUS_API_KEY` set, so the Nemotron step is real. If the key is not set the sidebar says "OFFLINE template"; then do not say "Nemotron" in the voice-over for that take. Keep the sidebar visible. Do not show the key, the Render env page or any email/phone.

| Time | Screen | Voice-over (plain, honest) |
|---|---|---|
| 0:00-0:08 | Title and yellow boundary banner | "AI can explain data fluently and still be wrong. This prototype shows an AI sentence only after the numbers check out. All data here is synthetic." |
| 0:08-0:20 | Pipeline line, signal plot, band-power bars | "Observer, calibrator, residual latch, verifier, archive. Two seeded synthetic sessions; every number carries an evidence tag." |
| 0:20-0:38 | Sidebar: choose "Baseline drift" | "Before any AI is asked, Spark's OES-32 residual latch checks every 32-sample block against the baseline. Drift trips it: LATCHED, AI not consulted." Click Generate: still no AI call. Switch back to No artifact: clean. |
| 0:38-0:58 | Click "Generate and verify" | "Now Nemotron, through Nebius Token Factory, explains the metrics. Each claim is re-computed. All pass, so the text is shown." Point at PASS rows and the narrative. |
| 0:58-1:12 | Tick "inject a false AI claim", run | "Here I append a false claim, theta at 90 percent. The verifier computes 8.9 percent, marks FAIL, and withholds the narrative." Show "Explanation REFUSED". |
| 1:12-1:25 | Evidence record hash, Download button | "Every run is archived with hashes, so any edit is detectable." |
| 1:25-1:30 | Repo URL, Apache-2.0 | "Open source, Apache-2.0. Synthetic, non-diagnostic, not a medical device." |

Numbers to expect at seed 42 and default sliders (SYNTHETIC): max S 0.286; follow-up alpha 68.7%, baseline 52.7%; follow-up alpha coherence 0.304; tamper claim computed 0.089. If Nemotron words things differently the PASS table is what matters.

Do not say: that this detects, diagnoses, treats or heals anything; that the latch measures a brain or person; that the thresholds are validated.
