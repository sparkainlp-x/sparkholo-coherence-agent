"""SparkHolo Coherence Agent: Streamlit UI. SYNTHETIC data only. Non-diagnostic."""
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sparkholo import nemotron
from sparkholo.signals import BANDS, session_metrics, synth_session
from sparkholo.verify import verify

FS = 256
st.set_page_config(page_title="SparkHolo Coherence Agent", page_icon="✨", layout="wide")
st.title("✨ SparkHolo Coherence Agent")
st.warning(
    "SYNTHETIC data only. Research prototype. Not a medical device. No diagnosis, treatment or "
    "health claim of any kind. AI text is shown only after independent numerical verification."
)

with st.sidebar:
    st.header("Synthetic session controls")
    seed = st.number_input("Seed", value=42, step=1)
    noise = st.slider("Noise", 0.0, 2.0, 0.5, 0.05)
    shared = st.slider("Shared source (drives coherence)", 0.0, 1.0, 0.6, 0.05)
    st.subheader("Follow-up session vs baseline")
    alpha_f = st.slider("Follow-up alpha gain", 0.0, 3.0, 1.5, 0.1)
    theta_f = st.slider("Follow-up theta gain", 0.0, 3.0, 1.0, 0.1)
    live = bool(os.environ.get("NEBIUS_API_KEY"))
    st.caption("Explainer: " + ("Nemotron via Nebius Token Factory" if live else "OFFLINE template (no API key set)"))
    tamper = st.checkbox("Demo: inject a false AI claim", value=False)

t, base = synth_session(seed=int(seed), noise=noise, shared=shared)
_, foll = synth_session(seed=int(seed) + 1, noise=noise, shared=shared, alpha_gain=alpha_f, theta_gain=theta_f)
metrics = {"baseline": session_metrics(base, FS), "followup": session_metrics(foll, FS)}

c1, c2 = st.columns(2)
with c1:
    st.subheader("Signals (first 4 s, channel 1)")
    n = FS * 4
    fig = go.Figure()
    fig.add_scatter(x=t[:n], y=base[0, :n], name="baseline")
    fig.add_scatter(x=t[:n], y=foll[0, :n], name="follow-up")
    fig.update_layout(xaxis_title="seconds (SYNTHETIC)", height=350)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.subheader("Relative band power")
    fig = go.Figure()
    for s in ("baseline", "followup"):
        fig.add_bar(x=list(BANDS), y=[metrics[s]["relative_power"][b] for b in BANDS], name=s)
    fig.update_layout(yaxis_tickformat=".0%", height=350, barmode="group")
    st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    pd.DataFrame({s: {**{f"{b} (rel)": round(m["relative_power"][b], 4) for b in BANDS},
                      "dominant band": m["dominant_band"], "alpha coherence": round(m["alpha_coherence"], 4)}
                  for s, m in metrics.items()}).astype(str),
    use_container_width=True,
)

st.header("🧠 AI explanation, verified before display")
if st.button("Generate and verify explanation", type="primary"):
    try:
        exp = nemotron.explain(metrics) if (live and not tamper) else nemotron.offline_explain(metrics, tamper=tamper)
        if tamper and live:
            exp = nemotron.offline_explain(metrics, tamper=True)
    except Exception as exc:  # fail-closed: no text is shown on any error
        st.error(f"Explainer unavailable; nothing displayed (fail-closed): {exc}")
        st.stop()
    verdict = verify(exp, metrics)
    rows = [{"claim": str(r.claim), "verified": "✅" if r.ok else "❌", "detail": r.reason} for r in verdict.results]
    st.table(pd.DataFrame(rows) if rows else pd.DataFrame([{"claim": "(none)", "verified": "❌", "detail": "no claims"}]))
    if verdict.ok:
        st.success("All claims independently verified against computed metrics.")
        st.write(exp["narrative"])
    else:
        st.error("Explanation REFUSED. The narrative is withheld because:")
        for p in verdict.problems:
            st.write("• " + p)
