"""SparkHolo Coherence Agent: Streamlit UI. SYNTHETIC data only. Non-diagnostic.

Pipeline: Observer -> Calibrator -> Residual-Latch -> Verifier -> Archive.
"""
import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sparkholo.evidence import BOUNDARY, PIPELINE, fmt
from sparkholo.latch import BLOCK, MARGIN
from sparkholo.pipeline import ARTIFACTS, FS, RunConfig, explain_and_verify, prepare
from sparkholo.signals import BANDS
from sparkholo.verify import COH_TOL, REL_TOL

S, T = "SYNTHETIC", "TARGET"
st.set_page_config(page_title="SparkHolo Coherence Agent", page_icon="✨", layout="wide")
st.title("✨ SparkHolo Coherence Agent")
st.warning(BOUNDARY + " AI text is shown only after a clean residual latch and independent numerical verification.")
st.markdown("**Pipeline:** " + " → ".join(f"`{p}`" for p in PIPELINE))
st.caption("Every number carries an evidence tag: SYNTHETIC (computed from seeded synthetic signals), "
           "TARGET (configured threshold), REPORTED (real measured data: none here), UNRUN (not run).")

with st.sidebar:
    st.header("Synthetic session controls")
    seed = st.number_input("Seed (SYNTHETIC)", value=42, step=1)
    noise = st.slider("Noise (SYNTHETIC)", 0.0, 2.0, 0.5, 0.05)
    shared = st.slider("Shared source, drives coherence (SYNTHETIC)", 0.0, 1.0, 0.6, 0.05)
    st.subheader("Follow-up session vs baseline")
    alpha_f = st.slider("Follow-up alpha gain (SYNTHETIC)", 0.0, 3.0, 1.5, 0.1)
    theta_f = st.slider("Follow-up theta gain (SYNTHETIC)", 0.0, 3.0, 1.0, 0.1)
    st.subheader("🔒 Residual-Latch demo")
    artifact = st.selectbox("Inject artifact / drift into follow-up", list(ARTIFACTS),
                            format_func=lambda k: ARTIFACTS[k])
    live = bool(os.environ.get("NEBIUS_API_KEY"))
    st.caption("Explainer: " + ("Nemotron via Nebius Token Factory" if live else "OFFLINE template (no API key set)"))
    tamper = st.checkbox("Demo: inject a false AI claim", value=False)

cfg = RunConfig(seed=int(seed), noise=float(noise), shared=float(shared), alpha_gain=float(alpha_f),
                theta_gain=float(theta_f), artifact=artifact, tamper=bool(tamper))
prep = prepare(cfg)
lat = prep.latch

c1, c2 = st.columns(2)
with c1:
    st.subheader("Observer: signals (first 4 s, channel 1)")
    n = FS * 4
    fig = go.Figure()
    fig.add_scatter(x=prep.t[:n], y=prep.baseline[0, :n], name="baseline")
    fig.add_scatter(x=prep.t[:n], y=prep.followup[0, :n], name="follow-up")
    fig.update_layout(xaxis_title="seconds [SYNTHETIC]", yaxis_title="amplitude, a.u. [SYNTHETIC]", height=350)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.subheader("Residual-Latch: block scores S per channel")
    fig = go.Figure()
    nb = prep.followup.shape[1] // BLOCK
    for h in ("followup",):
        for c in range(prep.followup.shape[0]):
            ss = [x.S for x in lat.fired if x.session == h and x.channel == c and x.S is not None]
            ks = [x.block for x in lat.fired if x.session == h and x.channel == c and x.S is not None]
            if ks:
                fig.add_scatter(x=ks, y=ss, mode="markers", name=f"fired ch {c + 1}")
    fig.add_hline(y=cfg.tau, line_dash="dash", annotation_text=f"tau {cfg.tau:.2f} [TARGET]")
    fig.update_layout(xaxis_title=f"follow-up block index (of {nb})", yaxis_title="S [SYNTHETIC]",
                      xaxis_range=[0, nb], yaxis_range=[0, max(1.0, (lat.max_S or 0) * 1.1)], height=350)
    st.plotly_chart(fig, use_container_width=True)

st.markdown(
    f"Calibrator: reference = baseline 32-sample block template per channel; "
    f"block {fmt(BLOCK, T)} samples, margin {fmt(MARGIN, T, 1)}, tau {fmt(cfg.tau, T, 2)}. "
    f"Blocks evaluated {fmt(lat.n_blocks, S)}, max R {fmt(lat.max_R, S)}, max S {fmt(lat.max_S, S)}."
)
if lat.latched:
    st.error(f"🔒 LATCHED: AI not consulted ({lat.reason}). {lat.detail}")
    st.table(pd.DataFrame([{"session": h.session, "channel": h.channel + 1, "block": h.block,
                            "t start": fmt(h.t_start_s, S, 3), "R": fmt(h.R, S), "S": fmt(h.S, S),
                            "reason": h.reason} for h in lat.fired[:25]]
                          or [{"session": "-", "reason": lat.detail}]).astype(str))
    if len(lat.fired) > 25:
        st.caption(f"…and {len(lat.fired) - 25} more fired blocks (all in the evidence record).")
else:
    st.success(f"🔒 Residual-Latch clean: all blocks below tau {fmt(cfg.tau, T, 2)}.")
    m = prep.metrics
    st.subheader("Relative band power")
    fig = go.Figure()
    for s in ("baseline", "followup"):
        fig.add_bar(x=list(BANDS), y=[m[s]["relative_power"][b] for b in BANDS], name=s)
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="relative power [SYNTHETIC]", height=320, barmode="group")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.DataFrame({s: {**{f"{b} (rel)": fmt(v["relative_power"][b], S) for b in BANDS},
                                   "dominant band": v["dominant_band"],
                                   "alpha coherence": fmt(v["alpha_coherence"], S)} for s, v in m.items()}).astype(str),
                 use_container_width=True)

st.header("🧠 AI explanation, verified before display")
st.caption(f"Verifier tolerances: relative power ±{fmt(REL_TOL, T, 2)}, coherence ±{fmt(COH_TOL, T, 2)}.")
key = json.dumps(cfg.__dict__, sort_keys=True)
if st.button("Generate and verify explanation", type="primary"):
    st.session_state["run"] = (key, explain_and_verify(prep, live=live))
res = st.session_state.get("run", (None, None))[1] if st.session_state.get("run", (None,))[0] == key else None

if res is not None:
    if res.status == "LATCHED":
        st.error("LATCHED: AI not consulted. The Residual-Latch fired, so no explanation was requested.")
        st.write("Blocks that fired: " + ", ".join(f"{h.session} ch {h.channel + 1} block {h.block}"
                                                   for h in res.prepared.latch.fired[:12])
                 + (" …" if len(res.prepared.latch.fired) > 12 else ""))
    elif res.status == "EXPLAINER_ERROR":
        st.error(f"Explainer unavailable; nothing displayed (fail-closed): {res.error}")
    else:
        v = res.verdict
        rows = [{"claim": str(r.claim), "verified": "✅" if r.ok else "❌", "detail": f"{r.reason} [SYNTHETIC]"}
                for r in v.results]
        st.table(pd.DataFrame(rows or [{"claim": "(none)", "verified": "❌", "detail": "no claims"}]))
        if v.ok:
            st.success("All claims independently verified against computed metrics.")
            st.write(res.explanation["narrative"])
            st.caption("All numbers in this narrative are SYNTHETIC.")
        else:
            st.error("Explanation REFUSED. The narrative is withheld because:")
            for p in v.problems:
                st.write("• " + p)
    st.subheader("🧾 Evidence record")
    st.caption(f"manifest SHA-256: `{res.manifest['manifest_sha256']}` · code {res.manifest['code_version']}")
    st.download_button("Download evidence record (JSON)", json.dumps(res.manifest, indent=2, sort_keys=True),
                       file_name=f"sparkholo-evidence-{res.manifest['manifest_sha256'][:12]}.json",
                       mime="application/json")
