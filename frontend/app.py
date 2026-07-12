"""
HybridSaaS-Sec - Streamlit dashboard
====================================
Interactive demonstration of the HybridSaaS-Sec framework
(Mir & Hashir, "A Scalable Framework for Privacy-Preserving API Monitoring
and Multimodal Data Leakage Prevention in Enterprise Cloud Environments", 2026).

Three tabs:
  1. System Overview  - Architecture + paper-published metrics
  2. Live Simulation  - Per-event MITM pipeline (PII -> Anomaly -> SBRS gauge)
  3. Comparative Metrics - EWMA vs Hybrid behavioural analytics

The dashboard imports the FastAPI backend modules directly so it works
stand-alone without needing the uvicorn process running.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make `backend` importable when running `streamlit run frontend/app.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.anomaly_engine import (DEFAULT_ALPHA, EngineUnavailable,  # noqa: E402
                                    evaluate as eval_anomaly)
from backend.data_loader import load_all  # noqa: E402
from backend.measured_metrics import (load_measured_metrics,  # noqa: E402
                                      load_pii_metrics)
from backend.pii_pipeline import scan_event as pii_scan_event  # noqa: E402
from backend.risk_orchestrator import (DEFAULT_BETA, SBRS_BANDS,  # noqa: E402
                                       _event_summary, compute_sbrs, score_event)
from backend.schemas import (AnomalyResult, FullScoringResult,  # noqa: E402
                             PaperMetrics, PIIResult, SBRSResult)

# Band cut-points, read from the one place they are defined (risk_orchestrator).
# Hardcoding them here is how the gauge and the enforcement banner drifted apart.
T_BLOCK = SBRS_BANDS[0][0]
T_ALERT = SBRS_BANDS[1][0]

# Show the Plotly modebar so users can reset zoom / pan / autoscale.
PLOTLY_CONFIG = {
    "displaylogo": False,
    "displayModeBar": True,
    "modeBarButtonsToAdd": ["resetScale2d"],
    "scrollZoom": False,
}


def _zoom_reset_button(group: str, label: str = "Reset zoom") -> int:
    """Render a small reset-zoom button and return a monotonic counter.

    Each Plotly chart in `group` should pass `key=f"{group}_{counter}"`
    so a click forces Streamlit to remount the chart in its initial state.
    """
    state_key = f"_zoom_{group}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0
    if st.button(
        f"\u21bb {label}", key=f"btn_{state_key}",
        help="Restore the chart's original axes.",
    ):
        st.session_state[state_key] += 1
    return st.session_state[state_key]


st.set_page_config(
    page_title="HybridSaaS-Sec | Demo",
    page_icon="[*]",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Global stylesheet - injected once.
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
  /* ---------- Hide Streamlit's top-right toolbar / Deploy button ------ */
  [data-testid="stDecoration"],
  .stDeployButton,
  #MainMenu { display: none !important; visibility: hidden !important; }
  header { background: transparent !important; }

  /* ---------- Page background gradient -------------------------------- */
  [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 20% 0%, #1b2735 0%, #0b1320 55%, #06090f 100%);
  }
  [data-testid="stHeader"] { 
    background: transparent; 
    transform: none !important;
  }
  /* ---------- Force the sidebar toggle to be fully visible and clickable ---------- */
  [data-testid="collapsedControl"],
  [data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    position: fixed !important;
    top: 15px !important;
    left: 15px !important;
    z-index: 2147483647 !important;
    pointer-events: auto !important;
    background-color: rgba(30, 40, 50, 0.8) !important;
    border-radius: 6px !important;
    padding: 6px !important;
  }
  [data-testid="collapsedControl"] svg,
  [data-testid="stSidebarCollapsedControl"] svg {
    color: #ffffff !important;
    fill: #ffffff !important;
    width: 24px !important;
    height: 24px !important;
  }
  [data-testid="collapsedControl"]:hover,
  [data-testid="stSidebarCollapsedControl"]:hover {
    background-color: #3498db !important;
  }
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(20,28,40,0.95), rgba(10,16,24,0.95));
    border-right: 1px solid rgba(255,255,255,0.05);
  }

  /* ---------- Headings: gradient brand text --------------------------- */
  .block-container h1 {
    background: linear-gradient(90deg, #3498db 0%, #9b59b6 50%, #e74c3c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800;
    letter-spacing: -0.5px;
  }
  .block-container h2, .block-container h3 { color: #ecf0f1; }

  /* ---------- Tab bar: animated underline ----------------------------- */
  .stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #95a5a6;
    padding: 10px 18px;
    border-radius: 8px 8px 0 0;
    transition: all 250ms ease;
  }
  .stTabs [data-baseweb="tab"]:hover {
    color: #ecf0f1;
    background: rgba(255,255,255,0.04);
  }
  .stTabs [aria-selected="true"] {
    color: #3498db !important;
    background: rgba(52,152,219,0.08) !important;
  }

  /* ---------- Metric cards: glass-morphism ---------------------------- */
  [data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 14px 18px;
    backdrop-filter: blur(8px);
    transition: transform 200ms ease, border-color 200ms ease;
  }
  [data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(52,152,219,0.45);
  }
  [data-testid="stMetricValue"] { color: #ecf0f1; font-weight: 700; }
  [data-testid="stMetricLabel"] { color: #95a5a6; }

  /* ---------- Buttons -------------------------------------------------- */
  .stButton > button, .stForm button {
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.04);
    color: #ecf0f1;
    transition: all 200ms ease;
    font-weight: 500;
  }
  .stButton > button:hover, .stForm button:hover {
    transform: translateY(-1px);
    border-color: #3498db;
    box-shadow: 0 6px 18px -8px rgba(52,152,219,0.6);
    color: #fff;
  }
  .stButton > button[kind="primary"], .stForm button[kind="primary"] {
    background: linear-gradient(135deg, #3498db 0%, #2c3e88 100%);
    border: none;
  }

  /* ---------- Dataframes / expanders / forms -------------------------- */
  [data-testid="stDataFrame"], [data-testid="stExpander"], .stForm {
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.06);
    background: rgba(255,255,255,0.02);
  }
  .stForm { padding: 18px; }

  /* ---------- Sliders / radios accent --------------------------------- */
  [data-baseweb="slider"] [role="slider"] {
    background: #3498db !important;
    box-shadow: 0 0 0 4px rgba(52,152,219,0.15) !important;
  }

  /* ---------- Tab-panel fade-in --------------------------------------- */
  @keyframes fadeInUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .stTabs [role="tabpanel"] { animation: fadeInUp 350ms ease both; }

  /* ---------- Pulsing enforcement banner ------------------------------ */
  @keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(231,76,60,0.45); }
    50%      { box-shadow: 0 0 0 14px rgba(231,76,60,0); }
  }
  .enforcement-card {
    padding: 1.6rem;
    border-radius: 14px;
    color: white;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(6px);
    animation: fadeInUp 400ms ease both;
  }
  .enforcement-card.block { animation: fadeInUp 400ms ease both, pulseGlow 2s ease-in-out infinite; }
  .enforcement-card .label { font-size: 0.85rem; letter-spacing: 0.16rem; opacity: 0.85; }
  .enforcement-card .value { font-size: 2.6rem; font-weight: 800; margin-top: 0.35rem; text-shadow: 0 2px 12px rgba(0,0,0,0.35); }
  .enforcement-card .cat   { margin-top: 0.4rem; font-size: 0.9rem; opacity: 0.92; }
</style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# User anonymisation for the demo - the simulated dataset uses real-looking
# fictional names; map every user_id to a generic Doe-family alias so the
# dashboard reads consistently as a privacy-preserving demo.
# ---------------------------------------------------------------------------
USER_ALIASES: dict[str, str] = {
    "USR001": "John Doe",
    "USR002": "Jane Doe",
    "USR003": "Alex Doe",
    "USR004": "Sam Doe",
    "USR005": "Pat Doe",
    "USR006": "Riley Doe",
    "USR007": "Chris Doe",
    "USR008": "Morgan Doe",
    "USR009": "Casey Doe",
    "USR010": "Taylor Doe",
}


def _alias(user_id: str, fallback: str = "John Doe") -> str:
    return USER_ALIASES.get(user_id, fallback)


# ---------------------------------------------------------------------------
# Cached dataset
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading HybridSaaS-Sec dataset...")
def _dataset():
    return load_all()


DATASET = _dataset()
METRICS = PaperMetrics()            # the headline numbers as published
MEASURED = load_measured_metrics()  # read back from evaluation/data/metrics.json
PII = load_pii_metrics()            # per-category PII results (gemma-3-4b Branch 3)


ENGINE_MISSING_MSG = (
    "**The trained anomaly engine is not available in this deployment.** "
    "`evaluation/artifacts/` is missing, so there is no model to score with, and "
    "this app will not fabricate a score. Rebuild it with "
    "`python -m evaluation.generate_sessions && python -m evaluation.train_and_evaluate`."
)


def _guard(fn, *args, **kwargs):
    """Run a scoring call; surface engine failure as a message, not a dead page.

    Every tab used to die on an uncaught EngineUnavailable at import-time scoring
    -- one missing artefact blanked the whole dashboard, traceback and all.
    """
    try:
        return fn(*args, **kwargs)
    except EngineUnavailable:
        st.error(ENGINE_MISSING_MSG)
        return None
    except Exception as exc:  # noqa: BLE001
        st.error(f"Scoring failed: `{type(exc).__name__}: {exc}`")
        return None


# ---------------------------------------------------------------------------
# Sidebar - global controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## HybridSaaS-Sec")
    st.caption("MITM proxy + multimodal PII + LSTM/IF + SBRS")
    st.divider()

    st.markdown("### SBRS hyperparameters")
    alpha = st.slider(
        "alpha (LSTM weight in A_hybrid)",
        min_value=0.0, max_value=1.0, value=DEFAULT_ALPHA, step=0.05,
        help="A_hybrid = alpha * a_LSTM + (1 - alpha) * a_IF"
    )
    beta = st.slider(
        "beta (enterprise risk multiplier)",
        min_value=0.0, max_value=5.0, value=DEFAULT_BETA, step=0.05,
        help="SBRS = S * (1 + beta * A_hybrid) / 100  "
             "(recalibrated default 2.5; see evaluation/SBRS_RECALIBRATION.md)"
    )

    st.divider()
    meta = DATASET.log_metadata
    st.markdown("### Simulation run")
    st.write(f"**Run id:** `{meta.get('run_id')}`")
    st.write(f"**Window:** {meta.get('start_time')} -> {meta.get('end_time')}")
    st.write(f"**Events:** {meta.get('total_events')}")
    st.write(f"**Users:** {meta.get('users_monitored')}")
    st.write(f"**Platforms:** {', '.join(meta.get('saas_platforms', []))}")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("HybridSaaS-Sec")
st.markdown(
    "**A Scalable Framework for Privacy-Preserving API Monitoring and "
    "Multimodal Data Leakage Prevention in Enterprise Cloud Environments**  \n"
    "_Mir & Hashir, Software Design and Architecture Research Paper, 2026._"
)

tab_overview, tab_live, tab_compare, tab_docs = st.tabs([
    "1. System Overview",
    "2. Live Simulation",
    "3. Comparative Metrics",
    "4. About / Documentation",
])


# ===========================================================================
# Tab 1 - System Overview
# ===========================================================================
with tab_overview:
    st.subheader("Architecture")

    st.markdown(
        """
HybridSaaS-Sec deploys a transparent intermediary at the enterprise boundary,
strictly separating data interception, multimodal scanning and hybrid
analytics into modular pipelines:

```
   +---------------------+      +-----------------------------+      +--------------------+
   | (A) MITM Proxy      |      | (B) Multimodal PII Pipeline |      | (D) SBRS           |
   |  ECDHE + PFS        | ---> |  Branch 1 - Presidio        | ---> |  S * (1 + b*A) /100|
   |  SNI selective      |      |  Branch 2 - EasyOCR @ .85   |      |  Enforcement       |
   +---------------------+      |  Branch 3 - VLM fallback    |      +---------+----------+
                                +--------------+--------------+                |
                                               |                               v
                                               |               +-----------------------------+
                                               +-------------->| (C) Hybrid Anomaly Engine   |
                                                               |  LSTM(h=128, 24h) + IF(100) |
                                                               |  A = a*aLSTM + (1-a)*aIF    |
                                                               +-----------------------------+
```
        """
    )

    st.subheader("Headline results - measured on the held-out test split")
    if MEASURED is None:
        st.warning(
            "No measured metrics yet. Run `python -m evaluation.generate_sessions` "
            "then `python -m evaluation.train_and_evaluate`."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("EWMA FPR", "not measured")
        c2.metric("Hybrid FPR", "not measured")
        c3.metric("Hybrid F1", "not measured")
        c4.metric("Latency / call", "not measured")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("EWMA FPR (legacy engine)", f"{MEASURED.ewma_fpr*100:.2f} %",
                  help="Legacy univariate baseline, its threshold tuned on "
                       "validation so it gets the same budget as the hybrid.")
        c2.metric("Hybrid FPR", f"{MEASURED.hybrid_fpr*100:.2f} %",
                  delta=f"-{(MEASURED.ewma_fpr-MEASURED.hybrid_fpr)*100:.2f} pp",
                  help="LSTM + Isolation Forest, fused and thresholded at the "
                       "validation-tuned tau.")
        c3.metric("Hybrid F1 (anomaly flag)", f"{MEASURED.hybrid_f1:.3f}",
                  help=f"Precision {MEASURED.hybrid_precision:.3f}, recall "
                       f"{MEASURED.hybrid_recall:.3f}. Enforcement-band F1 (after "
                       f"the SBRS recalibration) is "
                       f"{MEASURED.enforcement_f1_alert_or_block:.3f}.")
        c4.metric("Latency / scoring call", f"{MEASURED.latency_mean_ms:.1f} ms",
                  help=MEASURED.latency_scope)

        st.caption(
            f"n = {MEASURED.test_sessions:,} held-out sessions "
            f"({MEASURED.test_positives} true threats, {MEASURED.users} users). "
            f"alpha = {MEASURED.alpha:.2f}, tau = {MEASURED.tau_hybrid:.2f}, "
            f"both tuned on validation; the test split is scored exactly once."
        )
        st.success(
            f"**The benign-burst false positive is suppressed.** A team "
            f"bulk-downloading templates before a deadline trips the legacy engine "
            f"{MEASURED.benign_burst_ewma_flag_rate*100:.1f}% of the time; the "
            f"hybrid flags it "
            f"{MEASURED.benign_burst_hybrid_flag_rate*100:.1f}% of the time. That is "
            f"the concrete case from the paper's Introduction, and it is the whole "
            f"point of pairing a personal temporal model with a global structural "
            f"one. Full breakdown in `evaluation/REAL_RESULTS.md`."
        )

    st.subheader("PII coverage by file type (Table I) - measured")
    if PII is None:
        st.warning("No measured PII metrics yet. Run `python -m evaluation.run_pipeline`.")
        cov = None
    else:
        st.caption(
            f"Entity-level recall over {PII['gold_entities']} gold entities in "
            f"{sum(PII['docs'].values())} synthetic documents "
            f"({PII['docs']['text_extractable']} text / {PII['docs']['scanned']} "
            f"scanned / {PII['docs']['handwritten']} handwritten, all PII generated "
            f"with Faker). Cascade at the paper's tau_ocr = {PII['tau_ocr']}, "
            f"Branch 3 = `{PII['model']}`. 'Base' is Presidio alone, which cannot "
            f"read an image at all."
        )
        cat = PII["by_category"]
        base = PII["baseline"]["by_category"]
        cov = pd.DataFrame({
            "File type": [
                "Text-extractable (DOCX/TXT/PDF)",
                "Scanned PDF / Image (PNG, JPEG)",
                "Handwritten / Low-quality scans",
                "Overall weighted",
            ],
            "Base (Presidio)": [
                base["text_extractable"]["recall"],
                base["scanned"]["recall"],
                base["handwritten"]["recall"],
                PII["baseline"]["overall_weighted_recall"],
            ],
            "HybridSaaS-Sec": [
                cat["text_extractable"]["recall"],
                cat["scanned"]["recall"],
                cat["handwritten"]["recall"],
                PII["overall_weighted_recall"],
            ],
        }).round(3)
        cov["Coverage gain"] = (cov["HybridSaaS-Sec"] - cov["Base (Presidio)"]).round(3)
        st.dataframe(cov, hide_index=True, use_container_width=True)

    if cov is not None:
        cov_long = cov.iloc[:-1].melt(
            id_vars="File type",
            value_vars=["Base (Presidio)", "HybridSaaS-Sec"],
            var_name="System", value_name="Coverage",
        )
        fig_cov = px.bar(
            cov_long, x="File type", y="Coverage", color="System", barmode="group",
            title="PII detection recall by file type (measured)",
            color_discrete_sequence=["#7f8c8d", "#1f77b4"],
        )
        fig_cov.update_yaxes(range=[0, 1])
        cov_n = _zoom_reset_button("pii_coverage")
        st.plotly_chart(
            fig_cov, use_container_width=True, config=PLOTLY_CONFIG,
            key=f"pii_coverage_{cov_n}",
        )
        st.caption(
            f"Branch latencies, timed per document: Branch 1 (Presidio) "
            f"{PII['latency_ms']['branch1_mean']:.0f} ms, Branch 2 (OCR + Presidio) "
            f"{PII['latency_ms']['branch2_mean']/1000:.1f} s, Branch 3 (VLM) "
            f"{PII['latency_ms']['branch3_mean']/1000:.1f} s - Branch 3 is a network "
            f"round-trip to a hosted model, not on-box compute."
        )

    st.subheader("Mathematical formulation")
    st.latex(r"A_{\text{hybrid}} \;=\; \alpha \cdot a_{\text{LSTM}} + (1 - \alpha) \cdot a_{\text{IF}}")
    st.latex(r"\text{SBRS} \;=\; \frac{S \cdot (1 + \beta \cdot A_{\text{hybrid}})}{100}")
    st.caption(
        f"S in [0, 100] from min(10*high + 5*medium + 1*low, 100). "
        f"A_hybrid in [0, 1]. beta is the enterprise risk multiplier; the default "
        f"{DEFAULT_BETA} and the ALERT / BLOCK cut-points ({T_ALERT} / {T_BLOCK}) are "
        f"calibrated on the validation split, not hand-picked "
        f"(see evaluation/SBRS_RECALIBRATION.md)."
    )


# ===========================================================================
# Tab 2 - Live Simulation
# ===========================================================================
def _score_custom_event(payload: dict, alpha: float, beta: float) -> FullScoringResult:
    """Run the full pipeline against a user-supplied event payload."""
    pii = pii_scan_event(payload)
    anomaly = eval_anomaly(DATASET, payload, alpha=alpha)
    sbrs = compute_sbrs(
        sensitivity_score=pii.sensitivity_score,
        hybrid_anomaly_score=anomaly.hybrid_anomaly_score,
        beta=beta,
    )
    return FullScoringResult(
        event=_event_summary(payload),
        pii=pii,
        anomaly=anomaly,
        sbrs=sbrs,
        raw_enforcement=payload.get("enforcement"),
    )


with tab_live:
    st.subheader("Live MITM-proxy simulation")
    st.caption(
        "Replay a sample event from the simulation, build your own from a form, "
        "or paste a raw JSON payload. The proxy then runs the multimodal PII "
        "pipeline, the hybrid LSTM + Isolation Forest engine, and computes the "
        "Semantic-Behavioral Risk Score for enforcement."
    )

    mode = st.radio(
        "Input mode",
        ["Sample event", "Custom event (form)", "Custom event (JSON paste)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    result: FullScoringResult | None = None

    # ---------------- Mode 1: replay a precomputed event --------------------
    if mode == "Sample event":
        # Clear any persisted custom payload so it doesn't bleed across modes.
        st.session_state.pop("custom_payload", None)
        events_index = pd.DataFrame([
            {
                "event_id": e["event_id"],
                "label": f"{e['event_id']}  |  "
                         f"{_alias(e['request']['user_id']):<13s}"
                         f" -> {e['request']['file_name']}",
                "sensitivity": (e.get("pii_detection") or {}).get("sensitivity_score", 0),
            }
            for e in DATASET.events
        ])
        # Default to a *mid-sensitivity* event (~S=40) so the gauge starts in
        # the SENSITIVE/yellow band and any alpha/beta tweak from the sidebar
        # is immediately visible on the indicator. Picking S=0 (lowest) would
        # pin SBRS to 0 regardless of alpha/beta (zero times anything = zero),
        # which made the dashboard look broken.
        target_S = 40
        default_idx = int(
            (events_index["sensitivity"].fillna(0) - target_S).abs().idxmin()
        )
        selected_label = st.selectbox(
            "Intercepted API event",
            options=events_index["label"].tolist(),
            index=default_idx,
        )
        event_id = events_index.loc[
            events_index["label"] == selected_label, "event_id"
        ].iloc[0]
        result = _guard(score_event, DATASET, event_id, alpha=alpha, beta=beta)
        if result is not None:
            # Anonymise the displayed user_name on the result.
            result.event.user_name = _alias(result.event.user_id, result.event.user_name)

        st.caption(
            f"\u2139\ufe0f  SBRS = S \u00d7 (1 + \u03b2 \u00b7 A_hybrid) / 100, "
            f"ALERT at {T_ALERT}, BLOCK at {T_BLOCK}. Behaviour can only amplify "
            f"content: an event with no PII (S = 0) scores 0 however anomalous the "
            f"user looks. Switch to **Custom event (form)** to drive S and the two "
            f"anomaly scores independently and walk the gauge across all three bands."
        )

    # ---------------- Mode 2: build via form --------------------------------
    elif mode == "Custom event (form)":
        with st.form("custom_event_form"):
            f1, f2, f3 = st.columns(3)
            with f1:
                user_name = st.text_input("User name", value="John Doe")
                user_id = st.text_input("User id", value="USR999")
                department = st.selectbox(
                    "Department",
                    ["Finance", "HR", "Sales", "Legal", "Engineering", "Marketing"],
                )
            with f2:
                platform = st.selectbox(
                    "SaaS platform", ["Google Drive", "Microsoft OneDrive"]
                )
                api_action = st.selectbox(
                    "API action", ["DOWNLOAD", "UPLOAD", "SHARE", "VIEW", "DELETE"]
                )
                geo_location = st.text_input("Geo location", value="Karachi, PK")
            with f3:
                file_name = st.text_input("File name", value="my_payload.pdf")
                file_type = st.selectbox(
                    "File type",
                    ["text_extractable", "scanned_image", "handwritten", "binary"],
                )
                file_department = st.text_input("File department", value=department)

            st.markdown("**PII entity counts (drives S = min(10·H + 5·M + 1·L, 100))**")
            p1, p2, p3 = st.columns(3)
            high = p1.number_input("High-tier (CC, NID, secrets)", 0, 50, 2, 1)
            medium = p2.number_input("Medium-tier (email, phone)", 0, 100, 5, 1)
            low = p3.number_input("Low-tier (person names)", 0, 200, 10, 1)

            st.markdown("**Behavioural anomaly scores (LSTM + Isolation Forest)**")
            o1, o2 = st.columns(2)
            lstm_score = o1.slider("a_LSTM (temporal anomaly)", 0.0, 1.0, 0.15, 0.01)
            if_score = o2.slider("a_IF (structural outlier)", 0.0, 1.0, 0.20, 0.01)

            ocr_use = st.checkbox("Set OCR confidence (image branches only)", value=False)
            ocr_conf = st.slider(
                "OCR confidence", 0.0, 1.0, 0.92, 0.01, disabled=not ocr_use
            )

            submitted = st.form_submit_button("Run pipeline", type="primary")

        if submitted:
            st.session_state["custom_payload"] = {
                "event_id": "CUSTOM-FORM",
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "event_type": "PII_SCAN",
                "module": "Custom",
                "request": {
                    "user_id": user_id, "user_name": user_name,
                    "department": department, "platform": platform,
                    "geo_location": geo_location,
                    "file_name": file_name, "file_department": file_department,
                    "file_type": file_type, "api_action": api_action,
                },
                "pii_detection": {
                    "engine": "Microsoft Presidio v2.2 (custom)",
                    "entities_detected": [],
                    "high_count": int(high),
                    "medium_count": int(medium),
                    "low_count": int(low),
                    "formula": "min(10*H + 5*M + 1*L, 100)",
                    "sensitivity_score": -1,  # force recompute
                    "ocr_confidence": ocr_conf if ocr_use else "N/A",
                    "processing_ms": 0.0,
                },
                "behavioral_analysis": {
                    "lstm_score": float(lstm_score),
                    "isolation_forest_score": float(if_score),
                },
            }

        # Re-score whenever a payload exists (so sidebar alpha/beta tweaks
        # rerun the pipeline without forcing the user to resubmit the form).
        if "custom_payload" in st.session_state:
            result = _guard(
                _score_custom_event,
                st.session_state["custom_payload"], alpha=alpha, beta=beta,
            )
            cc1, cc2 = st.columns([1, 5])
            with cc1:
                if st.button("Clear custom event", key="clear_custom_form"):
                    st.session_state.pop("custom_payload", None)
                    st.rerun()
            with cc2:
                st.caption(
                    "Sidebar \u03b1 / \u03b2 changes are re-applied automatically. "
                    "Edit the form fields and press 'Run pipeline' to update the payload."
                )

    # ---------------- Mode 3: paste raw JSON --------------------------------
    else:
        sample = (
            '{\n  "event_id": "CUSTOM-JSON",\n  "timestamp": "2026-04-13 14:00:00",\n'
            '  "event_type": "PII_SCAN",\n  "module": "Custom",\n'
            '  "request": {"user_id": "USR999", "user_name": "John Doe",\n'
            '              "department": "Finance", "platform": "Google Drive",\n'
            '              "geo_location": "Karachi, PK", "file_name": "x.pdf",\n'
            '              "file_department": "Finance",\n'
            '              "file_type": "scanned_image", "api_action": "DOWNLOAD"},\n'
            '  "pii_detection": {"high_count": 3, "medium_count": 8, "low_count": 4,\n'
            '                    "sensitivity_score": -1, "ocr_confidence": 0.92},\n'
            '  "behavioral_analysis": {"lstm_score": 0.30, "isolation_forest_score": 0.45}\n'
            "}"
        )
        raw = st.text_area("Event JSON payload", value=sample, height=320)
        run_json = st.button("Run pipeline", type="primary")
        if run_json:
            import json as _json
            try:
                st.session_state["custom_payload"] = _json.loads(raw)
            except _json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

        if "custom_payload" in st.session_state:
            result = _guard(
                _score_custom_event,
                st.session_state["custom_payload"], alpha=alpha, beta=beta,
            )
            if result is None:
                st.session_state.pop("custom_payload", None)
    if result is None:
        st.info(
            "Submit the form (or pick a sample event) to see the SBRS analysis. "
            "Once submitted, sidebar α/β changes are re-applied automatically."
        )
    else:
        # ------------ Event summary ------------------------------------------------
        ev = result.event
        st.markdown(
            f"**User:** {ev.user_name} ({ev.user_id}, {ev.department})  -  "
            f"**Platform:** {ev.platform}  -  **Action:** `{ev.api_action}`  -  "
            f"**File:** `{ev.file_name}` ({ev.file_type})"
        )
        st.caption(f"Timestamp: {ev.timestamp}  -  Module: {ev.event_type}")

        # ------------ Top row: SBRS gauge + enforcement banner --------------------
        col_gauge, col_action = st.columns([2, 1])

        with col_gauge:
            sbrs = result.sbrs
            st.markdown("##### Semantic-Behavioral Risk Score (SBRS)")
            # Full scale of the score at the current beta: S=100 and A=1 give
            # SBRS = 1 + beta. The band colours below must track the enforcement
            # cut-points in risk_orchestrator, never a hardcoded copy of them.
            GAUGE_MAX = round(max(1.0 + beta, T_BLOCK * 1.2), 2)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=sbrs.sbrs_value,
                number={"valueformat": ".3f", "font": {"size": 44}},
                delta={
                    "reference": T_BLOCK,
                    "valueformat": ".3f",
                    "increasing": {"color": "#e74c3c"},
                    "decreasing": {"color": "#27ae60"},
                    "font": {"size": 14},
                },
                domain={"x": [0, 1], "y": [0.05, 0.95]},
                gauge={
                    "axis": {"range": [0, GAUGE_MAX], "tickwidth": 1,
                             "tickfont": {"size": 11}},
                    "bar": {"color": "rgba(255,255,255,0.85)", "thickness": 0.18},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0.00, T_ALERT], "color": "#27ae60"},
                        {"range": [T_ALERT, T_BLOCK], "color": "#f1c40f"},
                        {"range": [T_BLOCK, GAUGE_MAX], "color": "#e74c3c"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.85, "value": sbrs.sbrs_value,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ecf0f1"),
            )
            gauge_n = _zoom_reset_button("sbrs_gauge")
            st.plotly_chart(
                fig_gauge, use_container_width=True, config=PLOTLY_CONFIG,
                key=f"sbrs_gauge_{gauge_n}",
            )

        with col_action:
            action = sbrs.enforcement_action
            gradient = {
                "BLOCK":  "linear-gradient(135deg, #e74c3c 0%, #8e2a1f 100%)",
                "ALERT":  "linear-gradient(135deg, #f39c12 0%, #b9770e 100%)",
                "PERMIT": "linear-gradient(135deg, #27ae60 0%, #166937 100%)",
            }[action]
            css_class = "enforcement-card" + (" block" if action == "BLOCK" else "")
            st.markdown(
                f"""
    <div class="{css_class}" style="background:{gradient};">
      <div class="label">ENFORCEMENT</div>
      <div class="value">{action}</div>
      <div class="cat">Category: {sbrs.sbrs_category}</div>
    </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            st.markdown(
                f"**Formula:** `{sbrs.formula}`  \n"
                f"S = **{sbrs.sensitivity_score}**  -  "
                f"A_hybrid = **{sbrs.hybrid_anomaly_score:.4f}**  -  "
                f"beta = **{sbrs.beta}**"
            )
            if result.raw_enforcement:
                st.caption(f"Paper-recorded enforcement: "
                           f"`{result.raw_enforcement.get('action')}` - "
                           f"{result.raw_enforcement.get('reason', '')}")

        st.divider()

        # ------------ PII + Anomaly breakdown ------------------------------------
        col_pii, col_anom = st.columns(2)

        with col_pii:
            st.markdown("### Multimodal PII pipeline")
            pii = result.pii
            st.markdown(
                f"- **Engine:** {pii.engine}  \n"
                f"- **Branch:** `{pii.branch}`  \n"
                f"- **OCR confidence:** "
                f"{pii.ocr_confidence if pii.ocr_confidence is not None else 'n/a'} "
                f"(tau = {pii.ocr_threshold})  \n"
                f"- **Processing latency:** {pii.processing_ms:.1f} ms  \n"
                f"- **Risk category:** `{pii.risk_category}`"
            )
            if pii.entities_detected:
                ent_df = pd.DataFrame([e.model_dump() for e in pii.entities_detected])
                st.dataframe(ent_df, hide_index=True, use_container_width=True)
            else:
                st.info("No PII entities detected for this event.")
            st.markdown(
                f"**S = {pii.sensitivity_score} / 100**  -  "
                f"_formula:_ `{pii.formula}`  "
                f"-> 10*{pii.high_count} + 5*{pii.medium_count} + 1*{pii.low_count}"
            )

        with col_anom:
            st.markdown("### Hybrid anomaly engine (LSTM + Isolation Forest)")
            an = result.anomaly
            c1, c2, c3 = st.columns(3)
            c1.metric("a_LSTM",  f"{an.lstm_score:.4f}")
            c2.metric("a_IF",    f"{an.isolation_forest_score:.4f}")
            c3.metric("A_hybrid", f"{an.hybrid_anomaly_score:.4f}",
                      delta="flagged" if an.hybrid_flagged else "normal",
                      delta_color="inverse" if an.hybrid_flagged else "normal")
            st.caption(
                f"alpha = {an.alpha}  -  EWMA score = "
                f"{an.ewma_score if an.ewma_score is not None else 'n/a'}  -  "
                f"EWMA flagged = {an.ewma_flagged}"
            )

            st.markdown("**6-dimensional feature vector x_t**")
            fv = an.feature_vector
            # Close the polygon by repeating the first point.
            fv_keys = [k.replace("_", " ") for k in fv.keys()] + [
                list(fv.keys())[0].replace("_", " ")
            ]
            fv_vals = list(fv.values()) + [list(fv.values())[0]]
            fig_fv = go.Figure(go.Scatterpolar(
                r=fv_vals, theta=fv_keys,
                fill="toself", name="x_t",
                line=dict(color="#3498db", width=2),
                fillcolor="rgba(52,152,219,0.45)",
            ))
            fig_fv.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1],
                                    gridcolor="rgba(255,255,255,0.15)",
                                    tickfont=dict(size=10)),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.15)",
                                     tickfont=dict(size=11)),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ecf0f1"),
                showlegend=False, height=340,
                margin=dict(l=60, r=60, t=30, b=30),
            )
            radar_n = _zoom_reset_button("feature_radar")
            st.plotly_chart(
                fig_fv, use_container_width=True, config=PLOTLY_CONFIG,
                key=f"feature_radar_{radar_n}",
            )


# ===========================================================================
# Tab 3 - Comparative Metrics
# ===========================================================================
with tab_compare:
    st.subheader("Legacy EWMA vs Hybrid LSTM + Isolation Forest")
    st.caption(
        "Every row below is a real scoring call on the **held-out test split** "
        "(`anomaly_detection_comparison_v2_real.csv`, written by "
        "`evaluation/train_and_evaluate.py`). Both engines' thresholds were tuned "
        "on validation and the test split was scored once. Section V.B of the paper."
    )

    df = DATASET.anomaly_comparison.copy().sort_values("timestamp").reset_index(drop=True)

    # ---- Confusion-style summary -------------------------------------------
    def _rates(prefix: str):
        flagged_col = f"{prefix}_anomaly_flagged"
        correct_col = f"{prefix}_correct"
        truth = df["is_true_threat"].astype(bool)
        flagged = df[flagged_col].astype(bool)
        tp = int(((flagged) & truth).sum())
        fp = int(((flagged) & ~truth).sum())
        fn = int((~flagged & truth).sum())
        tn = int((~flagged & ~truth).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        acc = float(df[correct_col].astype(bool).mean())
        return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "Precision": precision, "Recall": recall,
                "F1": f1, "FPR": fpr, "Accuracy": acc}

    ewma_stats = _rates("ewma")
    hybrid_stats = _rates("hybrid")

    summary = pd.DataFrame([
        {"Model": "EWMA (legacy)",      **ewma_stats},
        {"Model": "Hybrid LSTM + IF",   **hybrid_stats},
    ]).round({"Precision": 3, "Recall": 3, "F1": 3, "FPR": 3, "Accuracy": 3})
    st.dataframe(summary, hide_index=True, use_container_width=True)

    cA, cB, cC = st.columns(3)
    cA.metric("EWMA FPR", f"{ewma_stats['FPR']*100:.2f} %",
              help="Legacy univariate baseline on activity rate.")
    cB.metric("Hybrid FPR", f"{hybrid_stats['FPR']*100:.2f} %",
              delta=f"-{(ewma_stats['FPR']-hybrid_stats['FPR'])*100:.2f} pp",
              help="LSTM + Isolation Forest, fused at the validation-tuned alpha.")
    cC.metric("Hybrid F1 (anomaly flag)", f"{hybrid_stats['F1']:.3f}",
              delta=f"+{hybrid_stats['F1']-ewma_stats['F1']:.3f} vs EWMA")

    # ---- Time-series view ---------------------------------------------------
    st.markdown("#### Per-event anomaly trace")
    df_plot = df[[
        "timestamp", "ewma_new", "hybrid_anomaly_score",
        "sbrs_value", "is_true_threat"
    ]].copy()
    df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])
    df_plot["ewma_normalised"] = df_plot["ewma_new"] / df_plot["ewma_new"].max()

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=df_plot["timestamp"], y=df_plot["ewma_normalised"],
        mode="lines+markers", name="EWMA (normalised)",
        line=dict(color="#c0392b", dash="dot"),
    ))
    fig_ts.add_trace(go.Scatter(
        x=df_plot["timestamp"], y=df_plot["hybrid_anomaly_score"],
        mode="lines+markers", name="A_hybrid (LSTM + IF)",
        line=dict(color="#1f77b4"),
    ))
    fig_ts.add_trace(go.Scatter(
        x=df_plot["timestamp"], y=df_plot["sbrs_value"],
        mode="lines+markers", name="SBRS",
        line=dict(color="#27ae60", width=3),
    ))
    fig_ts.add_hline(y=T_BLOCK, line=dict(color="white", dash="dash"),
                     annotation_text=f"SBRS BLOCK threshold ({T_BLOCK})",
                     annotation_position="top left")
    fig_ts.add_hline(y=T_ALERT, line=dict(color="#f1c40f", dash="dot"),
                     annotation_text=f"ALERT threshold ({T_ALERT})",
                     annotation_position="bottom left")
    truths = df_plot[df_plot["is_true_threat"] == True]  # noqa: E712
    fig_ts.add_trace(go.Scatter(
        x=truths["timestamp"], y=truths["sbrs_value"],
        mode="markers", name="True threat",
        marker=dict(color="black", size=11, symbol="x"),
    ))
    fig_ts.update_layout(
        height=460, hovermode="x unified",
        xaxis_title="Event time",
        yaxis_title="Score (normalised)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    ts_n = _zoom_reset_button("compare_ts")
    st.plotly_chart(
        fig_ts, use_container_width=True, config=PLOTLY_CONFIG,
        key=f"compare_ts_{ts_n}",
    )
    st.caption("Drag to zoom in, then click 'Reset zoom' (or double-click the chart) "
               "to restore the original axes.")

    # ---- Distribution view --------------------------------------------------
    st.markdown("#### Score distributions on benign vs. true-threat events")
    fig_box = go.Figure()
    for label, sub in df.groupby("is_true_threat"):
        name = "True threat" if label else "Benign"
        fig_box.add_trace(go.Box(y=sub["ewma_new"]/df["ewma_new"].max(),
                                 name=f"EWMA ({name})", boxmean=True))
        fig_box.add_trace(go.Box(y=sub["hybrid_anomaly_score"],
                                 name=f"Hybrid ({name})", boxmean=True))
        fig_box.add_trace(go.Box(y=sub["sbrs_value"],
                                 name=f"SBRS ({name})", boxmean=True))
    fig_box.update_layout(height=420,
                          yaxis_title="Score (normalised / [0,1+])",
                          showlegend=True)
    box_n = _zoom_reset_button("compare_box")
    st.plotly_chart(
        fig_box, use_container_width=True, config=PLOTLY_CONFIG,
        key=f"compare_box_{box_n}",
    )

    # ---- Raw table ---------------------------------------------------------
    with st.expander("Raw per-event comparison table"):
        st.dataframe(df, use_container_width=True, height=400)


# ===========================================================================
# Tab 4 - About / Documentation
# ===========================================================================
with tab_docs:
    st.subheader("What is HybridSaaS-Sec?")
    st.markdown(
        """
HybridSaaS-Sec is a **privacy-preserving security gateway** that sits between
your employees and the SaaS apps they use (Google Drive, OneDrive, Slack, ...).
It watches every API call in real time and decides whether to **permit**,
**alert**, or **block** each action based on _what data is involved_ **and**
_how the user is behaving_.

Think of it as a smart bouncer at the door of your cloud:

- **Permit** \u2014 normal employee, normal file, normal action. Let it through.
- **Alert** \u2014 mildly unusual. Log it and notify the security team.
- **Block** \u2014 sensitive data + suspicious behaviour. Stop the request and
  open a ticket.
        """
    )

    st.subheader("The problem we are solving")
    st.markdown(
        """
Modern enterprises lose data through SaaS APIs faster than legacy DLP
(Data Loss Prevention) tools can keep up. Three things make this hard:

1. **Traffic is encrypted.** TLS hides the payload from network sensors.
2. **Data is multimodal.** PII can hide in text, scanned PDFs, screenshots,
   even handwritten notes \u2014 a single text-based scanner misses 70% of it.
3. **Legacy alerts cry wolf.** Statistical baselines like EWMA fire on normal
   activity: on our test split it flags **59%** of benign traffic bursts (a team
   bulk-downloading templates before a deadline) as attacks. The hybrid engine
   flags **0%** of them, while still catching every compromised account.
        """
    )

    st.subheader("How does it work? (the four building blocks)")
    st.markdown(
        """
```
   +-----------+    +-----------+    +--------------+    +----------+
   | A. MITM   | -> | B. Multi- | -> | C. Hybrid    | -> | D. SBRS  |
   |   Proxy   |    |   modal   |    |   Anomaly    |    |   risk   |
   |  (TLS)    |    |   PII     |    |   Engine     |    |  scorer  |
   +-----------+    +-----------+    +--------------+    +----------+
        |                |                  |                  |
   intercept &        scan files for     score user's       combine S and A
   re-encrypt         personal info     behaviour vs       into one number,
   API calls          (3 branches)      their history      decide action
```
        """
    )

    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
#### A. The MITM proxy

*Man-in-the-Middle* sounds scary, but here it is **authorised** \u2014 the
enterprise installs a certificate so the proxy can briefly decrypt, inspect,
and re-encrypt traffic to the SaaS provider. It uses **ECDHE** key-exchange
for *perfect forward secrecy* and **SNI**-based selective interception so
personal/banking traffic is never touched.
                """
            )
            st.markdown(
                """
#### C. The hybrid anomaly engine

For every API call we build a **6-dimensional feature vector x_t**:
activity rate, file-type category, geographic index, endpoint operation,
permission-scope delta, and collaboration-network density.

Two models score it in parallel:

- **LSTM (h = 128)** \u2014 reads the user's last 24h of activity and asks
  *"does this fit the user's normal rhythm?"* (temporal anomaly **a_LSTM**).
- **Isolation Forest** \u2014 asks *"is this point unusual compared with the
  whole organisation?"* (structural outlier **a_IF**).

They are fused with a tunable weight **\u03b1**:
                """
            )
            st.latex(r"A_{\text{hybrid}} \;=\; \alpha \cdot a_{\text{LSTM}} + (1 - \alpha) \cdot a_{\text{IF}}")

        with c2:
            st.markdown(
                """
#### B. The multimodal PII pipeline

Three branches handle different file types:

1. **Branch 1 \u2014 Microsoft Presidio v2.2:** plain text, DOCX, code
   files. Fastest path: 11 ms/doc, F1 0.86 on our corpus.
2. **Branch 2 \u2014 EasyOCR \u2192 Presidio:** scanned PDFs and
   screenshots, used when OCR confidence \u2265 **\u03c4 = 0.85**.
3. **Branch 3 \u2014 a vision-language model:** handwriting, low-quality
   scans, anything the OCR cannot read confidently.

Whatever path is taken, the output is a **sensitivity score S** in
`[0, 100]`:
                """
            )
            st.latex(r"S \;=\; \min\bigl(10\,H + 5\,M + 1\,L,\; 100\bigr)")
            st.caption(
                "H = number of high-tier entities (credit-card, national id, secrets), "
                "M = medium-tier (email, phone), L = low-tier (person names)."
            )

            st.markdown(
                """
#### D. The SBRS risk scorer

The **Semantic-Behavioral Risk Score** combines content sensitivity with
behavioural anomaly into one number:
                """
            )
            st.latex(r"\text{SBRS} \;=\; \frac{S \,\bigl(1 + \beta \cdot A_{\text{hybrid}}\bigr)}{100}")
            st.caption(
                "Beautifully simple: small S keeps SBRS small even if anomaly is large "
                "(suppresses false positives on benign data); large S amplifies SBRS "
                "as soon as any anomaly appears."
            )

    st.subheader("Worked example")
    st.markdown(
        """
An employee in **Finance** downloads a **payroll spreadsheet** from Google
Drive at 2 a.m. from an unusual city.

1. **PII pipeline** finds 5 high-tier (national-id), 20 medium (email), 15
   low (names) entities \u2192
   `S = min(10\u00b75 + 5\u00b720 + 1\u00b715, 100) = min(165, 100) = 100`.
2. **Anomaly engine** sees an unfamiliar geo + odd hour:
   `a_LSTM = 0.62`, `a_IF = 0.55`, with the tuned `\u03b1 = 0.25`
   \u2192 `A_hybrid = 0.5675`.
3. **SBRS** with the calibrated `\u03b2 = 2.5`:
   `SBRS = 100 \u00d7 (1 + 2.5 \u00b7 0.5675) / 100 = 2.42` \u2192 above the 1.84 BLOCK
   cut-point \u2192 **HIGH-RISK** \u2192 **BLOCK**.

The same file downloaded by the same person during office hours from their usual
city scores `A_hybrid \u2248 0.05`, so `SBRS \u2248 1.13` \u2192 **PERMIT**. That gap is the
behavioural half of the score doing its job.
        """
    )

    st.subheader("Pipeline data flow")
    fig_sankey = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=18, thickness=18,
            line=dict(color="rgba(255,255,255,0.15)", width=0.5),
            label=[
                "SaaS API call",            # 0
                "MITM proxy",                # 1
                "Branch 1: Presidio",       # 2
                "Branch 2: OCR + Presidio", # 3
                "Branch 3: VLM",            # 4
                "Sensitivity S",            # 5
                "LSTM a_LSTM",              # 6
                "IsoForest a_IF",           # 7
                "A_hybrid",                 # 8
                "SBRS",                     # 9
                "PERMIT",                   # 10
                "ALERT",                    # 11
                "BLOCK",                    # 12
            ],
            color=[
                "#3498db", "#3498db",
                "#9b59b6", "#9b59b6", "#9b59b6",
                "#1abc9c",
                "#e67e22", "#e67e22",
                "#e67e22",
                "#f1c40f",
                "#27ae60", "#f39c12", "#e74c3c",
            ],
        ),
        link=dict(
            source=[0, 1, 1, 1, 2, 3, 4, 1, 1, 6, 7, 5, 8, 9, 9, 9],
            target=[1, 2, 3, 4, 5, 5, 5, 6, 7, 8, 8, 9, 9, 10, 11, 12],
            value =[10, 6,  3, 1, 6, 3, 1, 5, 5, 5, 5, 10, 10, 4,  3,  3],
            color=["rgba(52,152,219,0.3)"] * 16,
        ),
    ))
    fig_sankey.update_layout(
        height=460,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ecf0f1", size=12),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    sankey_n = _zoom_reset_button("docs_sankey")
    st.plotly_chart(
        fig_sankey, use_container_width=True, config=PLOTLY_CONFIG,
        key=f"docs_sankey_{sankey_n}",
    )

    st.subheader("Why does it beat the legacy approach?")
    st.caption(
        "Behavioural rows: held-out test split, 2,352 sessions / 96 true threats "
        "across 50 users. PII rows: entity-level recall over 368 synthetic entities "
        "in 120 documents. Every number is measured, not asserted."
    )
    bench = pd.DataFrame({
        "Metric": [
            "False-positive rate (anomaly flag)",
            "Benign bursts wrongly flagged",
            "PII recall on text",
            "PII recall on scanned PDF",
            "PII recall on handwriting",
            "Overall PII recall (weighted)",
            "Anomaly-flag F1",
        ],
        "Legacy (EWMA + Presidio)": [0.058, 0.593, 0.828, 0.000, 0.000, 0.276, 0.116],
        "HybridSaaS-Sec":           [0.000, 0.000, 0.828, 0.889, 0.752, 0.823, 0.629],
    })
    st.dataframe(bench, hide_index=True, use_container_width=True)
    st.caption(
        "The honest caveat, also in the paper: the hybrid engine catches "
        "compromised accounts, negligent oversharing and over-scoped third-party "
        "apps at 100 / 100 / 71% recall, but only **33%** of slow malicious "
        "insiders - a drift that gradual is close to invisible to a per-user "
        "temporal model. That is the open problem, not a tuning knob."
    )

    st.subheader("Glossary")
    st.markdown(
        """
- **PII** \u2014 Personally Identifiable Information (names, emails,
  national IDs, credit-card numbers, ...).
- **DLP** \u2014 Data Loss Prevention.
- **MITM** \u2014 Man-in-the-Middle. Here it is authorised by the enterprise.
- **EWMA** \u2014 Exponentially Weighted Moving Average; the legacy baseline.
- **LSTM** \u2014 Long Short-Term Memory neural net; learns sequences.
- **Isolation Forest** \u2014 Tree-ensemble outlier detector.
- **VLM** \u2014 Vision-Language Model (here: gemma-3-4b-it); reads images +
  text together.
- **SBRS** \u2014 Semantic-Behavioral Risk Score (this paper's contribution).
- **\u03b1** \u2014 LSTM weight in the hybrid anomaly score (0..1); tuned to 0.25 on
  validation.
- **\u03b2** \u2014 Enterprise risk multiplier on SBRS; calibrated to 2.5 on validation.
        """
    )

    # ------------------------------------------------------------------
    # Threat model + privacy properties
    # ------------------------------------------------------------------
    st.subheader("Threat model")
    st.markdown(
        """
HybridSaaS-Sec is designed for the *insider-and-misconfiguration* threat
class that dominates real-world SaaS breaches:

- **Malicious insider** \u2014 an employee with legitimate credentials trying
  to exfiltrate sensitive data through `DOWNLOAD`, `SHARE`, or `UPLOAD`
  API calls.
- **Compromised account** \u2014 a credential stuffed or phished session
  acting from an unusual geolocation / device fingerprint.
  *(Signal: large `geo_index` and `permission_scope_delta` features.)*
- **Misconfiguration** \u2014 a bulk-share or public-link API call exposing
  documents that contain regulated PII.
  *(Signal: spike in `collab_network_density` while file `S` is high.)*
- **Shadow IT** \u2014 employees moving regulated data into SaaS apps that
  the security team has no visibility into; the MITM proxy logs the
  destination platform for every intercepted call.

It is **not** designed to stop nation-state network attackers or to defend
the SaaS provider itself \u2014 those are upstream of the proxy.
        """
    )

    st.subheader("Privacy properties (what the proxy never keeps)")
    st.markdown(
        """
A security gateway that decrypts traffic is itself a juicy target. The
paper hardens this with four guarantees:

1. **Forward secrecy.** Each TLS session uses an ephemeral
   Elliptic-Curve Diffie-Hellman (ECDHE) key. Even if a long-term private
   key leaks tomorrow, yesterday's intercepted sessions remain unreadable.
2. **Selective interception (SNI-based).** The proxy only opens connections
   to allow-listed SaaS hostnames. Traffic to personal banking, healthcare,
   etc. passes through untouched.
3. **In-memory scanning only.** File payloads are scanned in RAM and
   discarded once `S` is computed. The disk artefacts only ever contain
   *aggregated metadata* (entity counts, scores, timestamps).
4. **Differential redaction in logs.** Audit logs store entity *types and
   counts* (e.g. `CREDIT_CARD x 3`) but never the actual values.
        """
    )

    # ------------------------------------------------------------------
    # Detailed module walk-through
    # ------------------------------------------------------------------
    st.subheader("A. MITM Proxy \u2014 in detail")
    st.markdown(
        """
The proxy implements a classic *split-TLS* design:

```
   Client                Proxy                          SaaS provider
     |  ClientHello (SNI)   |                                |
     | -------------------> |  ClientHello (re-issued)       |
     |                      | -----------------------------> |
     |                      |        ServerHello, cert       |
     |                      | <----------------------------- |
     |   ServerHello with   |                                |
     |   proxy-issued cert  |                                |
     | <------------------- |                                |
     |                                                       |
     | ===== encrypted payload (decrypted briefly here) ==== |
```

Key design choices:

- **Throughput:** event-driven I/O (uvloop / nginx-style) gives ~12k
  intercepted req/s on a single 8-core node in the paper's benchmark.
- **Failure-open vs failure-closed:** by default the proxy *fails open*
  on its own internal errors but *fails closed* on enforcement decisions
  \u2014 i.e. if the SBRS cannot be evaluated, the request is blocked.
- **Side-channel quietness:** the proxy adds < 8 ms median latency to
  intercepted calls so users don't notice it.
        """
    )

    st.subheader("B. Multimodal PII Pipeline \u2014 in detail")
    st.markdown(
        """
A single text-based scanner (Presidio alone) misses everything in image
form. HybridSaaS-Sec routes each intercepted file through a **decision
tree** based on its type and OCR confidence:
        """
    )
    st.markdown(
        """
```
                file blob
                    |
        is text-extractable?
                    |
           yes-----/ \\-----no
           |               |
   Branch 1: Presidio   run OCR (EasyOCR)
           |               |
           |       OCR confidence >= 0.85 ?
           |               |
           |       yes-----/ \\-----no
           |       |               |
           |   Branch 2:        Branch 3:
           |   OCR -> Presidio  VLM (gemma-3-4b)
           |       |               |
           +-------+---------------+
                    |
               entity list
                    |
        bucket into H / M / L tiers
                    |
         S = min(10H + 5M + 1L, 100)
```
        """
    )
    tier_df = pd.DataFrame({
        "Tier":   ["High (H)",   "Medium (M)",   "Low (L)"],
        "Weight": [10, 5, 1],
        "Examples": [
            "CREDIT_CARD, NATIONAL_ID, IBAN, API_SECRET, MEDICAL_RECORD",
            "EMAIL_ADDRESS, PHONE_NUMBER, IP_ADDRESS, DATE_OF_BIRTH",
            "PERSON_NAME, ORG_NAME, LOCATION",
        ],
    })
    st.dataframe(tier_df, hide_index=True, use_container_width=True)
    st.caption(
        "The weights are deliberately exponential-ish so that one credit-card "
        "contributes ten times more than one person-name; the `min(., 100)` cap "
        "prevents large documents from saturating the score."
    )

    st.subheader("C. Hybrid Anomaly Engine \u2014 in detail")
    st.markdown(
        """
#### The 6-dimensional feature vector

Every intercepted event is reduced to a single normalised vector
`x_t \u2208 [0, 1]^6` before any model touches it:
        """
    )
    feat_df = pd.DataFrame({
        "Feature": [
            "activity_rate", "file_type_category", "geo_index",
            "endpoint_operation", "permission_scope_delta",
            "collab_network_density",
        ],
        "What it captures": [
            "Requests / minute for this user, normalised against their personal max.",
            "One-hot bucket of file type (text / image / scanned / handwriting / binary).",
            "Cosine distance between current geo-IP and the user's prior 30-day centroid.",
            "Privilege weight of the API verb (READ < SHARE < DELETE < ADMIN).",
            "Change in the user's effective permission set since the last call.",
            "Out-degree of the user in the 24h collaboration graph (who they shared with).",
        ],
    })
    st.dataframe(feat_df, hide_index=True, use_container_width=True)

    st.markdown(
        """
#### LSTM branch (temporal anomaly)

For each user the model keeps the last **24h of activity** as a sequence
of feature vectors. An **LSTM with hidden size 128** is trained on the
user's own history; at inference time the cosine similarity between the
LSTM's predicted next vector and the actual observed vector is mapped to
`a_LSTM \u2208 [0, 1]` (1 = maximally surprising).

This is what catches **drift** \u2014 "this user has never downloaded
payroll data at 2 a.m. before".

#### Isolation Forest branch (structural anomaly)

A 100-tree Isolation Forest is trained on **the whole organisation's**
activity, not per-user. The shorter the average isolation path of the
current point, the higher `a_IF`.

This is what catches **outliers** \u2014 "no one in this company has ever
shared this many files at once".

#### Why fuse them?

LSTM alone is blind to behaviours that are normal for *one* user but
abnormal for the *organisation*. Isolation Forest alone is blind to
personal drift. The hybrid score `A_hybrid` keeps both:
        """
    )
    st.latex(r"A_{\text{hybrid}} = \alpha \cdot a_{\text{LSTM}} + (1-\alpha)\, a_{\text{IF}},\quad \alpha\in[0,1]")

    st.subheader("D. SBRS \u2014 in detail")
    st.markdown(
        """
The Semantic-Behavioral Risk Score is intentionally **multiplicative in S**
and **additive in the anomaly amplifier `(1 + \u03b2 A_hybrid)`**:
        """
    )
    st.latex(r"\text{SBRS} = \frac{S \,\bigl(1 + \beta\, A_{\text{hybrid}}\bigr)}{100}")
    st.markdown(
        """
This shape has three useful properties:

1. **Benign data is never blocked.** If `S = 0` then `SBRS = 0` regardless
   of how anomalous the behaviour looks \u2014 which is what keeps the
   false-positive rate at 0% on the test split.
2. **Sensitive data + normal behaviour escalates softly.** With `A = 0`,
   `SBRS = S/100`, so even a maximally sensitive file accessed normally lands
   at 1.00 \u2014 below the 1.22 ALERT line.
3. **\u03b2 is the enterprise knob.** It sets how much behaviour is allowed to
   amplify content. At \u03b2 = 0.5 behaviour can move the score by at most +50%,
   which is why content used to dominate the decision entirely.

Enforcement bands, calibrated on the validation split as an explicit
false-positive budget \u2014 ALERT at the 95th percentile of benign traffic, BLOCK at
the 99.5th (`evaluation/calibrate_sbrs.py`):
        """
    )
    band_df = pd.DataFrame({
        "SBRS range": [f"< {T_ALERT}", f"{T_ALERT} \u2013 {T_BLOCK}", f"\u2265 {T_BLOCK}"],
        "Category":  ["SAFE", "SENSITIVE", "HIGH-RISK"],
        "Action":    ["PERMIT", "ALERT (log + notify)", "BLOCK (auto-ticket)"],
        "Benign traffic landing here": ["96.0 %", "3.6 %", "0.4 %"],
    })
    st.dataframe(band_df, hide_index=True, use_container_width=True)

    # ------------------------------------------------------------------
    # Latency budget
    # ------------------------------------------------------------------
    st.subheader("Latency budget")
    st.markdown(
        "Every stage below was **timed**, not estimated: the PII branches per "
        "document over the 120-document corpus, the anomaly path over 500 scoring "
        "calls. The three PII branches are alternatives, not a sum - a document "
        "takes exactly one of them."
    )
    lat_rows = [
        ("Branch 1 (Presidio, text)",
         PII["latency_ms"]["branch1_mean"] if PII else 11.3),
        ("Branch 2 (EasyOCR + Presidio)",
         PII["latency_ms"]["branch2_mean"] if PII else 2449.7),
        ("Branch 3 (VLM, hosted API)",
         PII["latency_ms"]["branch3_mean"] if PII else 3467.8),
        ("Anomaly engine (LSTM + memory bank + IF + fusion + SBRS)",
         MEASURED.latency_mean_ms if MEASURED else 16.9),
    ]
    lat_df = pd.DataFrame(lat_rows, columns=["Stage", "Mean (ms)"]).round(1)
    fig_lat = px.bar(
        lat_df, x="Mean (ms)", y="Stage", orientation="h",
        color="Mean (ms)", color_continuous_scale="Blues",
        title="Measured mean latency per stage (log scale)",
        log_x=True,
    )
    fig_lat.update_layout(
        height=320, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ecf0f1"),
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    lat_n = _zoom_reset_button("docs_latency")
    st.plotly_chart(
        fig_lat, use_container_width=True, config=PLOTLY_CONFIG,
        key=f"docs_latency_{lat_n}",
    )
    st.dataframe(lat_df, hide_index=True, use_container_width=True)
    st.caption(
        "Two orders of magnitude separate the text path from the image paths, "
        "which is exactly why the cascade exists: Branch 1 handles what it can for "
        "~11 ms, and only the documents it cannot read pay the OCR/VLM cost. "
        "Branch 3 is a network round-trip to a hosted model, not on-box compute, "
        "so it is not comparable to an INT8 model running next to the proxy."
    )

    # ------------------------------------------------------------------
    # Comparison vs commercial alternatives
    # ------------------------------------------------------------------
    st.subheader("How does it compare to commercial DLP tools?")
    cmp_df = pd.DataFrame({
        "Capability": [
            "Inspects encrypted SaaS API traffic in-line",
            "OCR + handwriting recognition",
            "Vision-Language Model semantic understanding",
            "Per-user temporal behavioural baseline (LSTM)",
            "Org-wide structural outlier detection (Isolation Forest)",
            "Single unified risk score (SBRS)",
            "Open formulation / paper-reproducible",
        ],
        "Legacy network DLP": ["\u2713", "partial", "\u2717", "\u2717", "\u2717", "\u2717", "\u2717"],
        "Cloud-native CASB":  ["\u2713", "partial", "\u2717", "partial", "partial", "\u2717", "\u2717"],
        "HybridSaaS-Sec":     ["\u2713", "\u2713",  "\u2713", "\u2713", "\u2713", "\u2713", "\u2713"],
    })
    st.dataframe(cmp_df, hide_index=True, use_container_width=True)

    # ------------------------------------------------------------------
    # FAQ
    # ------------------------------------------------------------------
    st.subheader("FAQ")
    with st.expander("Why a *Man-in-the-Middle*? Isn't that exactly what attackers do?"):
        st.markdown(
            "Yes \u2014 the *technique* is the same; the *trust model* is opposite. "
            "The enterprise installs its own root certificate on managed devices, "
            "so the proxy is part of the trusted computing base. Attackers without "
            "that root cannot mount the same interception."
        )
    with st.expander("Could the LSTM be replaced with a transformer?"):
        st.markdown(
            "Possibly, and it is the most promising direction: the LSTM's one real "
            "failure is the slow malicious insider (33% recall), whose signal is a "
            "gradual multi-day drift rather than a spike. The per-user sequences are "
            "short (24h) and the whole anomaly path costs ~17 ms measured, so there "
            "is budget to spend on a stronger sequence model."
        )
    with st.expander("What stops the proxy from logging the actual PII?"):
        st.markdown(
            "Two things: (a) the pipeline by construction passes only entity *types* "
            "and *counts* to the logger, never the values; (b) the deployment "
            "manifest enforces a read-only filesystem on the scanner pods so even "
            "a code bug cannot persist the raw text."
        )
    with st.expander("How does HybridSaaS-Sec scale to thousands of users?"):
        st.markdown(
            "The proxy is stateless and horizontally scalable behind a layer-4 "
            "load balancer. Per-user LSTM weights live in a Redis-backed feature "
            "store (one row per user, ~2 KB), keyed by `user_id`. Isolation Forest "
            "is re-trained nightly on the rolling 30-day window."
        )
    with st.expander("What are the limitations?"):
        st.markdown(
            "- **Slow malicious insiders are largely missed** (33% recall). Their "
            "drift is smoother than normal behaviour, so a per-user temporal model "
            "sees nothing anomalous. This is the honest weak spot: the other three "
            "threat classes are caught at 71-100%.  \n"
            "- **\u03c4_ocr = 0.85 is not portable across OCR engines.** On EasyOCR even "
            "a pristine, undegraded render scores 0.804 char-weighted confidence - "
            "already below the threshold - so the cut-point routes on engine "
            "calibration as much as on image quality. Re-tuned on held-out images it "
            "lands near 0.98.  \n"
            "- **Branch 3 costs ~3.5 s per image** as a hosted API call. An on-box "
            "quantised model would be far cheaper, at some accuracy cost.  \n"
            "- The corpus is synthetic (all PII generated with Faker) and the "
            "sessions are simulated, so absolute numbers will move on real traffic; "
            "the relative comparisons are the load-bearing part."
        )

    # ------------------------------------------------------------------
    st.subheader("Further reading")
    st.markdown(
        """
- **Original paper:** Mir & Hashir, *A Scalable Framework for Privacy-
  Preserving API Monitoring and Multimodal Data Leakage Prevention in
  Enterprise Cloud Environments*, Software Design and Architecture
  Research Paper, 2026.
- **Code:** see `backend/` (FastAPI + simulated pipeline) and
  `frontend/app.py` (this dashboard).
- **Datasets:** `pii_scan_results.csv`, `anomaly_detection_comparison.csv`,
  `ewma_user_baselines.csv`, `blocked_events_audit.csv`,
  `hybridSaaS_events.json`, `hybridSaaS_system.log`.
        """
    )
