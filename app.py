from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BASE = Path(__file__).parent
MODEL_PATH = BASE / "fraud_model.pkl"
METRICS_PATH = BASE / "model_metrics.json"
DEMO_PATH = BASE / "demo_stream.csv"

st.set_page_config(page_title="Signal | Fraud intelligence", page_icon="◉", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#eef1f6; --muted:#919caf; --panel:#111720; --line:#202a37; --green:#a8efc1; --red:#ff8c82; --blue:#88b6ff; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:radial-gradient(ellipse at 80% -15%, #182b35 0, #0a0e13 38%); color:var(--ink); }
section[data-testid="stSidebar"] { background:#0d1219; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] > div { padding-top:1.4rem; }
.block-container { max-width:1420px; padding-top:2rem; padding-bottom:4rem; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.04em; }
.eyebrow { color:#9ba9bb; font:500 .72rem 'DM Mono',monospace; letter-spacing:.13em; text-transform:uppercase; }
.hero-title { font:700 clamp(2.15rem,4vw,3.5rem)/1.05 'Manrope',sans-serif; letter-spacing:-.065em; margin:.7rem 0 .55rem; color:#f4f6f9; }
.hero-copy { color:#9ba8b8; font-size:1rem; max-width:690px; line-height:1.65; }
.topline { height:1px; background:linear-gradient(90deg,#344052,transparent); margin:1.7rem 0 1.4rem; }
.metric-card { background:linear-gradient(145deg,rgba(20,28,39,.94),rgba(15,21,29,.90)); border:1px solid #25303e; border-radius:14px; padding:18px 20px; min-height:122px; box-shadow:0 12px 30px rgba(0,0,0,.13); }
.metric-label { color:#98a4b6; font-size:.77rem; letter-spacing:.04em; text-transform:uppercase; }
.metric-value { font:700 1.8rem 'Manrope',sans-serif; letter-spacing:-.05em; margin-top:9px; color:#f2f5f8; }
.metric-note { color:#8390a3; font-size:.77rem; margin-top:3px; }
.section-title { font:700 1.18rem 'Manrope',sans-serif; letter-spacing:-.035em; margin:1.5rem 0 .2rem; }
.section-sub { color:#8996a8; font-size:.86rem; margin-bottom:.8rem; }
.tag { display:inline-block; border:1px solid #2b3948; border-radius:99px; color:#b6c3d3; padding:5px 10px; font:.7rem 'DM Mono',monospace; letter-spacing:.04em; }
.tag-green { border-color:#2e6348; color:#a8efc1; background:#183327; }
.tag-red { border-color:#693d3a; color:#ff9b91; background:#351f20; }
.stButton>button { border-radius:9px; border:1px solid #355c4d; background:#a8efc1; color:#0d1712; font-weight:700; transition:all .18s ease; }
.stButton>button:hover { background:#c5f7d5; border-color:#a8efc1; color:#0b1510; transform:translateY(-1px); }
div[data-testid="stDataFrame"] { border:1px solid #26313e; border-radius:10px; overflow:hidden; }
.stTabs [data-baseweb="tab-list"] { gap:1rem; border-bottom:1px solid #293443; }
.stTabs [data-baseweb="tab"] { color:#a5afbd; }
.stTabs [aria-selected="true"] { color:#b7f1ca !important; }
.stAlert { border-radius:10px; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_bundle():
    with MODEL_PATH.open("rb") as source:
        return pickle.load(source)


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_demo():
    return pd.read_csv(DEMO_PATH)


if not MODEL_PATH.exists() or not METRICS_PATH.exists() or not DEMO_PATH.exists():
    st.error("Model files are missing. Run `python train_model.py --data path/to/creditcard.csv` first.")
    st.stop()

bundle = load_model_bundle()
model = bundle["model"]
FEATURES = bundle["features"]
DEFAULT_THRESHOLD = float(bundle["threshold"])
metrics = load_metrics()
demo = load_demo()


def score_rows(rows: pd.DataFrame, threshold: float) -> pd.DataFrame:
    data = rows.copy()
    probabilities = model.predict_proba(data[FEATURES].astype(float))[:, 1]
    data["fraud_probability"] = probabilities
    data["decision"] = np.where(probabilities >= threshold, "Review", "Clear")
    data["risk"] = np.select(
        [probabilities >= threshold, probabilities >= max(0.15, threshold * 0.55)],
        ["High", "Elevated"], default="Low"
    )
    return data


def metric_card(label: str, value: str, note: str = ""):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)


def section(title: str, subtitle: str):
    st.markdown(f'<div class="section-title">{title}</div><div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def chart_layout(fig, height=320):
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=8, r=8, t=22, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#acb7c6"),
        legend=dict(orientation="h", y=1.1, x=0),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#273240")
    fig.update_yaxes(gridcolor="#202a36", zeroline=False)
    return fig


with st.sidebar:
    st.markdown('<div class="eyebrow">◉ &nbsp; SIGNAL / CAPSTONE</div>', unsafe_allow_html=True)
    st.markdown("### Fraud intelligence")
    st.caption("REAL-TIME TRANSACTION MONITOR")
    page = st.radio("Workspace", ["Overview", "Live stream", "Score a transaction", "Model & data"], label_visibility="collapsed")
    st.markdown("<div class='topline'></div>", unsafe_allow_html=True)
    st.markdown('<span class="tag tag-green">● &nbsp; MODEL READY</span>', unsafe_allow_html=True)
    st.markdown("\n")
    st.caption(f"Random Forest · {metrics['trees']} trees")
    st.caption(f"Alert threshold · {DEFAULT_THRESHOLD:.3f}")
    st.markdown("<div class='topline'></div>", unsafe_allow_html=True)
    st.caption("BCA 2024–2027  ·  Capstone project")
    st.caption("Kalyan Shashidhar Reddy")


if page == "Overview":
    st.markdown('<div class="eyebrow">TRANSACTION OPERATIONS &nbsp; / &nbsp; OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">See the signal<br>in every transaction.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-copy">An end-to-end fraud detection prototype trained on anonymized card activity. Explore model performance, replay a transaction stream, and inspect alerts as they happen.</div>', unsafe_allow_html=True)
    st.markdown('<div class="topline"></div>', unsafe_allow_html=True)

    test = metrics["test"]
    c = st.columns(4, gap="medium")
    with c[0]: metric_card("Transactions analyzed", f"{metrics['rows']:,}", "Historical anonymized records")
    with c[1]: metric_card("Fraud prevalence", f"{metrics['fraud_rate'] * 100:.2f}%", f"{metrics['fraud_count']:,} labeled fraud cases")
    with c[2]: metric_card("Fraud recall", f"{test['recall'] * 100:.1f}%", "Held-out test set · default threshold")
    with c[3]: metric_card("PR-AUC", f"{test['pr_auc']:.3f}", "Precision–recall area under curve")

    left, right = st.columns([1.45, 1], gap="large")
    with left:
        section("Activity across the sample", "Historical transactions grouped into 2-hour windows")
        hour_frame = demo.copy()
        hour_frame["hour"] = (hour_frame["Time"] / 3600).astype(int)
        activity = pd.DataFrame(metrics["activity_by_hour"])
        activity = activity.melt(id_vars="hour", var_name="type", value_name="transactions")
        activity["type"] = activity["type"].map({"legitimate": "Legitimate", "fraud": "Fraud"})
        fig = px.area(activity, x="hour", y="transactions", color="type", color_discrete_map={"Legitimate": "#57708d", "Fraud": "#ff8c82"}, labels={"hour": "Elapsed hour", "transactions": "Transactions", "type": "Class"}, line_shape="linear")
        fig.update_traces(opacity=0.72)
        st.plotly_chart(chart_layout(fig, 330), use_container_width=True, config={"displayModeBar": False})
    with right:
        section("Detection quality", "The operating threshold was selected on a validation split")
        labels = ["Precision", "Recall", "F1 score"]
        values = [test["precision"], test["recall"], test["f1"]]
        fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker=dict(color=["#88b6ff", "#a8efc1", "#e8c184"], line=dict(width=0)), text=[f"{v:.1%}" for v in values], textposition="outside"))
        fig.update_xaxes(range=[0, 1.12], tickformat=".0%", gridcolor="#202a36")
        fig.update_yaxes(autorange="reversed", showgrid=False)
        st.plotly_chart(chart_layout(fig, 330), use_container_width=True, config={"displayModeBar": False})
    st.markdown('<div class="topline"></div>', unsafe_allow_html=True)
    section("Prototype pipeline", "The streaming layer is simulated here; Kafka and Spark are documented as the next integration stage.")
    pipeline = st.columns(5, gap="small")
    stages = [("01", "Historical data", "creditcard.csv"), ("02", "Train model", "Random Forest"), ("03", "Simulate", "Held-out sample"), ("04", "Classify", "Threshold scoring"), ("05", "Alert", "Live dashboard")]
    for col, (number, title, detail) in zip(pipeline, stages):
        with col:
            st.markdown(f'<div class="metric-card" style="min-height:96px;padding:14px 16px"><div class="eyebrow">{number}</div><div style="font-weight:700;margin-top:5px">{title}</div><div class="metric-note">{detail}</div></div>', unsafe_allow_html=True)


elif page == "Live stream":
    st.markdown('<div class="eyebrow">TRANSACTION OPERATIONS &nbsp; / &nbsp; STREAM SIMULATOR</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Live transaction stream</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-copy">Replay synthetic transaction profiles through the trained model. Each batch is scored with the current alert threshold, and suspicious profiles are added to the review queue.</div>', unsafe_allow_html=True)
    st.markdown('<div class="topline"></div>', unsafe_allow_html=True)
    threshold = st.slider("Alert threshold", min_value=0.05, max_value=0.95, value=DEFAULT_THRESHOLD, step=0.01, help="Transactions at or above this model probability are sent to review.")
    if "stream_cursor" not in st.session_state:
        st.session_state.stream_cursor = 0
    if "stream_results" not in st.session_state:
        st.session_state.stream_results = pd.DataFrame()
    if "stream_running" not in st.session_state:
        st.session_state.stream_running = False
    controls = st.columns([1, 1, 4])
    with controls[0]:
        if st.button("Pause stream" if st.session_state.stream_running else "▶  Start stream", use_container_width=True):
            st.session_state.stream_running = not st.session_state.stream_running
            st.rerun()
    with controls[1]:
        if st.button("Reset replay", use_container_width=True):
            st.session_state.stream_cursor = 0
            st.session_state.stream_results = pd.DataFrame()
            st.session_state.stream_running = False
            st.rerun()
    st.caption("Stream source: 1,200 synthetic profiles generated from class-level feature distributions · 2-second replay interval when running")

    @st.fragment(run_every="2s")
    def stream_view():
        if st.session_state.stream_running:
            start = st.session_state.stream_cursor
            if start >= len(demo):
                st.session_state.stream_cursor = 0
                start = 0
            stop = min(start + 12, len(demo))
            batch = demo.iloc[start:stop].copy()
            st.session_state.stream_cursor = stop
            scored = score_rows(batch, threshold)
            st.session_state.stream_results = pd.concat([st.session_state.stream_results, scored], ignore_index=True).tail(800)
        results = st.session_state.stream_results
        if results.empty:
            results = score_rows(demo.iloc[:12].copy(), threshold)
            st.info("Start the stream to begin processing transaction batches. The preview below shows the first sample batch.")
        frauds = int((results["decision"] == "Review").sum())
        a, b, c, d = st.columns(4, gap="medium")
        with a: metric_card("Processed", f"{st.session_state.stream_cursor:,}", f"of {len(demo):,} replay records")
        with b: metric_card("Review alerts", f"{frauds:,}", "In the visible rolling window")
        with c: metric_card("High risk", f"{int((results['risk'] == 'High').sum()):,}", "At or above threshold")
        with d: metric_card("Queue status", "Active" if st.session_state.stream_running else "Paused", "Simulated stream")
        main, side = st.columns([1.5, 1], gap="large")
        with main:
            section("Incoming activity", "Most recent transactions scored by the Random Forest")
            visible = results.tail(80).copy()
            visible["fraud_probability"] = (visible["fraud_probability"] * 100).map(lambda x: f"{x:.1f}%")
            visible["Amount"] = visible["Amount"].map(lambda x: f"${x:,.2f}")
            st.dataframe(visible[["transaction_id", "Amount", "decision", "risk", "fraud_probability"]].iloc[::-1], hide_index=True, use_container_width=True, height=350)
        with side:
            section("Review queue", "Highest-risk records currently in the window")
            alerts = results[results["decision"] == "Review"].sort_values("fraud_probability", ascending=False).head(12)
            if alerts.empty:
                st.markdown("<div class='metric-card'>No transactions crossed the alert threshold in the current batch.</div>", unsafe_allow_html=True)
            else:
                for _, row in alerts.iterrows():
                    st.markdown(f"<div class='metric-card' style='min-height:62px;padding:12px 15px;margin-bottom:8px'><span class='tag tag-red'>REVIEW</span> &nbsp; <b>{row['transaction_id']}</b><br><span class='metric-note'>${row['Amount']:,.2f} &nbsp;·&nbsp; {row['fraud_probability']:.1%} model score</span></div>", unsafe_allow_html=True)

    stream_view()


elif page == "Score a transaction":
    st.markdown('<div class="eyebrow">TRANSACTION OPERATIONS &nbsp; / &nbsp; SINGLE SCORE</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Score a transaction</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-copy">Try the model on one example or upload a batch. This project uses PCA-transformed V1–V28 inputs from the public dataset, so the anonymized profile values are retained for consistent scoring.</div>', unsafe_allow_html=True)
    st.markdown('<div class="topline"></div>', unsafe_allow_html=True)
    single, batch = st.tabs(["Single transaction", "Batch scoring"])
    with single:
        left, right = st.columns([1, 1], gap="large")
        with left:
            archetype = st.selectbox("Feature profile", ["Typical transaction profile", "Unusual transaction profile"])
            examples = demo[demo["profile_type"] == ("Typical profile" if archetype.startswith("Typical") else "Elevated profile")]
            source_row = examples.iloc[0]
            with st.form("single_score"):
                amount = st.number_input("Transaction amount", min_value=0.0, value=float(source_row["Amount"]), step=10.0, format="%.2f")
                elapsed = st.number_input("Elapsed time (seconds)", min_value=0.0, max_value=172800.0, value=float(source_row["Time"]), step=60.0)
                st.caption("The feature profile supplies the anonymized V1–V28 PCA values; amount and elapsed time use the values entered here.")
                submitted = st.form_submit_button("Run fraud assessment", use_container_width=True)
            if submitted:
                case = pd.DataFrame([{feature: source_row[feature] for feature in FEATURES}])
                case.loc[0, "Amount"] = amount
                case.loc[0, "Time"] = elapsed
                st.session_state.single_result = score_rows(case, DEFAULT_THRESHOLD).iloc[0].to_dict()
        with right:
            section("Assessment", "Model estimate and review guidance")
            result = st.session_state.get("single_result")
            if result:
                is_review = result["decision"] == "Review"
                st.markdown(f"<div class='metric-card' style='min-height:200px'><span class='tag {'tag-red' if is_review else 'tag-green'}'>{'REVIEW REQUIRED' if is_review else 'NO ALERT'}</span><div class='metric-value' style='font-size:2.7rem'>{result['fraud_probability']:.1%}</div><div class='metric-label'>Estimated fraud probability</div><div class='metric-note' style='margin-top:14px'>Threshold: {DEFAULT_THRESHOLD:.1%} · Risk band: {result['risk']}</div></div>", unsafe_allow_html=True)
                st.caption("This is a model score for an educational prototype. It is not a financial decision or a guarantee of fraud.")
            else:
                st.markdown("<div class='metric-card' style='min-height:200px'><div class='eyebrow'>READY TO ASSESS</div><div style='font:600 1.2rem Manrope;margin-top:16px'>Choose a profile and run the model</div><div class='metric-note' style='margin-top:8px'>Your score will appear here with the active alert threshold.</div></div>", unsafe_allow_html=True)
    with batch:
        st.markdown("Upload a CSV with columns `Time`, `V1`–`V28`, and `Amount`. The model returns a probability and a review decision for every row.")
        uploaded = st.file_uploader("Transaction CSV", type=["csv"], key="batch_upload")
        if uploaded is not None:
            try:
                incoming = pd.read_csv(uploaded)
                missing = [feature for feature in FEATURES if feature not in incoming.columns]
                if missing:
                    st.error("Missing required fields: " + ", ".join(missing))
                else:
                    if st.button("Score uploaded transactions", type="primary"):
                        scored = score_rows(incoming, DEFAULT_THRESHOLD)
                        st.success(f"Scored {len(scored):,} transactions · {int((scored['decision'] == 'Review').sum()):,} sent to review.")
                        view = scored.copy()
                        view["fraud_probability"] = view["fraud_probability"].map(lambda x: f"{x:.2%}")
                        st.dataframe(view[[*(["transaction_id"] if "transaction_id" in view else []), "Amount", "risk", "fraud_probability", "decision"]], hide_index=True, use_container_width=True)
                        st.download_button("Download scored results", scored.to_csv(index=False), "scored_transactions.csv", "text/csv")
            except Exception as exc:
                st.error(f"Could not read this file: {exc}")


elif page == "Model & data":
    st.markdown('<div class="eyebrow">PROJECT DETAILS &nbsp; / &nbsp; MODEL CARD</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Model & dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-copy">Training setup, evaluation results, and a transparent view of what this prototype can and cannot tell you.</div>', unsafe_allow_html=True)
    st.markdown('<div class="topline"></div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1], gap="large")
    with left:
        section("Training setup", "Reproducible experiment configuration")
        st.markdown(f"""
| Item | Value |
|---|---|
| Classifier | {metrics['model']} |
| Estimators | {metrics['trees']} trees |
| Class balancing | `{metrics['class_weight']}` |
| Input features | {metrics['feature_count']} (Time, V1–V28, Amount) |
| Training rows | {metrics['training_rows']:,} |
| Validation rows | {metrics['validation_rows']:,} |
| Held-out test rows | {metrics['test']['test_transactions']:,} |
| Split strategy | Stratified 70 / 15 / 15 |
| Decision threshold | {metrics['decision_threshold']:.4f} · tuned on validation F1 |
| Fit time | {metrics['training_seconds']:.1f} seconds |
""")
    with right:
        section("Held-out evaluation", "Threshold selected using validation data, then applied once to the test split")
        cm = metrics["test"]["confusion_matrix"]
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        fig = go.Figure(go.Heatmap(z=matrix, x=["Predicted clear", "Predicted review"], y=["Actual legitimate", "Actual fraud"], colorscale=[[0, "#16212c"], [0.5, "#386852"], [1, "#a8efc1"]], text=matrix, texttemplate="%{text:,}", showscale=False))
        st.plotly_chart(chart_layout(fig, 285), use_container_width=True, config={"displayModeBar": False})
        score_cols = st.columns(4)
        for col, label, value in zip(score_cols, ["Precision", "Recall", "F1", "ROC-AUC"], [metrics["test"]["precision"], metrics["test"]["recall"], metrics["test"]["f1"], metrics["test"]["roc_auc"]]):
            with col: metric_card(label, f"{value:.3f}")
    st.markdown('<div class="topline"></div>', unsafe_allow_html=True)
    section("Data notes & limitations", "Read these before interpreting an alert")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Dataset:** {metrics['dataset']} · {metrics['rows']:,} rows · {metrics['fraud_count']:,} fraud labels ({metrics['fraud_rate']:.3%}). Features V1–V28 are anonymized PCA components; they do not expose original cardholder attributes.")
        st.warning("The label is highly imbalanced. Precision, recall, F1, and PR-AUC are more informative than accuracy alone. The chosen threshold trades missed fraud against review volume.")
    with col2:
        st.info("**Streaming scope:** this website replays a held-out sample locally. The capstone outline names Kafka and Spark Structured Streaming for the full distributed pipeline; those services are not configured in this local demo.")
        st.warning("Historical test metrics do not guarantee future performance. This educational prototype is not connected to a bank, payment processor, or live financial data.")
        st.caption(f"Dataset amount context: median ${metrics['amount_median']:,.2f} overall · median ${metrics['fraud_amount_median']:,.2f} for labeled fraud cases. Model engine: {metrics['model_engine']}.")

