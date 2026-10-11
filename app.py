from __future__ import annotations
import hashlib,io,json,pickle
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE=Path(__file__).parent
MODEL,INFO,DEMO=BASE/"fraud_model.pkl",BASE/"model_metrics.json",BASE/"demo_stream.csv"
REPO="https://github.com/7t8Inferno/capstone-project-1"
PAGES=["Home","Live monitor","Analyze transactions","Model evaluation","Activity history","About","Contact","Privacy policy","Terms and conditions"]
st.set_page_config(page_title="Signal | Fraud monitoring",page_icon="S",layout="wide",initial_sidebar_state="expanded")
st.markdown("""
<style>
:root{color-scheme:dark;--bg:#0b0e10;--panel:#111719;--line:#293235;--text:#eef1ef;--muted:#a0aaa7;--accent:#a9d7b8}
html,body,[class*="css"]{font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif}
.stApp{background:var(--bg);color:var(--text)}
section[data-testid="stSidebar"]{background:#0e1214;border-right:1px solid var(--line)}
section[data-testid="stSidebar"]>div{padding:1.1rem 1rem 1.5rem}
.block-container{max-width:1320px;padding:2rem 2.4rem 2.5rem}
h1,h2,h3{color:var(--text)!important;letter-spacing:-.035em}
h1{font-size:clamp(2rem,3.4vw,3rem)!important;line-height:1.08!important}
p,li{color:#c5cdca;line-height:1.65} a{color:var(--accent)!important}
.eyebrow{color:#9da9a5;font:600 .68rem ui-monospace,monospace;letter-spacing:.12em;text-transform:uppercase}
.mark{display:inline-flex;width:1.6rem;height:1.6rem;align-items:center;justify-content:center;border:1px solid #67776f;border-radius:4px;color:var(--accent);font:700 .9rem ui-monospace,monospace;margin-right:.5rem}
.wordmark{color:#f0f3f1;font:700 .9rem ui-monospace,monospace;letter-spacing:.1em}
.badge{display:inline-block;color:#badcc5;background:#17231c;border:1px solid #354a3d;border-radius:4px;padding:.28rem .5rem;font:600 .66rem ui-monospace,monospace;letter-spacing:.07em}
.hero{max-width:760px;color:#acb6b2;font-size:1rem;line-height:1.65}
.rule{border:0;border-top:1px solid var(--line);margin:1.35rem 0}
.sec{color:#d9dfdc;font-weight:650;font-size:1rem;margin:.2rem 0}
.note{color:var(--muted);font-size:.83rem;margin:0 0 .7rem;line-height:1.5}
.card{height:100%;min-height:105px;padding:1rem;background:var(--panel);border:1px solid var(--line);border-radius:5px}
.label{color:#a8b2ae;font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.055em}
.value{color:var(--text);margin-top:.58rem;font-size:1.65rem;font-weight:700;letter-spacing:-.04em}
.sub{color:var(--muted);margin-top:.35rem;font-size:.76rem;line-height:1.45}
.notice{padding:.9rem 1rem;border:1px solid var(--line);border-left:3px solid #758d7d;border-radius:4px;background:var(--panel);color:#c2cbc6;line-height:1.55}
.footer{border-top:1px solid var(--line);margin-top:2.4rem;padding-top:1rem;color:var(--muted);font-size:.78rem}
.stButton>button,.stLinkButton>a,.stDownloadButton>button,[data-testid="stFormSubmitButton"] button{border-radius:4px;border:1px solid #51685a;background:var(--accent);color:#14231a;font-weight:650;min-height:2.5rem}
.stButton>button:hover,.stLinkButton>a:hover,.stDownloadButton>button:hover{background:#c1e4cc;color:#101b14}
.stButton>button:focus,.stLinkButton>a:focus,.stDownloadButton>button:focus{outline:2px solid #d5f1de;outline-offset:2px}
div[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:5px;overflow:hidden}
.stTabs [data-baseweb="tab-list"]{gap:.5rem;border-bottom:1px solid var(--line)}
.stTabs [aria-selected="true"]{color:var(--accent)!important}
footer{visibility:hidden}
@media(max-width:760px){.block-container{padding:1.3rem 1rem 2rem}h1{font-size:2rem!important}}
</style>""",unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with MODEL.open("rb") as f:return pickle.load(f)
@st.cache_data
def load_info():return json.loads(INFO.read_text(encoding="utf-8"))
@st.cache_data
def load_samples():return pd.read_csv(DEMO)
if not all(x.exists() for x in [MODEL,INFO,DEMO]):st.error("Model files are incomplete.");st.stop()
bundle=load_model();model=bundle["model"];FEATURES=bundle["features"];DEFAULT=float(bundle["threshold"]);info=load_info();demo=load_samples()

def section(title,note=""):
    st.markdown(f'<div class="sec">{title}</div><div class="note">{note}</div>',unsafe_allow_html=True)
def metric(label,value,note=""):
    st.markdown(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{note}</div></div>',unsafe_allow_html=True)
def navigate(name):
    st.session_state["nav"]=name;st.session_state["nav_param"]=name;st.query_params["view"]=name;st.rerun()
def sync_navigation():
    name=st.session_state["nav"]
    st.session_state["nav_param"]=name;st.query_params["view"]=name
def score(frame,threshold):
    data=frame.loc[:,FEATURES].astype(float);p=model.predict_proba(data)[:,1]
    risk=np.select([p>=threshold,p>=max(.15,threshold*.55)],["High","Elevated"],default="Low")
    return pd.DataFrame({"Row":np.arange(1,len(frame)+1),"Time (seconds)":data["Time"].to_numpy(),"Amount":data["Amount"].to_numpy(),"Fraud probability":p,"Risk band":risk,"Decision":np.where(p>=threshold,"Review","Clear")})
def validate(file):
    raw=file.getvalue()
    if not raw:raise ValueError("The selected CSV is empty.")
    try:frame=pd.read_csv(io.BytesIO(raw))
    except Exception as e:raise ValueError("Could not read this CSV. Check its format and encoding.") from e
    if frame.empty:raise ValueError("The CSV has no transaction rows.")
    if len(frame)>20000:raise ValueError("Upload no more than 20,000 rows at a time.")
    missing=[x for x in FEATURES if x not in frame]
    if missing:raise ValueError("Missing required fields: "+", ".join(missing))
    data=frame.loc[:,FEATURES].copy()
    for x in FEATURES:data[x]=pd.to_numeric(data[x],errors="coerce")
    if not np.isfinite(data.to_numpy(float)).all():raise ValueError("Every required feature must contain a finite number.")
    if (data["Time"]<0).any() or (data["Amount"]<0).any():raise ValueError("Time and Amount must be zero or greater.")
    return data
def log_activity(source,result):
    events=st.session_state.setdefault("activity",[]);review=int((result["Decision"]=="Review").sum())
    events.insert(0,{"Time (UTC)":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),"Source":source,"Rows":len(result),"Review":review,"Clear":len(result)-review});del events[100:]
def plot(fig,height=300):
    fig.update_layout(template="plotly_dark",height=height,margin=dict(l=8,r=12,t=16,b=8),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="Inter,sans-serif",color="#b8c1bd"),legend=dict(orientation="h",y=1.12,x=0))
    fig.update_xaxes(showgrid=False,zeroline=False,linecolor="#30393b");fig.update_yaxes(gridcolor="#252d2f",zeroline=False);return fig

requested=st.query_params.get("view")
if "nav_param" not in st.session_state:
    initial=requested if requested in PAGES else "Home"
    st.session_state["nav"]=initial;st.session_state["nav_param"]=initial;st.query_params["view"]=initial
elif requested in PAGES and requested!=st.session_state.get("nav_param"):
    st.session_state["nav"]=requested;st.session_state["nav_param"]=requested

with st.sidebar:
    st.markdown('<span class="mark">S</span><span class="wordmark">SIGNAL</span>',unsafe_allow_html=True)
    st.markdown("### Fraud monitoring");st.caption("CARD TRANSACTION ANALYSIS")
    page=st.radio("Navigate",PAGES,key="nav",on_change=sync_navigation)
    st.markdown("<hr class='rule'>",unsafe_allow_html=True)
    st.markdown('<span class="badge">MODEL READY</span>',unsafe_allow_html=True)
    st.caption(f"Random Forest, {info['trees']} trees");st.caption(f"Review threshold, {DEFAULT:.3f}")
    st.markdown("<hr class='rule'>",unsafe_allow_html=True);st.caption("Capstone research prototype");st.caption("No live payment connection")
x,y=st.columns([4,1])
with x:st.markdown('<div class="eyebrow">SIGNAL / FINANCIAL TRANSACTION MONITORING</div>',unsafe_allow_html=True)
with y:st.markdown('<div style="text-align:right"><span class="badge">RESEARCH PROTOTYPE</span></div>',unsafe_allow_html=True)

if page=="Home":
    st.title("Card transaction fraud monitoring")
    st.markdown('<div class="hero">Explore a Random Forest trained on anonymized card activity. Review held-out results, score a feature-only file, or run a clearly labeled synthetic simulation.</div>',unsafe_allow_html=True)
    st.markdown('<hr class="rule">',unsafe_allow_html=True);test=info["test"];cols=st.columns(4,gap="small")
    with cols[0]:metric("Source records",f"{info['rows']:,}","Anonymized historical dataset")
    with cols[1]:metric("Labeled fraud",f"{info['fraud_count']:,}",f"{info['fraud_rate']*100:.2f}% of source records")
    with cols[2]:metric("Test recall",f"{test['recall']:.1%}","Held-out test split")
    with cols[3]:metric("Test PR-AUC",f"{test['pr_auc']:.3f}","Precision-recall area")
    l,r=st.columns([1.2,1],gap="large")
    with l:
        section("Historical activity","Counts from the supplied dataset, grouped by elapsed time.")
        df=pd.DataFrame(info["activity_by_hour"]).melt(id_vars="hour",var_name="class",value_name="transactions");df["class"]=df["class"].map({"legitimate":"Legitimate","fraud":"Fraud"});fig=go.Figure()
        for name,color in [("Legitimate","#87938f"),("Fraud","#e9988e")]:
            sub=df[df["class"]==name];fig.add_trace(go.Scatter(x=sub["hour"],y=sub["transactions"],mode="lines",name=name,line=dict(color=color,width=2)))
        fig.update_layout(xaxis_title="Elapsed hour",yaxis_title="Transactions");st.plotly_chart(plot(fig),width="stretch",config={"displayModeBar":False})
    with r:
        section("Test set performance","Threshold selected on validation data.")
        vals=[test["precision"],test["recall"],test["f1"]];fig=go.Figure(go.Bar(x=vals,y=["Precision","Recall","F1"],orientation="h",marker_color=["#9aadc0","#a9d7b8","#d1b77e"],text=[f"{v:.1%}" for v in vals],textposition="outside"));fig.update_xaxes(range=[0,1.12],tickformat=".0%");fig.update_yaxes(autorange="reversed",showgrid=False);st.plotly_chart(plot(fig),width="stretch",config={"displayModeBar":False})
    st.markdown('<hr class="rule">',unsafe_allow_html=True);section("Choose a task")
    a,b,c=st.columns(3,gap="small")
    with a:
        st.markdown("**Run the monitor**");st.caption("Process synthetic profiles in user-triggered batches.")
        if st.button("Open live monitor",key="home_monitor"):navigate("Live monitor")
    with b:
        st.markdown("**Score transaction data**");st.caption("Upload one anonymized row or a batch.")
        if st.button("Analyze transactions",key="home_analyze"):navigate("Analyze transactions")
    with c:
        st.markdown("**Review model results**");st.caption("Inspect metrics, split, and limitations.")
        if st.button("Model evaluation",key="home_model"):navigate("Model evaluation")
    st.markdown('<div class="notice">The monitor uses synthetic example rows. It does not represent live card activity.</div>',unsafe_allow_html=True)

elif page=="Live monitor":
    st.title("Live monitor")
    st.markdown('<div class="hero">Step through the included synthetic sample and inspect decisions batch by batch. No event broker, payment gateway, or Spark service is connected.</div>',unsafe_allow_html=True)
    st.markdown('<hr class="rule">',unsafe_allow_html=True);st.markdown('<div class="notice">Simulation data only. These profiles are not real-time fraud activity or real transactions.</div>',unsafe_allow_html=True)
    threshold=st.slider("Review threshold",.05,.95,DEFAULT,.01,key="sim_threshold")
    if "cursor" not in st.session_state:st.session_state.cursor=0
    if "sim_result" not in st.session_state:st.session_state.sim_result=pd.DataFrame()
    l,r,_=st.columns([1,1,3])
    with l:run=st.button("Process next batch",key="process")
    with r:reset=st.button("Reset simulation",key="reset")
    if reset:st.session_state.cursor=0;st.session_state.sim_result=pd.DataFrame();st.rerun()
    if run:
        start=st.session_state.cursor
        if start>=len(demo):start=0
        end=min(start+24,len(demo));batch=score(demo.iloc[start:end],threshold)
        st.session_state.cursor=end;st.session_state.sim_result=pd.concat([st.session_state.sim_result,batch],ignore_index=True).tail(600);log_activity("Synthetic simulation",batch)
    result=st.session_state.sim_result;review=int((result["Decision"]=="Review").sum()) if len(result) else 0;cols=st.columns(3,gap="small")
    with cols[0]:metric("Rows processed",f"{st.session_state.cursor:,}",f"of {len(demo):,} synthetic sample rows")
    with cols[1]:metric("Review decisions",f"{review:,}","Session simulation window")
    with cols[2]:metric("Simulation state","Ready" if result.empty else "Paused","Runs on request")
    section("Recent scored rows","Only rows processed in this browser session appear here.")
    if result.empty:st.info("No rows processed. Select Process next batch to start.")
    else:
        view=result.tail(100).copy();view["Amount"]=view["Amount"].map(lambda z:"USD "+f"{z:,.2f}");view["Fraud probability"]=view["Fraud probability"].map(lambda z:f"{z:.1%}");st.dataframe(view.iloc[::-1],hide_index=True,width="stretch",height=340)
        alerts=result[result.Decision=="Review"].sort_values("Fraud probability",ascending=False);section("Review queue","Highest model scores in this session.")
        if alerts.empty:st.success("No processed row met the threshold.")
        else:st.dataframe(alerts.head(20),hide_index=True,width="stretch",height=250)

elif page=="Analyze transactions":
    st.title("Analyze transactions")
    st.markdown('<div class="hero">Score a single row or batch with the required anonymized feature schema. Results are estimates, not financial decisions.</div>',unsafe_allow_html=True)
    st.markdown('<hr class="rule">',unsafe_allow_html=True);st.markdown('<div class="notice">Remove names, card numbers, account IDs, and other personal details before upload. Required fields: Time, V1 through V28, Amount. PCA feature values must match the supplied dataset format.</div>',unsafe_allow_html=True)
    threshold=st.slider("Review threshold",.05,.95,DEFAULT,.01,key="threshold")
    one,many=st.tabs(["Single row","Batch file"])
    with one:
        section("Score one row","Upload a CSV containing exactly one feature row.");upload=st.file_uploader("Single transaction CSV",type=["csv"],key="single")
        if upload is None:st.info("Choose a one-row CSV. Download the blank schema to prepare a file.")
        else:
            digest=hashlib.sha256(upload.getvalue()).hexdigest()
            if st.button("Run assessment",key="assess"):
                try:
                    clean=validate(upload)
                    if len(clean)!=1:st.error("This workflow accepts one row. Use Batch file for multiple rows.");st.session_state.pop("single_result",None)
                    else:
                        result=score(clean,threshold);st.session_state.single_result=(digest,result);log_activity("Single row assessment",result)
                except ValueError as e:st.error(str(e))
                except Exception:st.error("The model could not score this file. Check its feature format and retry.")
            saved=st.session_state.get("single_result")
            if saved and saved[0]==digest:
                row=saved[1].iloc[0];a,b=st.columns(2)
                with a:metric("Fraud probability",f"{row['Fraud probability']:.1%}","Model estimate, not certainty")
                with b:metric("Decision",row.Decision,f"Risk band: {row['Risk band']}")
                st.download_button("Download assessment",saved[1].to_csv(index=False),"transaction_assessment.csv","text/csv",width="stretch")
        st.download_button("Download blank CSV schema",",".join(FEATURES)+"\n","transaction_schema.csv","text/csv",width="stretch")
    with many:
        section("Score a batch","Up to 20,000 feature rows per file.");upload=st.file_uploader("Batch transaction CSV",type=["csv"],key="batch")
        if upload is None:st.info("Choose a feature-only CSV to begin.")
        else:
            digest=hashlib.sha256(upload.getvalue()).hexdigest()
            if st.button("Score batch",key="score_batch"):
                try:
                    result=score(validate(upload),threshold);st.session_state.batch_result=(digest,result);log_activity("Batch assessment",result)
                except ValueError as e:st.error(str(e))
                except Exception:st.error("The model could not score this file. Check its feature format and retry.")
            saved=st.session_state.get("batch_result")
            if saved and saved[0]==digest:
                result=saved[1];cols=st.columns(3)
                with cols[0]:metric("Rows scored",f"{len(result):,}","This upload")
                with cols[1]:metric("Review",f"{int((result.Decision=='Review').sum()):,}","At threshold")
                with cols[2]:metric("Clear",f"{int((result.Decision=='Clear').sum()):,}","Below threshold")
                view=result.copy();view["Amount"]=view.Amount.map(lambda z:"USD "+f"{z:,.2f}");view["Fraud probability"]=view["Fraud probability"].map(lambda z:f"{z:.1%}");st.dataframe(view,hide_index=True,width="stretch",height=350)
                st.download_button("Download scored results",result.to_csv(index=False),"scored_transactions.csv","text/csv",width="stretch")

elif page=="Model evaluation":
    st.title("Model evaluation")
    st.markdown('<div class="hero">Held-out evaluation. The threshold was selected on validation data before test metrics were calculated.</div>',unsafe_allow_html=True)
    st.markdown('<hr class="rule">',unsafe_allow_html=True);test=info["test"];cols=st.columns(4)
    stats=[("Precision",test["precision"],"Review alerts labeled fraud"),("Recall",test["recall"],"Fraud cases found"),("F1 score",test["f1"],"Precision and recall balance"),("PR-AUC",test["pr_auc"],"Ranking on imbalanced labels")]
    for col,(name,value,note) in zip(cols,stats):
        with col:metric(name,f"{value:.3f}",note)
    l,r=st.columns(2,gap="large")
    with l:
        cm=test["confusion_matrix"];section("Confusion matrix",f"Test rows at threshold {info['decision_threshold']:.4f}.");matrix=np.array([[cm["tn"],cm["fp"]],[cm["fn"],cm["tp"]]])
        fig=go.Figure(go.Heatmap(z=matrix,x=["Predicted clear","Predicted review"],y=["Actual legitimate","Actual fraud"],colorscale=[[0,"#20292b"],[.55,"#556f60"],[1,"#a9d7b8"]],text=matrix,texttemplate="%{text:,}",showscale=False));st.plotly_chart(plot(fig,280),width="stretch",config={"displayModeBar":False})
    with r:
        section("Experiment configuration","Training data separated before fitting.")
        rows=[("Classifier",info["model"]),("Engine",info["model_engine"]),("Trees",info["trees"]),("Features",info["feature_count"]),("Training rows",f"{info['training_rows']:,}"),("Validation rows",f"{info['validation_rows']:,}"),("Test rows",f"{test['test_transactions']:,}"),("Split","Stratified 70 / 15 / 15"),("Class weight",info["class_weight"]),("Fit time",f"{info['training_seconds']:.1f} seconds")]
        st.dataframe(pd.DataFrame(rows,columns=["Setting","Value"]),hide_index=True,width="stretch")
    section("Interpretation","Positive labels are rare, so use multiple measures.")
    st.markdown(f"Test split: **{test['test_transactions']:,}** rows, including **{test['fraud_transactions']:,}** fraud cases. Precision **{test['precision']:.1%}**, recall **{test['recall']:.1%}**, F1 **{test['f1']:.1%}**, ROC-AUC **{test['roc_auc']:.3f}**. Historical results do not establish future performance.")
    st.warning("Educational use only. Do not use this model to approve, decline, block, or make decisions about real payment activity.")

elif page=="Activity history":
    st.title("Activity history")
    st.markdown('<div class="hero">Session-only summaries of scoring actions. Events are not written to a database.</div>',unsafe_allow_html=True)
    st.markdown('<hr class="rule">',unsafe_allow_html=True);events=st.session_state.get("activity",[])
    if not events:st.info("No scoring activity yet. Run an assessment or process a simulation batch to see a summary.")
    else:
        st.dataframe(pd.DataFrame(events),hide_index=True,width="stretch")
        if st.button("Clear session history",key="clear"):st.session_state.activity=[];st.rerun()

elif page=="About":
    st.title("About Signal")
    st.markdown('<div class="hero">A capstone research prototype for fraud classification on anonymized transaction features.</div>',unsafe_allow_html=True)
    st.markdown('<hr class="rule">',unsafe_allow_html=True);section("Project objective","Demonstrate a detection workflow and document its limits.")
    st.markdown("The app loads a trained Random Forest, reports held-out metrics, scores feature-only CSV files, and replays a synthetic sample. The outline describes Kafka and Spark Structured Streaming as future components; neither is connected here.")
    a,b=st.columns(2)
    with a:section("Model inputs","Time, Amount, and 28 PCA values.");st.markdown("V1 through V28 are anonymized components from the supplied dataset, not ordinary business attributes.")
    with b:section("Prototype limits","Research and demonstration only.");st.markdown("No bank, payment processor, event broker, production database, user accounts, or persistent customer records are connected.")
    section("Model card");st.markdown(f"**Classifier:** {info['model']}  \n**Training rows:** {info['training_rows']:,}  \n**Dataset:** {info['dataset']}  \n**Threshold:** {info['decision_threshold']:.4f}, selected on validation F1.")

elif page=="Contact":
    st.title("Contact and project source")
    st.markdown('<div class="hero">Review the implementation or raise a project issue through the public repository.</div>',unsafe_allow_html=True)
    st.markdown('<hr class="rule">',unsafe_allow_html=True);section("Questions and corrections","Do not include private transaction or account information.")
    st.link_button("Open project repository",REPO,width="stretch");st.link_button("Report an issue",REPO+"/issues/new",width="stretch")

elif page=="Privacy policy":
    st.title("Privacy policy");st.markdown('<div class="hero">How this hosted research prototype handles information.</div>',unsafe_allow_html=True);st.markdown('<hr class="rule">',unsafe_allow_html=True)
    section("Information processed")
    st.markdown("Scoring accepts CSV uploads with Time, Amount, and anonymized V1 through V28 values. The app validates and scores these values in the active session. Do not upload names, payment card numbers, account IDs, or other identifiers.")
    section("Storage and retention")
    st.markdown("This app has no account system or database and does not write uploads to the project repository. Uploaded files and results may remain in active session memory for display. Streamlit Community Cloud may process technical information needed to host the service under its own policies.")
    section("Sharing")
    st.markdown("Transaction rows are not sent to an external scoring API. This site and source code are public. Do not submit confidential data. Contact the project through its repository without including private information.")
    st.caption("Last updated: October 11, 2026.")

elif page=="Terms and conditions":
    st.title("Terms and conditions");st.markdown('<div class="hero">Conditions for using this capstone prototype.</div>',unsafe_allow_html=True);st.markdown('<hr class="rule">',unsafe_allow_html=True)
    section("Educational use");st.markdown("Signal is an academic demonstration. It is not a financial product, payment service, fraud guarantee, or replacement for professional controls.")
    section("Model outputs");st.markdown("Probabilities and review labels are estimates and may be wrong or unsuitable for new data. Do not use them to approve or decline payments, restrict accounts, or make decisions about a person.")
    section("Uploads");st.markdown("Only upload feature-only CSV files you are authorized to use. Remove direct personal identifiers first. You are responsible for checking file contents and exported results.")
    section("Availability");st.markdown("This educational service may change or become unavailable. No service-level commitment or warranty is offered.")
    st.caption("Last updated: October 11, 2026.")

st.markdown('<div class="footer">SIGNAL / CAPSTONE RESEARCH PROTOTYPE<br>Historical results do not guarantee future detection.</div>',unsafe_allow_html=True)
_,privacy,terms=st.columns([4,1,1])
with privacy:
    if st.button("Privacy",key="footer_privacy"):navigate("Privacy policy")
with terms:
    if st.button("Terms",key="footer_terms"):navigate("Terms and conditions")
