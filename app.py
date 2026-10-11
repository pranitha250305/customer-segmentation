"""Streamlit interface for beginner-friendly customer segmentation."""

from __future__ import annotations

import inspect
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from src.clustering import calculate_elbow, cluster_customers
from src.preprocessing import default_feature_columns, prepare_features
from src.visualization import cluster_profiles, plot_cluster_sizes, plot_clusters, plot_elbow


PROJECT_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = PROJECT_DIR / "data" / "sample_customers.csv"

st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="wide")

with st.sidebar:
    st.markdown(
        "<div class='brand-mark'><span class='brand-icon'>CS</span><div><strong>SEGMENT</strong><small>INSIGHTS STUDIO</small></div></div>",
        unsafe_allow_html=True,
    )
    theme_mode = st.radio("Dashboard appearance", ["Light", "Dark"], horizontal=True, key="theme_mode")

is_dark = theme_mode == "Dark"
if is_dark:
    page_bg = "#15172a"
    sidebar_bg = "#1b1e35"
    card_bg = "#222640"
    text_color = "#f5efff"
    muted_color = "#c2bad6"
    border_color = "#39314f"
    accent_color = "#ec6dcc"
else:
    page_bg = "#fffaff"
    sidebar_bg = "#f5e9fb"
    card_bg = "#ffffff"
    text_color = "#30233e"
    muted_color = "#74657f"
    border_color = "#f0dcef"
    accent_color = "#bd35c8"

st.markdown(
    f"""
    <style>
      :root {{
        --page-bg: {page_bg}; --sidebar-bg: {sidebar_bg}; --card-bg: {card_bg};
        --text-color: {text_color}; --muted-color: {muted_color};
        --border-color: {border_color}; --accent-color: {accent_color};
      }}
      [data-testid="stAppViewContainer"] {{
        background: radial-gradient(ellipse at 85% 4%, rgba(223, 116, 225, .12), transparent 27%),
                    radial-gradient(ellipse at 5% 55%, rgba(241, 142, 206, .08), transparent 25%), var(--page-bg);
        color: var(--text-color);
      }}
      [data-testid="stHeader"] {{ background: transparent; }}
      [data-testid="stSidebar"] {{
        background: linear-gradient(165deg, var(--sidebar-bg), color-mix(in srgb, var(--sidebar-bg) 88%, #e99add));
        border-right: 1px solid var(--border-color);
      }}
      [data-testid="stMainBlockContainer"] {{ max-width: 1500px; padding-top: 1.6rem; padding-bottom: 3rem; }}
      .brand-mark {{ display:flex; align-items:center; gap:.7rem; margin: .25rem 0 1.3rem; padding:.65rem .7rem; }}
      .brand-icon {{ display:grid; place-items:center; width:2.5rem; height:2.5rem; border-radius: .9rem; color:#fff; font-weight:800; background:linear-gradient(145deg,#8144d2,#ec52b4); box-shadow:0 5px 16px rgba(190,53,200,.3); }}
      .brand-mark strong {{ display:block; color:var(--text-color); font-size:.9rem; letter-spacing:.1em; }}
      .brand-mark small {{ display:block; color:var(--muted-color); font-size:.62rem; letter-spacing:.16em; margin-top:.12rem; }}
      .hero-card {{ position:relative; overflow:hidden; display:flex; justify-content:space-between; align-items:center; min-height:210px; margin:.1rem 0 1.35rem; padding:2rem 2.2rem; border:1px solid rgba(255,255,255,.28); border-radius:26px; color:white; background:linear-gradient(112deg,#532b9c 0%,#8b38b8 47%,#e24ba9 100%); box-shadow:0 18px 45px rgba(119,48,153,.23); }}
      .hero-card:after {{ content:''; position:absolute; width:260px; height:260px; right:8%; top:-135px; border:1px solid rgba(255,255,255,.22); border-radius:50%; box-shadow:0 0 0 25px rgba(255,255,255,.06),0 0 0 53px rgba(255,255,255,.04); }}
      .hero-copy {{ position:relative; z-index:1; max-width:760px; }}
      .hero-kicker {{ color:#f8dafa; font-size:.72rem; font-weight:750; letter-spacing:.2em; text-transform:uppercase; }}
      .hero-card h1 {{ margin:.55rem 0 .45rem; color:#fff !important; font-size:clamp(1.8rem,3.3vw,2.7rem); line-height:1.12; font-weight:780; letter-spacing:-.04em; }}
      .hero-card p {{ margin:0; color:#f8eaff !important; font-size:1rem; max-width:700px; line-height:1.6; }}
      .hero-tags {{ display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1rem; }}
      .hero-tags span {{ padding:.35rem .72rem; border:1px solid rgba(255,255,255,.28); border-radius:999px; color:white; background:rgba(255,255,255,.13); font-size:.74rem; }}
      .hero-art {{ position:relative; z-index:1; display:grid; place-items:center; width:118px; height:118px; margin:0 3.5rem 0 1rem; border:1px solid rgba(255,255,255,.3); border-radius:36px; background:linear-gradient(145deg,rgba(255,255,255,.27),rgba(255,255,255,.08)); transform:rotate(7deg); box-shadow:0 14px 34px rgba(43,15,72,.18); }}
      .hero-art span {{ font-size:3.1rem; transform:rotate(-7deg); filter:drop-shadow(0 6px 7px rgba(40,10,65,.25)); }}
      .section-heading {{ margin:1.75rem 0 .8rem; }}
      .score-card {{ background:var(--card-bg); border:1px solid var(--border-color); border-radius:16px; padding:.9rem 1rem; box-shadow:0 5px 18px rgba(115,42,139,.07); }}
      .score-eyebrow {{ color:var(--accent-color); font-size:.6rem; font-weight:800; letter-spacing:.15em; text-transform:uppercase; }}
      .score-heading {{ color:var(--text-color); font-size:1rem; font-weight:700; margin:.15rem 0 .2rem; letter-spacing:-.015em; }}
      .score-value {{ color:var(--text-color); font-size:1.65rem; line-height:1.15; font-weight:750; margin:.25rem 0 .35rem; }}
      .score-description {{ color:var(--muted-color) !important; font-size:.74rem !important; line-height:1.4 !important; margin:.25rem 0 0 !important; }}
      .section-heading span {{ color:var(--accent-color); font-size:.68rem; font-weight:800; letter-spacing:.18em; text-transform:uppercase; }}
      .section-heading h2 {{ color:var(--text-color); font-size:1.45rem; font-weight:740; margin:.18rem 0 .15rem; letter-spacing:-.025em; }}
      .section-heading p {{ color:var(--muted-color); margin:0; font-size:.9rem; }}
      [data-testid="stAppViewContainer"] h2, [data-testid="stAppViewContainer"] h3 {{ color: var(--text-color); }}
      [data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] label {{ color: var(--text-color); }}
      [data-testid="stCaptionContainer"] {{ color: var(--muted-color); }}
      [data-testid="stMetric"] {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 18px; padding: 1rem 1.2rem; box-shadow: 0 5px 20px rgba(115, 42, 139, .08); }}
      [data-testid="stMetricLabel"] {{ color: var(--muted-color); }}
      [data-testid="stMetricValue"] {{ color: var(--text-color); font-weight: 700; }}
      [data-testid="stDataFrame"], [data-testid="stTable"] {{ border: 1px solid var(--border-color); border-radius: 14px; overflow: hidden; }}
      [data-testid="stExpander"] {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 14px; }}
      [data-testid="stAlert"] {{ border-radius: 14px; }}
      [data-testid="stRadio"] input, [data-testid="stCheckbox"] input {{ accent-color:#d946c7; }}
      [data-testid="stRadio"] label {{ border-radius:10px; }}
      [data-testid="stSelectbox"] [data-baseweb="select"] > div, [data-testid="stMultiSelect"] [data-baseweb="select"] > div {{ background:var(--card-bg); border-color:var(--border-color); border-radius:12px; }}
      [data-testid="stSlider"] [role="slider"] {{ background:#d946c7; border-color:#fff; box-shadow:0 0 0 3px rgba(217,70,199,.18); }}
      [data-testid="stPlotlyChart"], [data-testid="stImage"] {{ border-radius:18px; }}
      [data-testid="stButton"] button, [data-testid="stDownloadButton"] button {{
        border: 0; border-radius: 999px; color: white; font-weight: 650;
        background: linear-gradient(100deg, #8b42d2, #db43b2);
        box-shadow: 0 5px 15px rgba(177, 49, 186, .2); transition: transform .15s ease, box-shadow .15s ease;
      }}
      [data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover {{
        color: white; transform: translateY(-1px); box-shadow: 0 8px 20px rgba(177, 49, 186, .3);
      }}
      [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {{
        background: var(--card-bg); color: var(--text-color); border-color: var(--border-color); border-radius: 10px;
      }}
      [data-testid="stSidebar"] [data-testid="stRadio"] label {{ color: var(--text-color); }}
      a {{ color: var(--accent-color); }}
      @media (max-width: 720px) {{ .hero-card {{ padding:1.5rem; min-height:0; }} .hero-art {{ display:none; }} }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


def read_upload(uploaded_file) -> pd.DataFrame:
    try:
        frame = pd.read_csv(uploaded_file)
    except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as error:
        raise ValueError(f"That file could not be read as a CSV: {error}") from error
    if frame.empty:
        raise ValueError("The CSV has headers but no customer rows.")
    if len(frame.columns) == 0:
        raise ValueError("The CSV does not contain any columns.")
    return frame


def data_signature(data: pd.DataFrame, features: list[str], k: int) -> tuple:
    row_hash = int(pd.util.hash_pandas_object(data, index=True).sum())
    return row_hash, tuple(features), k


def themed_plot(plot_function, *args):
    """Pass the selected theme when supported; tolerate a cached older module."""
    if "theme" in inspect.signature(plot_function).parameters:
        return plot_function(*args, theme=theme_mode.lower())
    return plot_function(*args)


st.markdown(
    """
    <div class="hero-card">
      <div class="hero-copy">
        <div class="hero-kicker">Customer analytics · Unsupervised learning</div>
        <h1>Customer Segmentation</h1>
        <p>Find meaningful customer groups, understand what makes each one distinct, and turn your data into clear, visual insights.</p>
        <div class="hero-tags"><span>✦ K-Means clustering</span><span>◉ Interactive insights</span><span>↧ CSV export</span></div>
      </div>
      <div class="hero-art"><span>✣</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.info("Cluster numbers (0, 1, 2, …) are arbitrary labels. They do not rank customers.")

with st.sidebar:
    st.header("1 · Choose customer data")
    source = st.radio("Data source", ["Built-in sample", "Upload a CSV"], index=0)
    uploaded = st.file_uploader("Choose a customer CSV", type=["csv"]) if source == "Upload a CSV" else None

try:
    if source == "Built-in sample":
        data = load_sample()
        st.caption("The built-in example is deterministic synthetic data created for this project.")
    elif uploaded is not None:
        data = read_upload(uploaded)
    else:
        st.info("Upload a CSV in the sidebar, or switch to the built-in sample to explore the app.")
        st.stop()
except (ValueError, OSError) as error:
    st.error(str(error))
    st.stop()

st.markdown("<div class='section-heading'><span>01 · DATA OVERVIEW</span><h2>Explore your customers</h2><p>Start by checking the shape and quality of the data you will analyze.</p></div>", unsafe_allow_html=True)
metric1, metric2 = st.columns(2)
metric1.metric("Customer rows", f"{len(data):,}")
metric2.metric("Columns", f"{len(data.columns):,}")
st.dataframe(data.head(20), use_container_width=True, hide_index=True)

with st.expander("Column types, missing values, and descriptive statistics"):
    overview = pd.DataFrame({
        "Data type": data.dtypes.astype(str),
        "Missing values": data.isna().sum(),
    })
    st.markdown("**Columns and missing values**")
    st.dataframe(overview, use_container_width=True)
    st.markdown("**Descriptive statistics**")
    st.dataframe(data.describe(include="all").transpose(), use_container_width=True)

numeric_candidates = data.select_dtypes(include="number").columns.tolist()
default_features = default_feature_columns(data)
if not numeric_candidates:
    st.error("This dataset has no numerical columns. Add numerical customer attributes, then upload it again.")
    st.stop()

st.markdown("<div class='section-heading'><span>02 · FEATURE PREPARATION</span><h2>Choose customer attributes</h2><p>Pick the measurements that should define similarity between customers.</p></div>", unsafe_allow_html=True)
st.write(
    "Select numerical customer attributes that could explain behaviour. Identifier columns such as "
    "CustomerID are excluded by default because their numbers identify rows, not customer similarity. "
    "Missing or infinite feature values are replaced with that feature's median."
)
features = st.multiselect(
    "Numerical features for clustering", options=numeric_candidates,
    default=[column for column in default_features if column in numeric_candidates],
)
if not features:
    st.warning("Select at least one numerical feature to continue.")
    st.stop()

try:
    scaled, scaler, cleaned_features = prepare_features(data, features)
except ValueError as error:
    st.error(str(error))
    st.stop()

if cleaned_features.isna().any().any():  # Defensive check: preparation should leave no missing values.
    st.error("Some selected features still contain missing values after preparation.")
    st.stop()

unique_points = len(pd.DataFrame(scaled).drop_duplicates())
max_k = min(10, len(data) - 1, unique_points)
if max_k < 2:
    st.error("At least three rows with two distinct feature patterns are needed for K-Means.")
    st.stop()

with st.sidebar:
    st.header("2 · Set clustering options")
    n_clusters = st.slider("Number of clusters (K)", min_value=2, max_value=max_k, value=min(5, max_k))
    st.caption("K is the number of groups K-Means will try to find.")

st.caption(
    f"StandardScaler puts each selected feature on a comparable scale. "
    f"It is fitted to these {len(data):,} rows; median imputation keeps customer rows in the analysis."
)
run = st.button("Run K-Means clustering", type="primary")
if run:
    try:
        with st.spinner("Preparing the elbow curve and finding customer groups…"):
            elbow = calculate_elbow(scaled, max_k=max_k)
            model, labelled, evaluation = cluster_customers(data, scaled, n_clusters)
            analysis_data = labelled.copy()
            # Keep source columns as-is in the download. Use cleaned feature values in analysis views.
            for feature in features:
                analysis_data[feature] = cleaned_features[feature]
        st.session_state["segmentation_result"] = {
            "signature": data_signature(data, features, n_clusters),
            "elbow": elbow,
            "model": model,
            "labelled": labelled,
            "analysis": analysis_data,
            "evaluation": evaluation,
        }
    except ValueError as error:
        st.error(str(error))

result = st.session_state.get("segmentation_result")
if not result or result["signature"] != data_signature(data, features, n_clusters):
    st.caption("Run clustering to see the elbow chart, cluster groups, and profiles.")
    st.stop()

st.markdown("<div class='section-heading'><span>03 · FIND A USEFUL K</span><h2>Elbow method</h2></div>", unsafe_allow_html=True)
st.write(
    "The chart shows how within-group variation (inertia) changes for each valid K. Look for a bend "
    "where improvement starts to slow. The elbow is an estimate, not a guaranteed perfect answer."
)
st.pyplot(themed_plot(plot_elbow, result["elbow"]), use_container_width=True)

st.markdown("<div class='section-heading'><span>04 · SEGMENT DISCOVERY</span><h2>Explore the customer groups</h2></div>", unsafe_allow_html=True)
if len(features) == 1:
    st.info("Choose a second feature to display a 2D scatter plot. Clustering can still use the selected feature set.")
else:
    axis1, axis2 = st.columns(2)
    with axis1:
        x_feature = st.selectbox("Horizontal axis", features, index=0)
    with axis2:
        y_feature = st.selectbox("Vertical axis", features, index=1 if len(features) > 1 else 0)
    if x_feature == y_feature:
        st.warning("Choose different features for the two axes to make the scatter plot easier to read.")
    else:
        st.pyplot(themed_plot(plot_clusters, result["analysis"], x_feature, y_feature), use_container_width=True)

size_col, score_col = st.columns([1.65, 0.9])
with size_col:
    st.pyplot(themed_plot(plot_cluster_sizes, result["labelled"]), use_container_width=True)
with score_col:
    score = result["evaluation"]["silhouette_score"]
    if score is None:
        note = escape(result["evaluation"]["silhouette_note"] or "A score is unavailable for this result.")
        st.markdown(
            f"<div class='score-card'><div class='score-eyebrow'>Cluster quality</div>"
            f"<div class='score-heading'>Silhouette score</div>"
            f"<p class='score-description'>{note}</p></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='score-card'><div class='score-eyebrow'>Cluster quality</div>"
            f"<div class='score-heading'>Silhouette score</div>"
            f"<div class='score-value'>{score:.3f}</div>"
            f"<p class='score-description'>Ranges from −1 to +1. Higher usually means customers fit their own group better and are more separated from others. It is not accuracy.</p></div>",
            unsafe_allow_html=True,
        )

st.markdown("<div class='section-heading'><span>05 · CUSTOMER PROFILES</span><h2>What makes each cluster different?</h2><p>Compare the groups using averages calculated from your selected features.</p></div>", unsafe_allow_html=True)
profiles, descriptions = cluster_profiles(result["analysis"], features)
st.dataframe(profiles.round(2), use_container_width=True)
st.caption("Feature averages use the cleaned values used in analysis. Each profile is calculated from this run's actual clusters.")
for cluster, description in descriptions.items():
    st.write(f"**Cluster {cluster}:** {description}")

st.markdown("<div class='section-heading'><span>06 · TAKE YOUR RESULTS</span><h2>Download the clustered data</h2></div>", unsafe_allow_html=True)
csv_bytes = result["labelled"].to_csv(index=False).encode("utf-8")
st.download_button(
    "Download clustered dataset (CSV)", data=csv_bytes,
    file_name="customer_segments.csv", mime="text/csv",
)
