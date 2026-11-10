"""
app.py  -  Film Revenue & ROI Forecast (Streamlit)
Run from the project root:  streamlit run app.py
Needs: models/rf_model.joblib, models/tier_cutoffs.json, data/processed/titles.csv
(created by src/clean.py and src/train.py)
Optional: .streamlit/config.toml for the matching light theme.
"""
import html as _html
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.charts import (bubble_chart, breakeven_chart, find_break_even,  # noqa: E402
                        genre_roi_chart, genre_trend_chart)

st.set_page_config(
    page_title="Film Revenue & ROI Forecast", page_icon="🎬", layout="wide",
    menu_items={"About": "Film Revenue & ROI Forecast: a decision-support prototype trained on TMDB films."},
)

# ---------------- design tokens ----------------
INK, MUTED, ACCENT, ACCENT_DARK, LINE = "#17243b", "#64748b", "#bd583b", "#933d2c", "#e4e9f1"
GREEN, RED, AMBER, SLATE = "#1f9d6b", "#c8503f", "#d9a441", "#8b9bb4"
PALETTE = ["#bd583b", "#17243b", "#d9a441", "#3b82a8", "#6a994e", "#8e6c8a",
           "#e07a5f", "#5c6b8a", "#2a9d8f", "#b5838d", "#9c6644", "#4a6fa5"]

st.markdown(
    """
    <style>
    :root {
        --ink: #17243b;
        --muted: #64748b;
        --accent: #bd583b;
        --accent-dark: #933d2c;
        --line: #e4e9f1;
        --surface: #ffffff;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 82% 0%, rgba(233, 202, 177, 0.22), transparent 34rem),
            #f5f6f8;
        color: var(--ink);
    }
    [data-testid="stHeader"] {background: transparent;}
    footer {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1320px;}
    h1, h2, h3 {color: var(--ink); letter-spacing: -0.035em;}
    h2 {font-weight: 700 !important;}
    h3 {font-weight: 700 !important; font-size: 1.28rem !important;}
    p, [data-testid="stCaptionContainer"] {color: var(--muted);}

    /* ---------- hero ---------- */
    .hero {
        position: relative; overflow: hidden;
        display: flex; flex-wrap: wrap; gap: 1.6rem;
        align-items: flex-end; justify-content: space-between;
        padding: 2.1rem 2.3rem; margin-bottom: 1rem; border-radius: 22px;
        background:
            radial-gradient(circle at 90% 8%, rgba(217, 119, 69, 0.55), transparent 22rem),
            linear-gradient(125deg, #17243b 0%, #22345a 62%, #2c3e66 100%);
        box-shadow: 0 18px 44px rgba(23, 36, 59, 0.22);
    }
    .hero-title {
        margin: 0.2rem 0 0; color: #ffffff !important; font-weight: 780;
        font-size: clamp(2rem, 3.6vw, 3rem); letter-spacing: -0.035em; line-height: 1.08;
    }
    .hero-sub {
        margin: 0.75rem 0 0; max-width: 46rem; font-size: 1.02rem; line-height: 1.55;
        color: rgba(255, 255, 255, 0.82) !important;
    }
    .hero-chips {display: flex; gap: 0.7rem; flex-wrap: wrap;}
    .hero-chip {
        min-width: 7.2rem; padding: 0.7rem 1rem; border-radius: 14px;
        background: rgba(255, 255, 255, 0.10); border: 1px solid rgba(255, 255, 255, 0.16);
        backdrop-filter: blur(6px);
    }
    .hero-chip b {display: block; color: #ffffff; font-size: 1.35rem; font-weight: 760; letter-spacing: -0.02em;}
    .hero-chip span {
        color: rgba(255, 255, 255, 0.72); font-size: 0.7rem; font-weight: 650;
        letter-spacing: 0.09em; text-transform: uppercase;
    }
    .studio-kicker {
        display: inline-flex; align-items: center; gap: 0.55rem;
        padding: 0.38rem 0.72rem; border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.2); background: rgba(255, 255, 255, 0.12);
        color: #f6d9c9; font-size: 0.72rem; font-weight: 750;
        letter-spacing: 0.12em; text-transform: uppercase;
    }
    .studio-kicker span {
        width: 0.45rem; height: 0.45rem; border-radius: 50%; background: #f0a06f;
        box-shadow: 0 0 0 3px rgba(240, 160, 111, 0.22);
    }

    /* ---------- how-to steps ---------- */
    .steps {display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8rem; margin: 0 0 1.2rem;}
    .step {
        display: flex; gap: 0.8rem; align-items: flex-start; padding: 0.9rem 1rem;
        border: 1px solid var(--line); border-radius: 14px; background: rgba(255, 255, 255, 0.78);
    }
    .step-n {
        flex: none; width: 1.7rem; height: 1.7rem; border-radius: 50%; display: grid; place-items: center;
        background: rgba(189, 88, 59, 0.12); color: var(--accent-dark); font-weight: 760; font-size: 0.85rem;
    }
    .step b {display: block; color: var(--ink); font-size: 0.92rem; margin-bottom: 0.1rem;}
    .step span {color: var(--muted); font-size: 0.82rem; line-height: 1.4;}

    /* ---------- input pills ---------- */
    .pill-row {display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 0.1rem 0 0.9rem;}
    .pill {
        padding: 0.28rem 0.72rem; border-radius: 999px; border: 1px solid var(--line);
        background: #ffffff; color: var(--ink); font-size: 0.8rem; font-weight: 620;
    }

    /* ---------- verdict ---------- */
    .verdict {
        display: flex; gap: 1rem; align-items: flex-start; margin-bottom: 1rem;
        padding: 1.15rem 1.3rem; border-radius: 16px; background: #ffffff;
        border: 1px solid var(--line); border-left-width: 6px;
        box-shadow: 0 8px 24px rgba(31, 47, 73, 0.055);
    }
    .verdict .badge {
        flex: none; padding: 0.3rem 0.8rem; border-radius: 999px; color: #ffffff; white-space: nowrap;
        font-size: 0.72rem; font-weight: 760; letter-spacing: 0.07em; text-transform: uppercase;
    }
    .verdict p {margin: 0; color: var(--ink); font-size: 1.02rem; line-height: 1.62;}
    .v-success {border-left-color: #1f9d6b;} .v-success .badge {background: #1f9d6b;}
    .v-info {border-left-color: #3b82a8;} .v-info .badge {background: #3b82a8;}
    .v-warning {border-left-color: #d9a441;} .v-warning .badge {background: #c4891f;}
    .v-error {border-left-color: #c8503f;} .v-error .badge {background: #c8503f;}

    /* ---------- KPI cards ---------- */
    .kpi-grid {display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.9rem; margin-bottom: 0.8rem;}
    .kpi-grid.three {grid-template-columns: repeat(3, 1fr);}
    .kpi {
        position: relative; overflow: hidden; padding: 1.1rem 1.2rem;
        border: 1px solid var(--line); border-radius: 16px;
        background: linear-gradient(145deg, #fff 0%, #fbfcfe 100%);
        box-shadow: 0 8px 24px rgba(31, 47, 73, 0.055);
        transition: transform 160ms ease, box-shadow 160ms ease;
    }
    .kpi::before {content: ""; position: absolute; inset: 0 auto 0 0; width: 4px; background: var(--accent);}
    .kpi.pos::before {background: #1f9d6b;}
    .kpi.neg::before {background: #c8503f;}
    .kpi:hover {transform: translateY(-2px); box-shadow: 0 12px 28px rgba(31, 47, 73, 0.1);}
    .kpi-top {
        display: flex; align-items: center; gap: 0.5rem; color: var(--muted);
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.065em; text-transform: uppercase;
    }
    .kpi-value {
        margin: 0.35rem 0 0.15rem; color: var(--ink); font-weight: 770; line-height: 1.1;
        font-size: clamp(1.5rem, 2.2vw, 2.05rem); letter-spacing: -0.03em;
    }
    .kpi.pos .kpi-value {color: #177a54;}
    .kpi.neg .kpi-value {color: #b13f2f;}
    .kpi-sub {color: var(--muted); font-size: 0.8rem;}

    /* ---------- sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(165deg, #17243b 0%, #202f49 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #eef2f8;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color: #b6c2d4;}
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #e4eaf3; font-size: 0.86rem; font-weight: 600;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="textarea"] > div {
        border-color: rgba(255, 255, 255, 0.15); border-radius: 10px; background: rgba(255, 255, 255, 0.96);
    }
    [data-testid="stSidebar"] input {border-radius: 10px;}
    [data-testid="stSlider"] [role="slider"] {box-shadow: 0 0 0 4px rgba(217, 119, 69, 0.12);}

    /* ---------- buttons ---------- */
    button[kind="primary"] {
        border: 0; border-radius: 10px; color: white; font-weight: 700;
        background: linear-gradient(110deg, #bd583b, #d17a4d);
        transition: transform 140ms ease, box-shadow 140ms ease;
    }
    button[kind="primary"]:hover {transform: translateY(-1px); box-shadow: 0 8px 18px rgba(189, 88, 59, 0.32);}
    [data-testid="stDownloadButton"] button, button[kind="secondary"] {
        border: 1px solid rgba(189, 88, 59, 0.3); border-radius: 10px; color: var(--accent-dark);
        font-weight: 650; transition: background 150ms ease, border-color 150ms ease;
    }
    [data-testid="stDownloadButton"] button:hover, button[kind="secondary"]:hover {
        border-color: var(--accent); background: rgba(189, 88, 59, 0.07); color: var(--accent-dark);
    }
    [data-testid="stSidebar"] button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.08); color: #f3d9cb; border-color: rgba(255, 255, 255, 0.22);
    }

    /* ---------- tabs, tables, charts ---------- */
    div[data-testid="stTabs"] [role="tablist"] {gap: 0.35rem; border-bottom: 1px solid var(--line);}
    div[data-testid="stTabs"] button[role="tab"] {
        border-radius: 10px 10px 0 0; color: #64748b; font-weight: 650; padding: 0.6rem 1rem;
    }
    div[data-testid="stTabs"] button[role="tab"]:hover {background: rgba(189, 88, 59, 0.05);}
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--accent-dark); background: rgba(189, 88, 59, 0.07);
    }
    div[data-testid="stAlert"] {
        border-radius: 14px; border: 1px solid var(--line); box-shadow: 0 5px 16px rgba(31, 47, 73, 0.035);
    }
    div[data-testid="stDataFrame"] {overflow: hidden; border: 1px solid var(--line); border-radius: 14px;}
    div[data-testid="stPlotlyChart"] {
        overflow: hidden; border: 1px solid rgba(228, 233, 241, 0.75); border-radius: 15px;
        background: rgba(255, 255, 255, 0.72);
    }
    [data-testid="stExpander"] {
        border: 1px solid var(--line); border-radius: 14px; background: rgba(255, 255, 255, 0.7);
    }

    /* ---------- footer ---------- */
    .site-footer {
        display: flex; justify-content: space-between; flex-wrap: wrap; gap: 0.6rem;
        margin-top: 2rem; padding: 1.1rem 0 0.4rem; border-top: 1px solid var(--line);
        color: var(--muted); font-size: 0.8rem;
    }

    @media (max-width: 1000px) {.kpi-grid {grid-template-columns: repeat(2, 1fr);}}
    @media (max-width: 800px) {.steps {grid-template-columns: 1fr;} .hero {padding: 1.5rem 1.3rem;}}
    @media (max-width: 560px) {.kpi-grid, .kpi-grid.three {grid-template-columns: 1fr;}}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- small helpers ----------------
def usd(x, d=0):
    """Dollar text that is safe inside Streamlit markdown (escapes the $ so it isn't read as LaTeX)."""
    return f"\\${x:,.{d}f}M"


def H(markup):
    """Render custom HTML. Collapses whitespace (so markdown never treats it as a code block)
    and escapes $ (so it never triggers LaTeX)."""
    flat = " ".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(flat.replace("$", "&#36;"), unsafe_allow_html=True)


def esc(x):
    return _html.escape(str(x), quote=True)


def show_plot(fig, toolbar=False):
    cfg = {"displayModeBar": toolbar, "displaylogo": False}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg)
    except Exception:                       # older Streamlit versions
        st.plotly_chart(fig, use_container_width=True, config=cfg)


def show_table(frame, column_config=None):
    kw = {"hide_index": True}
    if column_config:
        kw["column_config"] = column_config
    try:
        st.dataframe(frame, width="stretch", **kw)
    except Exception:
        st.dataframe(frame, use_container_width=True, **kw)


def style_fig(fig, height=None):
    """Give every Plotly figure the same look."""
    fig.update_layout(
        font=dict(family="Inter, 'Segoe UI', system-ui, sans-serif", size=13, color=INK),
        colorway=PALETTE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=14, r=18, t=32, b=14),
        hoverlabel=dict(bgcolor="white", bordercolor=LINE, font=dict(color=INK, size=13)),
    )
    fig.update_xaxes(gridcolor="#e9edf4", zeroline=False, linecolor=LINE)
    fig.update_yaxes(gridcolor="#e9edf4", zeroline=False, linecolor=LINE)
    if height:
        fig.update_layout(height=height)
    return fig


def kpi_card(icon, label, value, sub, tone="", tip=""):
    return (f'<div class="kpi {tone}" title="{esc(tip)}"><div class="kpi-top"><span>{icon}</span>'
            f'<span>{label}</span></div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>')


# ---------------- loaders (cached) ----------------
@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "rf_model.joblib")


@st.cache_data
def load_data():
    d = pd.read_csv(ROOT / "data" / "processed" / "titles.csv")
    d["text"] = d["text"].fillna("")
    d["genre"] = d["genre"].fillna("Unknown")
    if "genre_list" not in d.columns:
        d["genre_list"] = d["genre"]
    d["genre_list"] = d["genre_list"].fillna("").replace("", "Unknown")
    return d


@st.cache_data
def load_json(name):
    p = ROOT / "models" / name
    return json.loads(p.read_text()) if p.exists() else {}


try:
    model, df = load_model(), load_data()
except FileNotFoundError as err:
    st.error(f"Missing file: {err.filename}. Run `python src/clean.py` and `python src/train.py` first, "
             "then restart the app.")
    st.stop()
cutoffs, metrics = load_json("tier_cutoffs.json"), load_json("metrics.json")
if not cutoffs:
    st.error("models/tier_cutoffs.json is missing. Run `python src/train.py` first.")
    st.stop()

LATEST_YEAR = int(df["year"].max())   # the model can't extrapolate in time, so we score at the latest year it knows
Y_LO = int(df["year"].min())


# ---------------- cached charts that don't depend on the sidebar ----------------
@st.cache_data(show_spinner=False)
def market_bubble(years, genres):
    d = load_data()
    d = d[d["year"].between(years[0], years[1])]
    if genres:
        d = d[d["genre"].isin(list(genres))]
    if len(d) < 3:
        return None
    return bubble_chart(d), float(d["roi"].quantile(0.97)), len(d)


@st.cache_data(show_spinner=False)
def trend_figure():
    return genre_trend_chart(load_data())


@st.cache_data(show_spinner=False)
def genre_roi_figure():
    return genre_roi_chart(load_data())


# ---------------- model helpers ----------------
def make_rows(budgets_m, runtime, quarter, star, genre_list, text):
    budgets_m = np.atleast_1d(budgets_m).astype(float)
    return pd.DataFrame({
        "log_budget": np.log(budgets_m * 1e6), "runtime": runtime, "quarter": quarter,
        "year": LATEST_YEAR, "star_power": star, "genre_list": genre_list, "text": text,
    })


def predict_revenue_m(budgets_m, **kw):
    """Predicted revenue in $M for one or many budgets (other inputs fixed)."""
    return np.exp(model.predict(make_rows(budgets_m, **kw))) / 1e6


def tree_estimates_m(rows):
    """Revenue estimate from each tree in the forest ($M): shows how much the model's trees agree."""
    pre, rf = model.named_steps["pre"], model.named_steps["rf"]
    xt = pre.transform(rows)
    if hasattr(xt, "toarray"):
        xt = xt.toarray()
    return np.exp(np.array([t.predict(xt)[0] for t in rf.estimators_])) / 1e6


def tier_for(revenue_m):
    r = revenue_m * 1e6
    if r <= cutoffs["low_max"]:
        return "Low"
    if r <= cutoffs["medium_max"]:
        return "Medium"
    if r <= cutoffs["high_max"]:
        return "High"
    return "Breakout"


TIER_BAND = {"Low": "Bottom 40% of films", "Medium": "40th to 75th percentile",
             "High": "75th to 95th percentile", "Breakout": "Top 5% of films"}


def verdict_for(roi):
    if roi >= 1.0:
        return "success", "Strong return"
    if roi >= 0.2:
        return "success", "Likely profitable"
    if roi >= 0.0:
        return "info", "Marginal"
    if roi >= -0.3:
        return "warning", "At risk"
    return "error", "Likely loss"


# ---------------- inputs: defaults, examples, session state ----------------
b_min, b_max = float(df["budget"].min() / 1e6), float(df["budget"].max() / 1e6)
B_LO, B_HI = float(np.ceil(b_min)), float(np.floor(b_max))
all_genres = sorted({g for gl in df["genre_list"] for g in gl.split("|") if g})


def _g(*names):
    got = [n for n in names if n in all_genres]
    return got or all_genres[:1]


def _b(x):
    return float(np.clip(x, B_LO, B_HI))


DEFAULTS = dict(budget=_b(np.round(df["budget"].median() / 1e6)), star=50, genres=_g("Drama"), quarter=1,
                runtime=int(np.clip(df["runtime"].median(), 70, 200)), logline="")
PRESETS = {
    "Typical film (default)": DEFAULTS,
    "Summer action blockbuster": dict(budget=_b(180), star=85, genres=_g("Action", "Adventure"),
                                      quarter=2, runtime=130, logline=""),
    "Mid-budget crime thriller": dict(budget=_b(40), star=60, genres=_g("Crime", "Thriller"),
                                      quarter=4, runtime=110, logline=""),
    "Low-budget horror": dict(budget=_b(8), star=30, genres=_g("Horror"), quarter=4, runtime=92, logline=""),
    "Family animation": dict(budget=_b(100), star=70, genres=_g("Animation", "Family"),
                             quarter=2, runtime=98, logline=""),
    "Awards-season drama": dict(budget=_b(25), star=75, genres=_g("Drama"), quarter=4, runtime=120, logline=""),
}
PRESET_NAMES = list(PRESETS)
CTX_DEFAULT_YEARS = (max(Y_LO, LATEST_YEAR - 20), LATEST_YEAR)

for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
st.session_state.setdefault("preset", PRESET_NAMES[0])
st.session_state.setdefault("lab_budget", DEFAULTS["budget"])
st.session_state.setdefault("lab_quarter", DEFAULTS["quarter"])
st.session_state.setdefault("ctx_years", CTX_DEFAULT_YEARS)
st.session_state.setdefault("ctx_genres", [])
st.session_state.setdefault("pins", [])


def _load_values(values):
    for k, v in values.items():
        st.session_state[k] = v
    st.session_state["lab_budget"] = values["budget"]
    st.session_state["lab_quarter"] = values["quarter"]


def apply_preset():
    _load_values(PRESETS[st.session_state["preset"]])


def reset_inputs():
    st.session_state["preset"] = PRESET_NAMES[0]
    _load_values(DEFAULTS)


def sync_lab():
    st.session_state["lab_budget"] = st.session_state["budget"]
    st.session_state["lab_quarter"] = st.session_state["quarter"]


def pin_scenario(payload):
    pins = list(st.session_state.get("pins", []))
    if payload not in pins:
        pins.append(payload)
    st.session_state["pins"] = pins[-6:]
    st.session_state["flash"] = "Forecast pinned. Compare it in the Scenario lab tab."


def clear_pins():
    st.session_state["pins"] = []


QUARTER_LABELS = {1: "Q1 (Jan–Mar)", 2: "Q2 (Apr–Jun)", 3: "Q3 (Jul–Sep)", 4: "Q4 (Oct–Dec)"}

st.sidebar.header("1. Describe your project")
st.sidebar.caption("Results update automatically when you change an input.")
st.sidebar.selectbox("Start from an example", PRESET_NAMES, key="preset", on_change=apply_preset,
                     help="Loads a typical set of inputs so you can see how the tool works. "
                          "Change any value afterwards.")
budget_m = st.sidebar.slider("Production budget ($M)", B_LO, B_HI, step=1.0, key="budget",
                             help="Total cost to make the film, in millions of US dollars. Marketing is not included.")
star_pct = st.sidebar.slider("Star power (percentile)", 0, 100, step=1, key="star",
                             help="How established the top-billed cast is, measured by how many films in the dataset "
                                  "they appear in. 50 = typical cast, 90 = very established names.")
genres_sel = st.sidebar.multiselect("Genres (choose up to 3)", all_genres, key="genres", max_selections=3,
                                    help="The first genre you pick is used to find similar films.")
quarter = st.sidebar.selectbox("Release quarter", [1, 2, 3, 4], key="quarter",
                               format_func=lambda q: QUARTER_LABELS[q],
                               help="Box office is seasonal: summer and holiday releases usually earn more.")
runtime = st.sidebar.slider("Runtime (minutes)", 70, 200, step=1, key="runtime")
logline = st.sidebar.text_area("Logline / synopsis (optional)", key="logline",
                               placeholder="A rogue agent must stop a conspiracy…",
                               help="One or two sentences about the story. The model reads key words, "
                                    "but this has a small effect compared with budget and genre.")
st.sidebar.button("↺ Reset all inputs", on_click=reset_inputs, key="reset_btn")

if not genres_sel:
    H('<div class="hero"><div><div class="studio-kicker"><span></span> Studio intelligence</div>'
      '<div class="hero-title">Film Revenue &amp; ROI Forecast</div>'
      '<p class="hero-sub">Choose at least one genre in the sidebar to see a forecast.</p></div></div>')
    st.stop()

flash = st.session_state.pop("flash", None)
if flash:
    try:
        st.toast(flash)
    except Exception:
        pass

genre = genres_sel[0]
genre_txt = " / ".join(genres_sel)
article = "an" if genre_txt[0].upper() in "AEIOU" else "a"
q_short = QUARTER_LABELS[quarter].split(" ")[0]
star = float(np.quantile(df["star_power"], star_pct / 100))
inputs = dict(runtime=runtime, quarter=quarter, star=star, genre_list="|".join(genres_sel), text=logline)

# ---------------- forecast ----------------
rows = make_rows(budget_m, **inputs)
rev_m = float(predict_revenue_m(budget_m, **inputs)[0])
roi = rev_m / budget_m - 1
profit_m = rev_m - budget_m
tier = tier_for(rev_m)
per_tree = tree_estimates_m(rows)
lo, hi = (float(v) for v in np.percentile(per_tree, [10, 90]))
p_cover = float((per_tree >= budget_m).mean())
kind, label = verdict_for(roi)

# ---------------- hero ----------------
cv = metrics.get("cv_r2_log_revenue")
cv_txt = f"{cv:.2f}" if cv is not None else "n/a"
H(f"""
<div class="hero">
  <div>
    <div class="studio-kicker"><span></span> Studio intelligence · box office outlook</div>
    <div class="hero-title">Film Revenue &amp; ROI Forecast</div>
    <p class="hero-sub">Estimate how much a film could earn at the box office, and whether it is likely to cover
    its budget, before it is greenlit. Trained on real films from TMDB. It forecasts box-office revenue,
    not streaming viewership.</p>
  </div>
  <div class="hero-chips">
    <div class="hero-chip"><b>{len(df):,}</b><span>Films analysed</span></div>
    <div class="hero-chip"><b>{Y_LO}–{LATEST_YEAR}</b><span>Release years</span></div>
    <div class="hero-chip"><b>{cv_txt}</b><span>Model R² (CV)</span></div>
  </div>
</div>
<div class="steps">
  <div class="step"><div class="step-n">1</div><div><b>Describe your project</b>
    <span>Use the sidebar, or start from an example.</span></div></div>
  <div class="step"><div class="step-n">2</div><div><b>Read the forecast</b>
    <span>Verdict, headline numbers and likely range.</span></div></div>
  <div class="step"><div class="step-n">3</div><div><b>Explore and compare</b>
    <span>Test scenarios, pin options, check the market.</span></div></div>
</div>
""")

# ---------------- verdict + KPI cards ----------------
pills = "".join(f'<span class="pill">{x}</span>' for x in [
    esc(genre_txt), q_short, f"${budget_m:,.0f}M budget", f"{runtime} min", f"Star power {star_pct}th pct"])
H(f'<div class="pill-row">{pills}</div>')

H(f"""
<div class="verdict v-{kind}">
  <div class="badge">{label}</div>
  <p>At a <b>${budget_m:,.0f}M</b> budget, {article} {esc(genre_txt)} film released in {q_short} is projected to earn
  about <b>${rev_m:,.0f}M</b> (typical range ${lo:,.0f}M to ${hi:,.0f}M), a <b>{roi * 100:+,.0f}%</b> return.
  In about <b>{p_cover * 100:,.0f}%</b> of the model's internal estimates, revenue covers the budget.</p>
</div>
""")

tone_roi = "pos" if roi >= 0 else "neg"
H('<div class="kpi-grid">' + "".join([
    kpi_card("💰", "Estimated revenue", f"${rev_m:,.1f}M", f"Typical range ${lo:,.0f}M to ${hi:,.0f}M", "",
             "The model's central estimate of worldwide box-office gross."),
    kpi_card("📈", "Estimated ROI", f"{roi * 100:,.0f}%", "Above break-even" if roi >= 0 else "Below break-even",
             tone_roi, "ROI = (revenue - budget) / budget. 100% means the film earns back double its budget."),
    kpi_card("🧾", "Profit / loss", f"${profit_m:,.1f}M", "Revenue minus budget, before marketing", tone_roi,
             "Estimated revenue minus production budget."),
    kpi_card("🏆", "Revenue tier", tier, TIER_BAND[tier], "",
             f"Low: under ${cutoffs['low_max'] / 1e6:,.0f}M. Medium: up to ${cutoffs['medium_max'] / 1e6:,.0f}M. "
             f"High: up to ${cutoffs['high_max'] / 1e6:,.0f}M. Breakout: above that."),
]) + "</div>")

pin_payload = {
    "Scenario": f"{genre_txt} · {q_short} · ${budget_m:,.0f}M",
    "Budget ($M)": round(budget_m, 1), "Est. revenue ($M)": round(rev_m, 1),
    "Est. ROI (%)": round(roi * 100, 0), "Est. profit / loss ($M)": round(profit_m, 1),
    "Chance of covering budget (%)": round(p_cover * 100, 0),
    "Runtime (min)": runtime, "Star power (pct)": star_pct,
}
pc1, pc2 = st.columns([1, 3])
with pc1:
    st.button("📌 Pin this forecast", on_click=pin_scenario, args=(pin_payload,), key="pin_btn", type="primary")
with pc2:
    st.caption("Box-office revenue is what tickets earned in total, not what the studio keeps, and the budget "
               "excludes marketing, so real-world profit will be lower. Pin forecasts to compare options.")

# ---------------- tabs ----------------
tab_out, tab_lab, tab_sim, tab_mkt, tab_be = st.tabs(
    ["Outlook", "Scenario lab", "Similar films", "Market context", "Break-even"])

# ===== Outlook =====
with tab_out:
    left, right = st.columns([1.7, 1])
    with left:
        st.subheader("Range of likely outcomes")
        st.caption(f"Each bar counts how many of the model's {len(per_tree)} internal estimates landed in that revenue "
                   "range. The shaded band is the typical range (10th to 90th percentile); the dashed line is your budget.")
        hist = go.Figure(go.Histogram(
            x=per_tree, nbinsx=30, marker=dict(color=ACCENT, line=dict(color="white", width=1)), opacity=0.92,
            hovertemplate="About $%{x:,.0f}M<br>%{y} estimates<extra></extra>"))
        hist.add_vrect(x0=lo, x1=hi, fillcolor="rgba(189,88,59,0.10)", line_width=0, layer="below")
        hist.add_vline(x=budget_m, line_dash="dash", line_color=INK, annotation_text="Budget",
                       annotation_position="top left")
        hist.add_vline(x=rev_m, line_color=ACCENT_DARK, line_width=2, annotation_text="Forecast",
                       annotation_position="top right")
        hist.update_xaxes(title_text="Estimated revenue ($M)",
                          range=[0, max(float(np.percentile(per_tree, 99)) * 1.05, budget_m * 1.25)])
        hist.update_yaxes(title_text="Number of estimates")
        hist.update_layout(showlegend=False, bargap=0.04)
        show_plot(style_fig(hist, 340))
    with right:
        st.subheader("Chance of covering budget")
        st.caption("Share of the model's internal estimates where revenue is at least the budget.")
        bar_color = GREEN if p_cover >= 0.6 else (AMBER if p_cover >= 0.4 else RED)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=round(p_cover * 100, 1),
            number={"suffix": "%", "font": {"size": 46, "color": INK}},
            gauge={"axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": LINE},
                   "bar": {"color": bar_color, "thickness": 0.32}, "bgcolor": "white", "borderwidth": 0,
                   "steps": [{"range": [0, 40], "color": "#f7dfd9"}, {"range": [40, 60], "color": "#f8eed4"},
                             {"range": [60, 100], "color": "#d9efe5"}]}))
        gauge.update_layout(height=300, margin=dict(l=24, r=24, t=30, b=6), paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Inter, 'Segoe UI', system-ui, sans-serif", color=INK))
        show_plot(gauge)

    st.subheader("What moves this forecast")
    st.caption("Change in estimated revenue when one input is shifted and everything else stays as set. "
               "Bars to the right raise the forecast, bars to the left lower it.")
    s_lo_p, s_hi_p = max(0, star_pct - 25), min(100, star_pct + 25)
    r_lo, r_hi = max(70, runtime - 15), min(200, runtime + 15)
    b_lo_v, b_hi_v = _b(budget_m * 0.75), _b(budget_m * 1.25)
    full = dict(budgets_m=budget_m, runtime=runtime, quarter=quarter, star=star,
                genre_list=inputs["genre_list"], text=logline)
    cases = [
        ("Budget", f"Budget ${b_lo_v:,.0f}M", dict(budgets_m=b_lo_v)),
        ("Budget", f"Budget ${b_hi_v:,.0f}M", dict(budgets_m=b_hi_v)),
        ("Star power", f"Star power {s_lo_p}th pct", dict(star=float(np.quantile(df["star_power"], s_lo_p / 100)))),
        ("Star power", f"Star power {s_hi_p}th pct", dict(star=float(np.quantile(df["star_power"], s_hi_p / 100)))),
        ("Runtime", f"Runtime {r_lo} min", dict(runtime=r_lo)),
        ("Runtime", f"Runtime {r_hi} min", dict(runtime=r_hi)),
    ] + [("Release quarter", f"Release in {QUARTER_LABELS[q].split(' ')[0]}", dict(quarter=q)) for q in (1, 2, 3, 4)]
    case_rev = np.exp(model.predict(pd.concat([make_rows(**{**full, **o}) for _, _, o in cases],
                                              ignore_index=True))) / 1e6
    by_driver = {}
    for (drv, desc, _), r in zip(cases, case_rev):
        by_driver.setdefault(drv, []).append((float(r) - rev_m, desc))
    order = sorted(by_driver, key=lambda d: max(abs(v[0]) for v in by_driver[d]), reverse=True)
    weak = [min(by_driver[d])[0] for d in order]
    weak_t = [min(by_driver[d])[1] for d in order]
    strong = [max(by_driver[d])[0] for d in order]
    strong_t = [max(by_driver[d])[1] for d in order]
    tornado = go.Figure()
    tornado.add_trace(go.Bar(
        y=order, x=weak, orientation="h", name="Weaker case", marker_color=SLATE, customdata=weak_t,
        text=[f"{v:+,.0f}M" for v in weak], textposition="outside", cliponaxis=False,
        hovertemplate="%{customdata}<br>Revenue change: %{x:+,.1f}M<extra></extra>"))
    tornado.add_trace(go.Bar(
        y=order, x=strong, orientation="h", name="Stronger case", marker_color=ACCENT, customdata=strong_t,
        text=[f"{v:+,.0f}M" for v in strong], textposition="outside", cliponaxis=False,
        hovertemplate="%{customdata}<br>Revenue change: %{x:+,.1f}M<extra></extra>"))
    tornado.add_vline(x=0, line_color=INK, line_width=1)
    tornado.update_layout(barmode="overlay", legend=dict(orientation="h", y=1.12, x=0))
    tornado.update_xaxes(title_text="Change in estimated revenue ($M) vs. your forecast")
    tornado.update_yaxes(autorange="reversed")
    show_plot(style_fig(tornado, 310))

# ===== Scenario lab =====
with tab_lab:
    st.subheader("Scenario lab")
    st.caption("Play with a separate test scenario. These controls do not change your main forecast.")
    lab_left, lab_right, lab_btn = st.columns([1.2, 1, 0.6])
    with lab_left:
        lab_budget = st.slider("Test production budget ($M)", B_LO, B_HI, step=1.0, key="lab_budget",
                               help="Explore how the forecast responds to a different production budget.")
    with lab_right:
        lab_quarter = st.selectbox("Test release window", [1, 2, 3, 4], key="lab_quarter",
                                   format_func=lambda q: QUARTER_LABELS[q])
    with lab_btn:
        st.write("")
        st.button("Match main forecast", on_click=sync_lab, key="sync_lab_btn")

    lab_inputs = {**inputs, "quarter": lab_quarter}
    lab_revenue = float(predict_revenue_m(lab_budget, **lab_inputs)[0])
    lab_roi = lab_revenue / lab_budget - 1
    lab_profit = lab_revenue - lab_budget
    H('<div class="kpi-grid three">' + "".join([
        kpi_card("💰", "Scenario revenue", f"${lab_revenue:,.1f}M", f"{lab_revenue - rev_m:+,.1f}M vs. main forecast"),
        kpi_card("📈", "Scenario ROI", f"{lab_roi * 100:,.0f}%", f"{(lab_roi - roi) * 100:+,.0f} pts vs. main forecast",
                 "pos" if lab_roi >= 0 else "neg"),
        kpi_card("🧾", "Scenario profit / loss", f"${lab_profit:,.1f}M", "Revenue minus budget",
                 "pos" if lab_profit >= 0 else "neg"),
    ]) + "</div>")

    curve_budgets = np.linspace(B_LO, B_HI, 36)
    curve_revenues = predict_revenue_m(curve_budgets, **lab_inputs)
    lab_fig = go.Figure()
    lab_fig.add_trace(go.Scatter(
        x=curve_budgets, y=curve_revenues, mode="lines", name="Forecast revenue",
        line=dict(color=ACCENT, width=3), fill="tozeroy", fillcolor="rgba(189, 88, 59, 0.08)",
        hovertemplate="Budget: $%{x:,.1f}M<br>Revenue: $%{y:,.1f}M<extra></extra>"))
    lab_fig.add_trace(go.Scatter(
        x=curve_budgets, y=curve_budgets, mode="lines", name="Production cost",
        line=dict(color="#75839a", width=2, dash="dash"), hovertemplate="Budget: $%{x:,.1f}M<extra></extra>"))
    lab_fig.add_trace(go.Scatter(
        x=[lab_budget], y=[lab_revenue], mode="markers+text", name="Selected scenario",
        text=["Selected"], textposition="top center",
        marker=dict(color=INK, size=13, line=dict(color="white", width=2)),
        hovertemplate="Selected budget: $%{x:,.1f}M<br>Revenue: $%{y:,.1f}M<extra></extra>"))
    lab_fig.update_layout(xaxis_title="Production budget ($M)", yaxis_title="Estimated amount ($M)",
                          hovermode="x unified", legend=dict(orientation="h", y=1.12, x=0))
    show_plot(style_fig(lab_fig, 380))

    st.divider()
    st.subheader("Quick comparisons")
    st.caption("Each chart changes one input while keeping the rest of your main forecast fixed.")
    q_rev = [float(predict_revenue_m(budget_m, **{**inputs, "quarter": q})[0]) for q in [1, 2, 3, 4]]
    q_df = pd.DataFrame({
        "Quarter": [QUARTER_LABELS[q] for q in [1, 2, 3, 4]],
        "Budget ($M)": round(budget_m, 1),
        "Est. revenue ($M)": np.round(q_rev, 1),
        "Est. ROI (%)": np.round((np.array(q_rev) / budget_m - 1) * 100, 0),
    })
    q_df["Est. profit / loss ($M)"] = (q_df["Est. revenue ($M)"] - q_df["Budget ($M)"]).round(1)

    b_vals = np.clip(budget_m * np.array([0.5, 0.75, 1.0, 1.25, 1.5]), B_LO, B_HI)
    b_rev = predict_revenue_m(b_vals, **inputs)
    b_df = pd.DataFrame({
        "Budget change": ["-50%", "-25%", "As set", "+25%", "+50%"],
        "Budget ($M)": np.round(b_vals, 1),
        "Est. revenue ($M)": np.round(b_rev, 1),
        "Est. ROI (%)": np.round((b_rev / b_vals - 1) * 100, 0),
    })
    b_df["Est. profit / loss ($M)"] = (b_df["Est. revenue ($M)"] - b_df["Budget ($M)"]).round(1)

    def roi_bars(labels, values):
        fig = go.Figure(go.Bar(
            x=labels, y=values, marker_color=[GREEN if v >= 0 else RED for v in values],
            text=[f"{v:+,.0f}%" for v in values], textposition="outside", cliponaxis=False,
            hovertemplate="%{x}<br>ROI %{y:,.0f}%<extra></extra>"))
        fig.update_layout(yaxis_title="Estimated ROI (%)", showlegend=False)
        return style_fig(fig, 290)

    cmp_l, cmp_r = st.columns(2)
    with cmp_l:
        st.markdown("**Release quarter**")
        show_plot(roi_bars(list(q_df["Quarter"]), [float(v) for v in q_df["Est. ROI (%)"]]))
        show_table(q_df)
    with cmp_r:
        st.markdown("**Production budget**")
        show_plot(roi_bars(list(b_df["Budget change"]), [float(v) for v in b_df["Est. ROI (%)"]]))
        show_table(b_df)

    export = pd.concat([q_df.rename(columns={"Quarter": "Scenario"}).assign(Group="Release quarter"),
                        b_df.rename(columns={"Budget change": "Scenario"}).assign(Group="Budget")])
    st.download_button("Download these scenarios (CSV)", export.to_csv(index=False).encode("utf-8"),
                       file_name="film_roi_scenarios.csv", mime="text/csv", key="dl_scen")

    st.divider()
    st.subheader("Pinned forecasts")
    pins = st.session_state.get("pins", [])
    if not pins:
        st.info("Nothing pinned yet. Use **📌 Pin this forecast** under the headline numbers to save an option, "
                "change the sidebar, and pin another to compare them side by side.")
    else:
        pin_df = pd.DataFrame(pins)
        pin_fig = go.Figure(go.Bar(
            x=pin_df["Scenario"], y=pin_df["Est. ROI (%)"],
            marker_color=[GREEN if v >= 0 else RED for v in pin_df["Est. ROI (%)"]],
            text=[f"{v:+,.0f}%" for v in pin_df["Est. ROI (%)"]], textposition="outside", cliponaxis=False,
            hovertemplate="%{x}<br>ROI %{y:,.0f}%<extra></extra>"))
        pin_fig.update_layout(yaxis_title="Estimated ROI (%)", showlegend=False)
        show_plot(style_fig(pin_fig, 300))
        show_table(pin_df)
        d1, d2 = st.columns([1, 1])
        with d1:
            st.download_button("Download pinned forecasts (CSV)", pin_df.to_csv(index=False).encode("utf-8"),
                               file_name="pinned_forecasts.csv", mime="text/csv", key="dl_pins")
        with d2:
            st.button("Clear pinned forecasts", on_click=clear_pins, key="clear_pins_btn")

# ===== Similar films =====
with tab_sim:
    st.subheader(f"Historical {genre} films near your budget")
    st.caption("Real past films in the same genre with a budget within ±50% of yours. "
               "Revenue and ROI are historical figures, not model estimates.")
    comps_all = df[(df["genre"] == genre) & df["budget"].between(budget_m * 1e6 * 0.5, budget_m * 1e6 * 1.5)].copy()
    if comps_all.empty:
        st.info("No comparable films in the data for this genre and budget. Try a different budget or genre.")
    else:
        hit = float((comps_all["roi"] > 0).mean())
        med_roi = float(comps_all["roi"].median())
        H('<div class="kpi-grid three">' + "".join([
            kpi_card("🎞️", "Comparable films", f"{len(comps_all):,}", f"{genre}, budget within ±50%"),
            kpi_card("✅", "Covered their budget", f"{hit * 100:,.0f}%", "Share with ROI above 0%",
                     "pos" if hit >= 0.5 else "neg"),
            kpi_card("📊", "Median ROI", f"{med_roi * 100:,.0f}%", "Typical return of these films",
                     "pos" if med_roi >= 0 else "neg"),
        ]) + "</div>")
        comps = comps_all.assign(gap=(comps_all["budget"] / 1e6 - budget_m).abs()).sort_values("gap").head(10)
        table = pd.DataFrame({
            "Title": comps["title"], "Year": comps["year"],
            "Budget ($M)": (comps["budget"] / 1e6).round(1),
            "Revenue ($M)": (comps["revenue"] / 1e6).round(1),
            "ROI (%)": (comps["roi"] * 100).round(0),
        })
        try:
            cfg = {
                "Year": st.column_config.NumberColumn("Year", format="%d"),
                "Budget ($M)": st.column_config.NumberColumn("Budget ($M)", format="$%.1fM"),
                "Revenue ($M)": st.column_config.NumberColumn("Revenue ($M)", format="$%.1fM"),
                "ROI (%)": st.column_config.NumberColumn("ROI (%)", format="%d%%"),
            }
        except Exception:
            cfg = None
        st.markdown("**The 10 closest by budget**")
        show_table(table, cfg)

# ===== Market context =====
with tab_mkt:
    st.subheader("Where your project sits among past films")
    st.caption("Each bubble is a past film. Higher means a better return, bigger means more revenue, and films above "
               "the dashed line made money. The star is your project. Use the filters to narrow the view.")
    f1, f2 = st.columns([1.2, 1])
    with f1:
        years = st.slider("Release years shown", Y_LO, LATEST_YEAR, key="ctx_years")
    with f2:
        gsel = st.multiselect("Filter genres (leave empty for all)", all_genres, key="ctx_genres")
    res = market_bubble((int(years[0]), int(years[1])), tuple(sorted(gsel)))
    if res is None:
        st.info("Too few films match these filters. Widen the year range or clear the genre filter.")
    else:
        bubble, cap, n_shown = res
        bubble.add_trace(go.Scatter(
            x=[budget_m], y=[min(roi, cap)], mode="markers+text", name="Your project",
            marker=dict(symbol="star", size=24, color=INK, line=dict(width=2, color="white")),
            text=["Your project"], textposition="top center"))
        show_plot(style_fig(bubble, 520), toolbar=True)
        st.caption(f"Showing {n_shown:,} films. ROI is capped at the 97th percentile so a few extreme hits "
                   "don't flatten the chart.")

    c_l, c_r = st.columns(2)
    with c_l:
        trend_fig, y0, y1 = trend_figure()
        st.subheader(f"Genre mix, {y0}–{y1}")
        st.caption("Share of all releases that each genre made up in a given year.")
        show_plot(style_fig(trend_fig, 400))
    with c_r:
        st.subheader("Median ROI by genre")
        st.caption("The typical return of a film in each genre. Higher bars mean historically better returns.")
        show_plot(style_fig(genre_roi_figure(), 400))

# ===== Break-even =====
with tab_be:
    st.subheader("How much budget can this project carry?")
    st.caption("The solid line is predicted revenue at each budget, the dashed line is cost. "
               "Where revenue is above the dashed line the film is predicted to cover its budget.")
    budgets = np.linspace(max(b_min, 1.0), b_max, 60)
    revs = predict_revenue_m(budgets, **inputs)
    be, status = find_break_even(budgets, revs)
    if status == "crossover":
        st.success(f"Predicted revenue covers cost up to roughly **{usd(be)}** for these inputs. "
                   "Above that budget, the film is predicted to lose money.")
    elif status == "always_profitable":
        st.success("Predicted revenue exceeds cost across the whole budget range for these inputs.")
    else:
        st.warning("Predicted revenue does not cover cost at any budget in range for these inputs.")
    show_plot(style_fig(breakeven_chart(budgets, revs, budget_m, be), 470))
    st.caption("The model cannot extrapolate beyond budgets it has seen, so the curve flattens at the extremes.")

# ---------------- about + footer ----------------
with st.expander("About this forecast and its limits"):
    acc = f"explains about {cv * 100:.0f}% of the variation in film revenue" if cv is not None else "is moderately accurate"
    st.markdown(
        f"""
**Data.** TMDB 5000 Movie Dataset: films with known budgets and box-office revenue, released up to {LATEST_YEAR}.
Forecasts are made at {LATEST_YEAR} market levels. Films with missing or tiny revenue figures were removed, which
likely removes some real flops, so results lean optimistic.

**Model.** A Random Forest trained on budget, runtime, release quarter, release year, cast experience, genres and
logline keywords. In testing it {acc}. The rest depends on things nobody knows before release, such as marketing,
reviews, competition and word of mouth.

**The "typical range" and "chance of covering budget"** come from how much the forest's individual trees disagree.
They are a guide to the model's uncertainty, not a statistical guarantee.

**What it does not include.** Streaming viewership, marketing spend, maturity ratings, audience demographics and
the studio's share of ticket sales. Treat results as a starting point for discussion, not a final decision.
"""
    )

H(f"""
<div class="site-footer">
  <span>Built with Streamlit, scikit-learn and Plotly · Data: TMDB 5000 Movie Dataset</span>
  <span>Estimates are not guarantees · Forecasts scored at {LATEST_YEAR} market levels</span>
</div>
""")