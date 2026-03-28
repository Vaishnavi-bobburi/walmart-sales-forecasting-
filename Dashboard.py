"""
╔══════════════════════════════════════════════════════════════════╗
║   WALMART SALES INTELLIGENCE PLATFORM  —  Professional Edition  ║
║   Senior Data Scientist · Product Designer · Business Analyst   ║
║   Run: streamlit run dashboard.py                               ║
╚══════════════════════════════════════════════════════════════════╝
pip install streamlit plotly pandas numpy statsmodels scikit-learn
Files needed in same folder: train.csv  features.csv  stores.csv
"""

import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMA_OK = True
except ImportError:
    SARIMA_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Walmart Sales Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS  — Deep Navy + Saffron Orange (Indian business aesthetic)
# ══════════════════════════════════════════════════════════════════════════════
C = dict(
    navy    = "#0F1B2D",
    navy2   = "#162236",
    navy3   = "#1E3A5F",
    saffron = "#FF6B00",
    saff2   = "#FF8C38",
    gold    = "#FFB800",
    green   = "#00C853",
    green2  = "#00E676",
    red     = "#FF1744",
    red2    = "#FF6E7A",
    sky     = "#00B4D8",
    white   = "#FFFFFF",
    off     = "#F8FAFC",
    g1      = "#E2E8F0",
    g2      = "#94A3B8",
    g3      = "#475569",
    card    = "#FFFFFF",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=Space+Mono:wght@400;700&display=swap');

html,body,[class*="css"]{{
    font-family:'Plus Jakarta Sans',sans-serif;
    background:{C['off']};color:{C['navy']};
}}
#MainMenu,footer,header{{visibility:hidden;}}
.block-container{{padding:0 1.8rem 2rem;max-width:1700px;}}
::-webkit-scrollbar{{width:5px;height:5px;}}
::-webkit-scrollbar-track{{background:{C['g1']};}}
::-webkit-scrollbar-thumb{{background:{C['navy3']};border-radius:3px;}}

/* ── SIDEBAR ── */
[data-testid="stSidebar"]{{
    background:linear-gradient(175deg,{C['navy']} 0%,{C['navy2']} 55%,{C['navy3']} 100%);
    border-right:1px solid rgba(255,255,255,0.05);
}}
[data-testid="stSidebar"] *{{color:#CBD5E1 !important;}}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{{color:white !important;}}
[data-testid="stSidebar"] label{{
    color:#94A3B8 !important;font-size:0.7rem !important;
    font-weight:700 !important;letter-spacing:0.1em !important;text-transform:uppercase !important;
}}
[data-testid="stSidebar"] .stSelectbox>div>div,
[data-testid="stSidebar"] .stMultiSelect>div>div{{
    background:rgba(255,255,255,0.07) !important;
    border:1px solid rgba(255,255,255,0.12) !important;
    border-radius:8px !important;
}}

/* ── TOP BANNER ── */
.banner{{
    background:linear-gradient(120deg,{C['navy']} 0%,{C['navy3']} 70%,#1a4068 100%);
    padding:1.3rem 2rem;margin:0 -1.8rem 1.8rem -1.8rem;
    display:flex;align-items:center;justify-content:space-between;
    border-bottom:3px solid {C['saffron']};
    box-shadow:0 6px 30px rgba(15,27,45,0.18);
}}
.banner-left{{display:flex;align-items:center;gap:1.1rem;}}
.b-logo{{
    width:50px;height:50px;background:{C['saffron']};border-radius:14px;
    display:flex;align-items:center;justify-content:center;
    font-size:1.5rem;box-shadow:0 4px 16px rgba(255,107,0,0.4);
    flex-shrink:0;
}}
.b-title{{font-size:1.5rem;font-weight:900;color:white;letter-spacing:-0.5px;line-height:1.2;}}
.b-sub{{font-size:0.72rem;color:#94A3B8;font-weight:500;margin-top:3px;letter-spacing:0.03em;}}
.b-right{{display:flex;gap:0.8rem;align-items:center;flex-wrap:wrap;}}
.live{{
    background:rgba(0,200,83,0.15);border:1px solid {C['green']};
    color:{C['green']};font-size:0.66rem;font-weight:700;
    padding:0.28rem 0.75rem;border-radius:999px;
    letter-spacing:0.1em;text-transform:uppercase;
}}
.b-stat{{
    background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.12);
    color:white;font-size:0.7rem;font-weight:600;
    padding:0.28rem 0.85rem;border-radius:8px;
}}

/* ── SECTION HEADERS ── */
.sec{{
    display:flex;align-items:center;gap:0.8rem;
    padding:0.5rem 0;margin-bottom:1rem;
    border-bottom:2px solid {C['g1']};
}}
.sec-icon{{
    width:38px;height:38px;border-radius:10px;
    display:flex;align-items:center;justify-content:center;
    font-size:1.05rem;flex-shrink:0;
}}
.sec-title{{font-size:1.05rem;font-weight:800;color:{C['navy']};}}
.sec-sub{{font-size:0.73rem;color:{C['g2']};margin-top:1px;}}

/* ── KPI CARDS ── */
.kpi{{
    background:{C['card']};border-radius:16px;
    padding:1.25rem 1.35rem 1.1rem;
    border:1px solid {C['g1']};
    box-shadow:0 2px 14px rgba(15,27,45,0.06);
    transition:all .22s ease;position:relative;overflow:hidden;height:100%;
}}
.kpi::before{{
    content:'';position:absolute;top:0;left:0;right:0;height:3px;
    background:var(--ac,{C['saffron']});
}}
.kpi:hover{{transform:translateY(-3px);box-shadow:0 10px 32px rgba(15,27,45,0.13);}}
.kpi-top{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.8rem;}}
.kpi-icon{{
    width:46px;height:46px;border-radius:13px;
    background:var(--ib,#FFF3E0);
    display:flex;align-items:center;justify-content:center;
    font-size:1.3rem;flex-shrink:0;
}}
.kpi-lbl{{font-size:0.67rem;font-weight:700;letter-spacing:0.09em;
    text-transform:uppercase;color:{C['g2']};margin-bottom:0.28rem;}}
.kpi-val{{font-size:1.75rem;font-weight:900;line-height:1.1;
    color:var(--vc,{C['navy']});letter-spacing:-0.5px;}}
.kpi-delta{{
    display:inline-flex;align-items:center;gap:3px;
    font-size:0.7rem;font-weight:700;
    padding:0.16rem 0.48rem;border-radius:999px;margin-top:0.38rem;
}}
.up{{background:rgba(0,200,83,0.1);color:{C['green']};}}
.dn{{background:rgba(255,23,68,0.1);color:{C['red']};}}
.nu{{background:rgba(148,163,184,0.1);color:{C['g3']};}}
.kpi-hint{{font-size:0.7rem;color:{C['g2']};margin-top:0.22rem;font-weight:500;}}

/* ── CHART CARDS ── */
.cc{{background:{C['card']};border-radius:16px;padding:1.3rem 1.4rem;
    border:1px solid {C['g1']};
    box-shadow:0 2px 14px rgba(15,27,45,0.06);margin-bottom:1.2rem;}}
.cc-title{{font-size:0.93rem;font-weight:700;color:{C['navy']};margin-bottom:0.12rem;}}
.cc-sub{{font-size:0.72rem;color:{C['g2']};margin-bottom:0.9rem;}}

/* ── ALERTS ── */
.a-green{{background:linear-gradient(135deg,rgba(0,200,83,.08),rgba(0,200,83,.03));
    border:1px solid rgba(0,200,83,.25);border-left:4px solid {C['green']};
    border-radius:10px;padding:.8rem 1rem;margin:.3rem 0;font-size:.81rem;color:#064e3b;}}
.a-red{{background:linear-gradient(135deg,rgba(255,23,68,.08),rgba(255,23,68,.03));
    border:1px solid rgba(255,23,68,.25);border-left:4px solid {C['red']};
    border-radius:10px;padding:.8rem 1rem;margin:.3rem 0;font-size:.81rem;color:#7f1d1d;}}
.a-amber{{background:linear-gradient(135deg,rgba(255,184,0,.1),rgba(255,184,0,.03));
    border:1px solid rgba(255,184,0,.3);border-left:4px solid {C['gold']};
    border-radius:10px;padding:.8rem 1rem;margin:.3rem 0;font-size:.81rem;color:#78350f;}}
.a-blue{{background:linear-gradient(135deg,rgba(0,180,216,.08),rgba(0,180,216,.03));
    border:1px solid rgba(0,180,216,.25);border-left:4px solid {C['sky']};
    border-radius:10px;padding:.8rem 1rem;margin:.3rem 0;font-size:.81rem;color:#0c4a6e;}}
.at{{font-weight:700;margin-bottom:.22rem;font-size:.83rem;}}

/* ── RANK ROWS ── */
.rrow{{
    display:flex;align-items:center;padding:.6rem .9rem;border-radius:10px;
    margin:.25rem 0;gap:.9rem;background:{C['card']};
    border:1px solid {C['g1']};font-size:.8rem;
    transition:transform .15s;
}}
.rrow:hover{{transform:translateX(5px);box-shadow:0 2px 12px rgba(15,27,45,.08);}}
.rnum{{
    width:28px;height:28px;border-radius:8px;
    background:{C['navy']};color:white;font-weight:800;font-size:.73rem;
    display:flex;align-items:center;justify-content:center;flex-shrink:0;
}}
.gold{{background:linear-gradient(135deg,#FFD700,#FFA500)!important;}}
.silver{{background:linear-gradient(135deg,#C0C0C0,#909090)!important;}}
.bronze{{background:linear-gradient(135deg,#CD7F32,#A0522D)!important;}}

/* ── SUGGESTION CARDS ── */
.sug{{border-radius:12px;padding:.95rem 1.15rem;margin:.4rem 0;
    border-left:4px solid;font-size:.81rem;position:relative;}}
.sr{{background:#FFF1F2;border-color:{C['red']};color:#7f1d1d;}}
.sa{{background:#FFFBEB;border-color:{C['gold']};color:#78350f;}}
.sg{{background:#F0FDF4;border-color:{C['green']};color:#064e3b;}}
.sb{{background:#EFF6FF;border-color:{C['sky']};color:#1e3a5f;}}
.st{{font-weight:700;font-size:.84rem;margin-bottom:.22rem;}}
.sbadge{{
    position:absolute;top:.7rem;right:.9rem;
    font-size:.6rem;font-weight:700;text-transform:uppercase;
    padding:.13rem .45rem;border-radius:99px;letter-spacing:.08em;
}}
.urgent{{background:#FFE4E6;color:#9F1239;}}
.opp{{background:#DCFCE7;color:#14532D;}}
.tip{{background:#DBEAFE;color:#1E40AF;}}

/* ── WEEK TABLE ── */
.wrow{{
    display:grid;grid-template-columns:1.4fr 1fr 1fr 1fr 1fr 1.1fr;
    gap:.4rem;padding:.58rem .85rem;border-radius:8px;
    font-size:.77rem;align-items:center;margin:.2rem 0;
}}
.wh{{background:{C['navy']};color:white;font-weight:700;
    font-size:.67rem;letter-spacing:.07em;text-transform:uppercase;}}
.wg{{background:#F0FDF4;border:1px solid #BBF7D0;color:#0F1B2D;}}
.wl{{background:#FFF1F2;border:1px solid #FECDD3;color:#0F1B2D;}}
.wf{{background:#F8FAFC;border:1px solid {C['g1']};color:#0F1B2D;}}
.wrow *{{color:#0F1B2D;}}
.wh *{{color:white !important;}}
.tag{{display:inline-block;padding:.13rem .48rem;border-radius:99px;font-size:.64rem;font-weight:700;}}
.tg{{background:#DCFCE7;color:#14532D !important;}}
.tl{{background:#FFE4E6;color:#9F1239 !important;}}
.tf{{background:#F1F5F9;color:{C['g3']} !important;}}

/* ── FC PANEL ── */
.fcp{{
    background:linear-gradient(135deg,{C['navy']} 0%,{C['navy3']} 100%);
    border-radius:16px;padding:1.5rem 1.8rem;color:white;
    border:1px solid rgba(255,107,0,.3);
    box-shadow:0 8px 36px rgba(15,27,45,.22);margin-bottom:1.2rem;
}}
.fck{{
    text-align:center;background:rgba(255,255,255,.07);
    border:1px solid rgba(255,255,255,.1);border-radius:12px;padding:1rem;
}}
.fck-v{{font-size:1.55rem;font-weight:900;color:{C['saffron']};}}
.fck-l{{font-size:.68rem;color:#94A3B8;text-transform:uppercase;
    letter-spacing:.1em;margin-top:.22rem;}}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"]{{
    background:{C['card']};border-radius:12px;padding:.28rem;
    gap:.2rem;border:1px solid {C['g1']};
    box-shadow:0 1px 4px rgba(0,0,0,.05);
}}
.stTabs [data-baseweb="tab"]{{
    border-radius:8px;font-weight:600;font-size:.79rem;
    padding:.45rem 1rem;color:{C['g3']};
}}
.stTabs [aria-selected="true"]{{
    background:{C['navy']} !important;color:white !important;
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data():
    try:
        train    = pd.read_csv("train.csv",    parse_dates=["Date"])
        features = pd.read_csv("features.csv", parse_dates=["Date"])
        stores   = pd.read_csv("stores.csv")
    except FileNotFoundError as e:
        st.error(f"❌ Missing data file: {e}. Place train.csv, features.csv, stores.csv in same folder.")
        st.stop()

    df = (train
          .merge(features, on=["Store","Date","IsHoliday"], how="left")
          .merge(stores,   on="Store", how="left"))

    df["Year"]       = df["Date"].dt.year
    df["Month"]      = df["Date"].dt.month
    df["MonthName"]  = df["Date"].dt.strftime("%b")
    df["Quarter"]    = df["Date"].dt.quarter
    df["WeekNum"]    = df["Date"].dt.isocalendar().week.astype(int)
    df["YearMonth"]  = df["Date"].dt.to_period("M").astype(str)
    df["MonthYear"]  = df["Date"].dt.strftime("%b %Y")
    df["Season"]     = df["Month"].map({
        12:"Winter",1:"Winter",2:"Winter",
        3:"Spring",4:"Spring",5:"Spring",
        6:"Summer",7:"Summer",8:"Summer",
        9:"Autumn",10:"Autumn",11:"Autumn"
    })

    # Profit / Loss estimation (30% margin assumption — standard retail)
    MARGIN = 0.30
    df["Est_Profit"]  = df["Weekly_Sales"] * MARGIN
    df["IsLoss"]      = df["Weekly_Sales"] < 0

    # WoW change per store-dept
    df = df.sort_values(["Store","Dept","Date"])
    df["Prev_Sales"]  = df.groupby(["Store","Dept"])["Weekly_Sales"].shift(1)
    df["Sales_Chg"]   = df["Weekly_Sales"] - df["Prev_Sales"]
    df["Sales_Chg%"]  = (df["Sales_Chg"] / df["Prev_Sales"].abs() * 100).replace([np.inf,-np.inf],np.nan)

    return df

with st.spinner("🔄 Loading Walmart data..."):
    raw = load_data()

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def M(v):
    """Format as Indian Rupees."""
    v = float(v)
    if abs(v)>=1e9: return f"₹{v/1e9:.2f}B"
    if abs(v)>=1e6: return f"₹{v/1e6:.2f}M"
    if abs(v)>=1e3: return f"₹{v/1e3:.1f}K"
    return f"₹{v:.0f}"

def Md(v):
    """Format as Indian Rupees (same as M — both used throughout)."""
    v = float(v)
    if abs(v)>=1e9: return f"₹{v/1e9:.2f}B"
    if abs(v)>=1e6: return f"₹{v/1e6:.2f}M"
    if abs(v)>=1e3: return f"₹{v/1e3:.1f}K"
    return f"₹{v:.0f}"

def cs(fig, h=360, bgcolor=None):
    """Apply clean chart styling."""
    bg = bgcolor or "rgba(0,0,0,0)"
    fig.update_layout(
        height=h, font_family="Plus Jakarta Sans",
        paper_bgcolor=bg, plot_bgcolor=bg,
        margin=dict(t=30,b=30,l=5,r=5),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,
                    xanchor="right",x=1,bgcolor="rgba(0,0,0,0)",font_size=11),
        xaxis=dict(showgrid=False,linecolor=C["g1"],tickfont_size=11),
        yaxis=dict(gridcolor=C["g1"],linecolor="rgba(0,0,0,0)",tickfont_size=11),
    )
    return fig

def section(icon, title, subtitle, color=None):
    col = color or C["saffron"]
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:.8rem;padding:.5rem 0;
                margin-bottom:1rem;border-bottom:2px solid #E2E8F0;">
      <div style="width:38px;height:38px;border-radius:10px;background:{col}18;
                  display:flex;align-items:center;justify-content:center;font-size:1.05rem;flex-shrink:0">
        {icon}
      </div>
      <div>
        <div style="font-size:1.05rem;font-weight:800;color:#0F1B2D">{title}</div>
        <div style="font-size:.73rem;color:#94A3B8;margin-top:1px">{subtitle}</div>
      </div>
    </div>""", unsafe_allow_html=True)

def subsection(label, color="#0F1B2D"):
    """Mini heading inside a section for grouping charts."""
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:.5rem;
                margin:1.2rem 0 .6rem;padding:.35rem 0;
                border-left:3px solid {color};padding-left:.75rem;">
      <div style="font-size:.88rem;font-weight:800;color:{color}">{label}</div>
    </div>""", unsafe_allow_html=True)

def cc_open(title, subtitle=""):
    sub = f'<div class="cc-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="cc"><div class="cc-title">{title}</div>{sub}', unsafe_allow_html=True)

def cc_close():
    st.markdown('</div>', unsafe_allow_html=True)

def _kpi_html(icon, label, value, delta=None, hint=None, accent="#FF6B00",
               icon_bg="#FFF3E0", val_color="#0F1B2D"):
    """Return an HTML string for one KPI card (never calls st.markdown directly)."""
    vc = val_color or "#0F1B2D"
    dhtml = ""
    if delta is not None:
        try:
            dv   = float(str(delta).replace("%","").replace("+","")
                         .replace("₹","").replace(",","").strip())
            dcol = "#059669" if dv >= 0 else "#DC2626"
            dbg  = "rgba(5,150,105,0.1)" if dv >= 0 else "rgba(220,38,38,0.1)"
            darr = "▲" if dv >= 0 else "▼"
            dhtml = (f'<span style="display:inline-flex;align-items:center;gap:2px;' +
                     f'font-size:.69rem;font-weight:700;padding:.14rem .44rem;' +
                     f'border-radius:999px;margin-top:.35rem;background:{dbg};color:{dcol}">' +
                     f'{darr} {delta}</span>')
        except Exception:
            dhtml = f'<span style="font-size:.69rem;color:#64748B">{delta}</span>'
    hint_html = (f'<div style="font-size:.69rem;color:#94A3B8;margin-top:.2rem">{hint}</div>'
                 if hint else "")
    return (
        f'<div style="background:#FFFFFF;border-radius:14px;padding:1.2rem 1.3rem 1rem;' +
        f'border:1px solid #E2E8F0;border-top:3px solid {accent};' +
        f'box-shadow:0 2px 12px rgba(15,27,45,0.07);flex:1;min-width:0;">' +
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start">' +
        f'<div style="flex:1;min-width:0;">' +
        f'<div style="font-size:.64rem;font-weight:700;letter-spacing:.09em;' +
        f'text-transform:uppercase;color:#94A3B8;margin-bottom:.25rem">{label}</div>' +
        f'<div style="font-size:1.65rem;font-weight:900;line-height:1.1;' +
        f'color:{vc};letter-spacing:-0.5px;word-break:break-word">{value}</div>' +
        f'<div>{dhtml}</div>{hint_html}</div>' +
        f'<div style="width:44px;height:44px;border-radius:12px;background:{icon_bg};' +
        f'display:flex;align-items:center;justify-content:center;' +
        f'font-size:1.25rem;flex-shrink:0;margin-left:.8rem">{icon}</div>' +
        f'</div></div>'
    )

def kpi_row(cards):
    """Render a list of KPI card dicts as a single HTML flex row — avoids st.columns HTML bug.
    Each dict: icon, label, value, delta, hint, accent, icon_bg, val_color
    """
    html_cards = ""
    for c in cards:
        html_cards += _kpi_html(
            c.get("icon","📊"), c.get("label",""), c.get("value",""),
            c.get("delta"), c.get("hint"), c.get("accent","#FF6B00"),
            c.get("icon_bg","#FFF3E0"), c.get("val_color","#0F1B2D")
        )
    st.markdown(
        f'<div style="display:flex;gap:.9rem;margin-bottom:1rem;">{html_cards}</div>',
        unsafe_allow_html=True
    )

def kpi(icon, label, value, delta=None, hint=None, accent="#FF6B00",
        icon_bg="#FFF3E0", val_color=None):
    """Single KPI card — wraps in a flex row of 1."""
    kpi_row([dict(icon=icon,label=label,value=value,delta=delta,
                  hint=hint,accent=accent,icon_bg=icon_bg,
                  val_color=val_color or "#0F1B2D")])


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:1.4rem 0 .8rem">
      <div style="width:56px;height:56px;background:{C['saffron']};border-radius:16px;
                  display:inline-flex;align-items:center;justify-content:center;
                  font-size:1.7rem;box-shadow:0 6px 20px rgba(255,107,0,.4);">🛒</div>
      <div style="font-size:1rem;font-weight:800;color:white;margin-top:.7rem;">Walmart Analytics</div>
      <div style="font-size:.68rem;color:#64748B;letter-spacing:.12em;text-transform:uppercase;margin-top:3px;">
        Intelligence Platform</div>
    </div>
    <hr style="border-color:rgba(255,255,255,.07);margin:.6rem 0">
    """, unsafe_allow_html=True)

    st.markdown("#### 🏪 Store & Data Filters")

    all_stores  = sorted(raw["Store"].unique())
    all_types   = sorted(raw["Type"].unique())
    all_depts   = sorted(raw["Dept"].unique())

    store_type_sel = st.multiselect("Store Type", all_types, default=all_types,
                                     help="A=Large, B=Medium, C=Small format stores")
    valid_stores = sorted(raw[raw["Type"].isin(store_type_sel)]["Store"].unique()) if store_type_sel else all_stores
    store_sel  = st.multiselect("Stores",  valid_stores, default=valid_stores[:6],
                                 help="Select one or more Walmart stores")
    if not store_sel: store_sel = valid_stores[:6]

    dept_sel   = st.multiselect("Departments", all_depts, default=all_depts[:12],
                                 help="Select departments to analyse")
    if not dept_sel: dept_sel = all_depts[:12]

    mn, mx = raw["Date"].min().date(), raw["Date"].max().date()
    dr = st.date_input("Date Range", value=(mn, mx), min_value=mn, max_value=mx)
    s_date = pd.Timestamp(dr[0]) if len(dr)>0 else pd.Timestamp(mn)
    e_date = pd.Timestamp(dr[1]) if len(dr)>1 else pd.Timestamp(mx)

    hol_opt = st.radio("Holiday Filter",
                        ["All Weeks","Holiday Weeks Only","Non-Holiday Weeks Only"])

    st.markdown("<hr style='border-color:rgba(255,255,255,.07);margin:.7rem 0'>", unsafe_allow_html=True)
    st.markdown("#### 🔮 Forecast Settings")
    fc_store  = st.selectbox("Forecast Store",  all_stores)
    fc_dept   = st.selectbox("Forecast Dept",   all_depts)
    fc_weeks  = st.slider("Weeks to Forecast",  4, 26, 12)

    st.markdown("<hr style='border-color:rgba(255,255,255,.07);margin:.7rem 0'>", unsafe_allow_html=True)
    st.markdown("#### 📅 Weekly Deep-Dive")
    wa_store = st.selectbox("Weekly Store",  all_stores, key="wa_s")
    wa_dept  = st.selectbox("Weekly Dept",   all_depts,  key="wa_d")
    wa_n     = st.slider("Show Last N Weeks", 8, 52, 20)

    # Sidebar footer
    st.markdown(f"""
    <div style="margin-top:auto;padding:1rem 0 .5rem;text-align:center">
      <div style="font-size:.62rem;color:#475569;letter-spacing:.06em">
        DATA: Kaggle · Walmart Store Sales<br>
        © 2024 Sales Intelligence Platform
      </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FILTER MASTER DATASET
# ══════════════════════════════════════════════════════════════════════════════

mask = (
    raw["Store"].isin(store_sel) &
    raw["Dept"].isin(dept_sel) &
    (raw["Date"] >= s_date) &
    (raw["Date"] <= e_date)
)

if hol_opt == "Holiday Weeks Only":
    mask = mask & (raw["IsHoliday"] == True)
elif hol_opt == "Non-Holiday Weeks Only":
    mask = mask & (raw["IsHoliday"] == False)

dff = raw[mask].copy()

if dff.empty:
    st.warning("⚠️ No data matches your filters. Please broaden the selection.")
    st.stop()
real_min = dff["Date"].min()
real_max = dff["Date"].max()

date_range = f"{real_min.strftime('%d %b %Y')} → {real_max.strftime('%d %b %Y')}"
# ══════════════════════════════════════════════════════════════════════════════
# TOP BANNER
# ══════════════════════════════════════════════════════════════════════════════
last_dt  = dff["Date"].max().strftime("%d %b %Y")
n_stores = dff["Store"].nunique()
n_weeks  = dff["Date"].nunique()
n_depts  = dff["Dept"].nunique()


st.markdown(f"""
<div class="banner">
  <div class="banner-left">
    <div class="b-logo">🛒</div>
    <div>
      <div class="b-title">Walmart Sales Intelligence Platform</div>
      <div class="b-sub">Professional Analytics Dashboard · Powered by AI Forecasting · Data updated: {last_dt}</div>
    </div>
  </div>
  <div class="b-right">
    <span class="live">● Live</span>
    <span class="b-stat">📦 {n_stores} Stores</span>
    <span class="b-stat">🗂 {n_depts} Depts</span>
    <span class="b-stat">📅 {n_weeks} Weeks</span>
    <span class="b-stat">{date_range}</span>
    
  </div>
</div>
""", unsafe_allow_html=True)

real_min = dff["Date"].min()
real_max = dff["Date"].max()

date_range = f"{real_min.strftime('%d %b %Y')} → {real_max.strftime('%d %b %Y')}"

# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE MASTER METRICS
# ══════════════════════════════════════════════════════════════════════════════
MARGIN = 0.30
total_sales   = dff["Weekly_Sales"].sum()
total_profit  = total_sales * MARGIN
total_loss_v  = dff[dff["Weekly_Sales"] < 0]["Weekly_Sales"].sum()

weekly_ts     = dff.groupby("Date")["Weekly_Sales"].sum().sort_index()
avg_weekly    = weekly_ts.mean()
wow_last      = float(weekly_ts.pct_change().dropna().iloc[-1]*100) if len(weekly_ts)>1 else 0

store_tots    = dff.groupby("Store")["Weekly_Sales"].sum()
best_store    = int(store_tots.idxmax())
worst_store   = int(store_tots.idxmin())

dept_tots     = dff.groupby("Dept")["Weekly_Sales"].sum()
best_dept     = int(dept_tots.idxmax())
worst_dept    = int(dept_tots.idxmin())

# YoY growth
years = sorted(dff["Year"].unique())
if len(years) >= 2:
    y1 = dff[dff["Year"]==years[-2]]["Weekly_Sales"].sum()
    y2 = dff[dff["Year"]==years[-1]]["Weekly_Sales"].sum()
    yoy = (y2-y1)/y1*100 if y1 else 0
else:
    yoy = 0

# Gain/Loss weeks
chg_by_week = dff.groupby("Date")["Sales_Chg"].sum().dropna()
gain_weeks  = int((chg_by_week>0).sum())
loss_weeks  = int((chg_by_week<0).sum())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — KPI HEADLINE METRICS
# ══════════════════════════════════════════════════════════════════════════════
section("📊", "Business KPI Metrics",
        "Core performance indicators at a glance — everything a business owner needs to know",
        C["saffron"])

neg = dff[dff["Weekly_Sales"]<0]["Weekly_Sales"].sum()
kpi_row([
    dict(icon="💰", label="Total Sales Revenue",    value=Md(total_sales),
         delta=f"{wow_last:+.1f}% WoW",
         hint=f"Across {n_stores} stores · {n_depts} depts",
         accent=C["saffron"], icon_bg="#FFF3E0"),
    dict(icon="📈", label="Estimated Profit (30%)", value=Md(total_profit),
         delta=f"{yoy:+.1f}% YoY",
         hint="Based on 30% retail margin",
         accent=C["green"], icon_bg="#F0FDF4", val_color=C["green"]),
    dict(icon="📉", label="Loss / Negative Sales",  value=Md(abs(neg)) if neg<0 else "₹0",
         delta=f"{loss_weeks} loss weeks",
         hint="Weeks where sales turned negative",
         accent=C["red"], icon_bg="#FFF1F2",
         val_color=C["red"] if neg<0 else C["navy"]),
    dict(icon="🗓", label="Average Weekly Sales",   value=Md(avg_weekly),
         delta=f"Peak: {Md(weekly_ts.max())}",
         hint=f"Over {n_weeks} weeks of data",
         accent=C["sky"], icon_bg="#EFF6FF"),
])

st.markdown("<div style='margin-top:.9rem'></div>", unsafe_allow_html=True)

kpi_row([
    dict(icon="🏆", label="Best Performing Store",  value=f"Store {best_store}",
         delta=f"Revenue: {Md(store_tots.max())}",
         hint="Highest total sales in selection",
         accent=C["gold"], icon_bg="#FFFBEB"),
    dict(icon="🏅", label="Best Department",         value=f"Dept {best_dept}",
         delta=f"{Md(dept_tots.max())} total",
         hint="Top revenue-generating department",
         accent=C["saffron"], icon_bg="#FFF3E0"),
    dict(icon="📊", label="Sales Growth Rate (YoY)", value=f"{yoy:+.1f}%",
         delta=f"{years[-1]} vs {years[-2]}" if len(years)>=2 else "Single year",
         hint="Year-over-year comparison",
         accent=C["green"] if yoy>=0 else C["red"],
         icon_bg="#F0FDF4" if yoy>=0 else "#FFF1F2",
         val_color=C["green"] if yoy>=0 else C["red"]),
    dict(icon="🏪", label="Total Stores Analysed",  value=str(n_stores),
         delta=f"Types: {list(dff['Type'].unique())}",
         hint="A=Large, B=Medium, C=Small format",
         accent=C["navy3"], icon_bg="#EFF6FF"),
])

st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SALES TRENDS
# ══════════════════════════════════════════════════════════════════════════════
section("📈", "Sales Trends",
        "Weekly, Monthly, Quarterly & Yearly patterns — understand when sales peak and dip",
        C["sky"])

tab_t1, tab_t2, tab_t3, tab_t4 = st.tabs([
    "📅 Weekly Trend", "📆 Monthly Trend", "🗓️ Quarterly & Yearly", "🔥 Heatmap"
])

with tab_t1:
    subsection("📅 Total Sales Over Time — Weekly View", C["sky"])
    wkly = dff.groupby("Date")["Weekly_Sales"].sum().reset_index()
    wkly.columns = ["Date","Sales"]
    wkly["MA4"]  = wkly["Sales"].rolling(4, min_periods=1).mean()
    wkly["MA12"] = wkly["Sales"].rolling(12,min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wkly["Date"],y=wkly["Sales"],name="Weekly Sales",
        mode="lines",line=dict(color=C["sky"],width=1.5),
        fill="tozeroy",fillcolor="rgba(0,180,216,0.08)"))
    fig.add_trace(go.Scatter(x=wkly["Date"],y=wkly["MA4"],name="4-Week Average",
        line=dict(color=C["saffron"],width=2.5,dash="dot")))
    fig.add_trace(go.Scatter(x=wkly["Date"],y=wkly["MA12"],name="12-Week Trend",
        line=dict(color=C["green"],width=2.5,dash="dash")))
    # Holiday markers
    hol_dates = dff[dff["IsHoliday"]]["Date"].unique()
    for hd in hol_dates[:30]:
        fig.add_shape(type="line",
            x0=pd.Timestamp(hd).strftime("%Y-%m-%d"),
            x1=pd.Timestamp(hd).strftime("%Y-%m-%d"),
            y0=0,y1=1,xref="x",yref="paper",
            line=dict(color=C["gold"],width=1,dash="dot"),opacity=0.5)
    cs(fig, 380)
    fig.update_layout(hovermode="x unified",
        xaxis_title="Week",yaxis_title="Total Sales (₹)")
    cc_open("📅 Weekly Sales Trend",
            "Blue fill = actual sales · Orange dots = 4-week average · Green dashes = 12-week trend · Gold lines = holiday weeks")
    st.plotly_chart(fig, use_container_width=True)
    cc_close()

    # WoW growth bar
    wkly["Growth%"] = wkly["Sales"].pct_change()*100
    fig2 = go.Figure(go.Bar(
        x=wkly["Date"],y=wkly["Growth%"].fillna(0),
        marker_color=[C["green"] if v>=0 else C["red"] for v in wkly["Growth%"].fillna(0)],
        name="WoW Growth %"
    ))
    cs(fig2,240)
    fig2.update_layout(showlegend=False,yaxis_title="Growth %",hovermode="x unified")
    cc_open("📊 Week-over-Week Sales Growth %",
            "Green = sales grew vs last week · Red = sales declined · Helps spot momentum shifts")
    st.plotly_chart(fig2, use_container_width=True)
    cc_close()

with tab_t2:
    subsection("📆 Monthly Revenue Pattern", C["saffron"])
    mly = dff.groupby("YearMonth")["Weekly_Sales"].sum().reset_index()
    mly.columns = ["YearMonth","Sales"]
    mly["Date"] = pd.to_datetime(mly["YearMonth"])
    mly = mly.sort_values("Date")
    mly["MA3"] = mly["Sales"].rolling(3,min_periods=1).mean()

    fig_m = go.Figure()
    fig_m.add_trace(go.Bar(x=mly["Date"],y=mly["Sales"],name="Monthly Sales",
        marker_color=C["saffron"],opacity=0.85))
    fig_m.add_trace(go.Scatter(x=mly["Date"],y=mly["MA3"],name="3-Month Trend",
        line=dict(color=C["navy"],width=2.5),mode="lines"))
    cs(fig_m,380)
    fig_m.update_layout(hovermode="x unified",barmode="overlay",
        xaxis_title="Month",yaxis_title="Sales (₹)")
    cc_open("📆 Monthly Sales + 3-Month Moving Trend",
            "Orange bars = monthly total · Dark line = smoothed trend · Longer bars = better months")
    st.plotly_chart(fig_m, use_container_width=True)
    cc_close()

    # Month-name comparison
    mn_avg = dff.groupby(["Year","MonthName"])["Weekly_Sales"].sum().reset_index()
    mn_avg["MonthNum"] = pd.to_datetime(mn_avg["MonthName"],format="%b").dt.month
    mn_avg = mn_avg.sort_values(["Year","MonthNum"])
    fig_mn = px.line(mn_avg,x="MonthName",y="Weekly_Sales",color="Year",
        color_discrete_sequence=[C["sky"],C["saffron"],C["green"],C["gold"]],
        markers=True,labels={"Weekly_Sales":"Sales (₹)","MonthName":"Month"})
    cs(fig_mn,320)
    fig_mn.update_layout(hovermode="x unified")
    cc_open("📅 Year-over-Year Monthly Comparison",
            "Compare same months across different years — see if the business is growing year on year")
    st.plotly_chart(fig_mn, use_container_width=True)
    cc_close()

with tab_t3:
    col1, col2 = st.columns(2)
    with col1:
        qdf = dff.groupby(["Year","Quarter"])["Weekly_Sales"].sum().reset_index()
        qdf["QLabel"] = "Q"+qdf["Quarter"].astype(str)+" "+qdf["Year"].astype(str)
        fig_q = px.bar(qdf,x="QLabel",y="Weekly_Sales",color="Year",
            color_discrete_sequence=[C["navy3"],C["saffron"],C["sky"],C["green"]],
            barmode="group",labels={"Weekly_Sales":"Sales (₹)","QLabel":"Quarter"})
        cs(fig_q,380)
        cc_open("🔄 Quarterly Sales","Compare quarters across years to find seasonal patterns")
        st.plotly_chart(fig_q, use_container_width=True)
        cc_close()

    with col2:
        ydf = dff.groupby("Year")["Weekly_Sales"].sum().reset_index()
        ydf["PrevYear"] = ydf["Weekly_Sales"].shift(1)
        ydf["Growth%"]  = (ydf["Weekly_Sales"]-ydf["PrevYear"])/ydf["PrevYear"]*100
        fig_y = go.Figure()
        fig_y.add_trace(go.Bar(x=ydf["Year"].astype(str),y=ydf["Weekly_Sales"],
            name="Annual Sales",marker_color=C["navy3"],opacity=0.9,
            text=ydf["Weekly_Sales"].apply(Md),textposition="outside"))
        ax2 = go.Scatter(x=ydf["Year"].astype(str),y=ydf["Growth%"],
            name="YoY Growth %",mode="lines+markers",yaxis="y2",
            line=dict(color=C["saffron"],width=2.5),marker=dict(size=8))
        fig_y.add_trace(ax2)
        fig_y.update_layout(height=380,yaxis2=dict(overlaying="y",side="right",
            title="Growth %",showgrid=False),
            font_family="Plus Jakarta Sans",paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",margin=dict(t=30,b=30,l=5,r=5),
            legend=dict(orientation="h",y=1.05),
            xaxis=dict(showgrid=False),yaxis=dict(gridcolor=C["g1"]),
            hovermode="x unified")
        cc_open("📅 Annual Revenue & Year-over-Year Growth",
                "Blue bars = total annual sales · Orange line = growth % vs previous year")
        st.plotly_chart(fig_y, use_container_width=True)
        cc_close()

    # Season
    sdf = dff.groupby("Season")["Weekly_Sales"].agg(["sum","mean"]).reset_index()
    sdf.columns=["Season","Total","Average"]
    season_order = ["Spring","Summer","Autumn","Winter"]
    sdf["Season"] = pd.Categorical(sdf["Season"],categories=season_order,ordered=True)
    sdf = sdf.sort_values("Season")
    fig_s = px.bar(sdf,x="Season",y="Total",
        color="Season",text=sdf["Total"].apply(Md),
        color_discrete_sequence=[C["green"],C["saffron"],C["gold"],C["sky"]],
        labels={"Total":"Total Sales (₹)"})
    fig_s.update_traces(textposition="outside")
    cs(fig_s,300)
    fig_s.update_layout(showlegend=False)
    cc_open("🌿 Seasonal Sales Pattern","Which season brings the most revenue?")
    st.plotly_chart(fig_s, use_container_width=True)
    cc_close()

with tab_t4:
    hmap_pv = dff.pivot_table(values="Weekly_Sales",index="Store",
                               columns="WeekNum",aggfunc="sum",fill_value=0)
    top15 = hmap_pv.loc[hmap_pv.sum(axis=1).nlargest(min(15,len(hmap_pv))).index]
    fig_h = px.imshow(top15,aspect="auto",
        color_continuous_scale=[[0,"#EFF6FF"],[0.4,C["sky"]],[0.75,C["navy3"]],[1,C["navy"]]],
        labels=dict(x="Week of Year",y="Store",color="Sales (₹)"))
    fig_h.update_layout(height=440,font_family="Plus Jakarta Sans",
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30,b=30,l=5,r=5))
    cc_open("🔥 Store × Week Sales Heatmap",
            "Darker blue = higher sales that week. Quickly spot which stores and weeks perform best.")
    st.plotly_chart(fig_h, use_container_width=True)
    cc_close()

st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PROFIT & LOSS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
section("💰", "Profit & Loss Analysis",
        "Estimated profit, loss zones, declining stores — know where money is made and lost",
        C["green"])

# Compute P&L per store per month
pl_store = dff.groupby(["Store","YearMonth"]).agg(
    Sales=("Weekly_Sales","sum")).reset_index()
pl_store["Profit"]  = pl_store["Sales"] * MARGIN
pl_store["IsLoss"]  = pl_store["Sales"] < 0

# Store-level summary
store_pl = dff.groupby("Store").agg(
    TotalSales=("Weekly_Sales","sum"),
    AvgWeekly=("Weekly_Sales","mean"),
    Std=("Weekly_Sales","std"),
    NegWeeks=("IsLoss","sum"),
).reset_index()
store_pl["Profit"]  = store_pl["TotalSales"] * MARGIN
store_pl["CV"]      = store_pl["Std"] / store_pl["AvgWeekly"]
store_pl_sorted     = store_pl.sort_values("TotalSales", ascending=False)

col1, col2 = st.columns([3,2])

with col1:
    top10  = store_pl_sorted.head(10)
    colors = []
    for i,row in top10.iterrows():
        if row["TotalSales"]<0: colors.append(C["red"])
        elif row["Profit"] > store_pl["Profit"].quantile(0.75): colors.append(C["green"])
        else: colors.append(C["saffron"])

    subsection("💰 Store-wise Revenue vs Estimated Profit", C["green"])
    fig_pl = go.Figure()
    fig_pl.add_trace(go.Bar(
        y=top10["Store"].astype(str), x=top10["TotalSales"],
        orientation="h", name="Revenue",
        marker_color=colors,
        text=top10["TotalSales"].apply(Md), textposition="outside"
    ))
    fig_pl.add_trace(go.Bar(
        y=top10["Store"].astype(str), x=top10["Profit"],
        orientation="h", name="Est. Profit (30%)",
        marker_color=[
            "rgba({},{},{},0.45)".format(
                int(c[1:3],16), int(c[3:5],16), int(c[5:7],16)
            ) for c in colors
        ],
        text=top10["Profit"].apply(Md), textposition="outside"
    ))
    cs(fig_pl,420)
    fig_pl.update_layout(
        barmode="overlay",showlegend=True,
        yaxis=dict(autorange="reversed",tickprefix="Store "),
        xaxis_title="Amount (₹)",hovermode="y unified"
    )
    cc_open("💰 Top 10 Stores — Revenue vs Estimated Profit",
            "Solid bar = total revenue · Transparent bar = estimated 30% profit. Green = high profit zone.")
    st.plotly_chart(fig_pl, use_container_width=True)
    cc_close()

with col2:
    # P&L gauge-style donut
    pos_sales = float(dff[dff["Weekly_Sales"]>=0]["Weekly_Sales"].sum())
    neg_sales = float(abs(dff[dff["Weekly_Sales"]<0]["Weekly_Sales"].sum()))
    profit_est = pos_sales * MARGIN

    fig_do = go.Figure(go.Pie(
        values=[profit_est, pos_sales-profit_est, neg_sales if neg_sales>0 else 0.001],
        labels=["Est. Profit","Cost of Sales","Loss Zones"],
        hole=0.58,
        marker_colors=[C["green"],C["sky"],C["red"]],
        textfont_size=11, textposition="inside"
    ))
    fig_do.update_layout(height=280,
        font_family="Plus Jakarta Sans",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20,b=20,l=20,r=20),
        legend=dict(orientation="h",y=-0.05,font_size=10),
        annotations=[dict(text="P&L",x=0.5,y=0.5,
                          font_size=20,font_color=C["navy"],
                          font_family="Plus Jakarta Sans",showarrow=False)]
    )
    cc_open("🥧 Profit vs Cost vs Loss Breakdown","Estimated financial health of your selection")
    st.plotly_chart(fig_do, use_container_width=True)
    cc_close()

    # Declining stores
    st.markdown("<div style='margin-top:.5rem'></div>", unsafe_allow_html=True)
    cc_open("⚠️ Declining Sales Alerts")
    declining = store_pl[store_pl["NegWeeks"]>0].sort_values("NegWeeks",ascending=False).head(5)
    if declining.empty:
        st.markdown('<div class="a-green"><div class="at">✅ No Loss Weeks Detected</div>'
                    'All selected stores show positive sales across the period.</div>',
                    unsafe_allow_html=True)
    else:
        for _,r in declining.iterrows():
            st.markdown(f'<div class="a-red">'
                        f'<div class="at">🔴 Store {int(r["Store"])} — {int(r["NegWeeks"])} loss weeks</div>'
                        f'Avg weekly: {Md(r["AvgWeekly"])} · Needs immediate attention</div>',
                        unsafe_allow_html=True)
    cc_close()

# Profit trend over time
pl_monthly = dff.groupby("YearMonth")["Weekly_Sales"].sum().reset_index()
pl_monthly["Date"]     = pd.to_datetime(pl_monthly["YearMonth"])
pl_monthly             = pl_monthly.sort_values("Date")
pl_monthly["Profit"]   = pl_monthly["Weekly_Sales"] * MARGIN
pl_monthly["Loss"]     = pl_monthly["Weekly_Sales"].apply(lambda x: abs(x) if x<0 else 0)
pl_monthly["Status"]   = pl_monthly["Weekly_Sales"].apply(lambda x: "Profit" if x>=0 else "Loss")

fig_plt = go.Figure()
for status, color in [("Profit",C["green"]),("Loss",C["red"])]:
    sub = pl_monthly[pl_monthly["Status"]==status]
    fig_plt.add_trace(go.Bar(
        x=sub["Date"],y=sub["Weekly_Sales"].abs(),
        name=status,marker_color=color,opacity=0.85
    ))
cs(fig_plt,320)
fig_plt.update_layout(barmode="stack",hovermode="x unified",
    xaxis_title="Month",yaxis_title="Sales (₹)")
cc_open("📊 Monthly Profit & Loss Timeline",
        "Green = profitable months · Red = loss months · Tall bars = high volume")
st.plotly_chart(fig_plt, use_container_width=True)
cc_close()

# Department profit ranking
dept_pl = dff.groupby("Dept")["Weekly_Sales"].sum().reset_index()
dept_pl.columns = ["Dept","Sales"]
dept_pl["Profit"] = dept_pl["Sales"] * MARGIN
dept_pl = dept_pl.sort_values("Sales",ascending=False)

col1b, col2b = st.columns(2)
with col1b:
    top_dp = dept_pl.head(10)
    fig_dp = px.bar(top_dp,x="Dept",y="Sales",
        color="Profit",color_continuous_scale=[[0,C["sky"]],[0.5,C["green"]],[1,"#006400"]],
        text=top_dp["Sales"].apply(Md),labels={"Sales":"Sales (₹)","Dept":"Department"})
    fig_dp.update_traces(textposition="outside")
    cs(fig_dp,340)
    fig_dp.update_layout(showlegend=False,coloraxis_showscale=False,
        xaxis=dict(tickprefix="Dept "))
    cc_open("🏅 Top 10 Most Profitable Departments","Greener = higher profit estimate")
    st.plotly_chart(fig_dp, use_container_width=True)
    cc_close()

with col2b:
    bot_dp = dept_pl.tail(10)
    fig_bp = px.bar(bot_dp,x="Dept",y="Sales",
        color="Sales",color_continuous_scale=[[0,C["red"]],[0.5,C["red2"]],[1,C["gold"]]],
        text=bot_dp["Sales"].apply(Md),labels={"Sales":"Sales (₹)","Dept":"Department"})
    fig_bp.update_traces(textposition="outside")
    cs(fig_bp,340)
    fig_bp.update_layout(showlegend=False,coloraxis_showscale=False,
        xaxis=dict(tickprefix="Dept "))
    cc_open("⚠️ Bottom 10 Departments — Need Attention","These departments need marketing or restructuring")
    st.plotly_chart(fig_bp, use_container_width=True)
    cc_close()

st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — STORE PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
section("🏪", "Store Performance Analysis",
        "Top performers, worst stores, type comparison and store-by-store benchmarking",
        C["navy3"])

tab_s1, tab_s2, tab_s3 = st.tabs([
    "🏆 Rankings",  "📊 Comparison", "📦 Store Type"
])

with tab_s1:
    st_full = dff.groupby("Store").agg(
        Total=("Weekly_Sales","sum"),
        Avg=("Weekly_Sales","mean"),
        Best=("Weekly_Sales","max"),
        Weeks=("Date","nunique"),
    ).reset_index().sort_values("Total",ascending=False)
    st_full["Profit"] = st_full["Total"]*MARGIN
    st_full["Rank"]   = range(1,len(st_full)+1)

    col1c,col2c = st.columns([3,2])
    with col1c:
        top10s = st_full.head(10)
        rank_colors = [C["gold"],C["g2"],C["saffron"]]+[C["navy3"]]*20
        fig_r = go.Figure(go.Bar(
            y=[f"Store {int(s)}" for s in top10s["Store"]],
            x=top10s["Total"],
            orientation="h",
            marker_color=rank_colors[:len(top10s)],
            text=top10s["Total"].apply(Md),textposition="outside"
        ))
        cs(fig_r,400)
        fig_r.update_layout(showlegend=False,
            yaxis=dict(autorange="reversed"),xaxis_title="Total Sales (₹)")
        cc_open("🏆 Top 10 Performing Stores","Gold=1st, Silver=2nd, Orange=3rd. Size of bar shows revenue gap.")
        st.plotly_chart(fig_r, use_container_width=True)
        cc_close()

    with col2c:
        # Rank cards
        cc_open("🥇 Store Podium")
        for i,(_, row) in enumerate(st_full.head(5).iterrows()):
            num_class = "gold" if i==0 else "silver" if i==1 else "bronze" if i==2 else ""
            st.markdown(f"""
            <div class="rrow">
              <div class="rnum {num_class}">{i+1}</div>
              <div style="flex:1">
                <div style="font-weight:700;color:{C['navy']}">Store {int(row['Store'])}</div>
                <div style="font-size:.72rem;color:{C['g2']}">{row['Weeks']} weeks · Avg {Md(row['Avg'])}/wk</div>
              </div>
              <div style="text-align:right">
                <div style="font-weight:800;color:{C['green']};font-size:.95rem">{Md(row['Total'])}</div>
                <div style="font-size:.68rem;color:{C['g2']}">profit ~{Md(row['Profit'])}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        cc_close()

        # Worst 3
        cc_open("⚠️ Stores Needing Attention")
        for i,(_, row) in enumerate(st_full.tail(3).iterrows()):
            st.markdown(f"""
            <div class="rrow" style="border-color:#FECDD3">
              <div class="rnum" style="background:{C['red']}">{int(st_full.shape[0]-2+i)}</div>
              <div style="flex:1">
                <div style="font-weight:700;color:{C['navy']}">Store {int(row['Store'])}</div>
                <div style="font-size:.72rem;color:{C['g2']}">{row['Weeks']} weeks</div>
              </div>
              <div style="text-align:right">
                <div style="font-weight:800;color:{C['red']};font-size:.95rem">{Md(row['Total'])}</div>
                <div style="font-size:.68rem;color:{C['g2']}">Needs review</div>
              </div>
            </div>""", unsafe_allow_html=True)
        cc_close()

with tab_s2:
    # Store vs store weekly comparison
    top5s = st_full.head(5)["Store"].tolist()
    cmp_sel = st.multiselect("Choose stores to compare (max 6)",
                              all_stores, default=[int(s) for s in top5s[:5]],
                              max_selections=6)
    if cmp_sel:
        cmp_df = dff[dff["Store"].isin(cmp_sel)].groupby(["Date","Store"])["Weekly_Sales"].sum().reset_index()
        fig_cmp = px.line(cmp_df,x="Date",y="Weekly_Sales",color="Store",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"Weekly_Sales":"Sales (₹)"},markers=False)
        cs(fig_cmp,380)
        fig_cmp.update_layout(hovermode="x unified")
        cc_open("📊 Store vs Store Sales Race","Track how selected stores perform week by week relative to each other")
        st.plotly_chart(fig_cmp, use_container_width=True)
        cc_close()

        # Radar chart
        cats = ["Total Sales","Avg Weekly","Best Week","Consistency","Weeks Active"]
        fig_rad = go.Figure()
        for store in cmp_sel[:5]:
            row = st_full[st_full["Store"]==store]
            if row.empty: continue
            row = row.iloc[0]
            vals = [
                row["Total"]/st_full["Total"].max(),
                row["Avg"]/st_full["Avg"].max(),
                row["Best"]/st_full["Best"].max(),
                1-row.get("CV",0.5),
                row["Weeks"]/st_full["Weeks"].max(),
            ]
            fig_rad.add_trace(go.Scatterpolar(
                r=vals+[vals[0]],theta=cats+[cats[0]],
                name=f"Store {int(store)}",fill="toself",opacity=0.5
            ))
        fig_rad.update_layout(height=380,
            polar=dict(radialaxis=dict(visible=True,range=[0,1])),
            showlegend=True,font_family="Plus Jakarta Sans",
            paper_bgcolor="rgba(0,0,0,0)",margin=dict(t=30,b=30,l=30,r=30))
        cc_open("🕸️ Store Performance Radar","Bigger shape = better overall performance across all metrics")
        st.plotly_chart(fig_rad, use_container_width=True)
        cc_close()

with tab_s3:
    tp_data = dff.groupby("Type")["Weekly_Sales"].agg(["sum","mean","count"]).reset_index()
    tp_data.columns = ["Type","Total","Avg","Count"]
    tp_data["TypeName"] = tp_data["Type"].map({"A":"Type A (Large)","B":"Type B (Medium)","C":"Type C (Small)"})
    tp_data["Profit"] = tp_data["Total"]*MARGIN

    col1d,col2d = st.columns(2)
    with col1d:
        fig_tp = px.bar(tp_data,x="TypeName",y="Total",color="TypeName",
            color_discrete_sequence=[C["navy3"],C["saffron"],C["sky"]],
            text=tp_data["Total"].apply(Md),labels={"Total":"Total Sales (₹)","TypeName":"Store Type"})
        fig_tp.update_traces(textposition="outside")
        cs(fig_tp,360)
        fig_tp.update_layout(showlegend=False)
        cc_open("🏬 Total Sales by Store Type","Large format stores (Type A) typically generate more revenue")
        st.plotly_chart(fig_tp, use_container_width=True)
        cc_close()

    with col2d:
        fig_pie = go.Figure(go.Pie(
            values=tp_data["Total"].tolist(),
            labels=tp_data["TypeName"].tolist(),
            hole=0.5,
            marker_colors=[C["navy3"],C["saffron"],C["sky"]],
            textfont_size=12
        ))
        fig_pie.update_layout(height=360,
            font_family="Plus Jakarta Sans",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20,b=20,l=20,r=20),
            annotations=[dict(text="Sales Mix",x=0.5,y=0.5,
                font_size=14,font_color=C["navy"],showarrow=False)]
        )
        cc_open("🥧 Sales Contribution by Store Type","Pie shows which store format contributes most revenue")
        st.plotly_chart(fig_pie, use_container_width=True)
        cc_close()

    # Store size vs sales scatter
    sc_data = dff.groupby(["Store","Type","Size"])["Weekly_Sales"].agg(["sum","mean"]).reset_index() if "Size" in dff.columns else None
    if sc_data is not None:
        sc_data.columns=["Store","Type","Size","Total","Avg"]
        fig_sc = px.scatter(sc_data,x="Size",y="Total",color="Type",size="Total",
            color_discrete_sequence=[C["navy3"],C["saffron"],C["sky"]],
            hover_data=["Store"],size_max=40,
            labels={"Total":"Total Sales (₹)","Size":"Store Size (sq ft)"})
        cs(fig_sc,360)
        cc_open("📐 Store Size vs Total Revenue","Bigger stores tend to earn more — but efficiency matters")
        st.plotly_chart(fig_sc, use_container_width=True)
        cc_close()

st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DEPARTMENT PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
section("🏷", "Department Performance",
        "Which departments make money, which are declining, and where to focus resources",
        C["gold"])

dept_full = dff.groupby("Dept").agg(
    Total=("Weekly_Sales","sum"),
    Avg=("Weekly_Sales","mean"),
    Std=("Weekly_Sales","std"),
    Weeks=("Date","nunique"),
).reset_index().sort_values("Total",ascending=False)
dept_full["Profit"] = dept_full["Total"]*MARGIN
dept_full["CV"]     = dept_full["Std"]/dept_full["Avg"]

col1e, col2e = st.columns([2,1])
with col1e:
    top15d = dept_full.head(15)
    fig_dpt = go.Figure(go.Bar(
        y=[f"Dept {int(d)}" for d in top15d["Dept"]],
        x=top15d["Total"],orientation="h",
        marker_color=[C["green"] if p>dept_full["Profit"].quantile(0.75) else
                       C["saffron"] if p>dept_full["Profit"].median() else C["sky"]
                       for p in top15d["Profit"]],
        text=top15d["Total"].apply(Md),textposition="outside"
    ))
    cs(fig_dpt,460)
    fig_dpt.update_layout(showlegend=False,
        yaxis=dict(autorange="reversed"),xaxis_title="Total Sales (₹)")
    cc_open("🏆 Top 15 Departments by Revenue",
            "Green = top quartile profit · Orange = above median · Blue = below median")
    st.plotly_chart(fig_dpt, use_container_width=True)
    cc_close()

with col2e:
    # Dept matrix
    fig_dm = px.scatter(dept_full.head(30),x="Avg",y="Total",
        size="Total",color="CV",size_max=45,
        color_continuous_scale=[[0,C["green"]],[0.5,C["gold"]],[1,C["red"]]],
        hover_name=dept_full.head(30)["Dept"].apply(lambda x: f"Dept {int(x)}"),
        text="Dept",
        labels={"Avg":"Avg Weekly Sales","Total":"Total Sales","CV":"Volatility"})
    fig_dm.update_traces(textposition="top center",textfont_size=9)
    cs(fig_dm,460)
    fig_dm.update_layout(coloraxis_showscale=False,showlegend=False)
    cc_open("📊 Dept Performance Matrix",
            "Top-right = star depts (high total + high avg). Red dots = volatile/risky.")
    st.plotly_chart(fig_dm, use_container_width=True)
    cc_close()

# Dept growth trend
tab_d1, tab_d2 = st.tabs(["📈 Top 5 Dept Trends", "⚠️ Declining Departments"])

with tab_d1:
    top5d = dept_full.head(5)["Dept"].tolist()
    d5_sel = st.multiselect("Select departments", all_depts,
                             default=[int(d) for d in top5d[:5]], max_selections=6,
                             key="d5sel")
    if d5_sel:
        d5_df = dff[dff["Dept"].isin(d5_sel)].groupby(["YearMonth","Dept"])["Weekly_Sales"].sum().reset_index()
        d5_df["Date"] = pd.to_datetime(d5_df["YearMonth"])
        d5_df = d5_df.sort_values("Date")
        fig_d5 = px.line(d5_df,x="Date",y="Weekly_Sales",color="Dept",
            color_discrete_sequence=px.colors.qualitative.Set1,markers=True,
            labels={"Weekly_Sales":"Monthly Sales (₹)"})
        cs(fig_d5,380)
        fig_d5.update_layout(hovermode="x unified")
        cc_open("📈 Department Monthly Sales Trend","Track how individual departments perform month by month")
        st.plotly_chart(fig_d5, use_container_width=True)
        cc_close()

with tab_d2:
    # Departments with declining trend (negative avg WoW)
    dept_wow = dff.groupby(["Dept","Date"])["Weekly_Sales"].sum().reset_index()
    dept_wow = dept_wow.sort_values(["Dept","Date"])
    dept_wow["WoW"] = dept_wow.groupby("Dept")["Weekly_Sales"].pct_change()*100
    dept_trend = dept_wow.groupby("Dept")["WoW"].mean().reset_index()
    dept_trend.columns=["Dept","AvgWoW"]
    declining_d = dept_trend[dept_trend["AvgWoW"]<-0.5].sort_values("AvgWoW").head(10)

    if declining_d.empty:
        st.markdown('<div class="a-green"><div class="at">✅ No significantly declining departments</div>'
                    'All departments show stable or growing sales trends.</div>',
                    unsafe_allow_html=True)
    else:
        fig_dec = go.Figure(go.Bar(
            x=[f"Dept {int(d)}" for d in declining_d["Dept"]],
            y=declining_d["AvgWoW"],
            marker_color=C["red"],
            text=[f"{v:.1f}%" for v in declining_d["AvgWoW"]],
            textposition="outside"
        ))
        cs(fig_dec,340)
        fig_dec.update_layout(showlegend=False,yaxis_title="Avg WoW Growth %",
            title_text="⚠️ These departments are declining — intervention needed")
        cc_open("⚠️ Departments with Declining Sales Trend",
                "Negative % = sales falling week over week on average. Act now!")
        st.plotly_chart(fig_dec, use_container_width=True)
        cc_close()

st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — HOLIDAY & SEASONAL IMPACT
# ══════════════════════════════════════════════════════════════════════════════
section("🎄", "Holiday & Seasonal Impact",
        "How festivals and special events affect Walmart sales — plan your stock and staff accordingly",
        C["saffron"])

tab_h1, tab_h2 = st.tabs(["🎄 Holiday Analysis", "🌿 Seasonal Patterns"])

with tab_h1:
    ch1,ch2 = st.columns(2)
    with ch1:
        ha = dff.groupby("IsHoliday")["Weekly_Sales"].agg(["mean","median","sum"]).reset_index()
        ha["Label"] = ha["IsHoliday"].map({True:"🎄 Holiday Weeks",False:"📅 Normal Weeks"})
        fig_hb = go.Figure()
        fig_hb.add_bar(x=ha["Label"],y=ha["mean"],name="Average Sales",
            marker_color=[C["saffron"],C["navy3"]],
            text=ha["mean"].apply(Md),textposition="outside")
        fig_hb.add_bar(x=ha["Label"],y=ha["median"],name="Median Sales",
            marker_color=["#FFD6A5","#90B4CE"],
            text=ha["median"].apply(Md),textposition="outside")
        fig_hb.update_layout(barmode="group",height=360,
            font_family="Plus Jakarta Sans",
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=30,b=30,l=5,r=5),
            legend=dict(orientation="h",y=1.05),
            xaxis=dict(showgrid=False),yaxis=dict(gridcolor=C["g1"]))
        cc_open("🎄 Holiday vs Normal — Sales Comparison",
                "Do Walmart stores earn more during holidays? See the difference here.")
        st.plotly_chart(fig_hb, use_container_width=True)
        cc_close()

    with ch2:
        # Box plot
        fig_bx = px.box(dff,x="IsHoliday",y="Weekly_Sales",
            color="IsHoliday",
            color_discrete_map={True:C["saffron"],False:C["navy3"]},
            labels={"IsHoliday":"Is Holiday Week","Weekly_Sales":"Weekly Sales (₹)"},
            category_orders={"IsHoliday":[True,False]})
        fig_bx.update_layout(height=360,font_family="Plus Jakarta Sans",
            showlegend=False,paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",margin=dict(t=30,b=30,l=5,r=5),
            xaxis=dict(tickvals=[True,False],ticktext=["Holiday","Normal"],showgrid=False),
            yaxis=dict(gridcolor=C["g1"]))
        cc_open("📊 Sales Distribution — Holiday vs Normal",
                "Box shows spread and median. Higher box = higher typical sales that week.")
        st.plotly_chart(fig_bx, use_container_width=True)
        cc_close()

    # Holiday lift by store
    _has_both = dff["IsHoliday"].any() and (~dff["IsHoliday"]).any()
    if _has_both:
        hs_data = dff.groupby(["Store","IsHoliday"])["Weekly_Sales"].mean().reset_index()
        hp = hs_data.pivot(index="Store",columns="IsHoliday",
                           values="Weekly_Sales").reset_index().dropna()
        hp.columns = hp.columns.astype(str)
        rcols = {c:"NonHol" for c in hp.columns if str(c) in ("False","0")}
        rcols.update({c:"Hol" for c in hp.columns if str(c) in ("True","1")})
        hp = hp.rename(columns=rcols)
        if "Hol" in hp.columns and "NonHol" in hp.columns:
            hp["Lift%"] = (hp["Hol"]-hp["NonHol"])/hp["NonHol"]*100
            hp = hp.sort_values("Lift%",ascending=False)
            fig_lf = go.Figure(go.Bar(
                x=hp["Store"].astype(str),y=hp["Lift%"],
                marker_color=[C["green"] if v>=0 else C["red"] for v in hp["Lift%"]],
                text=[f"{v:+.1f}%" for v in hp["Lift%"]],
                textposition="outside",textfont_size=9
            ))
            cs(fig_lf,340)
            fig_lf.update_layout(showlegend=False,
                xaxis_title="Store",yaxis_title="Holiday Sales Lift (%)",
                xaxis=dict(tickprefix="S"))
            cc_open("📈 Holiday Sales Lift by Store",
                    "Green = store earns MORE in holiday weeks · Red = store earns LESS (unusual!)")
            st.plotly_chart(fig_lf, use_container_width=True)
            cc_close()

with tab_h2:
    col1f,col2f = st.columns(2)
    with col1f:
        # Season vs sales
        sea_df = dff.groupby(["Season","Year"])["Weekly_Sales"].sum().reset_index()
        fig_sea = px.bar(sea_df,x="Season",y="Weekly_Sales",color="Year",
            color_discrete_sequence=[C["navy3"],C["saffron"],C["sky"]],
            barmode="group",category_orders={"Season":["Spring","Summer","Autumn","Winter"]},
            labels={"Weekly_Sales":"Sales (₹)"})
        cs(fig_sea,360)
        cc_open("🌿 Seasonal Sales by Year","Which season performs best? Compare across years.")
        st.plotly_chart(fig_sea, use_container_width=True)
        cc_close()

    with col2f:
        # Month heatmap
        mheat = dff.groupby(["Year","Month"])["Weekly_Sales"].sum().reset_index()
        mheat_pv = mheat.pivot(index="Year",columns="Month",values="Weekly_Sales").fillna(0)
        mheat_pv.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                             "Jul","Aug","Sep","Oct","Nov","Dec"][:len(mheat_pv.columns)]
        fig_mh = px.imshow(mheat_pv,aspect="auto",
            color_continuous_scale=[[0,"#F0F9FF"],[0.5,C["sky"]],[1,C["navy"]]],
            labels=dict(x="Month",y="Year",color="Sales (₹)"))
        fig_mh.update_layout(height=360,font_family="Plus Jakarta Sans",
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=30,b=30,l=5,r=5))
        cc_open("🗓️ Month × Year Sales Heatmap","Darker = higher sales. Identify peak months every year.")
        st.plotly_chart(fig_mh, use_container_width=True)
        cc_close()

    # Weekly sales pattern within year
    wk_avg = dff.groupby("WeekNum")["Weekly_Sales"].mean().reset_index()
    wk_avg.columns = ["Week","AvgSales"]
    fig_wkp = go.Figure(go.Scatter(
        x=wk_avg["Week"],y=wk_avg["AvgSales"],mode="lines+markers",
        line=dict(color=C["saffron"],width=2.5),marker=dict(size=5),
        fill="tozeroy",fillcolor="rgba(255,107,0,0.08)"
    ))
    cs(fig_wkp,300)
    fig_wkp.update_layout(showlegend=False,
        xaxis_title="Week Number (1=Jan, 52=Dec)",yaxis_title="Avg Sales (₹)")
    cc_open("📅 Average Sales by Week of Year",
            "See which weeks of the year consistently perform best — plan promotions accordingly")
    st.plotly_chart(fig_wkp, use_container_width=True)
    cc_close()

st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — WEEKLY DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
section("📅", "Weekly Sales Deep Dive",
        f"Store {wa_store} · Dept {wa_dept} — Week-by-week gain/loss analysis with full breakdown",
        C["sky"])

wa_ts = (raw[(raw["Store"]==wa_store)&(raw["Dept"]==wa_dept)]
         .groupby("Date")["Weekly_Sales"].sum().sort_index()
         .tail(wa_n).reset_index())
wa_ts.columns = ["Date","Sales"]
wa_ts["Prev"]     = wa_ts["Sales"].shift(1)
wa_ts["Chg"]      = wa_ts["Sales"] - wa_ts["Prev"]
wa_ts["Chg%"]     = wa_ts["Chg"]/wa_ts["Prev"].abs()*100
wa_ts["Status"]   = wa_ts["Chg"].apply(
    lambda x: "GAIN" if (not np.isnan(x) and x>0)
               else ("LOSS" if (not np.isnan(x) and x<0) else "FLAT"))

tg = wa_ts[wa_ts["Status"]=="GAIN"]["Chg"].sum()
tl = wa_ts[wa_ts["Status"]=="LOSS"]["Chg"].sum()
nt = tg+tl

nc = C["green"] if nt>=0 else C["red"]
kpi_row([
    dict(icon="📅", label="Weeks Analysed",  value=str(len(wa_ts)),
         hint=f"Store {wa_store} · Dept {wa_dept}",
         accent=C["sky"], icon_bg="#EFF6FF"),
    dict(icon="✅", label="Total Gains",      value=Md(tg),
         delta=f"{(wa_ts['Status']=='GAIN').sum()} gain weeks",
         accent=C["green"], icon_bg="#F0FDF4", val_color=C["green"]),
    dict(icon="❌", label="Total Losses",     value=Md(abs(tl)),
         delta=f"{(wa_ts['Status']=='LOSS').sum()} loss weeks",
         accent=C["red"], icon_bg="#FFF1F2", val_color=C["red"]),
    dict(icon="📊", label="Net Change",       value=Md(abs(nt)),
         delta="Overall Gain" if nt>=0 else "Overall Loss",
         accent=nc, icon_bg="#F0FDF4" if nt>=0 else "#FFF1F2", val_color=nc),
])

st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

cw1,cw2 = st.columns([3,2])
with cw1:
    subsection("📊 Week-over-Week Change — Gain vs Loss Waterfall", C["green"])
    # Waterfall
    wf_colors = [C["green"] if c>0 else C["red"] if c<0 else C["g2"]
                 for c in wa_ts["Chg"].fillna(0)]
    fig_wf = go.Figure(go.Bar(
        x=wa_ts["Date"].dt.strftime("%b %d"),
        y=wa_ts["Chg"].fillna(0),
        marker_color=wf_colors,
        text=[f"{'+'if v>0 else ''}{v:,.0f}" for v in wa_ts["Chg"].fillna(0)],
        textposition="outside",textfont_size=9
    ))
    fig_wf.add_hline(y=0,line_width=2,line_color=C["g3"])
    cs(fig_wf,320)
    fig_wf.update_layout(showlegend=False,xaxis_title="Week",
        yaxis_title="Change vs Prior Week (₹)")
    cc_open("📊 Week-over-Week Change Waterfall",
            "Green bars = sales went UP vs last week · Red bars = sales went DOWN")
    st.plotly_chart(fig_wf, use_container_width=True)
    cc_close()

with cw2:
    # Cumulative
    wa_ts["Cumul"] = wa_ts["Sales"].cumsum()
    fig_cu = go.Figure(go.Scatter(
        x=wa_ts["Date"],y=wa_ts["Cumul"],mode="lines",
        line=dict(color=C["navy3"],width=2.5),
        fill="tozeroy",fillcolor="rgba(30,58,95,0.08)"
    ))
    cs(fig_cu,320)
    fig_cu.update_layout(showlegend=False,yaxis_title="Cumulative Sales (₹)")
    cc_open("📉 Cumulative Revenue Progress","Running total — steeper slope = faster growth")
    st.plotly_chart(fig_cu, use_container_width=True)
    cc_close()

# This week vs last week line
fig_tw = go.Figure()
mk = [C["green"] if s=="GAIN" else C["red"] if s=="LOSS" else C["g2"] for s in wa_ts["Status"]]
fig_tw.add_trace(go.Scatter(x=wa_ts["Date"],y=wa_ts["Prev"],
    name="Previous Week",mode="lines",
    line=dict(color=C["g2"],width=1.5,dash="dot"),opacity=0.8))
fig_tw.add_trace(go.Scatter(x=wa_ts["Date"],y=wa_ts["Sales"],
    name="This Week",mode="lines+markers",
    line=dict(color=C["sky"],width=2.5),
    marker=dict(size=8,color=mk)))
cs(fig_tw,300)
fig_tw.update_layout(hovermode="x unified",yaxis_title="Sales (₹)")
cc_open("📈 This Week vs Previous Week",
        "Blue line = current week · Dotted gray = prior week · Marker color = GAIN (green) or LOSS (red)")
st.plotly_chart(fig_tw, use_container_width=True)
cc_close()

# Detailed weekly table
cc_open("📋 Detailed Weekly Breakdown Table",
        "Complete row-by-row view — Date | Sales (₹) | Previous Week (₹) | Change Amount | Change % | GAIN or LOSS status")
st.markdown("""
<div style="display:grid;grid-template-columns:1.4fr 1fr 1fr 1fr 1fr 1.1fr;gap:.4rem;
            padding:.58rem .85rem;border-radius:8px;margin:.2rem 0;
            background:#0F1B2D;color:white;font-weight:700;
            font-size:.67rem;letter-spacing:.07em;text-transform:uppercase;">
  <div>📅 Week / Date</div>
  <div>💰 Sales (₹)</div>
  <div>⏮ Previous Week (₹)</div>
  <div>📊 Change Amount</div>
  <div>📈 Change %</div>
  <div>🏷 Status</div>
</div>""", unsafe_allow_html=True)

for _,row in wa_ts.sort_values("Date",ascending=False).iterrows():
    is_gain = row["Status"]=="GAIN"
    is_loss = row["Status"]=="LOSS"
    row_bg  = "#F0FDF4" if is_gain else "#FFF1F2" if is_loss else "#F8FAFC"
    row_bdr = "#BBF7D0" if is_gain else "#FECDD3" if is_loss else "#E2E8F0"
    gc      = "#059669" if is_gain else "#DC2626" if is_loss else "#475569"
    tag_bg  = "#DCFCE7" if is_gain else "#FFE4E6" if is_loss else "#F1F5F9"
    tag_col = "#14532D" if is_gain else "#9F1239" if is_loss else "#475569"
    tag_txt = "✅ GAIN" if is_gain else "❌ LOSS" if is_loss else "➡ FLAT"
    prev_s  = Md(row["Prev"]) if not np.isnan(row["Prev"]) else "—"
    chg_v   = row["Chg"]
    chg_s   = (f"+{Md(chg_v)}" if chg_v>0 else Md(chg_v)) if not np.isnan(chg_v) else "—"
    pct_v   = row["Chg%"]
    pct_s   = f"{pct_v:+.1f}%" if not np.isnan(pct_v) else "—"
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1.4fr 1fr 1fr 1fr 1fr 1.1fr;
                gap:.4rem;padding:.65rem .9rem;border-radius:8px;margin:.2rem 0;
                background:{row_bg};border:1px solid {row_bdr};
                font-size:.79rem;align-items:center;">
      <div style="color:#0F1B2D;font-weight:700">{row['Date'].strftime('%d %b %Y')}</div>
      <div style="color:#0F1B2D;font-weight:700">{Md(row['Sales'])}</div>
      <div style="color:#475569">{prev_s}</div>
      <div style="color:{gc};font-weight:700">{chg_s}</div>
      <div style="color:{gc};font-weight:700">{pct_s}</div>
      <div><span style="display:inline-block;padding:.15rem .55rem;border-radius:99px;
                         font-size:.67rem;font-weight:700;background:{tag_bg};color:{tag_col}">
           {tag_txt}</span></div>
    </div>""", unsafe_allow_html=True)
cc_close()

st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — AI SALES FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
section("🤖", "AI Sales Forecasting",
        f"SARIMA model predicts next {fc_weeks} weeks for Store {fc_store} · Dept {fc_dept}",
        C["green"])

fc_ts = (raw[(raw["Store"]==fc_store)&(raw["Dept"]==fc_dept)]
         .groupby("Date")["Weekly_Sales"].sum().sort_index())

if len(fc_ts) < 26:
    st.warning(f"⚠️ Not enough data for Store {fc_store} · Dept {fc_dept} "
               "(need 26+ weeks). Please choose a different combination.")
else:
    with st.spinner("🧠 AI model is learning from historical data and generating forecast..."):
        try:
            if not SARIMA_OK: raise ImportError("statsmodels not installed")
            mdl = SARIMAX(fc_ts, order=(1,1,1), seasonal_order=(1,1,0,52),
                          enforce_stationarity=False, enforce_invertibility=False)
            res = mdl.fit(disp=False, maxiter=200)
            fc_obj    = res.get_forecast(steps=fc_weeks)
            fc_mean   = fc_obj.predicted_mean
            ci        = fc_obj.conf_int(alpha=0.2)
            ci_lo, ci_hi = ci.iloc[:,0], ci.iloc[:,1]
            model_lbl = "SARIMA(1,1,1)(1,1,0)[52] — Seasonal AI Model"
        except Exception:
            st.info("ℹ️ Using trend-based fallback forecast (install statsmodels for full AI model).")
            last_v  = float(fc_ts.iloc[-1])
            trend_v = float(fc_ts.diff().dropna().mean())
            seasonal= (fc_ts - fc_ts.rolling(52,min_periods=1).mean()).tail(52)
            fdates  = pd.date_range(start=fc_ts.index[-1]+pd.Timedelta(weeks=1),
                                     periods=fc_weeks, freq="W")
            fvals   = [last_v+trend_v*(i+1)+float(seasonal.iloc[i%len(seasonal)])
                       for i in range(fc_weeks)]
            fc_mean  = pd.Series(fvals, index=fdates)
            std      = fc_ts.std()
            ci_lo, ci_hi = fc_mean-1.65*std, fc_mean+1.65*std
            model_lbl = "Trend + Seasonal Decomposition (Fallback)"

    # ── Forecast KPI banner ──
    hist_avg  = float(fc_ts.tail(fc_weeks).mean())
    fc_avg_v  = float(fc_mean.mean())
    fc_total  = float(fc_mean.sum())
    fc_peak   = float(fc_mean.max())
    fc_low    = float(fc_mean.min())
    vs_hist   = (fc_avg_v - hist_avg)/hist_avg*100 if hist_avg else 0
    pk_wk     = fc_mean.idxmax().strftime("%d %b %Y")
    trend_dir = "📈 Growing" if vs_hist>2 else ("📉 Declining" if vs_hist<-2 else "➡ Stable")

    st.markdown(f"""
    <div class="fcp">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1.2rem">
        <div>
          <div style="font-size:1.1rem;font-weight:800;color:white">
            🤖 AI Forecast: Store {fc_store} · Dept {fc_dept} · Next {fc_weeks} Weeks
          </div>
          <div style="font-size:.73rem;color:#94A3B8;margin-top:4px">
            {model_lbl} · 80% Confidence Interval
          </div>
        </div>
        <div style="background:rgba(255,107,0,.2);border:1px solid {C['saffron']};
                    border-radius:10px;padding:.5rem 1rem;color:{C['saffron']};
                    font-size:.8rem;font-weight:700">
          {trend_dir}
        </div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:1rem">
        <div class="fck"><div class="fck-v">{Md(fc_total)}</div>
          <div class="fck-l">Total Forecast Revenue</div></div>
        <div class="fck"><div class="fck-v">{Md(fc_avg_v)}</div>
          <div class="fck-l">Avg Per Week</div></div>
        <div class="fck"><div class="fck-v" style="color:{'#00E676' if vs_hist>=0 else '#FF6E7A'}">{vs_hist:+.1f}%</div>
          <div class="fck-l">vs Historical</div></div>
        <div class="fck"><div class="fck-v">{Md(fc_peak)}</div>
          <div class="fck-l">Peak Week ({pk_wk})</div></div>
        <div class="fck"><div class="fck-v">{Md(fc_peak-fc_low)}</div>
          <div class="fck-l">Forecast Range</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Main forecast chart ──
    subsection("🔮 AI Forecast Chart — Historical + Predicted", C["saffron"])
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=fc_ts.index, y=fc_ts.values,
        name="Historical Sales", mode="lines",
        line=dict(color=C["sky"],width=2)
    ))
    fig_fc.add_trace(go.Scatter(
        x=list(fc_mean.index)+list(fc_mean.index[::-1]),
        y=list(ci_hi.values)+list(ci_lo.values[::-1]),
        fill="toself", fillcolor="rgba(255,107,0,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="80% Confidence Band"
    ))
    fig_fc.add_trace(go.Scatter(
        x=fc_mean.index, y=fc_mean.values,
        name=f"AI Forecast ({fc_weeks}w)",
        mode="lines+markers",
        line=dict(color=C["saffron"],width=3,dash="dash"),
        marker=dict(size=6,color=C["saffron"])
    ))
    split_dt = pd.Timestamp(fc_ts.index[-1]).strftime("%Y-%m-%d")
    fig_fc.add_shape(type="line",x0=split_dt,x1=split_dt,y0=0,y1=1,
                     xref="x",yref="paper",
                     line=dict(color=C["g3"],width=2,dash="dot"))
    fig_fc.add_annotation(x=split_dt,y=1,xref="x",yref="paper",
                           text="▶ Forecast Starts",showarrow=False,
                           xanchor="left",yanchor="top",
                           font=dict(size=11,color=C["saffron"]),
                           bgcolor="white",borderpad=4)
    cs(fig_fc,440)
    fig_fc.update_layout(hovermode="x unified",
        xaxis_title="Date",yaxis_title="Weekly Sales (₹)")
    cc_open("🔮 AI Sales Forecast Chart",
            "Blue = actual historical data · Orange dashed = AI prediction · Shaded band = confidence range")
    st.plotly_chart(fig_fc, use_container_width=True)
    cc_close()

    cfc1, cfc2 = st.columns([3,2])
    with cfc1:
        # Forecast vs Historical side-by-side
        hist_comp = fc_ts.tail(fc_weeks).reset_index()
        hist_comp.columns = ["Date","Sales"]
        fc_comp   = fc_mean.reset_index()
        fc_comp.columns = ["Date","Sales"]

        fig_cfc = go.Figure()
        fig_cfc.add_trace(go.Bar(x=hist_comp["Date"],y=hist_comp["Sales"],
            name="Historical (Same Period)",marker_color=C["navy3"],opacity=0.75))
        fig_cfc.add_trace(go.Bar(x=fc_comp["Date"],y=fc_comp["Sales"],
            name="AI Forecast",marker_color=C["saffron"],opacity=0.9))
        cs(fig_cfc,340)
        fig_cfc.update_layout(barmode="overlay",hovermode="x unified")
        cc_open("📊 Forecast vs Historical Comparison",
                "Orange = what AI predicts · Blue = what actually happened last time")
        st.plotly_chart(fig_cfc, use_container_width=True)
        cc_close()

    with cfc2:
        # Forecast table
        fc_tbl = pd.DataFrame({
            "Week":      fc_mean.index.strftime("%d %b %Y"),
            "Forecast":  fc_mean.values.round(2),
            "Lower CI":  ci_lo.values.round(2),
            "Upper CI":  ci_hi.values.round(2),
        })
        fc_tbl["PrevFc"] = fc_tbl["Forecast"].shift(1)
        fc_tbl["Dir"]    = fc_tbl.apply(
            lambda r: "▲ Up" if r["Forecast"]>r["PrevFc"]
                      else ("▼ Down" if r["Forecast"]<r["PrevFc"] else "→ Flat")
                      if not np.isnan(r["PrevFc"]) else "—", axis=1)
        fc_tbl["Forecast"] = fc_tbl["Forecast"].apply(lambda x: f"₹{x:,.0f}")
        fc_tbl["Lower CI"] = fc_tbl["Lower CI"].apply(lambda x: f"₹{x:,.0f}")
        fc_tbl["Upper CI"] = fc_tbl["Upper CI"].apply(lambda x: f"₹{x:,.0f}")
        fc_tbl = fc_tbl[["Week","Forecast","Dir","Lower CI","Upper CI"]]
        fc_tbl.columns     = ["Week","Predicted (₹)","Direction","Min","Max"]
        cc_open("📋 Week-by-Week Forecast",f"Model: {model_lbl}")
        st.dataframe(fc_tbl, use_container_width=True, hide_index=True, height=330)
        cc_close()

    # AI insight message
    if vs_hist > 5:
        fc_msg = f'<div class="a-green"><div class="at">🚀 Strong Growth Expected!</div>AI model predicts <b>{vs_hist:.1f}% higher sales</b> than the historical average for this store-department. Plan extra inventory and staff for peak week around {pk_wk}.</div>'
    elif vs_hist > 0:
        fc_msg = f'<div class="a-blue"><div class="at">📈 Moderate Growth Forecast</div>Sales expected to grow by <b>{vs_hist:.1f}%</b> vs historical. Maintain current stock levels with slight buffer for peak week {pk_wk}.</div>'
    elif vs_hist > -5:
        fc_msg = f'<div class="a-amber"><div class="at">⚠️ Slight Decline Anticipated</div>AI predicts <b>{abs(vs_hist):.1f}% lower sales</b>. Consider running promotions or discounts to offset the expected dip.</div>'
    else:
        fc_msg = f'<div class="a-red"><div class="at">🔴 Significant Decline Forecasted</div>Sales may drop by <b>{abs(vs_hist):.1f}%</b>. Take urgent action: review pricing, run flash sales, and contact regional manager.</div>'
    st.markdown(fc_msg, unsafe_allow_html=True)

st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — BUSINESS INTELLIGENCE & SUGGESTIONS
# ══════════════════════════════════════════════════════════════════════════════
section("🧠", "Business Intelligence & AI Recommendations",
        "Automated insights and actionable suggestions generated from your sales data",
        C["navy3"])

# ── Compute intelligence ──────────────────────────────────────────────────────
store_rank   = dff.groupby("Store")["Weekly_Sales"].sum().sort_values(ascending=False)
dept_rank    = dff.groupby("Dept")["Weekly_Sales"].sum().sort_values(ascending=False)
top_store_id = int(store_rank.index[0])
bot_store_id = int(store_rank.index[-1])
top_dept_id  = int(dept_rank.index[0])
bot_dept_id  = int(dept_rank.index[-1])

# WoW trend for last 4 weeks
last4 = weekly_ts.tail(4)
declining_4w = (last4.diff().dropna() < 0).all()
growing_4w   = (last4.diff().dropna() > 0).all()

# Department declining
dept_wow2    = dff.groupby(["Dept","Date"])["Weekly_Sales"].sum().reset_index()
dept_wow2    = dept_wow2.sort_values(["Dept","Date"])
dept_wow2["WoW"] = dept_wow2.groupby("Dept")["Weekly_Sales"].pct_change()*100
dept_trend2  = dept_wow2.groupby("Dept")["WoW"].mean()
declining_depts = dept_trend2[dept_trend2<-1].index.tolist()
growing_depts   = dept_trend2[dept_trend2>1].index.tolist()

# Holiday lift
hol_lift_avg = 0
if _has_both and "hp" in dir() and not hp.empty and "Lift%" in hp.columns:
    hol_lift_avg = float(hp["Lift%"].mean())

# Volatility
cv_store  = dff.groupby("Store")["Weekly_Sales"].std()/dff.groupby("Store")["Weekly_Sales"].mean()
high_cv   = cv_store[cv_store > cv_store.quantile(0.85)].index.tolist()

# Profit leakage
wbs_l = dff.groupby(["Date","Store"])["Weekly_Sales"].sum().reset_index()
wbs_l = wbs_l.sort_values(["Store","Date"])
wbs_l["Prev"] = wbs_l.groupby("Store")["Weekly_Sales"].shift(1)
wbs_l["Drop"] = wbs_l["Weekly_Sales"] - wbs_l["Prev"]
wbs_l["DropPct"] = wbs_l["Drop"]/wbs_l["Prev"]*100
leakage_top  = wbs_l[wbs_l["DropPct"]<-20].sort_values("Drop").head(5)
total_leakage= float(leakage_top["Drop"].abs().sum()) if not leakage_top.empty else 0

# ── Insight Summary KPIs ──────────────────────────────────────────────────────
n_urgent = sum([declining_4w, len(declining_depts)>3, total_leakage>10000, loss_weeks>5])
n_opps   = sum([growing_4w, len(growing_depts)>0, hol_lift_avg>5])
trend_lbl = "📈 Growing" if yoy>2 else ("📉 Declining" if yoy<-2 else "➡ Stable")
kpi_row([
    dict(icon="🔴", label="Urgent Issues",   value=str(n_urgent),
         hint="Require immediate action",
         accent=C["red"], icon_bg="#FFF1F2", val_color=C["red"]),
    dict(icon="🟡", label="Opportunities",   value=str(n_opps),
         hint="High impact, act now",
         accent=C["gold"], icon_bg="#FFFBEB", val_color=C["gold"]),
    dict(icon="🟢", label="Growth Signals",  value=str(len(growing_depts)),
         hint="Departments showing growth",
         accent=C["green"], icon_bg="#F0FDF4", val_color=C["green"]),
    dict(icon="📊", label="Overall Trend",   value=trend_lbl,
         delta=f"{yoy:+.1f}% YoY",
         accent=C["green"] if yoy>0 else C["red"],
         icon_bg="#F0FDF4" if yoy>0 else "#FFF1F2",
         val_color=C["green"] if yoy>0 else C["red"]),
])

st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

# ── Insight tabs ──────────────────────────────────────────────────────────────
tab_i1, tab_i2, tab_i3 = st.tabs([
    "🔴 Urgent Actions", "🟡 Growth Opportunities", "🟢 Quick Wins & Tips"
])

with tab_i1:
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        st.markdown('<div class="cc"><div class="cc-title">🚨 Critical Alerts</div>'
                    '<div class="cc-sub">Issues requiring immediate attention from store managers</div>', unsafe_allow_html=True)

        if declining_4w:
            st.markdown(f"""<div class="sug sr">
            <span class="sbadge urgent">URGENT</span>
            <div class="st">📉 4-Week Sales Decline Detected</div>
            Total sales have been falling for 4 consecutive weeks.
            <b>Action:</b> Launch flash sales, email/SMS promotions, and review pricing strategy immediately.
            Contact district manager for support.
            </div>""", unsafe_allow_html=True)

        if loss_weeks > 5:
            st.markdown(f"""<div class="sug sr">
            <span class="sbadge urgent">ALERT</span>
            <div class="st">⚠️ {loss_weeks} Loss Weeks Detected</div>
            {loss_weeks} weeks showed negative or zero sales in the selection.
            <b>Action:</b> Audit Store {bot_store_id} operations, staff, and inventory.
            Check if there were supply chain issues or local competitive events.
            </div>""", unsafe_allow_html=True)

        if total_leakage > 0:
            st.markdown(f"""<div class="sug sr">
            <span class="sbadge urgent">LOSS</span>
            <div class="st">🔍 Profit Leakage: {Md(total_leakage)}</div>
            Top 5 worst single-week drops represent <b>{Md(total_leakage)}</b> in potential lost revenue.
            <b>Action:</b> Investigate these specific weeks for stockouts, weather events, or competitor promotions.
            Even recovering 20% = {Md(total_leakage*0.2)} extra revenue.
            </div>""", unsafe_allow_html=True)

        if len(declining_depts) > 0:
            dl_str = ", ".join([f"Dept {int(d)}" for d in declining_depts[:5]])
            st.markdown(f"""<div class="sug sr">
            <span class="sbadge urgent">TREND</span>
            <div class="st">📊 {len(declining_depts)} Departments Declining</div>
            {dl_str} show negative week-over-week growth trends.
            <b>Action:</b> Run category-specific promotions, review shelf placement,
            and renegotiate supplier pricing for these departments.
            </div>""", unsafe_allow_html=True)

        if high_cv:
            vc_str = ", ".join([f"Store {int(s)}" for s in high_cv[:3]])
            st.markdown(f"""<div class="sug sa">
            <span class="sbadge urgent">VOLATILITY</span>
            <div class="st">📈 High Sales Volatility: {vc_str}</div>
            These stores have highly unpredictable weekly sales.
            <b>Action:</b> Establish consistent weekly promotional calendar.
            Stable promotions smooth revenue and improve inventory planning.
            </div>""", unsafe_allow_html=True)

        if n_urgent == 0:
            st.markdown('<div class="a-green"><div class="at">✅ No Critical Issues Found</div>'
                        'Your selection looks healthy! Review the growth opportunities tab to scale further.</div>',
                        unsafe_allow_html=True)
        cc_close()

    with col_u2:
        st.markdown('<div class="cc"><div class="cc-title">🔍 Profit Leakage Details</div>'
                    '<div class="cc-sub">Biggest single-week drops — investigate these urgently</div>', unsafe_allow_html=True)
        if not leakage_top.empty:
            for _,lr in leakage_top.iterrows():
                st.markdown(f"""<div class="a-red">
                <div class="at">Store {int(lr['Store'])} — {lr['Date'].strftime('%d %b %Y')}</div>
                Sales dropped <b>{lr['DropPct']:.1f}%</b> vs prior week — lost <b>{Md(abs(lr['Drop']))}</b>
                </div>""", unsafe_allow_html=True)

            # Leakage bar
            lk_fig = go.Figure(go.Bar(
                y=[f"S{int(r['Store'])} {r['Date'].strftime('%b%d')}" for _,r in leakage_top.iterrows()],
                x=leakage_top["Drop"].abs(),
                orientation="h",marker_color=C["red"],
                text=leakage_top["Drop"].apply(lambda x: Md(abs(x))),
                textposition="outside"
            ))
            cs(lk_fig,260)
            lk_fig.update_layout(showlegend=False,xaxis_title="Revenue Lost (₹)",
                yaxis=dict(autorange="reversed"))
            st.plotly_chart(lk_fig, use_container_width=True)
        else:
            st.markdown('<div class="a-green">✅ No major drops detected (>20% in a single week)</div>',
                        unsafe_allow_html=True)
        cc_close()

with tab_i2:
    col_o1, col_o2 = st.columns(2)
    with col_o1:
        st.markdown('<div class="cc"><div class="cc-title">🚀 Revenue Growth Opportunities</div>'
                    '<div class="cc-sub">Where to focus resources for maximum sales impact</div>', unsafe_allow_html=True)

        top_s_total = float(store_rank.iloc[0])
        bot_s_total = float(store_rank.iloc[-1])
        gap = top_s_total - bot_s_total

        st.markdown(f"""<div class="sug sg">
        <span class="sbadge opp">OPPORTUNITY</span>
        <div class="st">🏪 Uplift Store {bot_store_id} — ₹{gap/1e6:.1f}M Gap</div>
        Store {bot_store_id} generates <b>{Md(bot_s_total)}</b> vs Store {top_store_id}
        at <b>{Md(top_s_total)}</b>. Closing just 25% of this gap adds <b>{Md(gap*0.25)}</b>.
        <b>Action:</b> Share top store's promotional calendar, product mix, and display strategy.
        Assign a store transformation mentor from Store {top_store_id}.
        </div>""", unsafe_allow_html=True)

        top_d_total = float(dept_rank.iloc[0])
        bot_d_total = float(dept_rank.iloc[-1])
        dept_gap_pct= (top_d_total-bot_d_total)/top_d_total*100

        st.markdown(f"""<div class="sug sg">
        <span class="sbadge opp">GROWTH</span>
        <div class="st">📦 Dept {bot_dept_id} Has {dept_gap_pct:.0f}% Gap vs Dept {top_dept_id}</div>
        Dept {top_dept_id} earns <b>{Md(top_d_total)}</b> vs Dept {bot_dept_id} at <b>{Md(bot_d_total)}</b>.
        A 10% improvement in Dept {bot_dept_id} = <b>{Md(bot_d_total*0.10)}</b> extra revenue.
        <b>Action:</b> Increase shelf space, improve product mix, run targeted promotions.
        </div>""", unsafe_allow_html=True)

        if growing_depts:
            gd_str = ", ".join([f"Dept {int(d)}" for d in growing_depts[:5]])
            st.markdown(f"""<div class="sug sb">
            <span class="sbadge opp">SCALE</span>
            <div class="st">⬆️ Scale These Growing Departments: {gd_str}</div>
            These departments show consistent positive WoW growth.
            <b>Action:</b> Increase inventory depth, negotiate better supplier terms,
            expand floor space allocation. Growth departments deserve MORE investment.
            </div>""", unsafe_allow_html=True)

        if hol_lift_avg > 3:
            st.markdown(f"""<div class="sug sg">
            <span class="sbadge opp">HOLIDAY</span>
            <div class="st">🎄 Holiday Boost: +{hol_lift_avg:.1f}% Avg Lift Available</div>
            Stores average <b>{hol_lift_avg:.1f}% more sales</b> during holiday weeks.
            <b>Action:</b> Pre-stock 15-20% more inventory before Super Bowl, Labor Day,
            Thanksgiving, and Christmas. Run "Early Holiday" deals 2 weeks ahead.
            </div>""", unsafe_allow_html=True)
        cc_close()

    with col_o2:
        st.markdown('<div class="cc"><div class="cc-title">📅 Seasonal Strategy Calendar</div>'
                    '<div class="cc-sub">When to run promotions for maximum impact</div>', unsafe_allow_html=True)

        best_month_val = dff.groupby("YearMonth")["Weekly_Sales"].sum().idxmax()
        worst_month_val= dff.groupby("YearMonth")["Weekly_Sales"].sum().idxmin()

        strategies = [
            (C["red"],   "PREP NOW",  f"🗓 Pre-stock for Peak Month ({best_month_val})",
             f"Your best month historically. Start ordering 6 weeks ahead. "
             f"Schedule maximum staff. Run 'Early Bird' promotions 3 weeks before."),
            (C["gold"],  "PLAN",      f"⚡ Recovery Plan for Slow Month ({worst_month_val})",
             "Create bundle deals, loyalty reward events, and referral campaigns "
             "to drive traffic during this typically slow period."),
            (C["green"], "WIN",       "🎄 Holiday Inventory Lock-in",
             "Secure supplier contracts for holiday stock NOW — avoid last-minute "
             "price hikes. Focus on top 5 departments that gain most from holidays."),
            (C["sky"],   "DIGITAL",   "📱 WhatsApp/SMS Marketing Push",
             f"Text promotions to local customer base 48 hours before expected "
             f"peak weeks. Indian consumers respond strongly to time-limited deals."),
            (C["saffron"],"LOYALTY",  "🏆 Rewards Program for Repeat Buyers",
             "Repeat customers spend 67% more than new ones. A simple points system "
             "can increase average basket size significantly."),
        ]
        for color, badge, title, body in strategies:
            st.markdown(f"""<div class="sug sb" style="border-color:{color}">
            <div class="st">{title}</div>
            <span class="sbadge tip">{badge}</span>
            {body}
            </div>""", unsafe_allow_html=True)
        cc_close()

with tab_i3:
    st.markdown('<div class="cc"><div class="cc-title">💡 Quick Wins & Business Tips</div>'
                '<div class="cc-sub">Low-effort, high-impact actions you can implement this week</div>', unsafe_allow_html=True)

    qw_tips = [
        (C["green"], "🏆", "Replicate Store "+str(top_store_id)+"'s success",
         f"Store {top_store_id} generates the highest revenue in your selection at {Md(store_rank.iloc[0])}. "
         "Document its promotional calendar, product assortment, store layout, and customer service practices. "
         "Apply to the bottom 3 stores for a fast win."),
        (C["saffron"],"📦", f"Department {top_dept_id} is your cash machine",
         f"Dept {top_dept_id} earns {Md(dept_rank.iloc[0])} total revenue. "
         "Ensure it never goes out of stock. Increase reorder point by 20%. "
         "Place it in high-traffic area of every store."),
        (C["sky"],"📊", "Weekly sales review meeting",
         "Hold a 15-minute Monday morning review using this dashboard. "
         "The act of reviewing data weekly has been shown to improve sales performance "
         "by 12-18% in retail businesses. Share the top 3 insights with store managers."),
        (C["gold"],"🎯", "Price anchoring on top departments",
         "Show a higher 'was' price alongside current price for top 5 products in "
         f"Dept {top_dept_id}. Indian consumers respond strongly to perceived discounts. "
         "Typical conversion improvement: 8-15%."),
        (C["navy3"],"📱", "WhatsApp Business for promotions",
         "Set up WhatsApp Business for each store. Send weekly 'deal of the week' "
         "to customer contacts. Indian users open WhatsApp messages at 98% rate vs "
         "20% for email. Zero cost, massive reach."),
        (C["green"],"🤝", "Supplier renegotiation for bottom depts",
         f"Dept {bot_dept_id} and similar low-performers likely have poor margins. "
         "Renegotiate supplier contracts — even a 5% cost reduction could turn "
         "a marginal department profitable."),
    ]

    col_q1, col_q2 = st.columns(2)
    for i,(col, icon, title, body) in enumerate(qw_tips):
        target_col = col_q1 if i%2==0 else col_q2
        with target_col:
            st.markdown(f"""<div class="sug sb" style="border-color:{col}">
            <div class="st">{icon} {title}</div>
            {body}
            </div>""", unsafe_allow_html=True)
    cc_close()

    # Auto-generated summary insight card
    summary_lines = []
    summary_lines.append(f"📊 <b>Total Revenue:</b> {Md(total_sales)} across {n_stores} stores and {n_depts} departments.")
    summary_lines.append(f"💰 <b>Estimated Profit:</b> {Md(total_profit)} at 30% retail margin.")
    if yoy>0:
        summary_lines.append(f"📈 <b>Business is growing</b> at {yoy:.1f}% year-over-year — strong positive signal.")
    else:
        summary_lines.append(f"📉 <b>Business is declining</b> at {yoy:.1f}% YoY — needs strategic intervention.")
    summary_lines.append(f"🏆 <b>Best Store:</b> Store {top_store_id} ({Md(store_rank.iloc[0])}) · "
                          f"<b>Best Dept:</b> Dept {top_dept_id} ({Md(dept_rank.iloc[0])}).")
    if len(growing_depts)>0:
        summary_lines.append(f"⬆️ <b>{len(growing_depts)} departments</b> are growing — invest more there.")
    if len(declining_depts)>0:
        summary_lines.append(f"⬇️ <b>{len(declining_depts)} departments</b> are declining — need promotional support.")
    if hol_lift_avg>0:
        summary_lines.append(f"🎄 <b>Holiday effect:</b> Stores earn {hol_lift_avg:.1f}% more during holiday weeks.")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{C['navy']} 0%,{C['navy3']} 100%);
                border-radius:16px;padding:1.5rem;margin-top:1rem;
                border:1px solid rgba(255,107,0,.25)">
      <div style="font-size:1rem;font-weight:800;color:white;margin-bottom:1rem">
        🤖 Auto-Generated Business Intelligence Summary
      </div>
      {''.join([f'<div style="font-size:.81rem;color:#CBD5E1;margin:.4rem 0;padding:.4rem 0;border-bottom:1px solid rgba(255,255,255,.06)">{l}</div>' for l in summary_lines])}
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="text-align:center;padding:2rem 0 1rem;
            border-top:1px solid {C['g1']};margin-top:2rem">
  <div style="font-size:.75rem;color:{C['g2']};">
    🛒 <b>Walmart Sales Intelligence Platform</b> &nbsp;·&nbsp;
    Built with ❤️ using Streamlit + Plotly + SARIMA AI &nbsp;·&nbsp;
    Data: Walmart Store Sales Forecasting (Kaggle) &nbsp;·&nbsp;
    Suitable for Final Year DS Project Presentation
  </div>
  <div style="font-size:.68rem;color:{C['g1']};margin-top:.5rem;">
    Designed for Indian Business Users · Not financial advice · Estimates based on 30% retail margin assumption
  </div>
</div>
""", unsafe_allow_html=True)