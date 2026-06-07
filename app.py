"""
SalesIQ Pro  |  Enterprise Sales Intelligence & Forecasting Platform
=====================================================================
Version  : 3.1.0
Stack    : Python 3.11+ · Streamlit · Plotly · Statsmodels · SciPy
Analytics: SARIMA(1,1,1)(1,1,1,7) · Holt-Winters Triple ES · Ensemble
           Monte Carlo GBM · Z-score Anomaly Detection · ADF Test
"""

import io
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SalesIQ Pro",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────────────────

C = {
    "page":       "#F4F6F9",
    "card":       "#FFFFFF",
    "sidebar_bg": "#0A1628",
    "text":       "#0D1117",
    "text_md":    "#374151",
    "text_muted": "#6B7280",
    "border":     "#E5E7EB",
    "blue":       "#1A56DB",
    "blue_lt":    "#EBF5FF",
    "blue_md":    "#3F83F8",
    "blue_dk":    "#1E429F",
    "green":      "#057A55",
    "green_lt":   "#DEF7EC",
    "red":        "#C81E1E",
    "red_lt":     "#FDE8E8",
    "amber":      "#B45309",
    "amber_lt":   "#FEF3C7",
    "cyan":       "#0694A2",
    "purple":     "#6D28D9",
    "purple_lt":  "#EDE9FE",
}

PALETTE = ["#1A56DB", "#0694A2", "#057A55", "#B45309", "#6D28D9", "#C81E1E"]

CHART_FONT = dict(family="DM Sans, sans-serif", size=12, color=C["text_md"])


def _chart_layout(**kwargs) -> dict:
    """Return a safe layout dict for any figure type."""
    base = dict(
        margin=dict(l=2, r=2, t=32, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FAFBFC",
        font=CHART_FONT,
        hoverlabel=dict(
            bgcolor="white",
            bordercolor=C["border"],
            font=CHART_FONT,
        ),
    )
    base.update(kwargs)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# STYLESHEET
# ─────────────────────────────────────────────────────────────────────────────

def _css() -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #F4F6F9 !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #0D1117 !important;
}
.block-container {
    padding: 1.75rem 2.5rem 4rem !important;
    max-width: 1680px !important;
}

/* Sidebar */
[data-testid="stSidebar"] { background: #0A1628 !important; border-right: 1px solid #0F1E38 !important; }
[data-testid="stSidebar"] > div:first-child { padding: 1.5rem 1.2rem; }
[data-testid="stSidebar"] * { color: #94A3B8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #F1F5F9 !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    font-size: .78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: .07em !important;
}
[data-testid="stSidebar"] hr { border-color: #1E3A5F !important; margin: 1rem 0 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,.05) !important;
    border: 1px solid #1E3A5F !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] { background: #1A56DB !important; border-radius: 3px !important; }
[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    background: rgba(255,255,255,.05) !important;
    border: 1px solid #1E3A5F !important;
    border-radius: 6px !important;
    color: #F1F5F9 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
    font-size: .7rem !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    color: #475569 !important;
}

/* Tabs */
[data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 2px !important;
    border-bottom: 1px solid #E5E7EB !important;
    padding: 0 !important;
    margin-bottom: 1.5rem !important;
}
[data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: .82rem !important;
    color: #6B7280 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: .65rem 1.1rem !important;
}
[data-baseweb="tab"]:hover { color: #1A56DB !important; background: #EBF5FF !important; }
button[aria-selected="true"][data-baseweb="tab"] {
    color: #1A56DB !important;
    border-bottom: 2px solid #1A56DB !important;
    background: transparent !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #FFFFFF;
    border-radius: 8px;
    padding: 16px 20px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 1px 2px rgba(0,0,0,.04);
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* Download button */
[data-testid="stDownloadButton"] button {
    background: #1A56DB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: .84rem !important;
    padding: .6rem 1.4rem !important;
}
[data-testid="stDownloadButton"] button:hover { background: #1E429F !important; }

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    background: #FFFFFF !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #D1D5DB; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _page_header(title: str, subtitle: str, badge: str = "") -> None:
    badge_html = (
        f'<span style="background:#EBF5FF;color:#1A56DB;font-size:.7rem;'
        f'font-weight:700;letter-spacing:.07em;text-transform:uppercase;'
        f'padding:3px 10px;border-radius:20px;border:1px solid #BFDBFE">{badge}</span>'
        if badge else ""
    )
    st.markdown(f"""
<div style="display:flex;align-items:flex-end;justify-content:space-between;
            padding-bottom:1.25rem;border-bottom:1px solid #E5E7EB;margin-bottom:1.75rem">
  <div>
    <h1 style="font-family:'Sora',sans-serif;font-size:1.6rem;font-weight:800;
               color:#0A1628;margin:0;letter-spacing:-.03em">{title}</h1>
    <p  style="font-size:.84rem;color:#6B7280;margin:5px 0 0">{subtitle}</p>
  </div>
  {badge_html}
</div>""", unsafe_allow_html=True)


def _section(title: str) -> None:
    st.markdown(
        f'<p style="font-family:\'Sora\',sans-serif;font-size:.85rem;font-weight:700;'
        f'color:#374151;text-transform:uppercase;letter-spacing:.08em;'
        f'margin:1.6rem 0 .8rem;padding-bottom:.5rem;'
        f'border-bottom:1px solid #E5E7EB">{title}</p>',
        unsafe_allow_html=True,
    )


def _kpi_row(cards: list) -> None:
    cols = st.columns(len(cards))
    _map = {
        "up":      (C["green"],  C["green_lt"]),
        "down":    (C["red"],    C["red_lt"]),
        "neutral": (C["blue"],   C["blue_lt"]),
    }
    for col, card in zip(cols, cards):
        color, bg = _map.get(card.get("delta_dir", "neutral"), _map["neutral"])
        with col:
            st.markdown(f"""
<div style="background:#FFFFFF;border:1px solid #E5E7EB;border-radius:10px;
            padding:20px 22px;box-shadow:0 1px 3px rgba(0,0,0,.05);
            border-top:3px solid {color}">
  <p style="font-size:.7rem;font-weight:700;text-transform:uppercase;
            letter-spacing:.09em;color:#6B7280;margin:0 0 10px">{card['label']}</p>
  <p style="font-family:'JetBrains Mono',monospace;font-size:1.75rem;
            font-weight:500;color:#0D1117;margin:0 0 8px;line-height:1">{card['value']}</p>
  <span style="background:{bg};color:{color};font-size:.73rem;font-weight:600;
               padding:2px 9px;border-radius:3px">{card['delta']}</span>
</div>""", unsafe_allow_html=True)


def _insight_card(title: str, body: str, kind: str = "info") -> None:
    styles = {
        "positive": (C["green"],  C["green_lt"],  "#86EFAC"),
        "warning":  (C["amber"],  C["amber_lt"],  "#FCD34D"),
        "critical": (C["red"],    C["red_lt"],    "#FCA5A5"),
        "info":     (C["blue"],   C["blue_lt"],   "#BFDBFE"),
    }
    tc, bg, border = styles.get(kind, styles["info"])
    st.markdown(f"""
<div style="background:{bg};border:1px solid {border};border-radius:8px;
            padding:14px 18px;margin-bottom:10px">
  <p style="font-size:.86rem;font-weight:700;color:{tc};margin:0 0 5px">{title}</p>
  <p style="font-size:.83rem;color:#374151;margin:0;line-height:1.65">{body}</p>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _build_dataset() -> pd.DataFrame:
    rng      = np.random.default_rng(42)
    start    = datetime(2023, 1, 1)
    n_days   = 540
    PRODUCTS = ["Enterprise Suite", "Growth Pack", "Starter Plan", "Add-ons"]
    REGIONS  = ["North", "South", "East", "West"]
    CHANNELS = ["Online", "Offline"]
    p_mult   = {"Enterprise Suite": 1.40, "Growth Pack": 1.00,
                "Starter Plan": 0.65, "Add-ons": 0.45}
    r_mult   = {"North": 1.10, "South": 0.96, "East": 0.91, "West": 1.05}
    c_mult   = {"Online": 1.25, "Offline": 0.82}

    rows = []
    for i in range(n_days):
        date    = start + timedelta(days=i)
        trend   = 60 + i * 0.22 + max(0, (i - 300) * 0.08)
        dow_eff = 14 * np.sin(2 * np.pi * date.weekday() / 7 - np.pi / 4)
        annual  = 28 * np.sin(2 * np.pi * i / 365 - np.pi / 2)
        for product in PRODUCTS:
            for region in REGIONS:
                for channel in CHANNELS:
                    base    = trend + dow_eff + annual
                    mult    = p_mult[product] * r_mult[region] * c_mult[channel]
                    revenue = max(5.0, round(base * mult + rng.normal(0, 4 + 0.04 * i), 2))
                    units   = max(1, int(revenue / rng.uniform(10, 18)))
                    cost    = round(revenue * rng.uniform(0.36, 0.54), 2)
                    rows.append({
                        "date":    pd.Timestamp(date),
                        "product": product,
                        "region":  region,
                        "channel": channel,
                        "revenue": revenue,
                        "units":   units,
                        "cost":    cost,
                        "profit":  round(revenue - cost, 2),
                    })

    df        = pd.DataFrame(rows)
    spike_idx = rng.choice(len(df), size=12, replace=False)
    dip_idx   = rng.choice(
        list(set(range(len(df))) - set(spike_idx.tolist())), size=8, replace=False
    )
    df.loc[spike_idx, "revenue"] = (df.loc[spike_idx, "revenue"] * rng.uniform(3.5, 6.0, 12)).round(2)
    df.loc[dip_idx,   "revenue"] = (df.loc[dip_idx,   "revenue"] * rng.uniform(0.04, 0.15, 8)).round(2)
    df["profit"] = (df["revenue"] - df["cost"]).round(2)
    return df


@st.cache_data(show_spinner=False)
def _parse_upload(raw: bytes, name: str) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw)) if name.endswith(".csv") else pd.read_excel(io.BytesIO(raw))
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date"], inplace=True)
    if "profit" not in df.columns and {"revenue", "cost"} <= set(df.columns):
        df["profit"] = df["revenue"] - df["cost"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _inr(v: float) -> str:
    if v >= 1e7:  return f"₹{v/1e7:.2f} Cr"
    if v >= 1e5:  return f"₹{v/1e5:.2f} L"
    return f"₹{v:,.0f}"


def _prepare_ts(df: pd.DataFrame) -> pd.Series:
    ts = df.groupby("date")["revenue"].sum().sort_index()
    ts.index = pd.DatetimeIndex(ts.index)
    full_range = pd.date_range(ts.index.min(), ts.index.max(), freq="D")
    ts = ts.reindex(full_range, fill_value=float(ts.median()))
    ts.index.freq = "D"
    return ts


def _adf_result(series: pd.Series) -> str:
    try:
        p = adfuller(series.dropna(), maxlag=10)[1]
        return f"Stationary (p={p:.4f})" if p < 0.05 else f"Non-stationary (p={p:.4f})"
    except Exception:
        return "Undetermined"


# ─────────────────────────────────────────────────────────────────────────────
# FORECASTING
# ─────────────────────────────────────────────────────────────────────────────

def _fc_sarima(series: pd.Series, horizon: int):
    idx = pd.date_range(series.index[-1] + timedelta(days=1), periods=horizon, freq="D")
    try:
        res  = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,7),
                       enforce_stationarity=False, enforce_invertibility=False
                       ).fit(disp=False, maxiter=150)
        fc   = res.get_forecast(steps=horizon)
        mean = pd.Series(fc.predicted_mean.values, index=idx)
        ci   = fc.conf_int(alpha=0.10)
        lo   = pd.Series(ci.iloc[:, 0].values, index=idx)
        hi   = pd.Series(ci.iloc[:, 1].values, index=idx)
        return mean, lo, hi
    except Exception:
        last = float(series.iloc[-1])
        m = pd.Series([last] * horizon, index=idx)
        return m, m * 0.88, m * 1.12


def _fc_holtwinters(series: pd.Series, horizon: int):
    idx = pd.date_range(series.index[-1] + timedelta(days=1), periods=horizon, freq="D")
    try:
        fit  = ExponentialSmoothing(series, trend="add", seasonal="add",
                                    seasonal_periods=7).fit(optimized=True)
        vals = fit.forecast(horizon)
        m    = pd.Series(vals.values, index=idx)
        return m, m * 0.90, m * 1.10
    except Exception:
        last = float(series.iloc[-1])
        m = pd.Series([last] * horizon, index=idx)
        return m, m * 0.90, m * 1.10


def _fc_dispatch(series: pd.Series, horizon: int, model: str):
    if model == "Holt-Winters":
        return _fc_holtwinters(series, horizon)
    if model == "Ensemble":
        sm, sl, sh = _fc_sarima(series, horizon)
        hm, hl, hh = _fc_holtwinters(series, horizon)
        return (sm+hm)/2, (sl+hl)/2, (sh+hh)/2
    return _fc_sarima(series, horizon)


def _backtest_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    e     = actual.values - predicted.values
    denom = np.where(np.abs(actual.values) < 1e-9, 1.0, actual.values)
    ss_tot = float(np.sum((actual.values - actual.mean()) ** 2))
    r2     = 1 - float(np.sum(e**2)) / ss_tot if ss_tot > 1e-9 else 0.0
    return {
        "MAE":    float(np.mean(np.abs(e))),
        "RMSE":   float(np.sqrt(np.mean(e**2))),
        "MAPE %": float(np.mean(np.abs(e / denom))) * 100,
        "R²":     r2,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _detect_anomalies(series: pd.Series, threshold: float = 2.8) -> pd.Series:
    rm  = series.rolling(window=21, min_periods=7, center=True).mean()
    rs  = series.rolling(window=21, min_periods=7, center=True).std().replace(0, 1e-9)
    return ((series - rm).abs() / rs) > threshold


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _vline(fig: go.Figure, x_val, label: str = "") -> None:
    """Vertical reference line using add_shape (avoids Timestamp arithmetic bug)."""
    x_str = x_val.isoformat() if hasattr(x_val, "isoformat") else str(x_val)
    fig.add_shape(type="line", x0=x_str, x1=x_str, y0=0, y1=1, yref="paper",
                  line=dict(color="#9CA3AF", dash="dash", width=1.4))
    if label:
        fig.add_annotation(x=x_str, y=0.97, yref="paper", text=label,
                           showarrow=False, xanchor="left", xshift=6,
                           font=dict(size=11, color="#6B7280", family="DM Sans"))


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def chart_main_trend(ts, fc_mean, fc_lo, fc_hi, anomalies) -> go.Figure:
    ma14 = ts.rolling(14, min_periods=1).mean()
    fig  = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name="Daily Revenue",
                             mode="lines", line=dict(color="#CBD5E1", width=1),
                             hovertemplate="%{x|%b %d, %Y}<br>₹%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=ma14.index, y=ma14.values, name="14-Day MA",
                             mode="lines", line=dict(color=C["text"], width=2.2),
                             hovertemplate="%{x|%b %d}<br>MA14: ₹%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=list(fc_hi.index) + list(reversed(list(fc_lo.index))),
        y=list(fc_hi.values) + list(reversed(list(fc_lo.values))),
        fill="toself", fillcolor="rgba(26,86,219,.10)",
        line=dict(width=0), name="90% CI", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc_mean.index, y=fc_mean.values, name="Forecast",
                             mode="lines", line=dict(color=C["blue"], dash="dot", width=2.2),
                             hovertemplate="%{x|%b %d}<br>Forecast: ₹%{y:,.0f}<extra></extra>"))
    if anomalies.any():
        ax = ts[anomalies]
        fig.add_trace(go.Scatter(x=ax.index, y=ax.values, name="Anomaly", mode="markers",
                                 marker=dict(color=C["red"], size=9, symbol="x",
                                             line=dict(width=2.5, color=C["red"])),
                                 hovertemplate="%{x|%b %d}<br>Anomaly: ₹%{y:,.0f}<extra></extra>"))
    _vline(fig, ts.index[-1], "Forecast Start")
    fig.update_layout(**_chart_layout(
        height=420, hovermode="x unified",
        legend=dict(orientation="h", y=-0.15, font_size=11, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", zeroline=False),
        yaxis=dict(gridcolor="#F3F4F6", zeroline=False, title="Revenue (₹)"),
    ))
    return fig


def chart_channel_trend(df: pd.DataFrame) -> go.Figure:
    ct  = df.groupby(["date", "channel"])["revenue"].sum().reset_index()
    fig = go.Figure()
    cm  = {"Online": C["blue"], "Offline": C["cyan"]}
    for ch in df["channel"].unique():
        sub = ct[ct["channel"] == ch].copy()
        sma = sub.set_index("date")["revenue"].rolling(7, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=sma.index, y=sma.values, name=ch, mode="lines",
                                 line=dict(color=cm.get(ch, PALETTE[0]), width=2.2),
                                 hovertemplate=f"{ch}<br>%{{x|%b %d}}<br>₹%{{y:,.0f}}<extra></extra>"))
    fig.update_layout(**_chart_layout(
        height=300, hovermode="x unified",
        legend=dict(orientation="h", y=-0.22, font_size=11, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="#F3F4F6", zeroline=False, title="Revenue (₹)"),
    ))
    return fig


def chart_channel_donut(df: pd.DataFrame) -> go.Figure:
    ch  = df.groupby("channel")["revenue"].sum()
    fig = go.Figure(go.Pie(
        labels=ch.index, values=ch.values, hole=0.60,
        marker=dict(colors=[C["blue"], C["cyan"]], line=dict(color="white", width=2)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<extra></extra>",
    ))
    fig.update_layout(**_chart_layout(
        height=280, showlegend=False,
        annotations=[dict(text="Channel<br>Split", x=0.5, y=0.5, showarrow=False,
                          font=dict(size=12, color=C["text_md"], family="DM Sans"))],
    ))
    return fig


def chart_dayofweek(ts: pd.Series) -> go.Figure:
    dow_df  = pd.DataFrame({"revenue": ts.values, "dow": ts.index.day_name()})
    order   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    avg     = dow_df.groupby("dow")["revenue"].mean().reindex(order)
    overall = avg.mean()
    colors  = [C["green"] if v > overall*1.05 else C["red"] if v < overall*0.95
               else C["blue"] for v in avg.values]
    fig = go.Figure(go.Bar(
        x=avg.index, y=avg.values, marker_color=colors,
        text=[_inr(v) for v in avg.values], textposition="outside",
        textfont=dict(size=11, family="JetBrains Mono"), cliponaxis=False,
        hovertemplate="%{x}<br>Avg: ₹%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=overall, line_dash="dash", line_color="#9CA3AF",
                  annotation_text="Period Avg",
                  annotation_font=dict(size=11, color="#6B7280"))
    fig.update_layout(**_chart_layout(
        height=300, bargap=0.35,
        yaxis=dict(gridcolor="#F3F4F6", zeroline=False, title="Avg Revenue (₹)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    ))
    return fig


def chart_monthly_heatmap(ts: pd.Series) -> go.Figure:
    df_h  = pd.DataFrame({"year": ts.index.year, "month": ts.index.month, "rev": ts.values})
    pivot = df_h.pivot_table(values="rev", index="year", columns="month", aggfunc="sum")
    names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    cols  = [names[c-1] for c in pivot.columns]
    fig   = go.Figure(go.Heatmap(
        z=pivot.values, x=cols, y=pivot.index.astype(str).tolist(),
        colorscale=[[0,"#EBF5FF"],[0.5,C["blue_md"]],[1,C["blue_dk"]]],
        text=[[_inr(v) for v in row] for row in pivot.values],
        texttemplate="%{text}", textfont=dict(size=10, family="JetBrains Mono"),
        hovertemplate="Month: %{x} %{y}<br>Revenue: %{text}<extra></extra>",
        showscale=True, colorbar=dict(thickness=12, len=0.8),
    ))
    fig.update_layout(**_chart_layout(
        height=180, xaxis=dict(side="top"), yaxis=dict(tickfont=dict(size=11)),
    ))
    return fig


def chart_decomposition(ts: pd.Series):
    if len(ts) < 14:
        return None
    try:
        dec = seasonal_decompose(ts, model="additive", period=7, extrapolate_trend="freq")
    except Exception:
        return None
    labels  = ["Observed", "Trend", "Seasonal", "Residual"]
    series  = [dec.observed, dec.trend, dec.seasonal, dec.resid]
    colours = [C["text_md"], C["blue"], C["cyan"], C["amber"]]
    fig     = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=labels, vertical_spacing=0.07)
    for i, (s, col) in enumerate(zip(series, colours), 1):
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                                 line=dict(color=col, width=1.8), showlegend=False),
                      row=i, col=1)
    # Use explicit layout — do NOT unpack CHART_BASE into subplot figures
    fig.update_layout(
        height=540,
        margin=dict(l=2, r=2, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FAFBFC",
        font=CHART_FONT,
        showlegend=False,
    )
    fig.update_annotations(font_size=11, font_color=C["text_muted"])
    fig.update_yaxes(gridcolor="#F3F4F6", zeroline=False)
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def chart_product_revenue(df: pd.DataFrame) -> go.Figure:
    prod = df.groupby("product")["revenue"].sum().sort_values(ascending=False).reset_index()
    fig  = go.Figure(go.Bar(
        x=prod["product"], y=prod["revenue"], marker_color=PALETTE[:len(prod)],
        text=[_inr(v) for v in prod["revenue"]], textposition="outside",
        textfont=dict(size=11, family="JetBrains Mono"), cliponaxis=False,
        hovertemplate="%{x}<br>₹%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(**_chart_layout(
        height=320, bargap=0.38,
        yaxis=dict(gridcolor="#F3F4F6", zeroline=False, title="Revenue (₹)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    ))
    return fig


def chart_margin_bar(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby("product").agg(revenue=("revenue","sum"), cost=("cost","sum")).reset_index()
    grp["m"] = (grp["revenue"] - grp["cost"]) / grp["revenue"] * 100
    grp = grp.sort_values("m", ascending=False)
    colors = [C["green"] if v >= 50 else C["blue"] if v >= 40 else C["amber"] for v in grp["m"]]
    fig = go.Figure(go.Bar(
        x=grp["product"], y=grp["m"], marker_color=colors,
        text=[f"{v:.1f}%" for v in grp["m"]], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}<br>Margin: %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=50, line_dash="dot", line_color="#9CA3AF",
                  annotation_text="50% Benchmark",
                  annotation_font=dict(size=10, color="#6B7280"))
    fig.update_layout(**_chart_layout(
        height=300, bargap=0.38,
        yaxis=dict(gridcolor="#F3F4F6", zeroline=False, title="Gross Margin (%)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    ))
    return fig


def chart_product_trend(df: pd.DataFrame) -> go.Figure:
    pt  = df.groupby(["date", "product"])["revenue"].sum().reset_index()
    fig = go.Figure()
    for i, prod in enumerate(df["product"].unique()):
        sub = pt[pt["product"] == prod]
        sma = sub.set_index("date")["revenue"].rolling(7, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=sma.index, y=sma.values, name=prod, mode="lines",
                                 line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                                 hovertemplate=f"{prod}<br>%{{x|%b %d}}<br>₹%{{y:,.0f}}<extra></extra>"))
    fig.update_layout(**_chart_layout(
        height=320, hovermode="x unified",
        legend=dict(orientation="h", y=-0.2, font_size=11, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="#F3F4F6", zeroline=False, title="Revenue (₹)"),
    ))
    return fig


def chart_region_bar(df: pd.DataFrame) -> go.Figure:
    reg = df.groupby("region")["revenue"].sum().reset_index().sort_values("revenue", ascending=True)
    fig = go.Figure(go.Bar(
        y=reg["region"], x=reg["revenue"], orientation="h",
        marker=dict(color=reg["revenue"],
                    colorscale=[[0,"#DBEAFE"],[1,C["blue_dk"]]], showscale=False),
        text=[_inr(v) for v in reg["revenue"]], textposition="inside",
        insidetextanchor="middle", textfont=dict(color="white", size=11, family="JetBrains Mono"),
        hovertemplate="%{y}<br>₹%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(**_chart_layout(
        height=260,
        xaxis=dict(gridcolor="#F3F4F6", zeroline=False, title="Revenue (₹)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    ))
    return fig


def chart_region_product_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = df.pivot_table(index="product", columns="region",
                           values="revenue", aggfunc="sum", fill_value=0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0,"#EBF5FF"],[0.5,C["blue_md"]],[1,C["blue_dk"]]],
        text=[[_inr(v) for v in row] for row in pivot.values],
        texttemplate="%{text}", textfont=dict(size=10, family="JetBrains Mono"),
        hovertemplate="Product: %{y}<br>Region: %{x}<br>%{text}<extra></extra>",
        colorbar=dict(thickness=12),
    ))
    fig.update_layout(**_chart_layout(height=280))
    return fig


def chart_region_trend(df: pd.DataFrame) -> go.Figure:
    rt  = df.groupby(["date", "region"])["revenue"].sum().reset_index()
    fig = go.Figure()
    for i, reg in enumerate(df["region"].unique()):
        sub = rt[rt["region"] == reg]
        sma = sub.set_index("date")["revenue"].rolling(7, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=sma.index, y=sma.values, name=reg, mode="lines",
                                 line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                                 hovertemplate=f"{reg}<br>%{{x|%b %d}}<br>₹%{{y:,.0f}}<extra></extra>"))
    fig.update_layout(**_chart_layout(
        height=300, hovermode="x unified",
        legend=dict(orientation="h", y=-0.22, font_size=11, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="#F3F4F6", zeroline=False, title="Revenue (₹)"),
    ))
    return fig


def chart_forecast_comparison(ts: pd.Series):
    split = max(14, int(len(ts) * 0.80))
    train, test = ts.iloc[:split], ts.iloc[split:]
    if len(test) < 7:
        return None, {}, {}
    sa_m, _, _ = _fc_sarima(train, len(test))
    hw_m, _, _ = _fc_holtwinters(train, len(test))
    sa_m.index = test.index
    hw_m.index = test.index
    sa_met = _backtest_metrics(test, sa_m)
    hw_met = _backtest_metrics(test, hw_m)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test.index, y=test.values, name="Actual",
                             mode="lines", line=dict(color=C["text"], width=2.5),
                             hovertemplate="%{x|%b %d}<br>Actual: ₹%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=test.index, y=sa_m.values,
                             name=f"SARIMA (MAPE {sa_met['MAPE %']:.1f}%)",
                             mode="lines", line=dict(color=C["blue"], dash="dash", width=2),
                             hovertemplate="%{x|%b %d}<br>SARIMA: ₹%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=test.index, y=hw_m.values,
                             name=f"Holt-Winters (MAPE {hw_met['MAPE %']:.1f}%)",
                             mode="lines", line=dict(color=C["amber"], dash="dot", width=2),
                             hovertemplate="%{x|%b %d}<br>HW: ₹%{y:,.0f}<extra></extra>"))
    # Explicit layout — no **CHART_BASE for subplot-safe usage
    fig.update_layout(
        height=360,
        margin=dict(l=2, r=2, t=32, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FAFBFC",
        font=CHART_FONT,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.18, font_size=11, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="#F3F4F6", zeroline=False, title="Revenue (₹)"),
    )
    return fig, sa_met, hw_met


def chart_monte_carlo(ts: pd.Series, horizon: int):
    N    = 1000
    last = float(ts.iloc[-1])
    ret  = ts.pct_change().dropna()
    mu   = float(ret.mean())
    sig  = max(float(ret.std()), 0.01)
    rng  = np.random.default_rng(7)
    idx  = pd.date_range(ts.index[-1] + timedelta(days=1), periods=horizon, freq="D")
    paths       = np.empty((N, horizon))
    paths[:, 0] = last
    for t in range(1, horizon):
        paths[:, t] = paths[:, t-1] * (1 + rng.normal(mu, sig, N))
    p10 = np.percentile(paths, 10, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p90 = np.percentile(paths, 90, axis=0)
    fig = go.Figure()
    for i in range(0, N, 25):
        fig.add_trace(go.Scatter(x=idx, y=paths[i], mode="lines",
                                 line=dict(color="rgba(26,86,219,.05)", width=1),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=idx, y=p90, name="P90 — Optimistic",
                             line=dict(color=C["green"], width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=idx, y=p50, name="P50 — Base Case",
                             line=dict(color=C["blue"], width=2.5)))
    fig.add_trace(go.Scatter(x=idx, y=p10, name="P10 — Pessimistic",
                             line=dict(color=C["red"], width=2, dash="dot")))
    fig.update_layout(**_chart_layout(
        height=360,
        legend=dict(orientation="h", y=-0.18, font_size=11, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="#F3F4F6", zeroline=False, title="Revenue (₹)"),
    ))
    return fig, float(p10[-1]), float(p50[-1]), float(p90[-1])


def chart_anomaly(ts: pd.Series, anomalies: pd.Series) -> go.Figure:
    rm  = ts.rolling(21, min_periods=7, center=True).mean()
    rs  = ts.rolling(21, min_periods=7, center=True).std().replace(0, 1e-9)
    hi  = rm + 2.8 * rs
    lo  = rm - 2.8 * rs
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(hi.index) + list(reversed(list(lo.index))),
        y=list(hi.values) + list(reversed(list(lo.values))),
        fill="toself", fillcolor="rgba(26,86,219,.06)",
        line=dict(width=0), name="Normal Corridor (±2.8σ)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name="Revenue", mode="lines",
                             line=dict(color=C["text_md"], width=1.8),
                             hovertemplate="%{x|%b %d}<br>₹%{y:,.0f}<extra></extra>"))
    if anomalies.any():
        ax = ts[anomalies]
        fig.add_trace(go.Scatter(x=ax.index, y=ax.values, name="Anomaly", mode="markers",
                                 marker=dict(color=C["red"], size=10, symbol="diamond",
                                             line=dict(width=2, color="white")),
                                 hovertemplate="%{x|%b %d}<br>Anomaly: ₹%{y:,.0f}<extra></extra>"))
    fig.update_layout(**_chart_layout(
        height=340, hovermode="x unified",
        legend=dict(orientation="h", y=-0.18, font_size=11, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="#F3F4F6", zeroline=False, title="Revenue (₹)"),
    ))
    return fig


def chart_volatility(ts: pd.Series) -> go.Figure:
    vol = ts.rolling(30, min_periods=10).std()
    fig = go.Figure(go.Scatter(
        x=vol.index, y=vol.values, fill="tozeroy",
        fillcolor="rgba(180,83,9,.10)", line=dict(color=C["amber"], width=2),
        name="30-Day Rolling Std Dev",
        hovertemplate="%{x|%b %d}<br>Std Dev: ₹%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(**_chart_layout(
        height=240,
        legend=dict(orientation="h", y=-0.22, font_size=11, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="#F3F4F6", zeroline=False, title="Std Dev (₹)"),
    ))
    return fig


def chart_correlation(df: pd.DataFrame) -> go.Figure:
    corr = df[["revenue", "units", "cost", "profit"]].corr().round(3)
    fig  = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale=[[0, C["red_lt"]], [0.5, "#F9FAFB"], [1, C["blue_lt"]]],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}", textfont=dict(size=12, family="JetBrains Mono"),
        colorbar=dict(thickness=12),
        hovertemplate="%{y} / %{x}<br>ρ = %{text}<extra></extra>",
    ))
    fig.update_layout(**_chart_layout(height=280))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# INSIGHT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _generate_insights(df, ts, target, anomaly_count):
    total_rev    = df["revenue"].sum()
    total_cost   = df["cost"].sum()
    total_profit = df["profit"].sum()
    margin_pct   = total_profit / total_rev * 100 if total_rev else 0
    growth_wow   = ts.pct_change(7).tail(14).mean() * 100
    cov          = float(ts.std() / ts.mean()) if ts.mean() > 0 else 0
    top_product  = df.groupby("product")["revenue"].sum().idxmax()
    top_region   = df.groupby("region")["revenue"].sum().idxmax()
    weak_region  = df.groupby("region")["revenue"].sum().idxmin()
    top_pct      = df[df["product"] == top_product]["revenue"].sum() / total_rev * 100
    target_pct   = total_rev / target * 100 if target > 0 else 0
    n_days       = max(1, (df["date"].max() - df["date"].min()).days + 1)
    avg_daily    = total_rev / n_days
    out = []

    if growth_wow > 5:
        out.append(dict(kind="positive", title=f"Strong Revenue Momentum (+{growth_wow:.1f}% WoW)",
            body=f"Rolling 7-day revenue is growing at <b>{growth_wow:.1f}%</b> week-over-week. "
                 f"<b>{top_region}</b> and the Online channel are the primary drivers. "
                 f"Scale acquisition in these vectors while the tailwind holds."))
    elif growth_wow < -3:
        out.append(dict(kind="critical", title=f"Revenue Contraction ({growth_wow:.1f}% WoW)",
            body=f"Weekly revenue trend has turned negative at <b>{growth_wow:.1f}%</b>. "
                 f"Activate dormant accounts, accelerate pipeline in <b>{top_region}</b>, "
                 f"and determine whether macro seasonality or churn is the root cause."))
    else:
        out.append(dict(kind="info", title=f"Revenue Stable ({growth_wow:+.1f}% WoW)",
            body=f"Growth is within a stable corridor at <b>{growth_wow:+.1f}%</b>. "
                 f"Conversion rate optimisation and upselling within <b>{top_product}</b> "
                 f"are the clearest near-term acceleration levers."))

    if margin_pct >= 50:
        out.append(dict(kind="positive", title=f"Healthy Margin Profile ({margin_pct:.1f}%)",
            body=f"Gross margin of <b>{margin_pct:.1f}%</b> exceeds the 50% benchmark. "
                 f"Maintain pricing discipline under promotional pressure. "
                 f"Current absolute profit: <b>{_inr(total_profit)}</b>."))
    elif margin_pct >= 35:
        out.append(dict(kind="info", title=f"Margin Acceptable ({margin_pct:.1f}%)",
            body=f"Gross margin is below 50% but above the 35% floor. Review infrastructure "
                 f"and supplier costs. Target 40%+ within two quarters."))
    else:
        out.append(dict(kind="warning", title=f"Margin Pressure ({margin_pct:.1f}%)",
            body=f"Gross margin of <b>{margin_pct:.1f}%</b> is below the 35% warning threshold. "
                 f"Cost base of <b>{_inr(total_cost)}</b> requires urgent review. "
                 f"Evaluate low-margin SKUs for rationalisation and model a 5–10% price adjustment."))

    if top_pct > 40:
        out.append(dict(kind="warning", title=f"Revenue Concentration Risk — {top_product}",
            body=f"<b>{top_product}</b> drives <b>{top_pct:.1f}%</b> of revenue, above the 30% "
                 f"safe threshold. Diversify cross-sell motions to reduce single-product dependency."))
    else:
        out.append(dict(kind="positive", title=f"{top_product} Leading, Portfolio Well-Distributed",
            body=f"<b>{top_product}</b> at <b>{top_pct:.1f}%</b> revenue share is within healthy bounds. "
                 f"Apply the <b>{top_region}</b> growth playbook to <b>{weak_region}</b> to lift the floor."))

    if cov > 0.35:
        out.append(dict(kind="warning", title=f"Elevated Revenue Volatility (CoV = {cov:.2f})",
            body=f"Coefficient of variation of <b>{cov:.2f}</b> indicates significant demand swings. "
                 f"Annual contracts, subscription billing, or advance-payment incentives "
                 f"would smooth the signal and improve forecast accuracy."))

    if anomaly_count >= 5:
        out.append(dict(kind="critical", title=f"{anomaly_count} Anomalies Require Review",
            body=f"<b>{anomaly_count}</b> outlier observations flagged via rolling Z-score analysis "
                 f"(2.8σ). May reflect bulk orders, data errors, or promotional spikes. "
                 f"Validate against CRM records and restate if necessary."))

    gap = target - total_rev
    if target_pct >= 100:
        out.append(dict(kind="positive", title=f"Target Exceeded ({target_pct:.0f}% Attainment)",
            body=f"Revenue target fully achieved at <b>{target_pct:.0f}%</b>. "
                 f"Set a revised stretch target of {_inr(target * 1.15)} and assess "
                 f"whether the annualised run-rate of <b>{_inr(avg_daily * 365)}</b> is sustainable."))
    else:
        out.append(dict(kind="warning", title=f"Target Gap: {_inr(gap)} ({100-target_pct:.1f}% remaining)",
            body=f"At <b>{target_pct:.0f}%</b> attainment, closing the <b>{_inr(gap)}</b> gap "
                 f"requires an additional {_inr(gap/n_days)} per day. Focus pipeline acceleration "
                 f"on <b>{top_product}</b> in <b>{top_region}</b> with a time-bound incentive campaign."))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def _sidebar(df=None) -> dict:
    """
    Two modes:
      df = None  -> upload-only, all filters disabled
      df = df    -> full filter controls populated from data
    """
    with st.sidebar:
        st.markdown(
            '<p style="font-family:Sora,sans-serif;font-size:1.15rem;font-weight:800;'
            'color:#F1F5F9;letter-spacing:-.02em;margin-bottom:2px">SalesIQ Pro</p>'
            '<p style="font-size:.72rem;color:#475569;text-transform:none;'
            'letter-spacing:0;font-weight:400;margin-bottom:0">Enterprise Sales Intelligence</p>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # Upload always visible
        st.markdown("**Data Source**")
        uploaded = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"],
                                    key="salesiq_file_uploader",
                                    label_visibility="collapsed")

        if df is None:
            st.markdown("---")
            st.markdown(
                '<p style="font-size:.75rem;color:#475569;text-transform:none;'
                'letter-spacing:0;font-weight:400;line-height:1.6">'
                'Upload a CSV or Excel file above to activate filters, '
                'forecasting controls, and risk settings.</p>',
                unsafe_allow_html=True,
            )
            st.markdown("---")
            st.caption("v3.1.0  ·  Python · Streamlit · Statsmodels · Plotly")
            return dict(uploaded=uploaded, date_range=None, products=[],
                        regions=[], channels=[], horizon=30, model="SARIMA",
                        anomaly=True, target=0)

        # Date range
        st.markdown("---")
        st.markdown("**Date Range**")
        d_min = df["date"].min().date()
        d_max = df["date"].max().date()
        date_range = st.date_input("", value=(d_min, d_max),
                                   min_value=d_min, max_value=d_max,
                                   label_visibility="collapsed")

        # Filters
        st.markdown("---")
        st.markdown("**Filters**")
        products = st.multiselect("Products", sorted(df["product"].unique().tolist()),
                                  default=sorted(df["product"].unique().tolist()))
        regions  = st.multiselect("Regions", sorted(df["region"].unique().tolist()),
                                  default=sorted(df["region"].unique().tolist()))
        channels = st.multiselect("Channels", sorted(df["channel"].unique().tolist()),
                                  default=sorted(df["channel"].unique().tolist()))

        # Forecast
        st.markdown("---")
        st.markdown("**Forecast**")
        horizon = st.slider("Horizon (days)", 7, 180, 30, 7)
        model   = st.selectbox("Model", ["SARIMA", "Holt-Winters", "Ensemble"])

        # Risk
        st.markdown("---")
        st.markdown("**Risk**")
        anomaly_on = st.checkbox("Anomaly Detection", value=True)
        rev_target = st.number_input(
            "Revenue Target (INR)", min_value=0,
            value=int(df.groupby("date")["revenue"].sum().sum() * 0.85),
            step=25_000,
        )
        st.markdown("---")
        st.caption("v3.1.0  ·  Python · Streamlit · Statsmodels · Plotly")

    return dict(uploaded=uploaded, date_range=date_range, products=products,
                regions=regions, channels=channels, horizon=horizon, model=model,
                anomaly=anomaly_on, target=rev_target)


# ─────────────────────────────────────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────────────────────────────────────

def _filter(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    dr = cfg["date_range"]
    if len(dr) < 2:
        return df
    d0, d1 = pd.Timestamp(dr[0]), pd.Timestamp(dr[1])
    return df[
        df["date"].between(d0, d1)
        & df["product"].isin(cfg["products"])
        & df["region"].isin(cfg["regions"])
        & df["channel"].isin(cfg["channels"])
    ].copy()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _upload_screen() -> None:
    """Shown when no file has been uploaded yet."""
    _page_header("Sales Intelligence Platform", "Upload your sales data to get started")
    st.markdown("""
<div style="max-width:640px;margin:4rem auto;background:#FFFFFF;border:1px solid #E5E7EB;
            border-radius:12px;padding:3rem;text-align:center;
            box-shadow:0 4px 24px rgba(0,0,0,.06)">
  <div style="width:56px;height:56px;background:#EBF5FF;border-radius:12px;
              display:flex;align-items:center;justify-content:center;margin:0 auto 1.5rem">
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#1A56DB"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="17 8 12 3 7 8"/>
      <line x1="12" y1="3" x2="12" y2="15"/>
    </svg>
  </div>
  <h2 style="font-family:'Sora',sans-serif;font-size:1.3rem;font-weight:700;
             color:#0A1628;margin:0 0 .6rem">Upload Your Sales Data</h2>
  <p style="font-size:.88rem;color:#6B7280;margin:0 0 2rem;line-height:1.7">
    Use the <strong>Data Source</strong> panel in the sidebar to upload a
    CSV or Excel file. The full dashboard — forecasts, anomaly detection,
    segment analysis, and strategic insights — will populate automatically.
  </p>
  <div style="background:#F8FAFC;border:1px solid #E5E7EB;border-radius:8px;
              padding:1.25rem 1.5rem;text-align:left">
    <p style="font-size:.75rem;font-weight:700;text-transform:uppercase;
              letter-spacing:.08em;color:#374151;margin:0 0 .75rem">
      Required Column Schema
    </p>
    <table style="width:100%;border-collapse:collapse;font-size:.82rem">
      <thead>
        <tr>
          <th style="padding:6px 10px;background:#F1F5F9;border:1px solid #E5E7EB;
                     color:#374151;font-weight:700;text-align:left">Column</th>
          <th style="padding:6px 10px;background:#F1F5F9;border:1px solid #E5E7EB;
                     color:#374151;font-weight:700;text-align:left">Type</th>
          <th style="padding:6px 10px;background:#F1F5F9;border:1px solid #E5E7EB;
                     color:#374151;font-weight:700;text-align:left">Example</th>
        </tr>
      </thead>
      <tbody>
        <tr><td style="padding:6px 10px;border:1px solid #E5E7EB;font-family:'JetBrains Mono',monospace">date</td>
            <td style="padding:6px 10px;border:1px solid #E5E7EB;color:#6B7280">Date</td>
            <td style="padding:6px 10px;border:1px solid #E5E7EB;font-family:'JetBrains Mono',monospace;color:#057A55">2024-01-15</td></tr>
        <tr><td style="padding:6px 10px;border:1px solid #E5E7EB;font-family:'JetBrains Mono',monospace">product</td>
            <td style="padding:6px 10px;border:1px solid #E5E7EB;color:#6B7280">Text</td>
            <td style="padding:6px 10px;border:1px solid #E5E7EB;font-family:'JetBrains Mono',monospace;color:#057A55">Enterprise Suite</td></tr>
        <tr><td style="padding:6px 10px;border:1px solid #E5E7EB;font-family:'JetBrains Mono',monospace">region</td>
            <td style="padding:6px 10px;border:1px solid #E5E7EB;color:#6B7280">Text</td>
            <td style="padding:6px 10px;border:1px solid #E5E7EB;font-family:'JetBrains Mono',monospace;color:#057A55">North</td></tr>
        <tr><td style="padding:6px 10px;border:1px solid #E5E7EB;font-family:'JetBrains Mono',monospace">channel</td>
            <td style="padding:6px 10px;border:1px solid #E5E7EB;color:#6B7280">Text</td>
            <td style="padding:6px 10px;border:1px solid #E5E7EB;font-family:'JetBrains Mono',monospace;color:#057A55">Online</td></tr>
        <tr><td style="padding:6px 10px;border:1px solid #E5E7EB;font-family:'JetBrains Mono',monospace">revenue</td>
            <td style="padding:6px 10px;border:1px solid #E5E7EB;color:#6B7280">Number</td>
            <td style="padding:6px 10px;border:1px solid #E5E7EB;font-family:'JetBrains Mono',monospace;color:#057A55">2450.75</td></tr>
        <tr><td style="padding:6px 10px;border:1px solid #E5E7EB;font-family:'JetBrains Mono',monospace">units</td>
            <td style="padding:6px 10px;border:1px solid #E5E7EB;color:#6B7280">Integer</td>
            <td style="padding:6px 10px;border:1px solid #E5E7EB;font-family:'JetBrains Mono',monospace;color:#057A55">42</td></tr>
        <tr><td style="padding:6px 10px;border:1px solid #E5E7EB;font-family:'JetBrains Mono',monospace">cost</td>
            <td style="padding:6px 10px;border:1px solid #E5E7EB;color:#6B7280">Number</td>
            <td style="padding:6px 10px;border:1px solid #E5E7EB;font-family:'JetBrains Mono',monospace;color:#057A55">1100.00</td></tr>
        <tr><td style="padding:6px 10px;border:1px solid #E5E7EB;font-family:'JetBrains Mono',monospace">profit</td>
            <td style="padding:6px 10px;border:1px solid #E5E7EB;color:#6B7280">Number</td>
            <td style="padding:6px 10px;border:1px solid #E5E7EB;font-family:'JetBrains Mono',monospace;color:#057A55">1350.75</td></tr>
      </tbody>
    </table>
    <p style="font-size:.75rem;color:#6B7280;margin:.75rem 0 0">
      Date format: <code style="background:#E5E7EB;padding:1px 5px;border-radius:3px">YYYY-MM-DD</code>
      &nbsp;·&nbsp; profit = revenue - cost (auto-calculated if missing)
    </p>
  </div>
</div>""", unsafe_allow_html=True)


REQUIRED_COLS = {"date", "product", "region", "channel", "revenue", "units", "cost"}


def _validate(df: pd.DataFrame) -> list[str]:
    """Return list of missing required columns."""
    return sorted(REQUIRED_COLS - set(df.columns))


def main() -> None:
    _css()

    # Retrieve cached dataframe from session state
    raw = st.session_state.get("salesiq_raw_df", None)

    # Render sidebar ONCE — with real data if available, upload-only otherwise
    cfg = _sidebar(df=raw)

    # Handle file upload / clear
    if cfg["uploaded"] is not None:
        new_name = cfg["uploaded"].name
        # Only re-parse when a genuinely new file is provided
        if st.session_state.get("salesiq_filename") != new_name:
            try:
                new_raw = _parse_upload(cfg["uploaded"].read(), new_name)
            except Exception as exc:
                st.error(f"Could not read file: {exc}")
                st.info("Make sure the file is a valid CSV or Excel (.xlsx) with the required columns.")
                return

            missing = _validate(new_raw)
            if missing:
                st.error(f"Missing required columns: **{', '.join(missing)}**")
                st.info("Required: date, product, region, channel, revenue, units, cost")
                return

            if new_raw["date"].isna().all():
                st.error("The date column could not be parsed. Use YYYY-MM-DD format (e.g. 2024-01-15).")
                return

            # Cache parsed df and rerun so sidebar re-renders with full filter controls
            st.session_state["salesiq_raw_df"] = new_raw
            st.session_state["salesiq_filename"] = new_name
            st.rerun()
    else:
        # File uploader cleared — wipe cached data so landing screen reappears
        if raw is not None:
            st.session_state.pop("salesiq_raw_df", None)
            st.session_state.pop("salesiq_filename", None)
            raw = None

    # No data — show the landing / upload screen
    if raw is None:
        _upload_screen()
        return

    # Apply sidebar filters to the cached dataframe
    df  = _filter(raw, cfg)

    if df.empty:
        st.warning("No records match the current filter selection. Adjust the sidebar controls.")
        return

    ts        = _prepare_ts(df)
    anomalies = _detect_anomalies(ts) if cfg["anomaly"] else pd.Series(False, index=ts.index)
    n_anom    = int(anomalies.sum())
    fc_mean, fc_lo, fc_hi = _fc_dispatch(ts, cfg["horizon"], cfg["model"])

    total_rev    = df["revenue"].sum()
    total_profit = df["profit"].sum()
    total_units  = int(df["units"].sum())
    margin_pct   = total_profit / total_rev * 100 if total_rev else 0
    n_days       = max(1, (df["date"].max() - df["date"].min()).days + 1)
    growth_wow   = ts.pct_change(7).tail(14).mean() * 100
    fc_total     = float(fc_mean.sum())
    target_pct   = total_rev / cfg["target"] * 100 if cfg["target"] > 0 else 0
    cov_val      = float(ts.std() / ts.mean()) if ts.mean() > 0 else 0

    date_span = (f"{df['date'].min().strftime('%d %b %Y')}  –  "
                 f"{df['date'].max().strftime('%d %b %Y')}")
    _page_header(
        "Sales Intelligence Platform",
        f"Analysing {len(df):,} transactions across {n_days} days  ·  {date_span}",
        badge=cfg["model"] + " Forecast",
    )

    wow_dir = "up" if growth_wow >= 0 else "down"
    _kpi_row([
        dict(label="Total Revenue",     value=_inr(total_rev),
             delta=f"{growth_wow:+.1f}% WoW",          delta_dir=wow_dir),
        dict(label="Gross Profit",      value=_inr(total_profit),
             delta=f"{margin_pct:.1f}% Margin",
             delta_dir="up" if margin_pct >= 40 else "down"),
        dict(label="Units Sold",        value=f"{total_units:,}",
             delta=f"{total_units//n_days:,} / day avg", delta_dir="neutral"),
        dict(label="Forecast Revenue",  value=_inr(fc_total),
             delta=f"Next {cfg['horizon']} days",        delta_dir="neutral"),
        dict(label="Target Attainment", value=f"{target_pct:.1f}%",
             delta="On Track" if target_pct >= 100 else f"{_inr(cfg['target']-total_rev)} gap",
             delta_dir="up" if target_pct >= 100 else "down"),
    ])

    tabs = st.tabs([
        "Overview",
        "Revenue Intelligence",
        "Forecasting Engine",
        "Segment Analysis",
        "Product Analytics",
        "Risk Monitor",
        "Strategic Insights",
        "Data Explorer",
    ])

    # ── Overview ─────────────────────────────────────────────────────
    with tabs[0]:
        _section("Revenue Trend with Forecast Overlay")
        st.plotly_chart(chart_main_trend(ts, fc_mean, fc_lo, fc_hi, anomalies),
                        use_container_width=True, config={"displayModeBar": False},
                        key="p_main_trend")

        c1, c2 = st.columns([3, 2])
        with c1:
            _section("Channel Revenue Trend (7-Day MA)")
            st.plotly_chart(chart_channel_trend(df),
                            use_container_width=True, config={"displayModeBar": False},
                            key="p_ch_trend")
        with c2:
            _section("Channel Revenue Split")
            st.plotly_chart(chart_channel_donut(df),
                            use_container_width=True, config={"displayModeBar": False},
                            key="p_ch_donut")

        _section("Revenue by Day of Week")
        st.plotly_chart(chart_dayofweek(ts),
                        use_container_width=True, config={"displayModeBar": False},
                        key="p_dow")

    # ── Revenue Intelligence ──────────────────────────────────────────
    with tabs[1]:
        _section("Monthly Revenue Heatmap")
        st.plotly_chart(chart_monthly_heatmap(ts),
                        use_container_width=True, config={"displayModeBar": False},
                        key="p_monthly_hm")

        _section("Time-Series Decomposition")
        st.caption("Additive decomposition — Trend, Seasonal (period=7), Residual.")
        dec_fig = chart_decomposition(ts)
        if dec_fig is not None:
            st.plotly_chart(dec_fig, use_container_width=True,
                            config={"displayModeBar": False}, key="p_decomp")
        else:
            st.info("Select a wider date range to enable decomposition.")

        with st.expander("Augmented Dickey-Fuller Stationarity Test"):
            st.write(f"**Result:** {_adf_result(ts)}")
            st.caption("ADF null hypothesis: the series has a unit root (non-stationary). "
                       "SARIMA handles non-stationarity via first-order differencing (d=1).")

        _section("Revenue Descriptive Statistics")
        st.dataframe(pd.DataFrame({
            "Metric": ["Mean","Median","Std Dev","Min","Max","Skewness","Kurtosis"],
            "Value":  [_inr(float(ts.mean())), _inr(float(ts.median())),
                       _inr(float(ts.std())), _inr(float(ts.min())),
                       _inr(float(ts.max())), f"{float(ts.skew()):.3f}",
                       f"{float(ts.kurt()):.3f}"],
        }), use_container_width=True, hide_index=True)

    # ── Forecasting Engine ────────────────────────────────────────────
    with tabs[2]:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model",              cfg["model"])
        m2.metric("Horizon",            f"{cfg['horizon']} days")
        m3.metric("Forecast Total",     _inr(fc_total))
        m4.metric("Forecast Daily Avg", _inr(float(fc_mean.mean())))

        _section(f"{cfg['model']} Forecast — Next {cfg['horizon']} Days")
        st.plotly_chart(chart_main_trend(ts, fc_mean, fc_lo, fc_hi, anomalies),
                        use_container_width=True, config={"displayModeBar": False},
                        key="p_fc_trend")

        _section("Back-Test: Model Accuracy on 20% Hold-Out")
        bt_fig, sa_m, hw_m = chart_forecast_comparison(ts)
        if bt_fig is not None:
            st.plotly_chart(bt_fig, use_container_width=True,
                            config={"displayModeBar": False}, key="p_backtest")
            b1, b2 = st.columns(2)
            with b1:
                st.markdown("**SARIMA Metrics**")
                st.dataframe(pd.DataFrame([(k, f"{v:.4f}") for k, v in sa_m.items()],
                                          columns=["Metric","Value"]),
                             use_container_width=True, hide_index=True)
            with b2:
                st.markdown("**Holt-Winters Metrics**")
                st.dataframe(pd.DataFrame([(k, f"{v:.4f}") for k, v in hw_m.items()],
                                          columns=["Metric","Value"]),
                             use_container_width=True, hide_index=True)
        else:
            st.info("Select a wider date range to enable back-testing.")

        _section(f"Monte Carlo Simulation — 1,000 Paths over {cfg['horizon']} Days")
        st.caption("Geometric Brownian Motion parameterised on historical daily return distribution.")
        mc_fig, p10, p50, p90 = chart_monte_carlo(ts, cfg["horizon"])
        st.plotly_chart(mc_fig, use_container_width=True,
                        config={"displayModeBar": False}, key="p_mc")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("P10 — Pessimistic", _inr(p10))
        mc2.metric("P50 — Base Case",   _inr(p50))
        mc3.metric("P90 — Optimistic",  _inr(p90))

        _section("Export Forecast")
        fc_df = pd.DataFrame({
            "date":     fc_mean.index.strftime("%Y-%m-%d"),
            "forecast": fc_mean.values.round(2),
            "lower_90": fc_lo.values.round(2),
            "upper_90": fc_hi.values.round(2),
        })
        buf = io.StringIO()
        fc_df.to_csv(buf, index=False)
        st.download_button(
            "Export Forecast as CSV", data=buf.getvalue().encode(),
            file_name=f"forecast_{cfg['model'].lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    # ── Segment Analysis ──────────────────────────────────────────────
    with tabs[3]:
        c1, c2 = st.columns([2, 3])
        with c1:
            _section("Revenue by Region")
            st.plotly_chart(chart_region_bar(df), use_container_width=True,
                            config={"displayModeBar": False}, key="p_reg_bar")
        with c2:
            _section("Regional Revenue Over Time (7-Day MA)")
            st.plotly_chart(chart_region_trend(df), use_container_width=True,
                            config={"displayModeBar": False}, key="p_reg_trend")

        _section("Product × Region Revenue Matrix")
        st.plotly_chart(chart_region_product_heatmap(df), use_container_width=True,
                        config={"displayModeBar": False}, key="p_reg_hm")

        _section("Region Summary")
        def _reg_summary(grp):
            rev = grp["revenue"].sum(); cst = grp["cost"].sum()
            return pd.Series({
                "Revenue":       _inr(rev),
                "Gross Profit":  _inr(grp["profit"].sum()),
                "Margin %":      f"{(rev-cst)/rev*100:.1f}%" if rev else "—",
                "Units":         f"{int(grp['units'].sum()):,}",
                "Avg Daily Rev": _inr(rev / n_days),
            })
        st.dataframe(
            df.groupby("region").apply(_reg_summary, include_groups=False).reset_index(),
            use_container_width=True, hide_index=True,
        )

    # ── Product Analytics ─────────────────────────────────────────────
    with tabs[4]:
        c1, c2 = st.columns(2)
        with c1:
            _section("Revenue by Product")
            st.plotly_chart(chart_product_revenue(df), use_container_width=True,
                            config={"displayModeBar": False}, key="p_prod_rev")
        with c2:
            _section("Gross Margin by Product")
            st.plotly_chart(chart_margin_bar(df), use_container_width=True,
                            config={"displayModeBar": False}, key="p_margin")

        _section("Product Revenue Over Time (7-Day MA)")
        st.plotly_chart(chart_product_trend(df), use_container_width=True,
                        config={"displayModeBar": False}, key="p_prod_trend")

        _section("Product Summary")
        def _prod_summary(grp):
            rev = grp["revenue"].sum(); cst = grp["cost"].sum()
            return pd.Series({
                "Revenue":       _inr(rev),
                "Gross Profit":  _inr(grp["profit"].sum()),
                "Margin %":      f"{(rev-cst)/rev*100:.1f}%" if rev else "—",
                "Units":         f"{int(grp['units'].sum()):,}",
                "Avg Order":     f"₹{rev/max(1,len(grp)):.0f}",
                "Revenue Share": f"{rev/total_rev*100:.1f}%",
            })
        st.dataframe(
            df.groupby("product").apply(_prod_summary, include_groups=False).reset_index(),
            use_container_width=True, hide_index=True,
        )

    # ── Risk Monitor ──────────────────────────────────────────────────
    with tabs[5]:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Anomalies Detected",      n_anom)
        r2.metric("Coefficient of Variation", f"{cov_val:.3f}")
        r3.metric("Max Anomaly Value",
                  _inr(float(ts[anomalies].max())) if anomalies.any() else "—")
        r4.metric("ADF Stationarity",         _adf_result(ts).split(" ")[0])

        if n_anom:
            st.warning(f"{n_anom} statistical anomalies detected (rolling Z-score, 2.8σ threshold).")
        else:
            st.success("No significant anomalies detected in the selected period.")

        _section("Anomaly Detection — Revenue with Rolling Normal Corridor")
        st.plotly_chart(chart_anomaly(ts, anomalies), use_container_width=True,
                        config={"displayModeBar": False}, key="p_anomaly")

        _section("30-Day Rolling Volatility")
        st.plotly_chart(chart_volatility(ts), use_container_width=True,
                        config={"displayModeBar": False}, key="p_volatility")

        if anomalies.any():
            _section("Anomaly Log")
            anom_log = (
                df[df["date"].isin(ts[anomalies].index)]
                .sort_values("revenue", ascending=False)
                [["date","product","region","channel","revenue","units","profit"]]
                .head(40).copy()
            )
            anom_log["revenue"] = anom_log["revenue"].apply(lambda x: f"₹{x:,.2f}")
            anom_log["profit"]  = anom_log["profit"].apply(lambda x: f"₹{x:,.2f}")
            st.dataframe(anom_log, use_container_width=True, hide_index=True)

    # ── Strategic Insights ────────────────────────────────────────────
    with tabs[6]:
        _section("AI-Generated Strategic Insights")
        st.caption("Derived from statistical analysis of revenue momentum, margin, "
                   "concentration risk, volatility, anomaly count, and target attainment.")
        for ins in _generate_insights(df, ts, cfg["target"], n_anom):
            _insight_card(ins["title"], ins["body"], ins["kind"])

        _section("Recommended Actions")
        items = [
            ("Resolve all flagged anomalies before period close",           n_anom > 5),
            (f"Improve margin — currently {margin_pct:.1f}% vs 50% benchmark", margin_pct < 50),
            ("Replicate top-region tactics in the weakest-performing region",  True),
            ("Run A/B pricing test on the second-ranked product",               True),
            ("Introduce annual billing to reduce revenue volatility",           cov_val > 0.30),
            ("Close target gap — accelerate pipeline in top segments",          target_pct < 100),
            ("Diversify product mix to reduce concentration risk",
             bool(total_rev and
                  df.groupby("product")["revenue"].sum().max() / total_rev > 0.40)),
        ]
        for text, urgent in items:
            flag   = "Critical" if urgent else "Recommended"
            colour = C["red"]   if urgent else C["green"]
            bg     = C["red_lt"] if urgent else C["green_lt"]
            st.markdown(
                f'<div style="padding:10px 14px;margin-bottom:6px;border-radius:6px;'
                f'background:{bg};border-left:3px solid {colour};">'
                f'<span style="font-size:.73rem;font-weight:700;color:{colour};'
                f'text-transform:uppercase;letter-spacing:.08em">{flag}</span>'
                f'<p style="font-size:.84rem;color:#374151;margin:3px 0 0">{text}</p></div>',
                unsafe_allow_html=True,
            )

    # ── Data Explorer ─────────────────────────────────────────────────
    with tabs[7]:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Records", f"{len(df):,}")
        d2.metric("Date Span",     f"{n_days} days")
        d3.metric("Products",      df["product"].nunique())
        d4.metric("Regions",       df["region"].nunique())

        _section("Pearson Correlation Matrix")
        st.plotly_chart(chart_correlation(df), use_container_width=True,
                        config={"displayModeBar": False}, key="p_corr")

        _section("Filtered Dataset — Latest 500 Rows")
        preview = df.sort_values("date", ascending=False).head(500).copy()
        for col in ["revenue", "cost", "profit"]:
            if col in preview.columns:
                preview[col] = preview[col].round(2)
        st.dataframe(preview, use_container_width=True, hide_index=True)

        _section("Export")
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "Export Filtered Data as CSV",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name=f"salesiq_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

        with st.expander("Descriptive Statistics"):
            st.dataframe(
                df[["revenue","units","cost","profit"]].describe().round(2),
                use_container_width=True,
            )


if __name__ == "__main__":
    main()