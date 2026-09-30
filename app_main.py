from data.stock_data import get_data
from core.indicators import calc_indicators
from core.trust_engine import get_trust_scores
from core.signals import get_signal
from ai.prompts import BOOK_KNOWLEDGE
from ui.styles import CUSTOM_CSS
from data.feedback import save_feedback, load_feedback
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
import pandas as pd
import numpy as np
from groq import Groq
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import heapq
import os
from newsapi import NewsApiClient
import yfinance as yf

@st.cache_data(ttl=300)
def get_stock_data(symbol, period):
    return yf.Ticker(symbol).history(period=period)

def section_title(text):
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:8px;margin:1rem 0 0.6rem;">
<div style="width:3px;height:15px;background:#00C9A7;border-radius:2px;"></div>
<span style="font-size:0.72rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:#8B90A0;">{text}</span>
</div>
    """, unsafe_allow_html=True)

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except:
    pass

    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    

st.set_page_config(
    page_title="BharatFinAI",
    page_icon="📈",
    layout="wide",

initial_sidebar_state="expanded"
)

#st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
# ============================================================
# BHARATFINAI — MINIMAL PREMIUM UI
# ============================================================

CUSTOM_CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ================= GLOBAL ================= */

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 80% 0%, rgba(255,255,255,0.025), transparent 30%),
        #0B0D10;
    color: #E8EAED;
}

/* Main container */
.block-container {
    max-width: 1450px;
    padding-top: 2.2rem;
    padding-bottom: 4rem;
}

/* Remove excessive Streamlit top spacing */
[data-testid="stHeader"] {
    background: transparent;
}

/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {
    background: #0E1013;
    border-right: 1px solid #1D2025;
}

section[data-testid="stSidebar"] > div {
    padding: 1.5rem 1.15rem;
}

section[data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif;
}

.sidebar-brand {
    padding: 4px 4px 24px 4px;
}

.sidebar-brand .brand-name {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.4px;
    color: #F4F5F6;
}

.sidebar-brand .brand-sub {
    margin-top: 5px;
    font-size: 11px;
    color: #858A93;
    letter-spacing: 0.2px;
}

/* ================= HEADINGS ================= */

h1, h2, h3 {
    color: #F3F4F6 !important;
    letter-spacing: -0.5px;
}

h1 {
    font-weight: 700 !important;
}

h2, h3 {
    font-weight: 600 !important;
}

p {
    color: #9A9FA8;
}

/* ================= CARDS ================= */

.bf-card {
    background: #111419;
    border: 1px solid #20242A;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 14px;
}

.bf-card:hover {
    border-color: #2A2F37;
}

.bf-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #777D87;
    font-weight: 600;
}

.bf-value {
    margin-top: 7px;
    font-size: 25px;
    font-weight: 600;
    color: #F2F3F5;
}

.bf-muted {
    color: #777D87;
    font-size: 12px;
}

/* ================= METRICS ================= */

[data-testid="stMetric"] {
    background: #111419;
    border: 1px solid #20242A;
    border-radius: 12px;
    padding: 16px;
}

[data-testid="stMetricLabel"] {
    color: #7F858F !important;
    font-size: 11px !important;
}

[data-testid="stMetricValue"] {
    color: #F2F3F5 !important;
    font-size: 22px !important;
    font-weight: 600 !important;
}

[data-testid="stMetricDelta"] {
    font-size: 12px !important;
}

/* ================= BUTTONS ================= */

.stButton > button {
    border-radius: 9px;
    border: 1px solid #292E35;
    background: #15181D;
    color: #E9EBEF;
    font-weight: 500;
    min-height: 40px;
    transition: all 0.15s ease;
}

.stButton > button:hover {
    border-color: #59616D;
    background: #1A1E24;
    color: #FFFFFF;
}

.stButton > button[kind="primary"] {
    background: #E8EAED;
    color: #0B0D10;
    border: none;
}

.stButton > button[kind="primary"]:hover {
    background: #FFFFFF;
}

/* ================= INPUTS ================= */

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stTextArea textarea {
    background: #111419 !important;
    color: #E8EAED !important;
    border: 1px solid #252A31 !important;
    border-radius: 9px !important;
}

.stTextInput input:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus {
    border-color: #555D68 !important;
    box-shadow: none !important;
}

/* ================= TABS ================= */

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: transparent;
    border-bottom: 1px solid #20242A;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #777D87;
    border-radius: 7px 7px 0 0;
    padding: 10px 14px;
    font-size: 12px;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #D8DBDF;
}

.stTabs [aria-selected="true"] {
    color: #F2F3F5 !important;
    background: #15181D;
}

/* ================= DIVIDERS ================= */

hr {
    border-color: #20242A !important;
}

/* ================= ALERTS ================= */

div[data-testid="stAlert"] {
    border-radius: 10px;
    border: 1px solid #252A31;
    background: #111419;
}

/* ================= PROGRESS ================= */

div[data-testid="stProgressBar"] {
    background: #20242A;
    border-radius: 99px;
}

div[data-testid="stProgressBar"] > div {
    border-radius: 99px;
}

/* ================= DATAFRAME ================= */

[data-testid="stDataFrame"] {
    border: 1px solid #20242A;
    border-radius: 10px;
    overflow: hidden;
}

/* ================= EXPANDER ================= */

.streamlit-expanderHeader {
    background: #111419 !important;
    border: 1px solid #20242A !important;
    border-radius: 9px !important;
}

/* ================= CAPTION ================= */

.stCaption {
    color: #6F757E !important;
}

/* ================= CUSTOM HEADER ================= */

.bf-header {
    padding: 4px 0 24px 0;
}

.bf-header-top {
    display: flex;
    align-items: center;
    gap: 10px;
}

.bf-logo {
    width: 34px;
    height: 34px;
    border-radius: 9px;
    background: #F1F3F5;
    color: #0B0D10;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 14px;
}

.bf-title {
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.8px;
    color: #F5F6F7;
}

.bf-subtitle {
    margin-top: 7px;
    font-size: 12px;
    color: #777D87;
}

/* ================= SECTION LABEL ================= */

.bf-section {
    display: flex;
    align-items: center;
    gap: 9px;
    margin: 22px 0 12px 0;
}

.bf-section-line {
    width: 3px;
    height: 17px;
    border-radius: 5px;
    background: #DDE1E6;
}

.bf-section-title {
    font-size: 12px;
    font-weight: 600;
    color: #C9CDD2;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ================= MOBILE ================= */

@media (max-width: 768px) {

    .block-container {
        padding: 1rem;
    }

    .bf-title {
        font-size: 22px;
    }

    [data-testid="stMetricValue"] {
        font-size: 18px !important;
    }

}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
#with st.sidebar:
   # st.markdown("## 📈 BharatFinAI")
    #st.markdown("*Hindi AI Stock Analyzer*")
    #st.divider()
    #st.markdown("**Quick Stocks:**")
    #st.markdown("RELIANCE • TCS • HDFCBANK\nINFY • WIPRO • SBIN\nTATAMOTORS • ADANIENT\nMARUTI • SUNPHARMA")
    #st.divider()
    #st.markdown("**Indicators Guide:**")
with st.sidebar:

    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-name">BharatFinAI</div>
        <div class="brand-sub">
            Financial Intelligence for Indian Markets
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    st.markdown(
        "<div class='bf-label'>Quick Access</div>",
        unsafe_allow_html=True
    )

    quick_stocks = [
        "RELIANCE",
        "TCS",
        "HDFCBANK",
        "INFY",
        "SBIN",
        "TATAMOTORS",
        "MARUTI",
        "SUNPHARMA"
    ]

    for stock in quick_stocks:
        if st.button(
            stock,
            key=f"quick_{stock}",
            use_container_width=True
        ):
            st.session_state["watch_stock"] = stock

    st.markdown("---")

    st.markdown(
        "<div class='bf-label'>Market Signals</div>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div style="
        font-size:12px;
        line-height:2;
        color:#858A93;
        margin-top:8px;
    ">
        <b style="color:#D8DBDF;">RSI</b> &nbsp; Momentum<br>
        <b style="color:#D8DBDF;">MACD</b> &nbsp; Trend confirmation<br>
        <b style="color:#D8DBDF;">Volume</b> &nbsp; Market participation<br>
        <b style="color:#D8DBDF;">SMA</b> &nbsp; Trend structure
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.caption(
        "Educational research platform. "
        "Always verify information independently."
    )

    WATCHLIST_FILE = "data/watchlist.csv"
    st.markdown("🔴 RSI > 70 = Overbought\n💚 RSI < 30 = Oversold\n📈 MACD Bullish = Uptrend\n🔊 Vol > 1.5x = Strong")
    st.divider()
    st.caption("⚠️ Sirf educational purpose.\nInvest apni research ke baad karein.")

    WATCHLIST_FILE = "data/watchlist.csv"

if not os.path.exists(WATCHLIST_FILE):
    pd.DataFrame(columns=["Symbol"]).to_csv(WATCHLIST_FILE, index=False)

PORTFOLIO_FILE = "data/portfolio.csv"

if not os.path.exists(PORTFOLIO_FILE):
    pd.DataFrame(
        columns=["Symbol", "Quantity", "BuyPrice"]
    ).to_csv(PORTFOLIO_FILE, index=False)

# Portfolio File
PORTFOLIO_FILE = "data/portfolio.csv"

# Portfolio Section
st.sidebar.divider()

st.sidebar.subheader("💼 Portfolio")

p_symbol = st.sidebar.text_input("Portfolio Stock", key="portfolio_stock")

p_qty = st.sidebar.number_input(
    "Quantity",
    min_value=1,
    step=1
)

p_buy = st.sidebar.number_input(
    "Buy Price",
    min_value=0.0,
    step=1.0
)

st.sidebar.markdown("---")

sell_symbol = st.sidebar.text_input(
    "Sell Stock",
    key="sell_stock"
)

if st.sidebar.button("Sell Stock"):
    df_port = pd.read_csv(PORTFOLIO_FILE)

    df_port = df_port[
        df_port["Symbol"] != sell_symbol.upper()
    ]

    df_port.to_csv(PORTFOLIO_FILE, index=False)

    st.sidebar.success(
        f"{sell_symbol.upper()} Sold!"
    )

if st.sidebar.button("Add To Portfolio"):

    df_port = pd.read_csv(PORTFOLIO_FILE)

    df_port.loc[len(df_port)] = [
        p_symbol.upper(),
        p_qty,
        p_buy
    ]

    df_port.to_csv(PORTFOLIO_FILE, index=False)





    st.sidebar.success("Portfolio Updated!")
    st.sidebar.markdown("### 📊 Current Portfolio")

df_port = pd.read_csv(PORTFOLIO_FILE)

if len(df_port) > 0:
    st.sidebar.dataframe(df_port, use_container_width=True)
    # Portfolio Summary

    total_investment = (df_port["Quantity"] * df_port["BuyPrice"]).sum()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Portfolio Summary")

    st.sidebar.metric(
        "Total Investment",
        f"₹{total_investment:,.2f}"
    )
    current_value = 0
    portfolio_rows = []

    for _, row in df_port.iterrows():
        try:
            stock_symbol = row["Symbol"]
            qty = row["Quantity"]

            yf_symbol = stock_symbol if stock_symbol.endswith(".NS") else stock_symbol + ".NS"
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period="5d")

            if not hist.empty:
                current_price = hist["Close"].iloc[-1]
                current_value += current_price * qty

                investment = row["Quantity"] * row["BuyPrice"]
                stock_value = current_price * qty
                stock_pl = stock_value - investment
                stock_pl_pct = (stock_pl / investment * 100) if investment > 0 else 0

                portfolio_rows.append({
                    "Symbol": stock_symbol,
                    "Qty": qty,
                    "Buy Price": row["BuyPrice"],
                    "Current Price": round(current_price, 2),
                    "Investment": round(investment, 2),
                    "Current Value": round(stock_value, 2),
                    "P/L": round(stock_pl, 2),
                    "P/L %": round(stock_pl_pct, 2)
                })

        except Exception as e:
            print(e)

    profit_loss = current_value - total_investment
    return_pct = (profit_loss / total_investment * 100) if total_investment > 0 else 0

    if portfolio_rows:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Portfolio Analytics")

        portfolio_df = pd.DataFrame(portfolio_rows)

        st.sidebar.dataframe(
            portfolio_df,
            use_container_width=True
        )

        st.sidebar.metric(
            "Current Value",
            f"₹{current_value:,.2f}"
        )

        st.sidebar.metric(
            "Profit / Loss",
            f"₹{profit_loss:,.2f}",
            f"{return_pct:.2f}%"
        )
        
# Portfolio Health Score
stock_count = len(df_port)
diversification_score = min(stock_count * 20, 100)

return_pct = float(return_pct) if "return_pct" in locals() else 0

profit_score = 100 if return_pct > 10 else 75 if return_pct > 0 else 50 if return_pct > -10 else 25

health_score = int((diversification_score * 0.4) + (profit_score * 0.6))

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Portfolio Health Score")

st.sidebar.metric(
    "Health Score",
    f"{health_score}/100"
)

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin:1.2rem 0 0.8rem;">
      <div style="width:3px;height:15px;background:#00C9A7;border-radius:2px;"></div>
      <span style="font-size:0.72rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:#8B90A0;">📌 Executive Portfolio Summary</span>
    </div>
    """, unsafe_allow_html=True)
    risk_level = "LOW RISK"
    portfolio_grade = "A"
    main_weakness = "No major weakness detected"
    suggested_action = "Portfolio looks stable"

    if health_score < 40:
        risk_level = "HIGH RISK"
        portfolio_grade = "D"
        main_weakness = "High risk compared to expected return"
        suggested_action = "Immediate rebalancing required"
    elif health_score < 60:
        risk_level = "MEDIUM RISK"
        portfolio_grade = "C"
        main_weakness = "Portfolio needs optimization"
        suggested_action = "Reduce risky exposure"
    elif health_score < 80:
        risk_level = "LOW-MEDIUM RISK"
        portfolio_grade = "B"
        main_weakness = "Minor concentration risk"
        suggested_action = "Improve diversification"

    st.markdown(f"""
    <div style="background:#16181F;border:1px solid #252836;border-radius:14px;padding:1rem 1.2rem;margin-bottom:0.8rem;">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;margin-bottom:0.8rem;">
        <div style="background:#1C1F2A;border-radius:8px;padding:0.7rem 1rem;">
          <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#555A6E;margin-bottom:4px;">Risk Level</div>
          <div style="font-size:1rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:#F59E0B;">{risk_level}</div>
        </div>
        <div style="background:#1C1F2A;border-radius:8px;padding:0.7rem 1rem;">
          <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#555A6E;margin-bottom:4px;">Portfolio Grade</div>
          <div style="font-size:1rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:#00C9A7;">{portfolio_grade}</div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;">
        <div style="background:#0D1F3C;border:1px solid rgba(59,130,246,0.3);border-radius:8px;padding:0.6rem 1rem;font-size:0.75rem;color:#BFDBFE;">⚠️ {main_weakness}</div>
        <div style="background:#2D2008;border:1px solid rgba(245,158,11,0.3);border-radius:8px;padding:0.6rem 1rem;font-size:0.75rem;color:#FFE0A3;">💡 {suggested_action}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    section_title("---")
    section_title("🎯 AI Confidence Meter")

    confidence = min(95, max(50, health_score))

    st.progress(confidence / 100)

    st.metric(
        "AI Confidence",
        f"{confidence}%"
    )

    if confidence >= 80:
        st.success("🟢 High Confidence")
    elif confidence >= 60:
        st.warning("🟡 Medium Confidence")
    else:
        st.error("🔴 Low Confidence")

st.sidebar.progress(health_score / 100)

if health_score >= 75:
    st.sidebar.success("Strong Portfolio ✅")
elif health_score >= 50:
    st.sidebar.warning("Average Portfolio ⚠️")
else:
    st.sidebar.error("Weak Portfolio 🚨")

st.sidebar.caption(
    f"Stocks: {stock_count} | Diversification Score: {diversification_score}/100"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚠️ Risk Meter")

if diversification_score < 40:
    risk_level = "HIGH RISK 🔴"
elif diversification_score < 70:
    risk_level = "MEDIUM RISK 🟡"
else:
    risk_level = "LOW RISK 🟢"

st.sidebar.metric("Risk Level", risk_level)

# AI Portfolio Suggestion
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 AI Portfolio Suggestion")

if stock_count == 1:
    st.sidebar.warning("Only 1 stock hai. Diversification low hai.")
    st.sidebar.info("Suggestion: 3-5 alag sector ke stocks add karo.")

elif stock_count < 4:
    st.sidebar.warning("Portfolio thoda concentrated hai.")
    st.sidebar.info("Suggestion: Banking, IT, FMCG, Auto jaise sectors mix karo.")

else:
    st.sidebar.success("Diversification better lag rahi hai.")

if return_pct < -10:
    st.sidebar.error("Portfolio loss high hai. Risk review karo.")
elif return_pct < 0:
    st.sidebar.warning("Portfolio negative hai. Panic sell mat karo, analysis check karo.")
elif return_pct > 10:
    st.sidebar.success("Portfolio profitable hai. Partial profit booking consider kar sakte ho.")
else:
    st.sidebar.info("Portfolio stable zone me hai.")
    
profit_loss = 0

st.sidebar.metric(
        "Profit / Loss",
        f"₹{profit_loss:,.2f}",
        f"{return_pct:.2f}%"
    )
#     # Portfolio Allocation Chart

# fig_pie = go.Figure(
#     data=[
#         go.Pie(
#             labels=df_port["Symbol"],
#             values=df_port["Quantity"] * df_port["BuyPrice"],
#             hole=0.45
#         )
#     ]
# )

# fig_pie.update_layout(
#     title="Portfolio Allocation",
#     height=300,
#     template="plotly_dark"
# )


            
#else:
 #   st.sidebar.info("No stocks added")

new_stock = st.sidebar.text_input("Add Stock Symbol", key="watchlist_stock_input")

if st.sidebar.button("Add To Watchlist"):
    df_watch = pd.read_csv(WATCHLIST_FILE)

    if new_stock.upper() not in df_watch["Symbol"].values:
        df_watch.loc[len(df_watch)] = [new_stock.upper()]
        df_watch.to_csv(WATCHLIST_FILE, index=False)
        st.sidebar.success("Added!")

if "watch_stock" in st.session_state:
    default_stock = st.session_state["watch_stock"]
else:
    default_stock = "RELIANCE"

df_watch = pd.read_csv(WATCHLIST_FILE)

st.sidebar.markdown("### My Watchlist")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Watchlist Live Prices")

for stock in df_watch["Symbol"]:
    try:
        yf_symbol = stock if stock.endswith(".NS") else stock + ".NS"
        hist = yf.Ticker(yf_symbol).history(period="1d")

        if not hist.empty:
            price = hist["Close"].iloc[-1]
            st.sidebar.metric(stock, f"₹{price:,.2f}")
        else:
            st.sidebar.warning(f"{stock} data nahi mila")

    except:
        st.sidebar.warning(f"{stock} error")

for stock in df_watch["Symbol"]:

    col1, col2, col3 = st.sidebar.columns([3, 1, 1])

    with col1:
        st.write(f"📈 {stock}")

    with col2:
        if st.button("Go", key=f"go_{stock}"):
            st.session_state["watch_stock"] = stock

    with col3:
        if st.button("❌", key=f"remove_{stock}"):
            df_watch = df_watch[df_watch["Symbol"] != stock]
            df_watch.to_csv(WATCHLIST_FILE, index=False)
            st.sidebar.success(f"{stock} removed")
            st.rerun()

with st.sidebar:
    st.markdown("### 📊 Feedback Records")

    feedback_df = load_feedback()

    if not feedback_df.empty:

        st.dataframe(


            feedback_df,
            use_container_width=True,
            hide_index=True
        )

        feedback_csv = feedback_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Feedback CSV Download",
            feedback_csv,
            "bharatfinai_feedback.csv",
            "text/csv"
        )

    else:
        st.info("Abhi koi feedback nahi mila.")

    st.markdown("### 💬 Feedback Section")
    st.caption("Aapka feedback BharatFinAI ko better banane me help karega.")

    name = st.text_input("Naam")
    city = st.text_input("City")

    user_type_fb = st.selectbox(
        "Aapka level",
        ["Beginner", "Student", "Investor", "Trader", "Other"]
    )

    rating = st.slider("Rating", 1, 5, 4)

    confusion = st.text_area("Kya confusing laga?")

    suggestion = st.text_area("Kya improve karna chahiye?")

    if st.button("✅ Feedback Submit"):

        save_feedback(
            name,
            city,
            user_type_fb,
            rating,
            confusion,
            suggestion
        )

        st.success("Thank you! Feedback save ho gaya ✅")   
st.markdown("""
<div class="bf-header">

<div class="bf-header-top">

<div class="bf-logo">
BF
</div>

<div class="bf-title">
BharatFinAI
</div>

</div>

<div class="bf-subtitle">
AI-powered financial intelligence for Indian markets
&nbsp;·&nbsp;
NSE / BSE
&nbsp;·&nbsp;
Research
&nbsp;·&nbsp;
Risk
</div>

</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7  = st.tabs([
    "Stock Analysis",
    "Market Scanner",
    " AI Comparison",
    "Quant",
    "Monte Carlo",
    "Risk",
    "Portfolio"  
])


# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
 
# ─────────────────────────────────────────
# BALANCE SHEET
# ─────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_balance_sheet(symbol):

    try:

        yf_symbol = symbol.upper()

        if not yf_symbol.endswith(".NS"):
            yf_symbol = yf_symbol + ".NS"

        ticker = yf.Ticker(yf_symbol)

        balance_sheet = ticker.balance_sheet

        if balance_sheet is None or balance_sheet.empty:
            return None

        return balance_sheet

    except Exception:
        return None

def get_news_sentiment(stock):
    try:
        newsapi = NewsApiClient(
            api_key=os.getenv("NEWS_API_KEY")
        )

        news = newsapi.get_everything(
            q=stock,
            language="en",
            sort_by="publishedAt",
            page_size=10
        )

        headlines = [
            article["title"]
            for article in news["articles"]
            if article.get("title")
        ]

        if not headlines:
            return "Neutral", 0

        total = 0

        for headline in headlines:
            total += TextBlob(headline).sentiment.polarity

        avg = total / len(headlines)

        if avg > 0.1:
            return "Positive", 10
        elif avg < -0.1:
            return "Negative", -10
        else:
            return "Neutral", 0

    except Exception:
        return "Neutral", 0


# ─────────────────────────────────────────
# NEWS + SENTIMENT ENGINE (helpers)
# ─────────────────────────────────────────
import re as _re

_POSITIVE_TERMS = [
    "surge", "surges", "jump", "jumps", "soar", "soars", "rally", "rallies",
    "gain", "gains", "record profit", "beats estimates", "beat estimates",
    "upgrade", "upgrades", "outperform", "bags order", "wins order",
    "order win", "strong growth", "profit rises", "profit jumps", "dividend",
    "bonus", "buyback", "expansion", "approval", "approved", "breakthrough",
    "all-time high", "top pick", "buy rating",
]

_NEGATIVE_TERMS = [
    "plunge", "plunges", "tumble", "tumbles", "slump", "slumps", "fall",
    "falls", "drop", "drops", "miss estimates", "misses estimates",
    "downgrade", "downgrades", "underperform", "probe", "fraud", "penalty",
    "fined", "lawsuit", "default", "loss widens", "profit falls",
    "profit drops", "resigns", "resignation", "raid", "ban", "recall",
    "weak growth", "sell rating", "layoff", "layoffs",
]

_POS_RE = [_re.compile(r"\b" + _re.escape(t) + r"\b") for t in _POSITIVE_TERMS]
_NEG_RE = [_re.compile(r"\b" + _re.escape(t) + r"\b") for t in _NEGATIVE_TERMS]

_EVENT_RES = {
    "📊 Earnings / Results": _re.compile(
        r"\b(q[1-4]|quarterly results?|quarter results?|earnings|net profit|profit after tax|pat|ebitda|results)\b",
        _re.IGNORECASE),
    "💰 Dividend": _re.compile(r"\b(dividend|payout|record date)\b", _re.IGNORECASE),
    "🎁 Bonus / Split": _re.compile(r"\b(bonus issue|bonus shares?|stock split|share split)\b", _re.IGNORECASE),
    "🔄 Buyback": _re.compile(r"\b(buyback|buy-back|share repurchase)\b", _re.IGNORECASE),
    "🤝 M&A / Deals": _re.compile(
        r"\b(acquisition|acquires?|merger|demerger|stake sale|takeover|joint venture)\b", _re.IGNORECASE),
    "📈 Analyst Rating": _re.compile(
        r"\b(upgrades?|downgrades?|target price|price target|initiates coverage|buy rating|sell rating|overweight|underweight)\b",
        _re.IGNORECASE),
    "⚖️ Regulatory / Legal": _re.compile(
        r"\b(sebi|penalty|fined?|probe|investigation|lawsuit|tribunal|nclt|show[- ]cause)\b", _re.IGNORECASE),
    "👔 Management Change": _re.compile(
        r"\b(resigns?|resignation|steps? down|appoints?|appointed|new (ceo|cfo|md|chairman))\b", _re.IGNORECASE),
    "📦 Orders / Contracts": _re.compile(
        r"\b(order win|bags order|wins order|secures order|order book|bags contract|wins contract)\b",
        _re.IGNORECASE),
    "💵 Fundraise": _re.compile(
        r"\b(qip|rights issue|preferential (issue|allotment)|fund ?raise|ncd|ipo|fpo)\b", _re.IGNORECASE),
}

NEWS_POSITIVE_CUTOFF = 0.12
NEWS_NEGATIVE_CUTOFF = -0.12


def score_news_text(text):
    """TextBlob polarity + finance keyword boost. Returns score in [-1, 1]."""
    if not text:
        return 0.0
    try:
        tb_score = TextBlob(text).sentiment.polarity
    except Exception:
        tb_score = 0.0
    low = text.lower()
    pos_hits = sum(1 for p in _POS_RE if p.search(low))
    neg_hits = sum(1 for p in _NEG_RE if p.search(low))
    keyword_score = max(-1.0, min(1.0, (pos_hits - neg_hits) * 0.35))
    return 0.4 * tb_score + 0.6 * keyword_score


def analyze_news_items(items):
    analyzed = []
    for it in items:
        text = f"{it['title']}. {it.get('description', '')}"
        score = score_news_text(text)
        if score > NEWS_POSITIVE_CUTOFF:
            label = "Positive"
        elif score < NEWS_NEGATIVE_CUTOFF:
            label = "Negative"
        else:
            label = "Neutral"
        events = [name for name, pat in _EVENT_RES.items() if pat.search(text)]
        analyzed.append({**it, "score": round(score, 2), "label": label, "events": events})
    return analyzed


# Ticker -> company display-name variants, jo headlines me actually use hote hain.
# NewsAPI ka "reliance" jaisa common-English-word ticker ke saath galat match na kare,
# isliye company ka pura naam bhi include kiya hai jahan pata hai.
_TICKER_NAME_VARIANTS = {
    "RELIANCE": ["Reliance Industries", "RIL"],
    "TCS": ["Tata Consultancy"],
    "INFY": ["Infosys"],
    "HDFCBANK": ["HDFC Bank"],
    "ICICIBANK": ["ICICI Bank"],
    "SBIN": ["SBI", "State Bank of India"],
    "KOTAKBANK": ["Kotak Mahindra Bank"],
    "TATAMOTORS": ["Tata Motors"],
    "MARUTI": ["Maruti Suzuki"],
    "SUNPHARMA": ["Sun Pharma"],
    "ITC": ["ITC Limited"],
    "HINDUNILVR": ["Hindustan Unilever", "HUL"],
    "WIPRO": ["Wipro"],
    "HCLTECH": ["HCL Technologies", "HCLTech"],
    "ADANIENT": ["Adani Enterprises"],
}

_NEWS_FINANCE_TERMS = (
    "stock OR shares OR share OR NSE OR BSE OR results OR profit OR revenue "
    "OR earnings OR Q1 OR Q2 OR Q3 OR Q4"
)


def _relevance_filter(items, base_symbol):
    """Extra safety net: sirf wo articles rakho jinke title me genuinely company ka naam ho."""
    variants = [base_symbol] + _TICKER_NAME_VARIANTS.get(base_symbol, [])
    variants_low = [v.lower() for v in variants]
    return [
        a for a in items
        if any(v in a["title"].lower() for v in variants_low)
    ]


def _normalize_news_articles(resp):
    items = []
    seen = set()
    for a in (resp or {}).get("articles", []):
        title = (a.get("title") or "").strip()
        if not title or title == "[Removed]" or title.lower() in seen:
            continue
        seen.add(title.lower())
        items.append({
            "title": title,
            "description": (a.get("description") or "").strip(),
            "source": (a.get("source") or {}).get("name", ""),
            "url": a.get("url"),
            "published": (a.get("publishedAt") or "")[:10],
        })
    return items


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_news_cached(base_symbol):
    # Error pe exception raise hota hai, taaki failed result cache na ho
    client = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))

    name_variants = [base_symbol] + _TICKER_NAME_VARIANTS.get(base_symbol, [])
    name_query = " OR ".join(f'"{v}"' for v in name_variants)

    company_resp = client.get_everything(
        # qInTitle: company ka naam HEADLINE me hona chahiye (body/description me nahi) —
        # isse "reliance" jaisa common word body me use hone se false-match nahi hoga.
        qintitle=f"({name_query}) AND ({_NEWS_FINANCE_TERMS})",
        language="en", sort_by="publishedAt", page_size=20
    )
    if company_resp.get("status") != "ok":
        raise RuntimeError(company_resp.get("message", "NewsAPI error"))

    market_articles = []
    try:
        market_resp = client.get_everything(
            q="(Sensex OR Nifty) AND (market OR stocks)",
            language="en", sort_by="publishedAt", page_size=10
        )
        if market_resp.get("status") == "ok":
            market_articles = _normalize_news_articles(market_resp)
    except Exception:
        market_articles = []

    company_items = _relevance_filter(_normalize_news_articles(company_resp), base_symbol)

    return {"company": company_items, "market": market_articles}


def fetch_news_bundle(stock_symbol):
    base = stock_symbol.replace(".NS", "").replace(".BO", "").upper()
    if not os.getenv("NEWS_API_KEY"):
        return {"company": [], "market": [], "error": "NEWS_API_KEY nahi mili (.env check karo)"}
    try:
        data = _fetch_news_cached(base)
        return {"company": data["company"], "market": data["market"], "error": None}
    except Exception as e:
        return {"company": [], "market": [], "error": str(e)}


@st.cache_data(ttl=1800, show_spinner=False)
def generate_news_ai_summary(base_symbol, headlines_block, user_level):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY nahi mili")

    client = Groq(api_key=api_key)

    prompt = f"""
Tu Indian stock market ka news analyst hai. {user_level} level ke user ko simple Hinglish me samjhao.

Stock: {base_symbol}

Neeche recent headlines hain (sentiment aur detected events ke saath).
SIRF in headlines ke basis pe jawab do. Koi naya fact, number ya date invent mat karo.
Agar headlines kam ya unclear hain to seedha bolo.

{headlines_block}

Format:
1. Key Takeaway (2 lines)
2. Important Events (bullets, sirf jo headlines me hain)
3. Investor ke liye kya dhyaan rakhna hai (2-3 bullets)

Ye financial advice nahi hai. 200 words max.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────
# TAB 1: SINGLE STOCK
# ─────────────────────────────────────────
with tab1:
    st.markdown("""
<div class="bf-section">
    <div class="bf-section-line"></div>
    <div class="bf-section-title">Market Research</div>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([3.2, 1.4, 1.4])

    with c1:
        symbol = st.text_input(
            "Stock",
            value=default_stock,
            placeholder="Search NSE stock...",
            label_visibility="collapsed",
            key="single"
        )

    with c2:
        user_type = st.selectbox(
            "Investor profile",
            ["College Student", "Beginner", "Experienced"],
            label_visibility="collapsed"
        )

    with c3:
        period = st.selectbox(
            "Period",
            ["3mo", "6mo", "1y"],
            index=1,
            label_visibility="collapsed"
        )

    st.markdown("")

    if st.button(
        "Analyze Stock",
        key="single_btn",
        type="primary",
        use_container_width="content"
    ):

            if not symbol:
                st.warning("⚠️ Stock symbol daalo!")
            else:
                with st.spinner("📡 Data fetch ho raha hai..."):
                    try:
                        df, info, sym = get_data(symbol, period)
                        if df.empty:
                            st.error("❌ Data nahi mila! Try: RELIANCE, TCS, HDFCBANK")
                            st.stop()
                        st.success(f"✅ {sym} — {len(df)} days data!")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        st.stop()

                df = calc_indicators(df)
                latest = df.iloc[-1]
                prev = df.iloc[-2]
                # ─────────────────────────────────────────
# FUNDAMENTAL METRICS
# ─────────────────────────────────────────

                try:
                    ticker = yf.Ticker(sym)

                    eps = ticker.info.get("trailingEps")
                    pe_ratio = ticker.info.get("trailingPE")
                    pb_ratio = ticker.info.get("priceToBook")
                    roe = ticker.info.get("returnOnEquity")

                except Exception:
                    eps = None
                    pe_ratio = None
                    pb_ratio = None
                    roe = None

                # ─────────────────────────────────────────
# ROCE CALCULATION
# ─────────────────────────────────────────

                roce = None

                try:
                    financials = ticker.financials
                    bs = ticker.balance_sheet

                    if financials is not None and not financials.empty:
                        if bs is not None and not bs.empty:

                            # EBIT / Operating Income
                            ebit = None

                            for row in ["EBIT", "Operating Income"]:
                                if row in financials.index:
                                    ebit = financials.loc[row].iloc[0]
                                    break

                            # Capital Employed = Total Assets - Current Liabilities
                            total_assets = None
                            current_liabilities = None

                            if "Total Assets" in bs.index:
                                total_assets = bs.loc["Total Assets"].iloc[0]

                            if "Current Liabilities" in bs.index:
                                current_liabilities = bs.loc["Current Liabilities"].iloc[0]

                            if (
                                ebit is not None
                                and total_assets is not None
                                and current_liabilities is not None
                            ):
                                capital_employed = total_assets - current_liabilities

                                if capital_employed != 0:
                                    roce = (ebit / capital_employed) * 100

                except Exception:
                    roce = None

                # ─────────────────────────────────────────
# FUNDAMENTAL ANALYSIS
# ─────────────────────────────────────────

                st.markdown("---")

                st.markdown("""
                <div class="bf-section">
                    <div class="bf-section-line"></div>
                    <div class="bf-section-title">Fundamental Analysis</div>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("📊 Balance Sheet")

                balance_sheet = get_balance_sheet(sym)

                total_assets_cr = None
                total_debt_cr = None
                cash_cr = None
                equity_cr = None 

                if balance_sheet is not None and not balance_sheet.empty:

                    # Latest 4 years
                    bs = balance_sheet.iloc[:, :4].copy()

                    # Important rows
                    row_aliases = {
                        "Total Assets": [
                            "Total Assets"
                        ],

                        "Current Assets": [
                            "Current Assets"
                        ],

                        "Cash": [
                            "Cash Cash Equivalents And Short Term Investments",
                            "Cash And Cash Equivalents",
                            "Cash Financial"
                        ],

                        "Inventory": [
                            "Inventory"
                        ],

                        "Current Liabilities": [
                            "Current Liabilities"
                        ],

                        "Total Liabilities": [
                            "Total Liabilities Net Minority Interest",
                            "Total Liabilities"
                        ],

                        "Total Debt": [
                            "Total Debt"
                        ],

                        "Shareholders Equity": [
                            "Stockholders Equity",
                            "Total Equity Gross Minority Interest",
                            "Common Stock Equity"
                        ]
                    }

                    selected_rows = {}

                    for display_name, aliases in row_aliases.items():

                        for alias in aliases:

                            if alias in bs.index:
                                selected_rows[display_name] = bs.loc[alias]
                                break

                    if selected_rows:

                        bs_display = pd.DataFrame(selected_rows).T

                        # Convert raw INR → ₹ Crore
                        bs_display = bs_display / 1e7

                        # Format column names
                        bs_display.columns = [
                            str(col.year) if hasattr(col, "year") else str(col)
                            for col in bs_display.columns
                        ]

                        st.dataframe(
                            bs_display.round(2),
                            use_container_width=True
                        )

                    else:

                        st.warning(
                            "Balance Sheet ke required fields available nahi hain."
                        )

                        # ─────────────────────────────────────────
# FINANCIAL HEALTH METRICS
# ─────────────────────────────────────────

                if balance_sheet is not None and not balance_sheet.empty:

                    def get_bs_value(names):

                        for name in names:

                            if name in balance_sheet.index:

                                value = balance_sheet.loc[name].iloc[0]

                                if pd.notna(value):
                                    return float(value)

                        return None


                    total_assets = get_bs_value([
                        "Total Assets"
                    ])

                    total_debt = get_bs_value([
                        "Total Debt"
                    ])

                    cash = get_bs_value([
                        "Cash Cash Equivalents And Short Term Investments",
                        "Cash And Cash Equivalents",
                        "Cash Financial"
                    ])

                    equity = get_bs_value([
                        "Stockholders Equity",
                        "Total Equity Gross Minority Interest",
                        "Common Stock Equity"
                    ])


                    # Convert to Crores

                    if total_assets is not None:
                        total_assets_cr = total_assets / 1e7
                    else:
                        total_assets_cr = None

                    if total_debt is not None:
                        total_debt_cr = total_debt / 1e7
                    else:
                        total_debt_cr = None

                    if cash is not None:
                        cash_cr = cash / 1e7
                    else:
                        cash_cr = None

                    if equity is not None:
                        equity_cr = equity / 1e7
                    else:
                        equity_cr = None


                    # Cards

                    c1, c2, c3, c4 = st.columns(4)

                    with c1:

                        if total_assets_cr is not None:
                            st.metric(
                                "Total Assets",
                                f"₹{total_assets_cr:,.0f} Cr"
                            )
                        else:
                            st.metric("Total Assets", "N/A")


                    with c2:

                        if total_debt_cr is not None:
                            st.metric(
                                "Total Debt",
                                f"₹{total_debt_cr:,.0f} Cr"
                            )
                        else:
                            st.metric("Total Debt", "N/A")


                    with c3:

                        if cash_cr is not None:
                            st.metric(
                                "Cash",
                                f"₹{cash_cr:,.0f} Cr"
                            )
                        else:
                            st.metric("Cash", "N/A")


                    with c4:

                        if equity_cr is not None:
                            st.metric(
                                "Shareholders Equity",
                                f"₹{equity_cr:,.0f} Cr"
                            )
                        else:
                            st.metric("Shareholders Equity", "N/A")

                            # ─────────────────────────────────────────
# VALUATION & PROFITABILITY METRICS
# ─────────────────────────────────────────

                st.markdown("### 📊 Fundamental Metrics")

                m1, m2, m3, m4, m5 = st.columns(5)

                with m1:
                    if eps is not None:
                        st.metric("EPS", f"₹{eps:.2f}")
                    else:
                        st.metric("EPS", "N/A")

                with m2:
                    if pe_ratio is not None:
                        st.metric("P/E", f"{pe_ratio:.2f}x")
                    else:
                        st.metric("P/E", "N/A")

                with m3:
                    if pb_ratio is not None:
                        st.metric("P/B", f"{pb_ratio:.2f}x")
                    else:
                        st.metric("P/B", "N/A")

                with m4:
                    if roe is not None:
                        st.metric("ROE", f"{roe * 100:.2f}%")
                    else:
                        st.metric("ROE", "N/A")

                with m5:
                    if roce is not None:
                        st.metric("ROCE", f"{roce:.2f}%")
                    else:
                        st.metric("ROCE", "N/A")
                    

                            # ─────────────────────────────────────────
# DEBT ANALYSIS
# ─────────────────────────────────────────

                st.markdown("### 🏦 Debt Analysis")

                if total_debt is not None and equity is not None and equity != 0:

                    debt_equity = total_debt / equity

                    st.metric(
                        "Debt / Equity",
                        f"{debt_equity:.2f}"
                    )

                    if debt_equity < 0.5:

                        st.info(
                            "Debt relatively low hai compared with equity."
                        )

                    elif debt_equity < 1:

                        st.warning(
                            "Debt aur equity moderate range me hain."
                        )

                    else:

                        st.error(
                            "Debt equity ke comparison me high hai."
                        )

                else:

                    st.info(
                        "Debt/Equity calculate karne ke liye sufficient data available nahi hai."
                    )

                st.warning(
                    f"⚠️ {sym} ke liye Balance Sheet data available nahi hai."
                )

                # ─────────────────────────────────────────
# INCOME STATEMENT
# ─────────────────────────────────────────

                st.markdown("---")
                st.subheader("📈 Income Statement")

                income_stmt = None

                try:
                    income_stmt = ticker.financials
                except Exception:
                    income_stmt = None

                if income_stmt is not None and not income_stmt.empty:

                    is_df = income_stmt.iloc[:, :4].copy()

                    is_row_aliases = {
                        "Revenue": [
                            "Total Revenue"
                        ],
                        "EBITDA": [
                            "EBITDA"
                        ],
                        "EBIT": [
                            "EBIT"
                        ],
                        "Operating Income": [
                            "Operating Income"
                        ],
                        "Net Profit": [
                            "Net Income",
                            "Net Income Common Stockholders"
                        ]
                    }

                    is_selected_rows = {}

                    for display_name, aliases in is_row_aliases.items():
                        for alias in aliases:
                            if alias in is_df.index:
                                is_selected_rows[display_name] = is_df.loc[alias]
                                break

                    if is_selected_rows:

                        is_display = pd.DataFrame(is_selected_rows).T

                        # Raw INR → ₹ Crore
                        is_display = is_display / 1e7

                        is_display.columns = [
                            str(col.year) if hasattr(col, "year") else str(col)
                            for col in is_display.columns
                        ]

                        st.dataframe(
                            is_display.round(2),
                            use_container_width=True
                        )

                    else:

                        st.warning(
                            "Income Statement ke required fields available nahi hain."
                        )

                    # ─────────────────────────────────────────
# INCOME STATEMENT METRICS (latest vs prior year)
# ─────────────────────────────────────────

                    def get_is_value(names, col_idx=0):

                        for name in names:

                            if name in income_stmt.index:

                                try:
                                    value = income_stmt.loc[name].iloc[col_idx]
                                except Exception:
                                    value = None

                                if value is not None and pd.notna(value):
                                    return float(value)

                        return None


                    revenue_latest = get_is_value(["Total Revenue"], 0)
                    revenue_prev = get_is_value(["Total Revenue"], 1)

                    ebitda_latest = get_is_value(["EBITDA"], 0)
                    ebit_latest = get_is_value(["EBIT"], 0)
                    op_income_latest = get_is_value(["Operating Income"], 0)

                    net_profit_latest = get_is_value(
                        ["Net Income", "Net Income Common Stockholders"], 0
                    )
                    net_profit_prev = get_is_value(
                        ["Net Income", "Net Income Common Stockholders"], 1
                    )

                    eps_latest = get_is_value(["Diluted EPS", "Basic EPS"], 0)
                    eps_prev = get_is_value(["Diluted EPS", "Basic EPS"], 1)


                    # Revenue Growth (YoY)
                    revenue_growth = None
                    if revenue_latest is not None and revenue_prev not in (None, 0):
                        revenue_growth = ((revenue_latest - revenue_prev) / abs(revenue_prev)) * 100

                    # Net Profit Growth (YoY)
                    net_profit_growth = None
                    if net_profit_latest is not None and net_profit_prev not in (None, 0):
                        net_profit_growth = ((net_profit_latest - net_profit_prev) / abs(net_profit_prev)) * 100

                    # EPS Growth (YoY)
                    eps_growth = None
                    if eps_latest is not None and eps_prev not in (None, 0):
                        eps_growth = ((eps_latest - eps_prev) / abs(eps_prev)) * 100

                    # Margins
                    ebitda_margin = None
                    if ebitda_latest is not None and revenue_latest not in (None, 0):
                        ebitda_margin = (ebitda_latest / revenue_latest) * 100

                    operating_margin = None
                    if op_income_latest is not None and revenue_latest not in (None, 0):
                        operating_margin = (op_income_latest / revenue_latest) * 100

                    net_profit_margin = None
                    if net_profit_latest is not None and revenue_latest not in (None, 0):
                        net_profit_margin = (net_profit_latest / revenue_latest) * 100


                    st.markdown("### 📊 Income Statement Metrics")

                    i1, i2, i3, i4 = st.columns(4)

                    with i1:
                        if revenue_latest is not None:
                            st.metric("Revenue", f"₹{revenue_latest/1e7:,.0f} Cr")
                        else:
                            st.metric("Revenue", "N/A")

                    with i2:
                        if revenue_growth is not None:
                            st.metric("Revenue Growth (YoY)", f"{revenue_growth:.2f}%")
                        else:
                            st.metric("Revenue Growth (YoY)", "N/A")

                    with i3:
                        if net_profit_latest is not None:
                            st.metric("Net Profit", f"₹{net_profit_latest/1e7:,.0f} Cr")
                        else:
                            st.metric("Net Profit", "N/A")

                    with i4:
                        if net_profit_growth is not None:
                            st.metric("Net Profit Growth (YoY)", f"{net_profit_growth:.2f}%")
                        else:
                            st.metric("Net Profit Growth (YoY)", "N/A")

                    i5, i6, i7, i8 = st.columns(4)

                    with i5:
                        if ebitda_margin is not None:
                            st.metric("EBITDA Margin", f"{ebitda_margin:.2f}%")
                        else:
                            st.metric("EBITDA Margin", "N/A")

                    with i6:
                        if operating_margin is not None:
                            st.metric("Operating Margin", f"{operating_margin:.2f}%")
                        else:
                            st.metric("Operating Margin", "N/A")

                    with i7:
                        if net_profit_margin is not None:
                            st.metric("Net Profit Margin", f"{net_profit_margin:.2f}%")
                        else:
                            st.metric("Net Profit Margin", "N/A")

                    with i8:
                        if eps_growth is not None:
                            st.metric("EPS Growth (YoY)", f"{eps_growth:.2f}%")
                        else:
                            st.metric("EPS Growth (YoY)", "N/A")

                    # ─────────────────────────────────────────
# AI INCOME STATEMENT COMMENTARY
# ─────────────────────────────────────────

                    is_notes = []

                    if revenue_growth is not None:
                        if revenue_growth > 15:
                            is_notes.append("✅ Revenue growth strong hai (>15% YoY).")
                        elif revenue_growth > 0:
                            is_notes.append("🟡 Revenue growth positive hai but moderate.")
                        else:
                            is_notes.append("🔴 Revenue de-grow ho raha hai YoY.")

                    if net_profit_margin is not None:
                        if net_profit_margin > 15:
                            is_notes.append("✅ Net Profit Margin healthy hai.")
                        elif net_profit_margin > 5:
                            is_notes.append("🟡 Net Profit Margin average range me hai.")
                        else:
                            is_notes.append("🔴 Net Profit Margin weak hai — cost structure check karo.")

                    if net_profit_growth is not None and revenue_growth is not None:
                        if net_profit_growth > revenue_growth:
                            is_notes.append("✅ Profit revenue se fast grow kar raha hai — operating leverage positive hai.")
                        elif net_profit_growth < 0 and revenue_growth > 0:
                            is_notes.append("⚠️ Revenue grow ho raha hai par profit gir raha hai — margins pressure me ho sakte hain.")

                    if is_notes:
                        for note in is_notes:
                            st.info(note)
                    else:
                        st.info("Income Statement analysis ke liye sufficient data available nahi hai.")

                else:
                    st.warning(
                        f"⚠️ {sym} ke liye Income Statement data available nahi hai."
                    )

                # ─────────────────────────────────────────
# CASH FLOW ANALYSIS
# ─────────────────────────────────────────

                st.markdown("---")
                st.subheader("💰 Cash Flow Analysis")

                cashflow_stmt = None

                try:
                    cashflow_stmt = ticker.cashflow
                except Exception:
                    cashflow_stmt = None

                if cashflow_stmt is not None and not cashflow_stmt.empty:

                    cf_df = cashflow_stmt.iloc[:, :4].copy()

                    cf_row_aliases = {
                        "Operating Cash Flow": [
                            "Operating Cash Flow",
                            "Cash Flow From Continuing Operating Activities"
                        ],
                        "Investing Cash Flow": [
                            "Investing Cash Flow",
                            "Cash Flow From Continuing Investing Activities"
                        ],
                        "Financing Cash Flow": [
                            "Financing Cash Flow",
                            "Cash Flow From Continuing Financing Activities"
                        ],
                        "Capital Expenditure": [
                            "Capital Expenditure"
                        ],
                        "Free Cash Flow": [
                            "Free Cash Flow"
                        ],
                        "Net Income": [
                            "Net Income",
                            "Net Income From Continuing Operations"
                        ]
                    }

                    cf_selected_rows = {}

                    for display_name, aliases in cf_row_aliases.items():
                        for alias in aliases:
                            if alias in cf_df.index:
                                cf_selected_rows[display_name] = cf_df.loc[alias]
                                break

                    if cf_selected_rows:

                        cf_display = pd.DataFrame(cf_selected_rows).T

                        # Raw INR → ₹ Crore
                        cf_display = cf_display / 1e7

                        cf_display.columns = [
                            str(col.year) if hasattr(col, "year") else str(col)
                            for col in cf_display.columns
                        ]

                        st.dataframe(
                            cf_display.round(2),
                            use_container_width=True
                        )

                    else:

                        st.warning(
                            "Cash Flow ke required fields available nahi hain."
                        )

                    # ─────────────────────────────────────────
# CASH FLOW METRICS (latest + FCF trend)
# ─────────────────────────────────────────

                    def get_cf_value(names, col_idx=0):

                        for name in names:

                            if name in cashflow_stmt.index:

                                try:
                                    value = cashflow_stmt.loc[name].iloc[col_idx]
                                except Exception:
                                    value = None

                                if value is not None and pd.notna(value):
                                    return float(value)

                        return None


                    ocf_latest = get_cf_value(
                        ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"], 0
                    )
                    icf_latest = get_cf_value(
                        ["Investing Cash Flow", "Cash Flow From Continuing Investing Activities"], 0
                    )
                    fcf_activity_latest = get_cf_value(
                        ["Financing Cash Flow", "Cash Flow From Continuing Financing Activities"], 0
                    )
                    capex_latest = get_cf_value(["Capital Expenditure"], 0)

                    fcf_latest = get_cf_value(["Free Cash Flow"], 0)

                    # Agar Free Cash Flow row directly available nahi hai to manually calculate karo
                    if fcf_latest is None and ocf_latest is not None and capex_latest is not None:
                        fcf_latest = ocf_latest + capex_latest  # capex yfinance me negative hota hai

                    net_income_cf_latest = get_cf_value(
                        ["Net Income", "Net Income From Continuing Operations"], 0
                    )

                    st.markdown("### 📊 Cash Flow Metrics")

                    cf1, cf2, cf3, cf4 = st.columns(4)

                    with cf1:
                        if ocf_latest is not None:
                            st.metric("Operating Cash Flow", f"₹{ocf_latest/1e7:,.0f} Cr")
                        else:
                            st.metric("Operating Cash Flow", "N/A")

                    with cf2:
                        if icf_latest is not None:
                            st.metric("Investing Cash Flow", f"₹{icf_latest/1e7:,.0f} Cr")
                        else:
                            st.metric("Investing Cash Flow", "N/A")

                    with cf3:
                        if fcf_activity_latest is not None:
                            st.metric("Financing Cash Flow", f"₹{fcf_activity_latest/1e7:,.0f} Cr")
                        else:
                            st.metric("Financing Cash Flow", "N/A")

                    with cf4:
                        if fcf_latest is not None:
                            st.metric("Free Cash Flow", f"₹{fcf_latest/1e7:,.0f} Cr")
                        else:
                            st.metric("Free Cash Flow", "N/A")

                    # ─────────────────────────────────────────
# CFO vs NET PROFIT — EARNINGS QUALITY
# ─────────────────────────────────────────

                    st.markdown("### 🧪 Earnings Quality — CFO vs Net Profit")

                    if ocf_latest is not None and net_income_cf_latest not in (None, 0):

                        cfo_np_ratio = ocf_latest / net_income_cf_latest

                        eq1, eq2 = st.columns(2)

                        with eq1:
                            st.metric("Net Profit", f"₹{net_income_cf_latest/1e7:,.0f} Cr")

                        with eq2:
                            st.metric("CFO / Net Profit Ratio", f"{cfo_np_ratio:.2f}x")

                        if cfo_np_ratio >= 1:
                            st.success(
                                "✅ Earnings quality achhi hai — operating cash flow, net profit se zyada ya barabar hai."
                            )
                        elif cfo_np_ratio >= 0.7:
                            st.warning(
                                "🟡 Earnings quality moderate hai — CFO thoda net profit se kam hai, monitor karo."
                            )
                        else:
                            st.error(
                                "🔴 Earnings quality weak hai — net profit accounting-heavy ho sakta hai, actual cash generation kam hai."
                            )

                    else:
                        st.info(
                            "CFO vs Net Profit comparison ke liye sufficient data available nahi hai."
                        )

                    # ─────────────────────────────────────────
# FCF TREND (multi-year)
# ─────────────────────────────────────────

                    st.markdown("### 📈 FCF Trend")

                    fcf_row = None

                    for alias in ["Free Cash Flow"]:
                        if alias in cf_df.index:
                            fcf_row = cf_df.loc[alias]
                            break

                    if fcf_row is None:
                        ocf_row = None
                        capex_row = None

                        for alias in ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]:
                            if alias in cf_df.index:
                                ocf_row = cf_df.loc[alias]
                                break

                        for alias in ["Capital Expenditure"]:
                            if alias in cf_df.index:
                                capex_row = cf_df.loc[alias]
                                break

                        if ocf_row is not None and capex_row is not None:
                            fcf_row = ocf_row + capex_row

                    if fcf_row is not None:

                        fcf_trend_df = (fcf_row / 1e7).round(2)

                        fcf_trend_df.index = [
                            str(col.year) if hasattr(col, "year") else str(col)
                            for col in fcf_trend_df.index
                        ]

                        fig_fcf = px.bar(
                            x=fcf_trend_df.index,
                            y=fcf_trend_df.values,
                            labels={"x": "Year", "y": "Free Cash Flow (₹ Cr)"},
                            title="Free Cash Flow Trend"
                        )

                        st.plotly_chart(
                            fig_fcf,
                            use_container_width="stretch",
                            key="fcf_trend_chart"
                        )

                        if len(fcf_trend_df.values) >= 2:
                            if fcf_trend_df.values[0] > fcf_trend_df.values[1]:
                                st.info("✅ FCF improve ho raha hai year-on-year.")
                            else:
                                st.warning("⚠️ FCF decline ho raha hai year-on-year — CapEx ya working capital check karo.")

                    else:
                        st.info(
                            "FCF Trend ke liye sufficient historical data available nahi hai."
                        )

                else:
                    st.warning(
                        f"⚠️ {sym} ke liye Cash Flow data available nahi hai."
                    )

                # ─────────────────────────────────────────
# DEBT TREND ANALYSIS
# ─────────────────────────────────────────

                st.markdown("---")
                st.subheader("📉 Debt Trend Analysis")

                if balance_sheet is not None and not balance_sheet.empty:

                    debt_row = None

                    for alias in ["Total Debt"]:
                        if alias in balance_sheet.index:
                            debt_row = balance_sheet.loc[alias]
                            break

                    if debt_row is not None:

                        debt_trend = (debt_row.iloc[:4] / 1e7).round(2)

                        debt_trend.index = [
                            str(col.year) if hasattr(col, "year") else str(col)
                            for col in debt_trend.index
                        ]

                        # Chronological order (oldest → newest) chart ke liye
                        debt_trend_sorted = debt_trend[::-1]

                        fig_debt = px.bar(
                            x=debt_trend_sorted.index,
                            y=debt_trend_sorted.values,
                            labels={"x": "Year", "y": "Total Debt (₹ Cr)"},
                            title="Total Debt Trend"
                        )

                        st.plotly_chart(
                            fig_debt,
                            use_container_width="stretch",
                            key="debt_trend_chart"
                        )

                        # ─────────────────────────────────────────
# DEBT GROWTH / REDUCTION (YoY)
# ─────────────────────────────────────────

                        debt_latest = debt_trend.iloc[0]
                        debt_prev = debt_trend.iloc[1] if len(debt_trend) > 1 else None

                        debt_growth = None

                        if debt_prev not in (None, 0):
                            debt_growth = ((debt_latest - debt_prev) / abs(debt_prev)) * 100

                        d1, d2, d3 = st.columns(3)

                        with d1:
                            st.metric("Total Debt (Latest)", f"₹{debt_latest:,.0f} Cr")

                        with d2:
                            if debt_growth is not None:
                                st.metric("Debt Growth (YoY)", f"{debt_growth:.2f}%")
                            else:
                                st.metric("Debt Growth (YoY)", "N/A")

                        with d3:
                            if debt_growth is not None:
                                if debt_growth < 0:
                                    st.metric("Debt Trend", "🟢 Reducing")
                                elif debt_growth < 10:
                                    st.metric("Debt Trend", "🟡 Stable")
                                else:
                                    st.metric("Debt Trend", "🔴 Rising")
                            else:
                                st.metric("Debt Trend", "N/A")

                        if debt_growth is not None:
                            if debt_growth < 0:
                                st.success(
                                    "✅ Company apna debt reduce kar rahi hai — deleveraging trend positive hai."
                                )
                            elif debt_growth < 10:
                                st.info(
                                    "🟡 Debt roughly stable hai, controlled range me."
                                )
                            else:
                                st.warning(
                                    "⚠️ Debt fast grow ho raha hai — leverage risk badh sakta hai, interest cost aur cash flow ke against check karo."
                                )

                    else:
                        st.info(
                            "Debt Trend ke liye Total Debt data available nahi hai."
                        )

                    # ─────────────────────────────────────────
# INTEREST COVERAGE RATIO
# ─────────────────────────────────────────

                    st.markdown("### 🛡️ Interest Coverage Ratio")

                    interest_expense = None
                    ebit_for_icr = None

                    if income_stmt is not None and not income_stmt.empty:

                        for alias in ["Interest Expense", "Interest Expense Non Operating"]:
                            if alias in income_stmt.index:
                                try:
                                    value = income_stmt.loc[alias].iloc[0]
                                    if pd.notna(value):
                                        interest_expense = float(value)
                                        break
                                except Exception:
                                    pass

                        for alias in ["EBIT"]:
                            if alias in income_stmt.index:
                                try:
                                    value = income_stmt.loc[alias].iloc[0]
                                    if pd.notna(value):
                                        ebit_for_icr = float(value)
                                        break
                                except Exception:
                                    pass

                    if ebit_for_icr is not None and interest_expense not in (None, 0):

                        interest_coverage = ebit_for_icr / abs(interest_expense)

                        st.metric("Interest Coverage Ratio", f"{interest_coverage:.2f}x")

                        if interest_coverage > 5:
                            st.success(
                                "✅ Interest obligations comfortably cover ho rahe hain — low default risk."
                            )
                        elif interest_coverage > 2:
                            st.warning(
                                "🟡 Interest coverage moderate hai — monitor karte raho."
                            )
                        else:
                            st.error(
                                "🔴 Interest coverage weak hai — debt servicing risk high ho sakta hai."
                            )

                    else:
                        st.info(
                            "Interest Coverage Ratio calculate karne ke liye sufficient data available nahi hai."
                        )

                else:
                    st.warning(
                        f"⚠️ {sym} ke liye Debt Trend data available nahi hai."
                    )

                # ─────────────────────────────────────────
# HISTORICAL FINANCIAL TRENDS
# ─────────────────────────────────────────

                st.markdown("---")
                st.subheader("📜 Historical Financial Trends")

                st.caption(
                    "Note: yfinance annual data typically 3-4 saal tak available hota hai (free tier). Jitna data mila hai, utna hi trend/CAGR yahan dikhaya ja raha hai."
                )

                def get_row_series(df, aliases):
                    if df is None or df.empty:
                        return None
                    for alias in aliases:
                        if alias in df.index:
                            return df.loc[alias]
                    return None

                revenue_series = get_row_series(income_stmt, ["Total Revenue"])
                net_profit_series = get_row_series(
                    income_stmt, ["Net Income", "Net Income Common Stockholders"]
                )
                eps_series = get_row_series(income_stmt, ["Diluted EPS", "Basic EPS"])
                ebit_series = get_row_series(income_stmt, ["EBIT"])

                total_debt_series = get_row_series(balance_sheet, ["Total Debt"])
                equity_series = get_row_series(
                    balance_sheet,
                    ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"]
                )
                total_assets_series = get_row_series(balance_sheet, ["Total Assets"])
                current_liab_series = get_row_series(balance_sheet, ["Current Liabilities"])

                fcf_series = get_row_series(cashflow_stmt, ["Free Cash Flow"])

                if fcf_series is None:
                    ocf_series = get_row_series(
                        cashflow_stmt,
                        ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]
                    )
                    capex_series = get_row_series(cashflow_stmt, ["Capital Expenditure"])

                    if ocf_series is not None and capex_series is not None:
                        fcf_series = ocf_series + capex_series

                # ─────────────────────────────────────────
# ROE / ROCE PER YEAR (aligned by column date)
# ─────────────────────────────────────────

                roe_series = None

                if net_profit_series is not None and equity_series is not None:

                    aligned_years = [
                        c for c in net_profit_series.index if c in equity_series.index
                    ]

                    roe_values = {}

                    for c in aligned_years:
                        np_val = net_profit_series[c]
                        eq_val = equity_series[c]

                        if pd.notna(np_val) and pd.notna(eq_val) and eq_val != 0:
                            roe_values[c] = (np_val / eq_val) * 100

                    if roe_values:
                        roe_series = pd.Series(roe_values)

                roce_series = None

                if (
                    ebit_series is not None
                    and total_assets_series is not None
                    and current_liab_series is not None
                ):

                    aligned_years = [
                        c for c in ebit_series.index
                        if c in total_assets_series.index and c in current_liab_series.index
                    ]

                    roce_values = {}

                    for c in aligned_years:
                        ebit_val = ebit_series[c]
                        ta_val = total_assets_series[c]
                        cl_val = current_liab_series[c]

                        if pd.notna(ebit_val) and pd.notna(ta_val) and pd.notna(cl_val):
                            cap_employed = ta_val - cl_val

                            if cap_employed != 0:
                                roce_values[c] = (ebit_val / cap_employed) * 100

                    if roce_values:
                        roce_series = pd.Series(roce_values)

                # ─────────────────────────────────────────
# FORMAT SERIES FOR CHART/TABLE
# ─────────────────────────────────────────

                def format_year_series(series, divide_cr=False):

                    if series is None:
                        return None

                    s = series.copy().dropna()

                    if s.empty:
                        return None

                    if divide_cr:
                        s = s / 1e7

                    s = s.sort_index()

                    s.index = [
                        str(c.year) if hasattr(c, "year") else str(c)
                        for c in s.index
                    ]

                    return s

                trend_data = {
                    "Revenue (₹ Cr)": format_year_series(revenue_series, divide_cr=True),
                    "Net Profit (₹ Cr)": format_year_series(net_profit_series, divide_cr=True),
                    "EPS (₹)": format_year_series(eps_series, divide_cr=False),
                    "ROE (%)": format_year_series(roe_series, divide_cr=False),
                    "ROCE (%)": format_year_series(roce_series, divide_cr=False),
                    "Total Debt (₹ Cr)": format_year_series(total_debt_series, divide_cr=True),
                    "FCF (₹ Cr)": format_year_series(fcf_series, divide_cr=True),
                }

                available_metrics = {
                    k: v for k, v in trend_data.items()
                    if v is not None and len(v) >= 2
                }

                if available_metrics:

                    metric_choice = st.selectbox(
                        "Metric select karo",
                        list(available_metrics.keys()),
                        key="hist_trend_metric"
                    )

                    chosen_series = available_metrics[metric_choice]

                    fig_trend = px.line(
                        x=chosen_series.index,
                        y=chosen_series.values,
                        markers=True,
                        labels={"x": "Year", "y": metric_choice},
                        title=f"{metric_choice} — Historical Trend"
                    )

                    st.plotly_chart(
                        fig_trend,
                        use_container_width="stretch",
                        key="historical_trend_chart"
                    )

                    # CAGR (oldest vs newest available data point)
                    start_value = chosen_series.iloc[0]
                    end_value = chosen_series.iloc[-1]
                    n_years = len(chosen_series) - 1

                    cagr = None

                    if n_years > 0 and start_value not in (None, 0) and start_value > 0 and end_value > 0:
                        cagr = ((end_value / start_value) ** (1 / n_years) - 1) * 100

                    latest_yoy = None

                    if len(chosen_series) >= 2:
                        prev_value = chosen_series.iloc[-2]

                        if prev_value not in (None, 0):
                            latest_yoy = ((end_value - prev_value) / abs(prev_value)) * 100

                    t1, t2, t3 = st.columns(3)

                    with t1:
                        st.metric(f"{metric_choice} — Latest", f"{end_value:,.2f}")

                    with t2:
                        if cagr is not None:
                            st.metric(f"CAGR ({n_years}Y)", f"{cagr:.2f}%")
                        else:
                            st.metric(f"CAGR ({n_years}Y)", "N/A")

                    with t3:
                        if latest_yoy is not None:
                            st.metric("Latest YoY Growth", f"{latest_yoy:.2f}%")
                        else:
                            st.metric("Latest YoY Growth", "N/A")

                    # ─────────────────────────────────────────
# ALL METRICS — CAGR SUMMARY TABLE
# ─────────────────────────────────────────

                    st.markdown("### 📋 All Metrics — CAGR Summary")

                    summary_rows = []

                    for name, series in available_metrics.items():

                        s_val = series.iloc[0]
                        e_val = series.iloc[-1]
                        yrs = len(series) - 1

                        row_cagr = None

                        if yrs > 0 and s_val not in (None, 0) and s_val > 0 and e_val > 0:
                            row_cagr = ((e_val / s_val) ** (1 / yrs) - 1) * 100

                        summary_rows.append({
                            "Metric": name,
                            "Oldest": round(s_val, 2),
                            "Latest": round(e_val, 2),
                            "CAGR": f"{row_cagr:.2f}%" if row_cagr is not None else "N/A"
                        })

                    summary_df = pd.DataFrame(summary_rows)

                    st.dataframe(
                        summary_df,
                        use_container_width=True
                    )

                    positive_cagr_count = sum(
                        1 for row in summary_rows
                        if row["CAGR"] != "N/A" and float(row["CAGR"].replace("%", "")) > 0
                    )

                    if positive_cagr_count >= len(summary_rows) * 0.6:
                        st.success(
                            "✅ Zyadatar fundamentals me positive long-term growth trend hai."
                        )
                    else:
                        st.warning(
                            "⚠️ Fundamentals ka growth trend mixed/weak hai — detailed analysis karo."
                        )

                else:
                    st.info(
                        "Historical Financial Trends ke liye sufficient multi-year data available nahi hai."
                    )

                # ─────────────────────────────────────────
# VALUATION ENGINE
# ─────────────────────────────────────────

                st.markdown("---")
                st.subheader("💹 Valuation Engine")

                enterprise_value = None
                ev_ebitda = None
                peg_ratio = None
                dividend_yield = None
                forward_pe = None

                try:
                    enterprise_value = ticker.info.get("enterpriseValue")
                    ev_ebitda = ticker.info.get("enterpriseToEbitda")
                    peg_ratio = ticker.info.get("pegRatio")
                    dividend_yield = ticker.info.get("dividendYield")
                    forward_pe = ticker.info.get("forwardPE")
                except Exception:
                    pass

                # Manual PEG fallback: P/E ÷ Net Profit Growth rate
                if peg_ratio is None and pe_ratio is not None:
                    try:
                        if net_profit_growth not in (None, 0) and net_profit_growth > 0:
                            peg_ratio = pe_ratio / net_profit_growth
                    except NameError:
                        peg_ratio = None

                v1, v2, v3, v4, v5 = st.columns(5)

                with v1:
                    if pe_ratio is not None:
                        st.metric("P/E (Trailing)", f"{pe_ratio:.2f}x")
                    else:
                        st.metric("P/E (Trailing)", "N/A")

                with v2:
                    if pb_ratio is not None:
                        st.metric("P/B", f"{pb_ratio:.2f}x")
                    else:
                        st.metric("P/B", "N/A")

                with v3:
                    if ev_ebitda is not None:
                        st.metric("EV/EBITDA", f"{ev_ebitda:.2f}x")
                    else:
                        st.metric("EV/EBITDA", "N/A")

                with v4:
                    if peg_ratio is not None:
                        st.metric("PEG Ratio", f"{peg_ratio:.2f}")
                    else:
                        st.metric("PEG Ratio", "N/A")

                with v5:
                    if dividend_yield is not None:

                        # yfinance kabhi fraction (0.0049) deta hai, kabhi already-percent (0.49) —
                        # dono interpretation try karke jo plausible range (0-20%) me aaye wahi use karo
                        dy_as_fraction = dividend_yield * 100
                        dy_as_percent = dividend_yield

                        if 0 <= dy_as_fraction <= 20:
                            dy_display = dy_as_fraction
                        elif 0 <= dy_as_percent <= 20:
                            dy_display = dy_as_percent
                        else:
                            dy_display = None

                        if dy_display is not None:
                            st.metric("Dividend Yield", f"{dy_display:.2f}%")
                        else:
                            st.metric("Dividend Yield", "N/A")
                    else:
                        st.metric("Dividend Yield", "N/A")
                # ─────────────────────────────────────────
# AI VALUATION VERDICT (absolute thresholds)
# ─────────────────────────────────────────

                val_notes = []

                if pe_ratio is not None:
                    if pe_ratio < 15:
                        val_notes.append("✅ P/E relatively low hai — value zone ho sakta hai (ya market ko growth ka doubt hai, dono check karo).")
                    elif pe_ratio > 40:
                        val_notes.append("⚠️ P/E high hai — market future growth expectations already price-in kar chuka hai, risk zyada hai.")

                if peg_ratio is not None:
                    if peg_ratio < 1:
                        val_notes.append("✅ PEG < 1 — growth ke hisab se stock undervalued lag raha hai.")
                    elif peg_ratio > 2:
                        val_notes.append("⚠️ PEG > 2 — growth ke comparison me valuation stretched hai.")

                if val_notes:
                    for note in val_notes:
                        st.info(note)
                else:
                    st.info("Valuation verdict ke liye sufficient data available nahi hai.")

                # ─────────────────────────────────────────
# HISTORICAL VALUATION RANGE (approx, within selected period)
# ─────────────────────────────────────────

                st.markdown("### 📊 Historical Valuation Range")

                if eps is not None and eps != 0:

                    hist_pe_series = (df["Close"] / eps).dropna()

                    if not hist_pe_series.empty:

                        pe_min = hist_pe_series.min()
                        pe_max = hist_pe_series.max()
                        pe_current = hist_pe_series.iloc[-1]
                        pe_avg = hist_pe_series.mean()

                        hp1, hp2, hp3, hp4 = st.columns(4)

                        with hp1:
                            st.metric("Period Low P/E", f"{pe_min:.2f}x")

                        with hp2:
                            st.metric("Period Avg P/E", f"{pe_avg:.2f}x")

                        with hp3:
                            st.metric("Period High P/E", f"{pe_max:.2f}x")

                        with hp4:
                            st.metric("Current P/E", f"{pe_current:.2f}x")

                        fig_pe_range = go.Figure()

                        fig_pe_range.add_trace(go.Scatter(
                            x=hist_pe_series.index,
                            y=hist_pe_series.values,
                            mode="lines",
                            name="Approx P/E"
                        ))

                        fig_pe_range.add_hline(
                            y=pe_avg,
                            line_dash="dash",
                            annotation_text="Average P/E"
                        )

                        fig_pe_range.update_layout(
                            title="P/E Trend — Selected Period (approx.)",
                            xaxis_title="Date",
                            yaxis_title="P/E"
                        )

                        st.plotly_chart(
                            fig_pe_range,
                            use_container_width="stretch",
                            key="pe_range_chart"
                        )

                        st.caption(
                            "Note: Ye approximate P/E hai — trailing EPS ko constant maan ke calculate kiya gaya hai "
                            "(quarterly EPS updates account nahi kiye gaye). Directional trend ke liye useful hai, exact historical P/E ke liye nahi."
                        )

                        if pe_max != pe_min:
                            pe_position_pct = ((pe_current - pe_min) / (pe_max - pe_min)) * 100
                        else:
                            pe_position_pct = 50

                        if pe_position_pct < 30:
                            st.success(
                                f"✅ Stock apni {period} range ke lower zone me trade kar raha hai (range ke {pe_position_pct:.0f}% pe)."
                            )
                        elif pe_position_pct < 70:
                            st.info(
                                f"🟡 Stock apni {period} range ke beech me trade kar raha hai (range ke {pe_position_pct:.0f}% pe)."
                            )
                        else:
                            st.warning(
                                f"⚠️ Stock apni {period} range ke top ke paas trade kar raha hai (range ke {pe_position_pct:.0f}% pe) — valuation stretched ho sakta hai."
                            )

                    else:
                        st.info("Historical P/E range calculate nahi ho paya.")

                else:
                    st.info("EPS available nahi hai, historical valuation range calculate nahi ho sakta.")

                # ─────────────────────────────────────────
# SECTOR PEER COMPARISON
# ─────────────────────────────────────────

                st.markdown("### 🏢 Sector Peer Comparison")

                sector_peer_map = {
                    "RELIANCE": ["ONGC", "IOC", "BPCL"],
                    "TCS": ["INFY", "WIPRO", "HCLTECH"],
                    "INFY": ["TCS", "WIPRO", "HCLTECH"],
                    "WIPRO": ["TCS", "INFY", "HCLTECH"],
                    "HCLTECH": ["TCS", "INFY", "WIPRO"],
                    "HDFCBANK": ["ICICIBANK", "KOTAKBANK", "SBIN"],
                    "ICICIBANK": ["HDFCBANK", "KOTAKBANK", "SBIN"],
                    "SBIN": ["HDFCBANK", "ICICIBANK", "KOTAKBANK"],
                    "KOTAKBANK": ["HDFCBANK", "ICICIBANK", "SBIN"],
                    "TATAMOTORS": ["MARUTI", "M&M", "EICHERMOT"],
                    "MARUTI": ["TATAMOTORS", "M&M", "EICHERMOT"],
                    "SUNPHARMA": ["DRREDDY", "CIPLA", "DIVISLAB"],
                    "DRREDDY": ["SUNPHARMA", "CIPLA", "DIVISLAB"],
                    "ITC": ["HINDUNILVR", "NESTLEIND", "BRITANNIA"],
                    "HINDUNILVR": ["ITC", "NESTLEIND", "BRITANNIA"],
                }

                base_symbol = sym.replace(".NS", "").upper()

                peer_list = sector_peer_map.get(base_symbol)

                

                def get_full_fundamentals(ticker_symbol):

                    result = {
                        "P/E": None,
                        "P/B": None,
                        "ROE (%)": None,
                        "ROCE (%)": None,
                        "Debt/Equity": None,
                        "Net Profit Margin (%)": None,
                        "Revenue Growth (%)": None,
                        "EPS Growth (%)": None,
                        "FCF (₹ Cr)": None,
                    }

                    try:
                        t = yf.Ticker(ticker_symbol)
                        info = t.info

                        result["P/E"] = info.get("trailingPE")
                        result["P/B"] = info.get("priceToBook")

                        roe_raw = info.get("returnOnEquity")
                        if roe_raw is not None:
                            result["ROE (%)"] = roe_raw * 100

                        de_raw = info.get("debtToEquity")
                        if de_raw is not None:
                            result["Debt/Equity"] = de_raw / 100

                        fin = t.financials
                        bs = t.balance_sheet
                        cf = t.cashflow

                        net_income = None
                        revenue_latest_p = None
                        revenue_prev_p = None
                        ebit_p = None
                        eps_latest_p = None
                        eps_prev_p = None

                        if fin is not None and not fin.empty:

                            for alias in ["Net Income", "Net Income Common Stockholders"]:
                                if alias in fin.index:
                                    net_income = fin.loc[alias].iloc[0]
                                    break

                            if "Total Revenue" in fin.index:
                                rev_row = fin.loc["Total Revenue"]
                                if len(rev_row) >= 1:
                                    revenue_latest_p = rev_row.iloc[0]
                                if len(rev_row) >= 2:
                                    revenue_prev_p = rev_row.iloc[1]

                            if "EBIT" in fin.index:
                                ebit_p = fin.loc["EBIT"].iloc[0]

                            for alias in ["Diluted EPS", "Basic EPS"]:
                                if alias in fin.index:
                                    eps_row = fin.loc[alias]
                                    if len(eps_row) >= 1:
                                        eps_latest_p = eps_row.iloc[0]
                                    if len(eps_row) >= 2:
                                        eps_prev_p = eps_row.iloc[1]
                                    break

                        if (
                            revenue_latest_p is not None and pd.notna(revenue_latest_p)
                            and revenue_prev_p not in (None, 0) and pd.notna(revenue_prev_p)
                        ):
                            result["Revenue Growth (%)"] = (
                                (revenue_latest_p - revenue_prev_p) / abs(revenue_prev_p)
                            ) * 100

                        if (
                            eps_latest_p is not None and pd.notna(eps_latest_p)
                            and eps_prev_p not in (None, 0) and pd.notna(eps_prev_p)
                        ):
                            result["EPS Growth (%)"] = (
                                (eps_latest_p - eps_prev_p) / abs(eps_prev_p)
                            ) * 100

                        if (
                            net_income is not None and pd.notna(net_income)
                            and revenue_latest_p not in (None, 0) and pd.notna(revenue_latest_p)
                        ):
                            result["Net Profit Margin (%)"] = (net_income / revenue_latest_p) * 100

                        if bs is not None and not bs.empty and ebit_p is not None and pd.notna(ebit_p):

                            total_assets_p = None
                            current_liab_p = None

                            if "Total Assets" in bs.index:
                                total_assets_p = bs.loc["Total Assets"].iloc[0]

                            if "Current Liabilities" in bs.index:
                                current_liab_p = bs.loc["Current Liabilities"].iloc[0]

                            if (
                                total_assets_p is not None and pd.notna(total_assets_p)
                                and current_liab_p is not None and pd.notna(current_liab_p)
                            ):
                                cap_employed_p = total_assets_p - current_liab_p

                                if cap_employed_p != 0:
                                    result["ROCE (%)"] = (ebit_p / cap_employed_p) * 100

                        if result["ROE (%)"] is None and net_income is not None and bs is not None and not bs.empty:

                            equity_p = None

                            for alias in ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"]:
                                if alias in bs.index:
                                    equity_p = bs.loc[alias].iloc[0]
                                    break

                            if equity_p not in (None, 0) and pd.notna(net_income) and pd.notna(equity_p):
                                result["ROE (%)"] = (net_income / equity_p) * 100

                        if cf is not None and not cf.empty:

                            fcf_val = None

                            if "Free Cash Flow" in cf.index:
                                fcf_val = cf.loc["Free Cash Flow"].iloc[0]
                            else:
                                ocf_val = None
                                capex_val = None

                                for alias in ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]:
                                    if alias in cf.index:
                                        ocf_val = cf.loc[alias].iloc[0]
                                        break

                                if "Capital Expenditure" in cf.index:
                                    capex_val = cf.loc["Capital Expenditure"].iloc[0]

                                if ocf_val is not None and capex_val is not None and pd.notna(ocf_val) and pd.notna(capex_val):
                                    fcf_val = ocf_val + capex_val

                            if fcf_val is not None and pd.notna(fcf_val):
                                result["FCF (₹ Cr)"] = fcf_val / 1e7

                    except Exception:
                        pass

                    return result


                if peer_list:

                    peer_symbols = [f"{base_symbol}.NS"] + [f"{p}.NS" for p in peer_list]
                    peer_names = [base_symbol] + peer_list

                    peer_comparison_rows = []

                    with st.spinner("Peers ka fundamental data fetch ho raha hai..."):

                        for p_name, p_ticker_sym in zip(peer_names, peer_symbols):

                            fundamentals = get_full_fundamentals(p_ticker_sym)

                            row = {"Stock": p_name}

                            for key, value in fundamentals.items():
                                row[key] = round(value, 2) if value is not None else "N/A"

                            peer_comparison_rows.append(row)

                    peer_comparison_df = pd.DataFrame(peer_comparison_rows)

                    st.dataframe(
                        peer_comparison_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    csv_peer = peer_comparison_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Peer Comparison CSV Download",
                        csv_peer,
                        "peer_comparison.csv",
                        "text/csv",
                        key="peer_comparison_csv"
                    )

                    peer_pe_numeric = pd.to_numeric(peer_comparison_df.iloc[1:]["P/E"], errors="coerce").dropna()
                    peer_roe_numeric = pd.to_numeric(peer_comparison_df.iloc[1:]["ROE (%)"], errors="coerce").dropna()

                    base_pe_val = peer_comparison_df.iloc[0]["P/E"]
                    base_roe_val = peer_comparison_df.iloc[0]["ROE (%)"]

                    peer_verdict_notes = []

                    if not peer_pe_numeric.empty and base_pe_val != "N/A":

                        sector_avg_pe_full = peer_pe_numeric.mean()

                        if base_pe_val < sector_avg_pe_full * 0.85:
                            peer_verdict_notes.append(f"✅ {base_symbol} apne peers ke comparison me P/E discount pe trade kar raha hai.")
                        elif base_pe_val > sector_avg_pe_full * 1.15:
                            peer_verdict_notes.append(f"⚠️ {base_symbol} apne peers ke comparison me P/E premium pe trade kar raha hai.")
                        else:
                            peer_verdict_notes.append(f"🟡 {base_symbol} apne peers ke roughly in-line P/E pe trade kar raha hai.")

                    if not peer_roe_numeric.empty and base_roe_val != "N/A":

                        sector_avg_roe_full = peer_roe_numeric.mean()

                        if base_roe_val > sector_avg_roe_full * 1.1:
                            peer_verdict_notes.append(f"✅ {base_symbol} ka ROE peers ke average se better hai — capital efficiency strong.")
                        elif base_roe_val < sector_avg_roe_full * 0.9:
                            peer_verdict_notes.append(f"⚠️ {base_symbol} ka ROE peers ke average se weak hai.")
                        else:
                            peer_verdict_notes.append(f"🟡 {base_symbol} ka ROE peers ke roughly in-line hai.")

                    for note in peer_verdict_notes:
                        if note.startswith("✅"):
                            st.success(note)
                        elif note.startswith("🟡"):
                            st.info(note)
                        else:
                            st.warning(note)

                else:
                    st.info(
                        f"{base_symbol} ke liye abhi peer mapping available nahi hai — sector comparison skip ho raha hai."
                    )

                # ─────────────────────────────────────────
# RISK ENGINE (STOCK-LEVEL)
# ─────────────────────────────────────────

                st.markdown("---")
                st.subheader("⚠️ Risk Engine")

                RISK_FREE_RATE = 0.065  # India ~10Y G-Sec assumption, annualized

                st.caption(
                    f"Risk-free rate assumption: {RISK_FREE_RATE*100:.1f}% (India 10Y G-Sec approx.) — "
                    "Sharpe/Sortino calculations is par based hain."
                )

                @st.cache_data(ttl=3600)
                def get_nifty_data(period):
                    try:
                        nifty = yf.Ticker("^NSEI").history(period=period)
                        return nifty
                    except Exception:
                        return None

                stock_returns = df["Close"].pct_change().dropna()

                if len(stock_returns) < 20:

                    st.info(
                        "Risk metrics calculate karne ke liye sufficient data available nahi hai. Longer period try karo."
                    )

                else:

                    # ─────────────────────────────────────────
# VOLATILITY (annualized)
# ─────────────────────────────────────────

                    daily_std = stock_returns.std()
                    annualized_volatility = daily_std * np.sqrt(252) * 100

                    # ─────────────────────────────────────────
# MAX DRAWDOWN
# ─────────────────────────────────────────

                    cum_returns = (1 + stock_returns).cumprod()
                    running_max = cum_returns.cummax()
                    drawdown_series = (cum_returns / running_max) - 1
                    stock_max_drawdown = drawdown_series.min() * 100

                    # ─────────────────────────────────────────
# SHARPE RATIO (annualized)
# ─────────────────────────────────────────

                    daily_rf = RISK_FREE_RATE / 252
                    excess_returns = stock_returns - daily_rf

                    if daily_std != 0:
                        sharpe_ratio = (excess_returns.mean() / daily_std) * np.sqrt(252)
                    else:
                        sharpe_ratio = None

                    # ─────────────────────────────────────────
# SORTINO RATIO (downside deviation based)
# ─────────────────────────────────────────

                    downside_returns = excess_returns[excess_returns < 0]

                    if len(downside_returns) > 0 and downside_returns.std() != 0:
                        downside_deviation = downside_returns.std()
                        sortino_ratio = (excess_returns.mean() / downside_deviation) * np.sqrt(252)
                        annualized_downside_risk = downside_deviation * np.sqrt(252) * 100
                    else:
                        sortino_ratio = None
                        annualized_downside_risk = None

                    # ─────────────────────────────────────────
# ATR-BASED VOLATILITY
# ─────────────────────────────────────────

                    high_low = df["High"] - df["Low"]
                    high_close = (df["High"] - df["Close"].shift()).abs()
                    low_close = (df["Low"] - df["Close"].shift()).abs()

                    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    atr_14 = true_range.rolling(14).mean()

                    latest_atr = atr_14.iloc[-1] if pd.notna(atr_14.iloc[-1]) else None
                    latest_close_price = df["Close"].iloc[-1]

                    if latest_atr is not None and latest_close_price != 0:
                        atr_pct = (latest_atr / latest_close_price) * 100
                    else:
                        atr_pct = None

                    # ─────────────────────────────────────────
# BETA + CORRELATION (vs NIFTY)
# ─────────────────────────────────────────

                    nifty_data = get_nifty_data(period)

                    stock_beta = None
                    stock_correlation = None

                    if nifty_data is not None and not nifty_data.empty:

                        nifty_returns = nifty_data["Close"].pct_change().dropna()

                        aligned = pd.concat(
                            [stock_returns, nifty_returns],
                            axis=1,
                            join="inner"
                        )
                        aligned.columns = ["stock", "nifty"]
                        aligned = aligned.dropna()

                        if len(aligned) > 20:

                            covariance = aligned["stock"].cov(aligned["nifty"])
                            nifty_variance = aligned["nifty"].var()

                            if nifty_variance != 0:
                                stock_beta = covariance / nifty_variance

                            stock_correlation = aligned["stock"].corr(aligned["nifty"])

                    # ─────────────────────────────────────────
# DISPLAY — RISK METRICS
# ─────────────────────────────────────────

                    r1, r2, r3, r4 = st.columns(4)

                    with r1:
                        st.metric("Volatility (Annualized)", f"{annualized_volatility:.2f}%")

                    with r2:
                        st.metric("Max Drawdown", f"{stock_max_drawdown:.2f}%")

                    with r3:
                        if sharpe_ratio is not None:
                            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                        else:
                            st.metric("Sharpe Ratio", "N/A")

                    with r4:
                        if sortino_ratio is not None:
                            st.metric("Sortino Ratio", f"{sortino_ratio:.2f}")
                        else:
                            st.metric("Sortino Ratio", "N/A")

                    r5, r6, r7, r8 = st.columns(4)

                    with r5:
                        if atr_pct is not None:
                            st.metric("ATR (14) % of Price", f"{atr_pct:.2f}%")
                        else:
                            st.metric("ATR (14) % of Price", "N/A")

                    with r6:
                        if stock_beta is not None:
                            st.metric("Beta (vs NIFTY)", f"{stock_beta:.2f}")
                        else:
                            st.metric("Beta (vs NIFTY)", "N/A")

                    with r7:
                        if stock_correlation is not None:
                            st.metric("Correlation (vs NIFTY)", f"{stock_correlation:.2f}")
                        else:
                            st.metric("Correlation (vs NIFTY)", "N/A")

                    with r8:
                        if annualized_downside_risk is not None:
                            st.metric("Downside Risk (Annualized)", f"{annualized_downside_risk:.2f}%")
                        else:
                            st.metric("Downside Risk (Annualized)", "N/A")

                    # ─────────────────────────────────────────
# DRAWDOWN CHART
# ─────────────────────────────────────────

                    st.markdown("### 📉 Drawdown Chart")

                    fig_dd = go.Figure()

                    fig_dd.add_trace(go.Scatter(
                        x=drawdown_series.index,
                        y=drawdown_series.values * 100,
                        mode="lines",
                        fill="tozeroy",
                        name="Drawdown %"
                    ))

                    fig_dd.update_layout(
                        title=f"Drawdown — {period} Period",
                        xaxis_title="Date",
                        yaxis_title="Drawdown (%)"
                    )

                    st.plotly_chart(
                        fig_dd,
                        use_container_width="stretch",
                        key="stock_drawdown_chart"
                    )

                    # ─────────────────────────────────────────
# AI RISK VERDICT
# ─────────────────────────────────────────

                    risk_notes = []

                    if annualized_volatility > 40:
                        risk_notes.append("🔴 Volatility high hai (>40% annualized) — price swings zyada hain.")
                    elif annualized_volatility > 25:
                        risk_notes.append("🟡 Volatility moderate hai.")
                    else:
                        risk_notes.append("✅ Volatility relatively low hai — stable price movement.")

                    if stock_beta is not None:
                        if stock_beta > 1.2:
                            risk_notes.append(f"🔴 Beta {stock_beta:.2f} — market se zyada volatile hai (high-beta stock).")
                        elif stock_beta < 0.8:
                            risk_notes.append(f"✅ Beta {stock_beta:.2f} — market se kam volatile hai (defensive stock).")
                        else:
                            risk_notes.append(f"🟡 Beta {stock_beta:.2f} — roughly market ke saath move karta hai.")

                    if stock_max_drawdown < -30:
                        risk_notes.append("🔴 Max Drawdown severe hai (>30% fall dekha gaya hai is period me).")
                    elif stock_max_drawdown < -15:
                        risk_notes.append("🟡 Max Drawdown moderate hai.")

                    if sharpe_ratio is not None:
                        if sharpe_ratio > 1:
                            risk_notes.append(f"✅ Sharpe Ratio {sharpe_ratio:.2f} — risk ke against achha risk-adjusted return.")
                        elif sharpe_ratio < 0:
                            risk_notes.append(f"🔴 Sharpe Ratio negative ({sharpe_ratio:.2f}) — risk-free rate se bhi kam return mila is period me.")

                    for note in risk_notes:
                        if note.startswith("✅"):
                            st.success(note)
                        elif note.startswith("🟡"):
                            st.info(note)
                        else:
                            st.warning(note)

                # ─────────────────────────────────────────
# MULTI-FACTOR ENGINE
# ─────────────────────────────────────────

                st.markdown("---")
                st.subheader("🧬 Multi-Factor Engine")

                st.caption(
                    "Roadmap weights: Quality 25% • Growth 20% • Valuation 20% • Momentum 15% • Risk 10% • Cash Flow 10%. "
                    "Ye weights abhi historical validation se prove nahi hue — starting framework hai."
                )

                def score_from_thresholds(value, thresholds, reverse=False):
                    """
                    thresholds: list of (cutoff, score) pairs, sorted best-to-worst by cutoff logic.
                    reverse=True → lower value = better (jaise P/E, Volatility).
                    """
                    if value is None:
                        return None

                    if reverse:
                        for cutoff, score in thresholds:
                            if value <= cutoff:
                                return score
                        return thresholds[-1][1]
                    else:
                        for cutoff, score in thresholds:
                            if value >= cutoff:
                                return score
                        return thresholds[-1][1]


                # ─────────────────────────────────────────
# QUALITY SCORE (25%) — ROE, ROCE, Net Profit Margin
# ─────────────────────────────────────────

                quality_components = []

                if roe is not None:
                    quality_components.append(
                        score_from_thresholds(roe * 100, [(20, 100), (15, 80), (10, 60), (5, 40), (0, 20)])
                    )

                if roce is not None:
                    quality_components.append(
                        score_from_thresholds(roce, [(20, 100), (15, 80), (10, 60), (5, 40), (0, 20)])
                    )

                try:
                    if net_profit_margin is not None:
                        quality_components.append(
                            score_from_thresholds(net_profit_margin, [(15, 100), (10, 80), (5, 60), (2, 40), (0, 20)])
                        )
                except NameError:
                    pass

                quality_score = (
                    sum(quality_components) / len(quality_components)
                    if quality_components else None
                )

                # ─────────────────────────────────────────
# GROWTH SCORE (20%) — Revenue Growth, Net Profit Growth (YoY)
# ─────────────────────────────────────────

                growth_components = []

                try:
                    if revenue_growth is not None:
                        growth_components.append(
                            score_from_thresholds(revenue_growth, [(20, 100), (12, 80), (5, 60), (0, 40), (-100, 20)])
                        )
                    if net_profit_growth is not None:
                        growth_components.append(
                            score_from_thresholds(net_profit_growth, [(20, 100), (12, 80), (5, 60), (0, 40), (-100, 20)])
                        )
                except NameError:
                    pass

                growth_score = (
                    sum(growth_components) / len(growth_components)
                    if growth_components else None
                )

                # ─────────────────────────────────────────
# VALUATION SCORE (20%) — P/E (lower better), PEG (lower better)
# ─────────────────────────────────────────

                valuation_components = []

                if pe_ratio is not None and pe_ratio > 0:
                    valuation_components.append(
                        score_from_thresholds(pe_ratio, [(15, 100), (25, 80), (35, 60), (50, 40), (9999, 20)], reverse=True)
                    )

                try:
                    if peg_ratio is not None and peg_ratio > 0:
                        valuation_components.append(
                            score_from_thresholds(peg_ratio, [(1, 100), (1.5, 80), (2, 60), (3, 40), (9999, 20)], reverse=True)
                        )
                except NameError:
                    pass

                valuation_score = (
                    sum(valuation_components) / len(valuation_components)
                    if valuation_components else None
                )

                # ─────────────────────────────────────────
# MOMENTUM SCORE (15%) — RSI, MACD, Price vs SMA_50
# ─────────────────────────────────────────

                momentum_components = []

                try:
                    if pd.notna(latest["RSI"]):
                        rsi_val_mf = latest["RSI"]
                        if 40 <= rsi_val_mf <= 60:
                            momentum_components.append(70)
                        elif 30 <= rsi_val_mf < 40 or 60 < rsi_val_mf <= 70:
                            momentum_components.append(55)
                        elif rsi_val_mf < 30:
                            momentum_components.append(85)  # oversold, potential bounce
                        else:
                            momentum_components.append(30)  # overbought
                except Exception:
                    pass

                try:
                    if pd.notna(latest["MACD"]) and pd.notna(latest["MACD_Signal"]):
                        if latest["MACD"] > latest["MACD_Signal"]:
                            momentum_components.append(75)
                        else:
                            momentum_components.append(35)
                except Exception:
                    pass

                try:
                    if pd.notna(latest["SMA_50"]) and latest["SMA_50"] != 0:
                        if latest["Close"] > latest["SMA_50"]:
                            momentum_components.append(70)
                        else:
                            momentum_components.append(30)
                except Exception:
                    pass

                momentum_score = (
                    sum(momentum_components) / len(momentum_components)
                    if momentum_components else None
                )

                # ─────────────────────────────────────────
# RISK SCORE (10%) — Sharpe Ratio, Volatility (lower vol better)
# ─────────────────────────────────────────

                risk_components = []

                try:
                    if sharpe_ratio is not None:
                        risk_components.append(
                            score_from_thresholds(sharpe_ratio, [(1.5, 100), (1, 80), (0.5, 60), (0, 40), (-100, 20)])
                        )
                    if annualized_volatility is not None:
                        risk_components.append(
                            score_from_thresholds(annualized_volatility, [(20, 100), (30, 80), (40, 60), (55, 40), (9999, 20)], reverse=True)
                        )
                except NameError:
                    pass

                risk_score = (
                    sum(risk_components) / len(risk_components)
                    if risk_components else None
                )

                # ─────────────────────────────────────────
# CASH FLOW SCORE (10%) — CFO / Net Profit Ratio (Earnings Quality)
# ─────────────────────────────────────────

                cashflow_components = []

                try:
                    if cfo_np_ratio is not None:
                        cashflow_components.append(
                            score_from_thresholds(cfo_np_ratio, [(1.2, 100), (1, 80), (0.8, 60), (0.5, 40), (0, 20)])
                        )
                except NameError:
                    pass

                cashflow_score = (
                    sum(cashflow_components) / len(cashflow_components)
                    if cashflow_components else None
                )

                # ─────────────────────────────────────────
# COMPOSITE WEIGHTED SCORE (missing factors ka weight redistribute hota hai)
# ─────────────────────────────────────────

                factor_weights = {
                    "Quality": (quality_score, 0.25),
                    "Growth": (growth_score, 0.20),
                    "Valuation": (valuation_score, 0.20),
                    "Momentum": (momentum_score, 0.15),
                    "Risk": (risk_score, 0.10),
                    "Cash Flow": (cashflow_score, 0.10),
                }

                available_factor_weight = sum(
                    weight for score, weight in factor_weights.values() if score is not None
                )

                if available_factor_weight > 0:

                    composite_score = sum(
                        score * weight for score, weight in factor_weights.values() if score is not None
                    ) / available_factor_weight

                else:
                    composite_score = None

                # ─────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────

                if composite_score is not None:

                    mf1, mf2 = st.columns([1, 2])

                    with mf1:
                        st.metric("Composite Score", f"{composite_score:.1f} / 100")

                        if composite_score >= 80:
                            grade = "A — Excellent"
                        elif composite_score >= 65:
                            grade = "B — Good"
                        elif composite_score >= 50:
                            grade = "C — Average"
                        elif composite_score >= 35:
                            grade = "D — Weak"
                        else:
                            grade = "F — Poor"

                        st.metric("Grade", grade)

                    with mf2:

                        radar_categories = []
                        radar_values = []

                        for factor_name, (score, weight) in factor_weights.items():
                            radar_categories.append(factor_name)
                            radar_values.append(score if score is not None else 0)

                        radar_categories.append(radar_categories[0])
                        radar_values.append(radar_values[0])

                        fig_radar = go.Figure()

                        fig_radar.add_trace(go.Scatterpolar(
                            r=radar_values,
                            theta=radar_categories,
                            fill="toself",
                            name=base_symbol if "base_symbol" in dir() else sym
                        ))

                        fig_radar.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                            showlegend=False,
                            title="Multi-Factor Breakdown"
                        )

                        st.plotly_chart(
                            fig_radar,
                            use_container_width="stretch",
                            key="multi_factor_radar"
                        )

                    st.markdown("### 📋 Factor Score Breakdown")

                    factor_table_rows = []

                    for factor_name, (score, weight) in factor_weights.items():
                        factor_table_rows.append({
                            "Factor": factor_name,
                            "Weight": f"{weight*100:.0f}%",
                            "Score": f"{score:.1f}" if score is not None else "N/A"
                        })

                    st.dataframe(
                        pd.DataFrame(factor_table_rows),
                        use_container_width=True
                    )

                    if available_factor_weight < 0.7:
                        st.warning(
                            f"⚠️ Sirf {available_factor_weight*100:.0f}% factor-weight ka data available tha — "
                            "score partial data pe based hai, poora confidence mat rakho."
                        )

                    # AI verdict
                    if composite_score >= 65:
                        st.success(
                            "✅ Overall multi-factor profile strong hai — fundamentals, valuation, aur momentum ka combination positive hai."
                        )
                    elif composite_score >= 50:
                        st.info(
                            "🟡 Overall profile average hai — kuch factors strong hain, kuch weak. Individual sections detail me dekho."
                        )
                    else:
                        st.warning(
                            "🔴 Overall multi-factor score weak hai — is stock ko carefully evaluate karo before deciding."
                        )

                else:
                    st.info(
                        "Multi-Factor Score calculate karne ke liye sufficient data available nahi hai."
                    )

                # ─────────────────────────────────────────
# MACRO-ECONOMIC ENGINE
# ─────────────────────────────────────────

                st.markdown("---")
                st.subheader("🌍 Macro-Economic Engine")

                st.caption(
                    "NIFTY, VIX, USD/INR, Gold, Crude — live data (yfinance). "
                    "CPI, Repo Rate, GDP — free market API se available nahi, neeche manually update karo."
                )

                @st.cache_data(ttl=3600)
                def get_macro_market_data():

                    macro_tickers = {
                        "NIFTY 50": "^NSEI",
                        "India VIX": "^INDIAVIX",
                        "USD/INR": "INR=X",
                        "Gold (USD/oz)": "GC=F",
                        "Crude Oil (USD/bbl)": "CL=F",
                    }

                    macro_results = {}

                    for label, tkr in macro_tickers.items():
                        try:
                            hist = yf.Ticker(tkr).history(period="5d")
                            if not hist.empty and len(hist) >= 2:
                                latest_val = hist["Close"].iloc[-1]
                                prev_val = hist["Close"].iloc[-2]
                                change_pct = ((latest_val - prev_val) / prev_val) * 100
                                macro_results[label] = (latest_val, change_pct)
                            else:
                                macro_results[label] = (None, None)
                        except Exception:
                            macro_results[label] = (None, None)

                    return macro_results


                macro_data = get_macro_market_data()

                mc1, mc2, mc3, mc4, mc5 = st.columns(5)

                macro_cols = [mc1, mc2, mc3, mc4, mc5]

                for col, (label, (value, change)) in zip(macro_cols, macro_data.items()):

                    with col:
                        if value is not None:
                            st.metric(
                                label,
                                f"{value:,.2f}",
                                f"{change:+.2f}%" if change is not None else None
                            )
                        else:
                            st.metric(label, "N/A")

                # ─────────────────────────────────────────
# INDIA VIX INTERPRETATION
# ─────────────────────────────────────────

                vix_value = macro_data.get("India VIX", (None, None))[0]

                if vix_value is not None:

                    st.markdown("### 😰 Market Fear Gauge (India VIX)")

                    if vix_value > 20:
                        st.warning(
                            f"🔴 India VIX {vix_value:.2f} — high fear/uncertainty market me hai. Volatility zyada expected hai."
                        )
                    elif vix_value > 14:
                        st.info(
                            f"🟡 India VIX {vix_value:.2f} — moderate uncertainty, normal range me hai."
                        )
                    else:
                        st.success(
                            f"✅ India VIX {vix_value:.2f} — low fear, market relatively calm/complacent hai."
                        )

                # ─────────────────────────────────────────
# MANUAL MACRO INDICATORS (CPI, Repo Rate, GDP)
# ─────────────────────────────────────────

                st.markdown("### 📝 Manual Macro Indicators (RBI/MOSPI)")

                st.caption(
                    "Ye values yahan store nahi hoti (session-only) — latest figures RBI website ya MOSPI se check karke daalo."
                )

                mm1, mm2, mm3 = st.columns(3)

                with mm1:
                    manual_repo_rate = st.number_input(
                        "Repo Rate (%)", min_value=0.0, max_value=15.0, value=6.5, step=0.25, key="macro_repo"
                    )

                with mm2:
                    manual_cpi = st.number_input(
                        "CPI Inflation (%)", min_value=-5.0, max_value=20.0, value=5.0, step=0.1, key="macro_cpi"
                    )

                with mm3:
                    manual_gdp_growth = st.number_input(
                        "GDP Growth (%)", min_value=-10.0, max_value=15.0, value=6.5, step=0.1, key="macro_gdp"
                    )

                macro_notes = []

                if manual_repo_rate > 7:
                    macro_notes.append("⚠️ Repo Rate high hai — borrowing cost badha hua hai, high-debt companies pe negative impact ho sakta hai.")
                elif manual_repo_rate < 5:
                    macro_notes.append("✅ Repo Rate low hai — cheap borrowing environment, growth-stocks ke liye favorable.")

                if manual_cpi > 6:
                    macro_notes.append("⚠️ CPI inflation RBI ke comfort zone (~4-6%) se upar hai — rate hikes ka risk badh sakta hai.")
                elif manual_cpi < 2:
                    macro_notes.append("🟡 CPI bahut low hai — demand weakness ka signal ho sakta hai.")

                if manual_gdp_growth > 7:
                    macro_notes.append("✅ GDP growth strong hai — overall economic environment supportive hai.")
                elif manual_gdp_growth < 4:
                    macro_notes.append("⚠️ GDP growth weak hai — corporate earnings pe broad pressure ho sakta hai.")

                for note in macro_notes:
                    if note.startswith("✅"):
                        st.success(note)
                    elif note.startswith("🟡"):
                        st.info(note)
                    else:
                        st.warning(note)

                if not macro_notes:
                    st.info("Macro indicators neutral/comfortable range me hain.")

                st.markdown("---")
                st.subheader("🏦 Institutional Smart Money Tracker")

                smart_money_score = 50

                avg_volume = df["Volume"].tail(30).mean()
                latest_volume = latest["Volume"]

                if latest_volume > avg_volume * 1.5:
                    smart_money_score += 25
                    smart_signal = "Strong Institutional Buying"
                elif latest_volume > avg_volume:
                    smart_money_score += 10
                    smart_signal = "Moderate Institutional Activity"
                else:
                    smart_money_score -= 10
                    smart_signal = "Weak Institutional Interest"

                st.metric("Smart Money Score", f"{smart_money_score}/100")
                st.progress(smart_money_score / 100)
                st.info(f"Signal: {smart_signal}")

                change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
                avg_vol = df["Volume"].tail(20).mean()

                if avg_vol > 0:
                    vol_ratio = latest["Volume"] / avg_vol
                else:
                    vol_ratio = 1
                week52_pos = ((latest['Close'] - df['Close'].min()) /
                            (df['Close'].max() - df['Close'].min()) * 100)
                rsi_val = latest['RSI']
                company = info.get('longName', sym) if info else sym

                # AI Buy / Hold / Sell Signal
                signal_score = 0

                if rsi_val < 35:
                    signal_score += 2
                elif rsi_val < 60:
                    signal_score += 1

                if latest["MACD"] > latest["MACD_Signal"]:
                    signal_score += 2

                if vol_ratio > 1.2:
                    signal_score += 1

                if week52_pos < 40:
                    signal_score += 1

                if signal_score >= 5:
                    ai_signal = "BUY 🟢"
                    signal_confidence = 85
                elif signal_score >= 3:
                    ai_signal = "HOLD 🟡"
                    signal_confidence = 65
                else:
                    ai_signal = "AVOID / SELL 🔴"
                    signal_confidence = 45

                st.markdown("### 🤖 AI Buy / Hold / Sell Signal")
                st.metric("AI Signal", ai_signal)
                st.progress(signal_confidence / 100)
                st.caption(f"Confidence: {signal_confidence}% | Score: {signal_score}/6")

                st.markdown(f"### {company}")

                st.markdown("### Research Notes")

                reasons = []

                if rsi_val < 35:
                    reasons.append("RSI low hai, stock oversold zone me hai.")

                if latest["MACD"] > latest["MACD_Signal"]:
                    reasons.append("MACD bullish crossover dikh raha hai.")

                if vol_ratio > 1.2:
                    reasons.append("Volume strong hai.")

                if week52_pos < 40:
                    reasons.append("52 week range ke lower zone me trade kar raha hai.")

                if not reasons:
                    reasons.append("Technical indicators mixed signals de rahe hain.")

                for r in reasons:
                    st.info(r)

                # Metrics
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("💰 Price", f"₹{latest['Close']:.2f}", f"{change:+.2f}%")
                m2.metric("📊 RSI", f"{rsi_val:.1f}",
                        "🔴 Overbought" if rsi_val > 70 else "🟢 Oversold" if rsi_val < 30 else "🟡 Neutral")
                m3.metric("🔊 Volume", f"{vol_ratio:.1f}x",
                        "🔥 High" if vol_ratio > 1.5 else "📉 Low" if vol_ratio < 0.7 else "Normal")
                m4.metric("📍 52W Pos", f"{week52_pos:.0f}%",
                        f"H:₹{df['Close'].max():.0f}")
                m5.metric("📈 MACD",
                        "Bullish ✅" if latest['MACD'] > latest['MACD_Signal'] else "Bearish ⚠️",
                        f"{latest['MACD']:.2f}")
                
                current_price = latest["Close"]
                
                target_price = round(current_price * 1.10, 2)
                upside = round(((target_price-current_price)/current_price)*100,2)

                st.metric(
                    "🎯 AI Target Price",
                    f"₹{target_price}",
                    f"{upside}% Upside"
                )

                # Chart
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                    row_heights=[0.55, 0.22, 0.23],
                                    vertical_spacing=0.03,
                                    subplot_titles=[f'{sym} Price', 'RSI', 'MACD'])

                fig.add_trace(go.Candlestick(
                    x=df.index, open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close'],
                    increasing_line_color='#00d68f',
                    decreasing_line_color='#ff4d6d', name='Price'), row=1, col=1)

                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'],
                    line=dict(color='#f0a500', width=1.5), name='50 SMA'), row=1, col=1)

                if df['SMA_200'].notna().any():
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'],
                        line=dict(color='#8b5cf6', width=1.5), name='200 SMA'), row=1, col=1)

                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'],
                    line=dict(color='rgba(67,97,238,0.4)', width=1),
                    name='BB', showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'],
                    line=dict(color='rgba(67,97,238,0.4)', width=1),
                    fill='tonexty', fillcolor='rgba(67,97,238,0.05)',
                    showlegend=False), row=1, col=1)

                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'],
                    line=dict(color='#4361ee', width=2),
                    name='RSI', showlegend=False), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,77,109,0.5)", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,214,143,0.5)", row=2, col=1)

                colors = ['#00d68f' if v >= 0 else '#ff4d6d' for v in df['MACD_Hist'].fillna(0)]
                fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'],
                    marker_color=colors, showlegend=False), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD'],
                    line=dict(color='#4361ee', width=1.5),
                    name='MACD', showlegend=False), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'],
                    line=dict(color='#f0a500', width=1.5),
                    name='Signal', showlegend=False), row=3, col=1)

                fig.update_layout(
                    template='plotly_dark', paper_bgcolor='#0c0c14',
                    plot_bgcolor='#0c0c14', height=580,
                    margin=dict(t=30, b=10, l=10, r=10),
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1, font=dict(size=10)))
                st.plotly_chart(fig, use_container_width="stretchS",key="nain_stock_chart")

                st.markdown("---")
                st.subheader("🏛 FII / DII Flow")

                fii_score = 60 if vol_ratio > 1 else 40

                st.metric(
                    "Institutional Flow Score",
                    fii_score
                )

                if fii_score > 55:
                    st.success("🟢 FII Buying Interest")
                else:
                    st.warning("🟡 Weak Institutional Flow")

    # PASTE HERE 👇

                    st.markdown("---")
                    st.divider()
                    st.markdown("### 🏦 FII / DII Smart Money Analysis")

                    fii_score = 40

                    if latest["MACD"] > latest["MACD_Signal"]:
                        fii_score += 20

                    if rsi_val > 50:
                        fii_score += 20

                    if vol_ratio > 1:
                        fii_score += 20

                    if fii_score >= 80:
                        flow_status = "🟢 Strong Institutional Buying"
                    elif fii_score >= 60:
                        flow_status = "🟡 Moderate Institutional Interest"
                    else:
                        flow_status = "🔴 Weak Institutional Flow"

                    st.metric("Institutional Flow Score", f"{fii_score}/100")
                    st.info(flow_status)

                    if fii_score >= 80:
                        st.success(
                            "Smart Money View: Institutions accumulation phase me dikh rahe hain."
                        )
                    elif fii_score >= 60:
                        st.warning(
                            "Smart Money View: Mixed institutional activity dikh rahi hai."
                        )
                    else:
                        st.error(
                            "Smart Money View: Institutions aggressively buy karte nahi dikh rahe."
                        )

                        st.markdown("---")
                    st.subheader("📊 Multi-Timeframe Analysis")

                    daily_trend = "🟢 Bullish" if latest["MACD"] > latest["MACD_Signal"] else "🔴 Bearish"

                    weekly_trend = "🟢 Bullish" if rsi_val > 50 else "🟡 Neutral"

                    sma50_value = df["Close"].rolling(50).mean().iloc[-1]
                    monthly_trend = "🟢 Bullish" if latest["Close"] > sma50_value else "🔴 Bearish"
                    
                    trend_score = 0

                    if "Bullish" in daily_trend:
                        trend_score += 35

                    if "Bullish" in weekly_trend:
                        trend_score += 35

                    if "Bullish" in monthly_trend:
                        trend_score += 30

                    st.metric("Overall Trend Strength", f"{trend_score}/100")

                    st.info(f"Daily Trend: {daily_trend}")
                    st.info(f"Weekly Trend: {weekly_trend}")
                    st.info(f"Monthly Trend: {monthly_trend}")

                    st.subheader("📊 Signal-Based Sentiment")

                    sentiment_score = signal_score * 15

                    if sentiment_score >= 70:
                        sentiment = "🟢 Positive"
                    elif sentiment_score >= 40:
                        sentiment = "🟡 Neutral"
                    else:
                        sentiment = "🔴 Negative"

                    st.metric("Signal Sentiment Score", f"{sentiment_score}/100")
                    st.info(f"Market Sentiment: {sentiment}")
                    st.caption("Ye technical signal score se nikla hai, news se nahi. Real news sentiment neeche 'News + Sentiment Engine' me hai.")

                    st.markdown("---")
                    st.subheader("⚡ Stock Strength Meter")

                    strength = int((rsi_val))if pd.notna(rsi_val) else 50

                    st.progress(strength / 100)

                    st.metric(
                        "Stock Strength",
                        f"{strength}/100"
                    )

                    if strength >= 75:
                        st.success("🟢 Strong Stock")
                    elif strength >= 50:
                        st.warning("🟡 Average Strength")
                    else:
                        st.error("🔴 Weak Stock")

                        st.markdown("---")
                        st.subheader("🩺 Portfolio Doctor")

                    if 'risk_score' not in locals():
                        risk_score = 100 - strength 
                        st.write("Risk Score:", risk_score)
                        st.write("Health Score:", health_score)

                        portfolio_score = max(0, min(100, int((100 - risk_score) + health_score/2)))
                        st.metric("Portfolio Health Score", f"{portfolio_score}/100")

                        if portfolio_score >= 80:
                            st.success("✅ Healthy Portfolio")
                        elif portfolio_score >= 60:
                            st.warning("🟡 Portfolio Needs Improvement")
                        else:
                            st.error("🔴 Portfolio Risky")

                            if portfolio_score < 60:

                                st.warning("""
                                Portfolio Problems:
                                
                                • Risk jyada hai
                                
                                • Diversification kam hai
                                
                                • Rebalancing ki zarurat hai
                                """)
                            else:
                                st.info("""
                                Portfolio Stable Hai
                                
                                • Risk manageable hai
                                
                                • Long term holding possible hai
                                """)
                            st.markdown("---")
                            st.subheader("📈 AI Investment Thesis Generator")

                        with st.expander("Generate Investment Thesis"):

                    # 52 Week Position
                            try:
                                week52_high = df['High'].tail(252).max()
                                week52_low = df['Low'].tail(252).min()
                                current_price = df['Close'].iloc[-1]
                                if week52_high != week52_low:
                                    weeks52_pos = ((current_price - week52_low) / 
                                                (week52_high - week52_low)) * 100
                                else:
                                    weeks52_pos = 50.0
                            except:
                                weeks52_pos = 50.0

                            thesis = f"""
                            STOCK: {company}

                            Current Price: ₹{current_price:.2f}

                            RSI: {rsi_val:.1f}

                            AI VIEW:

                            Strengths:
                            - Strong market presence
                            - Established business model
                            - Technical indicators monitored

                            Risks:
                            - Market volatility
                            - Sector specific risks
                            - Economic slowdown impact

                            Investment Thesis:
                            This stock should be evaluated based on long-term fundamentals,
                            technical momentum and risk profile.
                        """

                        if strength >= 80:
                            verdict = "🟢 STRONG BUY"
                        elif strength >= 65:
                            verdict = "🟢 BUY"
                        elif strength >= 45:
                            verdict = "🟡 HOLD"
                        else:
                            verdict = "🔴 AVOID"

                        thesis += f"""

                            Verdict:
                            {verdict}
                            """

                        st.write(thesis)

                        st.markdown("---")
                        st.subheader("🎯 AI Recommendation")
                        

                        st.success(f"""
                        Final Verdict: {verdict}

                        Confidence Score: {strength}/100

                        AI Reason:
                        • RSI: {rsi_val:.1f}
                        • MACD: {'Bullish' if latest['MACD'] > latest['MACD_Signal'] else 'Bearish'}
                        • 52W Position: {weeks52_pos:.0f}%
                        • Signal Sentiment: {sentiment}
                        """) 

                    # AI Analysis
                st.divider()
                st.markdown("""
                <div class="bf-section">
                    <div class="bf-section-line"></div>
                    <div class="bf-section-title">AI Research</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### Analysis in Hindi")

                api_key = os.getenv('GROQ_API_KEY')
                if not api_key:
                    st.error("❌ GROQ_API_KEY nahi mili! .env file check karo")
                    st.stop()

                with st.spinner("🧠 AI analysis kar raha hai..."):
                    try:
                        #st.write("API Loaded:", bool(api_key))
                        #st.write("Key Start:", api_key[:10])

                        client = Groq(api_key=api_key)
                        sma50 = latest['SMA_50'] if pd.notna(latest['SMA_50']) else 0
                        p_vs_50 = ((latest['Close'] - sma50) / sma50 * 100) if sma50 else 0

                        fund_total_assets = (
                            f"₹{total_assets_cr:.2f} Cr"
                            if total_assets_cr is not None
                            else "N/A"
                        )

                        fund_total_debt = (
                            f"₹{total_debt_cr:.2f} Cr"
                            if total_debt_cr is not None
                            else "N/A"
                        )

                        fund_cash = (
                            f"₹{cash_cr:.2f} Cr"
                            if cash_cr is not None
                            else "N/A"
                        )

                        fund_equity = (
                            f"₹{equity_cr:.2f} Cr"
                            if equity_cr is not None
                            else "N/A"
                        )

                        prompt = f"""
    Tu expert Indian stock market analyst hai.
    {user_type} ko simple Hindi mein samjhao.

    Stock: {sym}
    Price: Rs{latest['Close']:.2f} ({change:+.2f}% aaj)
    RSI: {rsi_val:.1f}
    MACD: {latest['MACD']:.3f} (Signal: {latest['MACD_Signal']:.3f})
    Price vs 50 SMA: {p_vs_50:.1f}%
    Volume: {vol_ratio:.1f}x average
    52W Position: {week52_pos:.0f}%
    52W High: Rs{df['Close'].max():.2f}
    52W Low: Rs{df['Close'].min():.2f}

    FUNDAMENTAL ANALYSIS

    Balance Sheet ko analyze karo:

    - Total Assets: {fund_total_assets}
    - Total Debt: {fund_total_debt}
    - Cash: {fund_cash}
    - Shareholders Equity: {fund_equity}

    Debt level aur equity ke comparison ko explain karo.
    Cash position ko explain karo.
    Overall balance sheet strength ko simple Hindi mein explain karo.
    Agar data incomplete ho to clearly "Data unavailable" bolo.

    Book Knowledge: {BOOK_KNOWLEDGE}

    FUNDAMENTAL DATA

    Total Assets:
    {fund_total_assets}
    Total Debt:
    {fund_total_debt}
    Cash: {fund_cash}
    Shareholders Equity:
    {fund_equity}

    Is format mein SIRF HINDI mein likho:

    TECHNICAL PICTURE
    [2-3 lines current situation]

    FUNDAMENTAL PICTURE
    [Balance Sheet ke basis par 2-3 lines analysis]

    DEBT & FINANCIAL HEALTH
    [Debt, Equity aur Cash position explain karo]

    OVERALL VIEW
    [Technical + Fundamental picture ko combine karke 2-3 lines]

    PSYCHOLOGY CHECK
    [FOMO, greed, fear warning]

    BOOK INSIGHT
    [Relevant lesson]

    RECOMMENDATION
    Signal: Buy/Hold/Avoid
    Entry: Rs[price]
    Stop Loss: Rs[price]
    Target: Rs[price]
    Risk: LOW/MEDIUM/HIGH
    Confidence: [%]

    SUMMARY
    [1 line seedhi baat]

    300 words max.
    """
                        response = client.chat.completions.create(
                            model="openai/gpt-oss-120b",
                            max_tokens=1000,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        analysis = response.choices[0].message.content
                        st.markdown(analysis)

                        report_text = f"""
                        BHARATFINAI STOCK REPORT

                        Stock: {sym}
                        Price: ₹{latest['Close']:.2f}
                        RSI: {rsi_val:.1f}

                        AI Analysis:
                        {analysis}
                        """

                        # YAHAN PASTE KARO 👇

                        st.markdown("### 🎯 AI Investment Thesis")

                        confidence = int((93 + 70 + 77) / 3)

                        high_52 = df["High"].rolling(252).max().iloc[-1]
                        low_52 = df["Low"].rolling(252).min().iloc[-1]

                        weeks52_pos = ((latest["Close"] - low_52) /
                                    (high_52 - low_52)) * 100
                        
                        if pd.isna(weeks52_pos): weeks52_pos = 50

                        bull_case = [
                            "MACD Bullish" if latest['MACD'] > latest['MACD_Signal'] else "MACD Weak",
                            f"52W Position {weeks52_pos:.0f}%",
                            f"RSI {rsi_val:.1f}"
                        ]

                        bear_case = [
                            "Low Volume" if vol_ratio < 1 else "Healthy Volume",
                            "Weak Institutional Flow"
                        ]

                        signal = "Hold" if signal_score >= 3 else "Avoid"
                        if signal_score >= 5:
                            signal = "Buy"
                        elif signal_score >= 3:
                            signal = "Hold"
                        else:
                            signal = "Avoid"

                        st.info(f"""
                        📈 Bull Case:
                        • {bull_case[0]}
                        • {bull_case[1]}
                        • {bull_case[2]}

                        📉 Bear Case:
                        • {bear_case[0]}
                        • {bear_case[1]}

                        🎯 Verdict: {signal}

                        🔥 Confidence: {confidence}%
                        """)

                        # FIR DOWNLOAD BUTTON
                        

                        st.download_button(
                            "📄 Download Analysis Report",
                            data=report_text,
                            file_name=f"{sym}_report.txt",
                            mime="text/plain"
                        )

                        # ─────────────────────────────────────────
# NEWS + SENTIMENT ENGINE
# ─────────────────────────────────────────

                        st.divider()
                        st.markdown("### 📰 News + Sentiment Engine")

                        st.caption(
                            "Company + market news (NewsAPI). Sentiment: TextBlob + finance keywords, "
                            "events: keyword detection. Headline/description level analysis hai — approximate hai, financial advice nahi."
                        )

                        nws_base = sym.replace(".NS", "").replace(".BO", "").upper()
                        nws_bundle = fetch_news_bundle(sym)

                        # Backward-compat variables (purane code ke liye)
                        news_sentiment, news_score = "Neutral", 0

                        if nws_bundle["error"]:

                            st.warning(f"⚠️ News fetch nahi ho paya: {nws_bundle['error']}")

                        elif not nws_bundle["company"]:

                            st.info("Is stock ke liye recent news nahi mili.")

                        else:

                            nws_company = analyze_news_items(nws_bundle["company"])
                            nws_market = analyze_news_items(nws_bundle["market"])

                            nws_avg = sum(a["score"] for a in nws_company) / len(nws_company)

                            nws_pos = sum(1 for a in nws_company if a["label"] == "Positive")
                            nws_neu = sum(1 for a in nws_company if a["label"] == "Neutral")
                            nws_neg = sum(1 for a in nws_company if a["label"] == "Negative")

                            if nws_avg > 0.10:
                                nws_overall = "🟢 Positive"
                                news_sentiment, news_score = "Positive", 10
                            elif nws_avg < -0.10:
                                nws_overall = "🔴 Negative"
                                news_sentiment, news_score = "Negative", -10
                            else:
                                nws_overall = "🟡 Neutral"

                            if nws_market:
                                nws_market_avg = sum(a["score"] for a in nws_market) / len(nws_market)
                                if nws_market_avg > 0.10:
                                    nws_market_mood = "🟢 Positive"
                                elif nws_market_avg < -0.10:
                                    nws_market_mood = "🔴 Negative"
                                else:
                                    nws_market_mood = "🟡 Neutral"
                            else:
                                nws_market_mood = "N/A"

                            nm1, nm2, nm3, nm4 = st.columns(4)

                            with nm1:
                                st.metric("Company News Sentiment", nws_overall)

                            with nm2:
                                st.metric("Avg Score (-1 to +1)", f"{nws_avg:+.2f}")

                            with nm3:
                                st.metric("Pos / Neu / Neg", f"{nws_pos} / {nws_neu} / {nws_neg}")

                            with nm4:
                                st.metric("Market News Mood", nws_market_mood)

                            if len(nws_company) < 5:
                                st.caption(
                                    f"⚠️ Sirf {len(nws_company)} articles mile — sample chhota hai, sentiment reliable nahi ho sakta."
                                )

                            nws_dist_df = pd.DataFrame({
                                "Sentiment": ["Positive", "Neutral", "Negative"],
                                "Articles": [nws_pos, nws_neu, nws_neg]
                            })

                            fig_nws = px.bar(
                                nws_dist_df,
                                x="Sentiment",
                                y="Articles",
                                color="Sentiment",
                                color_discrete_map={
                                    "Positive": "#00d68f",
                                    "Neutral": "#f0a500",
                                    "Negative": "#ff4d6d"
                                },
                                title="Company News — Sentiment Distribution"
                            )

                            st.plotly_chart(
                                fig_nws,
                                use_container_width="stretch",
                                key="news_sentiment_dist"
                            )

                            # ─────────────────────────────────────────
                            # EVENT DETECTION
                            # ─────────────────────────────────────────

                            st.markdown("### 🔔 Detected Events")

                            nws_event_rows = []

                            for a in nws_company:
                                for ev in a["events"]:
                                    nws_event_rows.append({
                                        "Event": ev,
                                        "Sentiment": a["label"],
                                        "Headline": a["title"],
                                        "Source": a["source"],
                                        "Date": a["published"],
                                    })

                            if nws_event_rows:

                                nws_event_counts = {}

                                for row in nws_event_rows:
                                    nws_event_counts[row["Event"]] = nws_event_counts.get(row["Event"], 0) + 1

                                st.info(
                                    "Events mile: " + " • ".join(
                                        f"{name} ×{count}" for name, count in nws_event_counts.items()
                                    )
                                )

                                st.dataframe(
                                    pd.DataFrame(nws_event_rows),
                                    use_container_width=True,
                                    hide_index=True
                                )

                                nws_legal_neg = [
                                    a for a in nws_company
                                    if "⚖️ Regulatory / Legal" in a["events"] and a["label"] == "Negative"
                                ]

                                if nws_legal_neg:
                                    st.warning(
                                        f"⚠️ {len(nws_legal_neg)} negative regulatory/legal headline(s) mili — "
                                        "upar events table me details dekho."
                                    )

                            else:
                                st.info("Recent headlines me koi major corporate event detect nahi hua.")

                            # ─────────────────────────────────────────
                            # ALL HEADLINES
                            # ─────────────────────────────────────────

                            with st.expander(f"📄 Saari headlines ({len(nws_company)})"):

                                for a in nws_company:

                                    nws_emoji = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}[a["label"]]

                                    nws_title = (
                                        a["title"]
                                        .replace("[", "(")
                                        .replace("]", ")")
                                        .replace("$", "\\$")
                                    )

                                    nws_link = f"[{nws_title}]({a['url']})" if a.get("url") else nws_title

                                    st.markdown(
                                        f"{nws_emoji} {nws_link}  \n"
                                        f"{a['source']} · {a['published']} · score {a['score']:+.2f}"
                                    )

                            # ─────────────────────────────────────────
                            # AI NEWS EXPLANATION (Groq, 30 min cached)
                            # ─────────────────────────────────────────

                            nws_ai_lines = []

                            for a in nws_company[:12]:
                                nws_ev_text = (
                                    ", ".join(e.split(" ", 1)[1] for e in a["events"])
                                    if a["events"] else "-"
                                )
                                nws_ai_lines.append(
                                    f"- [{a['label']} | {nws_ev_text}] {a['title']} ({a['source']}, {a['published']})"
                                )

                            st.markdown("### 🧠 AI News Explanation")

                            with st.spinner("News ka AI explanation ban raha hai..."):
                                try:
                                    nws_ai_text = generate_news_ai_summary(
                                        nws_base,
                                        "\n".join(nws_ai_lines),
                                        user_type
                                    )
                                    st.markdown(nws_ai_text)
                                except Exception as e:
                                    st.info(f"AI explanation abhi available nahi hai ({e})")

                        # AI Risk Meter
                        st.divider()
                        st.subheader("🛡️ AI Risk Meter")

                        volatility = abs(latest["Close"] - latest["Open"]) / latest["Close"] * 100

                        if volatility < 2:
                            risk_level = "🟢 Low Risk"
                            risk_score = 85
                        elif volatility < 5:
                            risk_level = "🟡 Medium Risk"
                            risk_score = 65
                        else:
                            risk_level = "🔴 High Risk"
                            risk_score = 35

                        max_downside = round(latest["Close"] * 0.90, 2)

                        st.metric("Risk Score", f"{risk_score}/100")
                        st.info(f"Risk Level: {risk_level}")
                        st.warning(f"Maximum Downside Estimate: ₹{max_downside}")

                        if risk_score >= 80:
                            st.success("Capital Protection Strong Hai.")
                        elif risk_score >= 60:
                            st.warning("Moderate Risk Present Hai.")
                        else:
                            st.error("Risk High Hai, Position Size Kam Rakho.")

                            # Hedge Fund Conviction Dashboard
                        st.divider()
                        st.subheader("🏦 Hedge Fund Conviction Dashboard")

                        conviction_score = int((80 + fii_score + 70 + risk_score) / 4)

                        if conviction_score >= 80:
                            conviction = "🟢 HIGH CONVICTION"
                            action = "Accumulation candidate"
                        elif conviction_score >= 60:
                            conviction = "🟡 MEDIUM CONVICTION"
                            action = "Watchlist / partial position"
                        else:
                            conviction = "🔴 LOW CONVICTION"
                            action = "Avoid / wait for confirmation"

                        c1, c2 = st.columns(2)

                        with c1:
                            st.metric("Conviction Score", f"{conviction_score}/100")
                            st.info(conviction)

                        with c2:
                            st.metric("Suggested Action", action)
                            st.warning("Use position sizing. Not financial advice.")

                        st.caption(
                            "Based on AI Thesis, Institutional Flow, Multi-Timeframe Trend and Risk Meter."
                        )

                        # Trust Engine
                        st.divider()
                        st.markdown("""
                            <div class="bf-section">
                                <div class="bf-section-line"></div>
                                <div class="bf-section-title">Trust Engine</div>
                            </div>
                            """, unsafe_allow_html=True)

                        rsi_sc, macd_sc, risk_sc, trust = get_trust_scores(
                            rsi_val, latest['MACD'], latest['MACD_Signal'], week52_pos)

                        tc1, tc2, tc3 = st.columns(3)
                        #with tc1:
                        #   st.metric("📊 Technical", f"{rsi_sc:.0f}/100",
                        #            "Strong ✅" if rsi_sc > 60 else "Weak ⚠️")
                        #with tc2:
                        #   st.metric("⚡ Momentum", f"{macd_sc:.0f}/100",
                                    #  "Bullish ✅" if macd_sc > 50 else "Bearish ⚠️")
                        #with tc3:
                        #   st.metric("🛡️ Risk", f"{risk_sc:.0f}/100",
                        #            "Safe ✅" if risk_sc > 50 else "Risky ⚠️")

                        st.metric("Technical", f"{rsi_sc:.0f}/100")
                        st.metric("Momentum", f"{macd_sc:.0f}/100")
                        st.metric("Risk", f"{risk_sc:.0f}/100")

                        st.progress(int(trust) / 100)
                        if trust > 65:
                            st.success(f"✅ {trust:.0f}/100 — High Confidence")
                        elif trust > 40:
                            st.warning(f"⚠️ {trust:.0f}/100 — Medium Confidence")
                        else:
                            st.error(f"🚨 {trust:.0f}/100 — Low Confidence — Avoid!")

                        st.markdown("### 💡 WHY Ye Signal Diya?")
                        st.markdown(f"""
    **RSI {rsi_val:.1f}** → {"🔴 Overbought — Caution!" if rsi_val > 70 else "🟢 Oversold — Opportunity!" if rsi_val < 30 else "🟡 Neutral — Wait for confirmation"}

    **52W Position: {week52_pos:.0f}%** → {"🚨 FOMO Zone — Near 52W High!" if week52_pos > 80 else "✅ Safe Zone" if week52_pos < 40 else "⚠️ Middle Zone — Caution"}

    **MACD** → {"✅ Bullish Momentum" if latest['MACD'] > latest['MACD_Signal'] else "⚠️ Bearish Momentum"}

    **Max Downside Risk:** ₹{latest['Close'] * 0.05:.0f} — ₹{latest['Close'] * 0.10:.0f}

    **Suggested Stop Loss:** ₹{latest['Close'] * 0.95:.0f}
                        """)

                    except Exception as e:
                        st.error(f"❌ AI Error: {e}")

                        st.divider()
                    st.subheader("🎯 AI Entry / Exit Zone")

                    entry_zone_low = latest["Close"] * 0.98
                    entry_zone_high = latest["Close"] * 1.00

                    breakout_level = latest["Close"] * 1.02

                    target1 = latest["Close"] * 1.10
                    target2 = latest["Close"] * 1.20

                    stop_loss = latest["Close"] * 0.95

                    st.success(
                        f"""
                    🎯 Ideal Buy Zone: ₹{entry_zone_low:.2f} - ₹{entry_zone_high:.2f}

                    🚀 Breakout Level: ₹{breakout_level:.2f}

                    💰 Target 1: ₹{target1:.2f}

                    💰 Target 2: ₹{target2:.2f}

                    🛑 Stop Loss: ₹{stop_loss:.2f}
                    """
                    )

                    risk_reward = (target1 - latest["Close"]) / (latest["Close"] - stop_loss)

                    st.info(f"Risk Reward Ratio: 1 : {risk_reward:.1f}")

                    st.divider()
                    st.subheader("📊 Portfolio Health Score")

                    portfolio_score = int(
                        (trust * 0.4) +
                        (70 * 0.3) +
                        (fii_score * 0.3)
                    )

                    st.metric("Portfolio Health", f"{portfolio_score}/100")

                    if portfolio_score >= 80:
                        st.success("🟢 Institutional Grade Portfolio")
                    elif portfolio_score >= 60:
                        st.warning("🟡 Moderate Quality Portfolio")
                    else:
                        st.error("🔴 Weak Portfolio Structure")

                        st.markdown("---")
                        st.subheader("🏆 Final AI Grade")

                        if portfolio_score >= 90:
                            st.success("🏆 Grade A+ | Elite Portfolio")
                        elif portfolio_score >= 80:
                            st.success("🥇 Grade A | Strong Portfolio")
                        elif portfolio_score >= 70:
                            st.info("🥈 Grade B | Good Portfolio")
                        elif portfolio_score >= 60:
                            st.warning("🥉 Grade C | Average Portfolio")
                        else:
                            st.error("⚠️ Grade D | High Risk Portfolio")

            st.markdown("---")
            st.subheader("🩺 AI Portfolio Doctor")

            strengths = []
            weaknesses = []

            if rsi_val > 50:
                strengths.append("Momentum Strong")

            if latest["MACD"] > latest["MACD_Signal"]:
                strengths.append("Bullish MACD")

            if week52_pos < 80:
                strengths.append("Safe Distance From 52W High")

            if rsi_val < 40:
                weaknesses.append("Weak Momentum")

            if week52_pos > 90:
                weaknesses.append("Near 52W High Risk")

            st.success("💪 Strengths: " + ", ".join(strengths))

            if weaknesses:
                st.warning("⚠ Weaknesses: " + ", ".join(weaknesses))
            else:
                st.success("✅ No Major Weakness Found")

            if portfolio_score >= 80:
                st.success("🚀 Action: Strong Hold")
            elif portfolio_score >= 60:
                st.info("👀 Action: Watchlist / Partial Position")
            else:
                st.error("🛑 Action: Avoid")

            st.markdown("---")
            st.subheader("🏆 Warren Buffett Quality Score")

            buffett_score = 0

            if week52_pos < 80:
                buffett_score += 25

            if rsi_val > 50:
                buffett_score += 25

            if latest["MACD"] > latest["MACD_Signal"]:
                buffett_score += 25

            if health_score >= 70:
                buffett_score += 25

            st.metric("Buffett Score", f"{buffett_score}/100")

            if buffett_score >= 80:
                st.success("🟢 Grade A | Buffett Style Compounder")
            elif buffett_score >= 60:
                st.info("🔵 Grade B | Good Long Term Candidate")
            elif buffett_score >= 40:
                st.warning("🟡 Grade C | Average Business Quality")
            else:
                st.error("🔴 Grade D | Not Buffett Style")

                # CSV Export
                with st.expander("📊 Raw Data + Export"):
                    cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'SMA_50']
                    display_df = df.tail(30)[cols].round(2)
                    st.dataframe(display_df, use_container_width="stretch")
                    csv = display_df.to_csv().encode('utf-8')
                    st.download_button(
                        "📥 CSV Download Karo",
                        csv,
                        f"{sym}_data.csv",
                        "text/csv"
                    )

# ─────────────────────────────────────────
# TAB 2: MULTI SCANNER
# ─────────────────────────────────────────
with tab2:
    
    st.markdown("### 🔭 Multi-Stock Scanner")
    st.caption("Ek saath multiple stocks analyze karo — Heap-based ranking")

    scanner_input = st.text_input(
        "Stocks daalo (comma separated)",
        value="RELIANCE,TCS,INFY,SBIN,MARUTI,ADANIENT,HDFCBANK,WIPRO",
        key="multi"
    )

    sc1, sc2 = st.columns(2)
    with sc1:
        scan_period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=1, key="sp")
    with sc2:
        top_n = st.selectbox("Top N stocks", [3, 5, 10], index=1, key="tn")

    with st.expander("🔍 Fundamental Filters (optional)"):

        st.caption(
            "Defaults wide-open hain — koi filter apply nahi hoga jab tak tum values change na karo."
        )

        f1, f2, f3 = st.columns(3)

        with f1:
            max_pe_filter = st.number_input(
                "Max P/E", min_value=0.0, value=100.0, step=1.0, key="filter_max_pe"
            )
            min_roe_filter = st.number_input(
                "Min ROE (%)", min_value=-50.0, value=-50.0, step=1.0, key="filter_min_roe"
            )

        with f2:
            max_debt_equity_filter = st.number_input(
                "Max Debt/Equity", min_value=0.0, value=999.0, step=0.1, key="filter_max_de"
            )
            min_revenue_growth_filter = st.number_input(
                "Min Revenue Growth (%)", min_value=-100.0, value=-100.0, step=1.0, key="filter_min_rg"
            )

        with f3:
            min_market_cap_filter = st.number_input(
                "Min Market Cap (₹ Cr)", min_value=0.0, value=0.0, step=100.0, key="filter_min_mcap"
            )

        st.caption(
            "⚠️ Fundamental data fetch karne se scan thoda slow hoga (extra API calls per stock)."
        )

    if st.button("🚀 Scanner Chalao", key="scan_btn"):

        if not scanner_input:
            st.warning("Stocks daalo!")
        else:
            stocks_list = [s.strip().upper() for s in scanner_input.split(",") if s.strip()]
            st.markdown(f"**Scanning {len(stocks_list)} stocks...**")

            results = []
            heap_opportunities = []  # Min heap for top opportunities
            heap_risky = []          # Max heap for risky stocks

            progress_bar = st.progress(0)
            status = st.empty()

            for idx, stk in enumerate(stocks_list):
                try:
                    status.text(f"⏳ Analyzing {stk}... ({idx+1}/{len(stocks_list)})")
                    sym_s = stk if stk.endswith('.NS') else stk + '.NS'
                    df_s = get_stock_data(sym_s, scan_period)

                    if df_s.empty:
                        continue

                    delta = df_s['Close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(14).mean()
                    loss = -delta.where(delta < 0, 0).rolling(14).mean()
                    rs = gain / loss
                    df_s['RSI'] = 100 - (100 / (1 + rs))

                    ema12 = df_s['Close'].ewm(span=12, adjust=False).mean()
                    ema26 = df_s['Close'].ewm(span=26, adjust=False).mean()
                    macd_line = ema12 - ema26
                    signal_line = macd_line.ewm(span=9, adjust=False).mean()
                    macd_s = pd.DataFrame({"MACD_12_26_9": macd_line, "MACDs_12_26_9": signal_line})
                    df_s['MACD'] = macd_s['MACD_12_26_9']
                    df_s['MACD_Signal'] = macd_s['MACDs_12_26_9']

                    lat = df_s.iloc[-1]
                    prev_s = df_s.iloc[-2]
                    chg = ((lat['Close'] - prev_s['Close']) / prev_s['Close']) * 100
                    rsi = lat['RSI']
                    w52 = ((lat['Close'] - df_s['Close'].min()) /
                           (df_s['Close'].max() - df_s['Close'].min()) * 100)

                    rsi_sc, macd_sc, risk_sc, trust = get_trust_scores(
                        rsi, lat['MACD'], lat['MACD_Signal'], w52)

                    signal = get_signal(rsi, w52, trust)

                                        # ─────────────────────────────────────────
                    # FUNDAMENTAL DATA (P/E, ROE, Debt/Equity, Revenue Growth, Market Cap)
                    # ─────────────────────────────────────────

                    fund_pe = None
                    fund_roe = None
                    fund_debt_equity = None
                    fund_revenue_growth = None
                    fund_market_cap_cr = None

                    try:
                        fund_ticker = yf.Ticker(sym_s)
                        finfo = fund_ticker.info

                        fund_pe = finfo.get("trailingPE")

                        raw_roe = finfo.get("returnOnEquity")
                        if raw_roe is not None:
                            fund_roe = raw_roe * 100

                        raw_de = finfo.get("debtToEquity")
                        if raw_de is not None:
                            # yfinance ka debtToEquity already percentage-scale hota hai (jaise 45.2 = 0.452 ratio)
                            fund_debt_equity = raw_de / 100

                        raw_mcap = finfo.get("marketCap")
                        if raw_mcap is not None:
                            fund_market_cap_cr = raw_mcap / 1e7

                        fund_financials = fund_ticker.financials

                        if fund_financials is not None and not fund_financials.empty:
                            if "Total Revenue" in fund_financials.index:
                                rev_row = fund_financials.loc["Total Revenue"]
                                if len(rev_row) >= 2:
                                    rev_latest_s = rev_row.iloc[0]
                                    rev_prev_s = rev_row.iloc[1]
                                    if pd.notna(rev_latest_s) and pd.notna(rev_prev_s) and rev_prev_s != 0:
                                        fund_revenue_growth = ((rev_latest_s - rev_prev_s) / abs(rev_prev_s)) * 100

                    except Exception:
                        pass

                    result = {
                        "Stock": stk,
                        "Price": f"₹{lat['Close']:.2f}",
                        "Change": f"{chg:+.2f}%",
                        "RSI": round(rsi, 1),
                        "52W%": f"{w52:.0f}%",
                        "Trust": round(trust, 0),
                        "Signal": signal,
                        "Stop Loss": f"₹{lat['Close'] * 0.95:.2f}",
                        "P/E": round(fund_pe, 2) if fund_pe is not None else None,
                        "ROE (%)": round(fund_roe, 2) if fund_roe is not None else None,
                        "Debt/Equity": round(fund_debt_equity, 2) if fund_debt_equity is not None else None,
                        "Revenue Growth (%)": round(fund_revenue_growth, 2) if fund_revenue_growth is not None else None,
                        "Market Cap (₹ Cr)": round(fund_market_cap_cr, 0) if fund_market_cap_cr is not None else None,
                    }
                    results.append(result)

                    # Heap operations — DSA!
                    # Max heap for opportunities (negate for max)
                    heapq.heappush(heap_opportunities, (-trust, stk, signal))
                    # Min heap for risky (low trust = risky)
                    heapq.heappush(heap_risky, (trust, stk, signal))

                except Exception:
                    pass

                progress_bar.progress((idx + 1) / len(stocks_list))

            status.empty()
            progress_bar.empty()

            if results:
                # Top opportunities from heap
                st.divider()
                st.markdown("###  Top Picks & Risk Alerts")

                top_opp = []
                temp_heap = heap_opportunities.copy()
                for _ in range(min(top_n, len(temp_heap))):
                    if temp_heap:
                        neg_trust, stk, sig = heapq.heappop(temp_heap)
                        top_opp.append((stk, -neg_trust, sig))

                top_risk = []
                temp_heap2 = heap_risky.copy()
                for _ in range(min(3, len(temp_heap2))):
                    if temp_heap2:
                        trust_val, stk, sig = heapq.heappop(temp_heap2)
                        top_risk.append((stk, trust_val, sig))

                col_opp, col_risk = st.columns(2)

                with col_opp:
                    st.markdown("#### 🟢 Top Opportunities")
                    for i, (stk, trust, sig) in enumerate(top_opp, 1):
                        st.success(f"#{i} **{stk}** — Trust: {trust:.0f}/100 | {sig}")

                with col_risk:
                    st.markdown("#### 🔴 Avoid These")
                    for i, (stk, trust, sig) in enumerate(top_risk, 1):
                        st.error(f"#{i} **{stk}** — Trust: {trust:.0f}/100 | {sig}")

                # Full results table
                st.divider()
                st.markdown("### 📊 Complete Scan Results")

                df_results = pd.DataFrame(results)

                # Color the trust column
                def color_trust(val):
                    if val >= 65:
                        return 'background-color: #1a3a1a; color: #00d68f'
                    elif val >= 40:
                        return 'background-color: #3a3a1a; color: #f0a500'
                    else:
                        return 'background-color: #3a1a1a; color: #ff4d6d'

                styled = df_results.style.map(color_trust, subset=['Trust'])
                st.dataframe(styled, use_container_width="stretch", hide_index=True)

                                # Export
                csv_scan = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Scan Results CSV Download",
                    csv_scan,
                    "scanner_results.csv",
                    "text/csv"
                )

                st.caption(f"✅ {len(results)}/{len(stocks_list)} stocks successfully scanned")

                # ─────────────────────────────────────────
                # FUNDAMENTAL SCREENER — FILTERED RESULTS
                # ─────────────────────────────────────────

                st.divider()
                st.markdown("### 🧮 Fundamental Screener — Filtered Results")

                filter_mask = pd.Series(True, index=df_results.index)

                if "P/E" in df_results.columns:
                    filter_mask &= df_results["P/E"].apply(
                        lambda x: x is not None and x <= max_pe_filter
                    )

                if "ROE (%)" in df_results.columns:
                    filter_mask &= df_results["ROE (%)"].apply(
                        lambda x: x is not None and x >= min_roe_filter
                    )

                if "Debt/Equity" in df_results.columns:
                    filter_mask &= df_results["Debt/Equity"].apply(
                        lambda x: x is not None and x <= max_debt_equity_filter
                    )

                if "Revenue Growth (%)" in df_results.columns:
                    filter_mask &= df_results["Revenue Growth (%)"].apply(
                        lambda x: x is not None and x >= min_revenue_growth_filter
                    )

                if "Market Cap (₹ Cr)" in df_results.columns:
                    filter_mask &= df_results["Market Cap (₹ Cr)"].apply(
                        lambda x: x is not None and x >= min_market_cap_filter
                    )

                filtered_df = df_results[filter_mask]

                if filtered_df.empty:
                    st.warning(
                        "⚠️ Koi stock in filters ko match nahi kar raha (ya fundamental data available nahi tha kuch stocks ke liye). "
                        "Filters relax karo ya expander me check karo."
                    )
                else:
                    st.success(f"✅ {len(filtered_df)}/{len(df_results)} stocks in filters pe match kar rahe hain.")

                    st.dataframe(
                        filtered_df,
                        use_container_width="stretch",
                        hide_index=True
                    )

                    csv_filtered = filtered_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Filtered Results CSV Download",
                        csv_filtered,
                        "filtered_screener_results.csv",
                        "text/csv",
                        key="filtered_csv_download"
                    )

        # ====================================
# TAB 4 : AI COMPARISON
# ====================================

with tab3:

    st.markdown("### 🤖 AI Stock Comparison")
    st.caption("2 stocks ko compare karo")

    col1, col2 = st.columns(2)

    with col1:
        stock1 = st.text_input("Stock 1", value="RELIANCE", key="cmp1")

    with col2:
        stock2 = st.text_input("Stock 2", value="TCS", key="cmp2")

    if st.button("⚔️ Compare Stocks", key="compare_btn"):

        s1 = stock1.upper().strip()
        s2 = stock2.upper().strip()

        sym1 = s1 if s1.endswith(".NS") else s1 + ".NS"
        sym2 = s2 if s2.endswith(".NS") else s2 + ".NS"

        try:
            df1 = yf.Ticker(sym1).history(period="1y")
            df2 = yf.Ticker(sym2).history(period="1y")

            if df1.empty or df2.empty:
                st.error("Stock data nahi mila.")
                st.stop()

            price1 = df1["Close"].iloc[-1]
            price2 = df2["Close"].iloc[-1]

           # rsi1 = ta.rsi(df1["Close"], length=14).dropna().iloc[-1]
            #rsi2 = ta.rsi(df2["Close"], length=14).dropna().iloc[-1]

            #macd1 = ta.macd(df1["Close"])
            #macd2 = ta.macd(df2["Close"])
            rsi1= 50 
            rsi2 = 50

            m1 = 0
            ms1 = 0

            m2 = 0
            ms2 = 0

            #m1 = macd1["MACD_12_26_9"].dropna().iloc[-1]
            #ms1 = macd1["MACDs_12_26_9"].dropna().iloc[-1]
            #m2 = macd2["MACD_12_26_9"].dropna().iloc[-1]
            #ms2 = macd2["MACDs_12_26_9"].dropna().iloc[-1]

            w52_1 = ((price1 - df1["Close"].min()) / (df1["Close"].max() - df1["Close"].min())) * 100
            w52_2 = ((price2 - df2["Close"].min()) / (df2["Close"].max() - df2["Close"].min())) * 100

            _, _, _, trust1 = get_trust_scores(rsi1, m1, ms1, w52_1)
            _, _, _, trust2 = get_trust_scores(rsi2, m2, ms2, w52_2)

            sent1, bonus1 = get_news_sentiment(s1)
            sent2, bonus2 = get_news_sentiment(s2)

            final1 = trust1 + bonus1
            final2 = trust2 + bonus2

            compare_df = pd.DataFrame({
                "Metric": ["Price", "RSI", "52W Position", "Trust Score", "Sentiment", "Final Score"],
                s1: [round(price1, 2), round(rsi1, 1), f"{w52_1:.0f}%", round(trust1, 0), sent1, round(final1, 0)],
                s2: [round(price2, 2), round(rsi2, 1), f"{w52_2:.0f}%", round(trust2, 0), sent2, round(final2, 0)]
            })

            st.dataframe(compare_df, use_container_width="stretch")

            winner = s1 if final1 > final2 else s2
            winner_score = final1 if final1 > final2 else final2

            st.success(f"🏆 Better Pick: {winner} | Final Score: {winner_score:.0f}/100")

            st.markdown("### ⚔️ Multi-Stock Battle")

            loser = s2 if winner == s1 else s1
            loser_score = final2 if winner == s1 else final1
            score_gap = abs(final1 - final2)

            st.success(f"🥇 Winner: {winner} | Score: {winner_score:.0f}/100")
            st.warning(f"🥈 Runner Up: {loser} | Score: {loser_score:.0f}/100")
            st.info(f"📊 Score Difference: {score_gap:.0f} points")


        except Exception as e:
            st.error(f"Error: {e}")

# -----------------------------------
# PORTFOLIO TRACKER UPGRADE
# -----------------------------------

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Portfolio Tracker")

df_port = pd.read_csv(PORTFOLIO_FILE)

if len(df_port) > 0:
    portfolio_rows = []
    total_investment = 0
    current_value = 0

    for _, row in df_port.iterrows():
        symbol = str(row["Symbol"]).upper()
        qty = float(row["Quantity"])
        buy_price = float(row["BuyPrice"])

        try:
            yf_symbol = symbol if symbol.endswith(".NS") else symbol + ".NS"
            hist = yf.Ticker(yf_symbol).history(period="5d")

            if not hist.empty:
                current_price = hist["Close"].iloc[-1]
                investment = qty * buy_price
                value = qty * current_price
                pnl = value - investment
                pnl_pct = (pnl / investment) * 100 if investment > 0 else 0

                total_investment += investment
                current_value += value

                portfolio_rows.append({
                    "Stock": symbol,
                    "Qty": qty,
                    "Buy": round(buy_price, 2),
                    "Current": round(current_price, 2),
                    "Investment": round(investment, 2),
                    "Value": round(value, 2),
                    "P/L ₹": round(pnl, 2),
                    "P/L %": round(pnl_pct, 2)
                })

        except Exception:
            pass

    if portfolio_rows:
        portfolio_df = pd.DataFrame(portfolio_rows)

        total_pnl = current_value - total_investment
        total_pnl_pct = (total_pnl / total_investment) * 100 if total_investment > 0 else 0

        st.sidebar.metric("Total Investment", f"₹{total_investment:,.2f}")
        st.sidebar.metric("Current Value", f"₹{current_value:,.2f}")
        st.sidebar.metric("Profit / Loss", f"₹{total_pnl:,.2f}", f"{total_pnl_pct:.2f}%")

        st.sidebar.dataframe(portfolio_df, use_container_width="stretch")
    else:
        st.sidebar.info("Portfolio empty hai.")
else:
    st.sidebar.info("No stocks added.")

            # -----------------------------------
# WATCHLIST SCANNER
# -----------------------------------

st.sidebar.markdown("---")
st.sidebar.markdown("### 👀 Watchlist Scanner")

watch_df = pd.read_csv(WATCHLIST_FILE)

signals = []

for stock in watch_df["Symbol"]:
    try:
        symbol = str(stock).upper()

        yf_symbol = symbol if symbol.endswith(".NS") else symbol + ".NS"

        df = yf.Ticker(yf_symbol).history(period="6mo")

        if len(df) < 50:
            continue

        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        current = df["Close"].iloc[-1]
        sma50 = df["Close"].rolling(50).mean().iloc[-1]
        rsi = df["RSI"].iloc[-1]

        signal = "HOLD"

        score = 50

        if current > sma50:
            score += 20

        if rsi < 70:
            score += 15

        if rsi > 40:
            score += 15

        if current > sma50 and rsi < 70:
            signal = "BUY"

        elif current < sma50 and rsi > 30:
            signal = "AVOID"

        signals.append({
    "Stock": symbol,
    "Price": round(current, 2),
    "RSI": round(rsi, 1),
    "Signal": signal,
    "Score": score
})

    except:
        pass

if signals:

    watchlist_table = pd.DataFrame(signals)

    watchlist_table = watchlist_table.sort_values(
        by="Score",
        ascending=False
    )

    # --------------------------------
# WATCHLIST BATTLE ROYALE
# --------------------------------

watchlist_table = []

if len(watchlist_table) >= 3:

    top1 = watchlist_table.iloc[0]
    top2 = watchlist_table.iloc[1]
    top3 = watchlist_table.iloc[2]

    st.sidebar.markdown("### 🏆 Watchlist Battle Royale")

    st.sidebar.success(
        f"🥇 #1 {top1['Stock']} | Score: {top1['Score']}"
    )

    st.sidebar.info(
        f"🥈 #2 {top2['Stock']} | Score: {top2['Score']}"
    )

    st.sidebar.warning(
        f"🥉 #3 {top3['Stock']} | Score: {top3['Score']}"
    )

    st.sidebar.caption(
        "AI ranked opportunities from your watchlist."
    )

    st.sidebar.dataframe(
        watchlist_table,
        use_container_width="stretch"
    )

    top_pick = watchlist_table.iloc[0]

    oversold_stocks = watchlist_table[watchlist_table["RSI"] < 40]

st.sidebar.markdown("### 🚨 Oversold Alerts")

import pandas as pd

oversold_stocks = pd.DataFrame()

if not oversold_stocks.empty:
    for _, row in oversold_stocks.iterrows():
        st.sidebar.warning(
            f"🚨 {row['Stock']} oversold zone me hai | RSI: {row['RSI']}"
        )
else:
    st.sidebar.info("Abhi koi oversold opportunity nahi mili.")

st.sidebar.markdown("### 🏆 Top Watchlist Pick")

import pandas as pd

if "top_pick" not in locals():
    top_pick = {"Score": 0, "Stock": "N/A"}


if top_pick["Score"] >= 80:
    st.sidebar.success(
        f"🟢 {top_pick['Stock']} strongest opportunity hai | Score: {top_pick['Score']}"
    )
elif top_pick["Score"] >= 60:
    st.sidebar.info(
        f"🟡 {top_pick['Stock']} decent watchlist pick hai | Score: {top_pick['Score']}"
    )
else:
    st.sidebar.warning(
        f"⚠️ Abhi strong opportunity nahi hai | Best: {top_pick['Stock']} | Score: {top_pick['Score']}"
    )

# -----------------------------------
# AI PORTFOLIO ADVISOR
# -----------------------------------

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 AI Portfolio Advisor")

try:
    if "portfolio_df" in locals() and not portfolio_df.empty:
        best_stock = portfolio_df.sort_values("P/L %", ascending=False).iloc[0]
        worst_stock = portfolio_df.sort_values("P/L %", ascending=True).iloc[0]

        st.sidebar.success(
            f"🏆 Best: {best_stock['Stock']} ({best_stock['P/L %']}%)"
        )

        st.sidebar.error(
            f"⚠️ Weak: {worst_stock['Stock']} ({worst_stock['P/L %']}%)"
        )

        # Portfolio Health Score

        health_score = 0

        if len(portfolio_df) >= 3:
            health_score += 30

        if worst_stock["P/L %"] > -10:
            health_score += 30

        if best_stock["P/L %"] > 0:
            health_score += 40

        st.sidebar.markdown("### ❤️ Portfolio Health")

        if health_score >= 80:
            st.sidebar.success(f"{health_score}/100 Excellent")
        elif health_score >= 60:
            st.sidebar.info(f"{health_score}/100 Good")
        elif health_score >= 40:
            st.sidebar.warning(f"{health_score}/100 Average")
        else:
            st.sidebar.error(f"{health_score}/100 Weak")

        if len(portfolio_df) < 3:
            st.sidebar.warning("Diversification low hai. 3-5 stocks rakho.")
        else:
            st.sidebar.info("Portfolio diversification decent hai.")

        if worst_stock["P/L %"] < -10:
            st.sidebar.warning(
                f"{worst_stock['Stock']} me loss high hai. Review karo."
            )

        if best_stock["P/L %"] > 10:
            st.sidebar.success(
                f"{best_stock['Stock']} strong performer hai. Hold/Trail SL."
            )

            # Portfolio Rebalancing Advisor
        st.sidebar.markdown("### ⚖️ Rebalancing Advisor")

        if worst_stock["P/L %"] < -10:
            st.sidebar.warning(
                f"📉 Reduce: {worst_stock['Stock']} exposure by 10-15%"
            )

        if best_stock["P/L %"] > 0:
            st.sidebar.success(
                f"📈 Increase/Hold: {best_stock['Stock']} strong hai"
            )

        if len(portfolio_df) < 5:
            st.sidebar.info(
                "🏦 Add 1-2 new sectors: Banking, FMCG, Auto ya Pharma"
            )

        st.sidebar.caption("Goal: loss control + diversification improve karna.")

    else:
        st.sidebar.info("Portfolio advisor ke liye stocks add karo.")

except Exception as e:
    st.sidebar.error(f"Advisor Error: {e}")

    # -----------------------------------
# MARKET MOOD INDEX
# -----------------------------------

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Market Mood Index")

try:
    nifty = yf.Ticker("^NSEI").history(period="5d")

    if not nifty.empty:
        nifty_change = (
            (nifty["Close"].iloc[-1] - nifty["Close"].iloc[-2])
            / nifty["Close"].iloc[-2]
        ) * 100

        if nifty_change > 0.5:
            st.sidebar.success(f"🟢 Bullish Market | NIFTY: {nifty_change:.2f}%")
        elif nifty_change < -0.5:
            st.sidebar.error(f"🔴 Bearish Market | NIFTY: {nifty_change:.2f}%")
        else:
            st.sidebar.warning(f"🟡 Sideways Market | NIFTY: {nifty_change:.2f}%")

except Exception as e:
    st.sidebar.info("Market mood data unavailable.")

    # -----------------------------------
# EXPORT REPORT
# -----------------------------------

st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 Export Report")

if "best_stock" not in locals():
    best_stock = {"Stock": "N/A", "P/L %": 0}

if "worst_stock" not in locals():
    worst_stock = {"Stock": "N/A", "P/L %": 0}
    
if "best_stock" not in locals():
    best_stock = {"Stock": "N/A", "P/L %": 0}

if "worst_stock" not in locals():
    worst_stock = {"Stock": "N/A", "P/L %": 0}

if "portfolio_df" not in locals():
    portfolio_df = []

report = f"""
BHARATFINAI REPORT

Portfolio Health Score : {health_score}/100

Best Stock : {best_stock['Stock']}
Worst Stock : {worst_stock['Stock']}

Diversification Status :
{"Good" if len(portfolio_df)>=3 else "Low"}

Generated by BharatFinAI
"""

st.sidebar.download_button(
    label="📥 Download Report",
    data=report,
    file_name="bharatfinai_report.txt",
    mime="text/plain"
)
with tab4:
    st.markdown("### 📊 Phase 4 — Quant Systems")
    q_symbol = st.text_input("Backtest Stock", value="RELIANCE", key="q_stock")
    q_period = st.selectbox("Backtest Period", ["6mo", "1y", "2y", "5y"], index=1)

    if st.button("Run Backtest"):
        sym = q_symbol.upper()
        yf_symbol = sym if sym.endswith(".NS") else sym + ".NS"
        df = yf.Ticker(yf_symbol).history(period=q_period)

        if df.empty:
            st.error("Data nahi mila")
        else:
            df["SMA20"] = df["Close"].rolling(20).mean()
            df["SMA50"] = df["Close"].rolling(50).mean()

            df["Signal"] = 0
            df.loc[df["SMA20"] > df["SMA50"], "Signal"] = 1
            df.loc[df["SMA20"] <= df["SMA50"], "Signal"] = 0

            df["Return"] = df["Close"].pct_change()
            df["Strategy"] = df["Signal"].shift(1) * df["Return"]

            # Trade Statistics

            trade_changes = df["Signal"].diff().fillna(0)

            total_trades = int((trade_changes != 0).sum())

            winning_trades = int(
                ((df["Strategy"] > 0) & (trade_changes != 0)).sum()
            )

            win_rate = (
                winning_trades / total_trades * 100
                if total_trades > 0
                else 0
            )

            total_return = (1 + df["Strategy"].fillna(0)).prod() - 1
            days = (df.index[-1] - df.index[0]).days

            cagr = (
                ((1 + total_return) ** (365 / days) - 1) * 100
                if days > 0
                else 0
            )

            rolling_max = df["Close"].cummax()
            drawdown = (df["Close"] - rolling_max) / rolling_max

            max_drawdown = abs(drawdown.min()) * 100

            calmar = (
                cagr / max_drawdown
                if max_drawdown != 0
                else 0
            )

            buy_hold = (
                (df["Close"].dropna().iloc[-1] /
                df["Close"].dropna().iloc[0]) - 1
            )

            nifty = yf.Ticker("^NSEI").history(period=q_period)

            if not nifty.empty:
                nifty = nifty.reindex(df.index).ffill()

                nifty_return = (
                    nifty["Close"].dropna().iloc[-1] /
                    nifty["Close"].dropna().iloc[0]
                ) - 1
            else:
                nifty_return = 0

            sharpe = (
                df["Strategy"].mean() / df["Strategy"].std() * (252 ** 0.5)
                if df["Strategy"].std() != 0 else 0
            )

            downside = df[df["Strategy"] < 0]["Strategy"].std()

            sortino = (
                df["Strategy"].mean() / downside * (252 ** 0.5)
                if downside != 0 and not pd.isna(downside)
                else 0
            )

            # Value at Risk (95%)
            var_95 = np.percentile(df["Strategy"].dropna(), 5) * 100

            # Conditional VaR (Expected Shortfall)
            cvar_95 = (
                df["Strategy"][df["Strategy"] <= np.percentile(df["Strategy"].dropna(), 5)]
                .mean() * 100
            )

            c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = st.columns(12)
            c1.metric(
                "Strategy Return",
                f"{total_return*100:.2f}%"
            )

            c2.metric(
                "Buy & Hold",
                f"{buy_hold*100:.2f}%"
            )

            c3.metric(
                "Sharpe Ratio",
                f"{sharpe:.2f}"
            )

            c4.metric(
                "Win Rate",
                f"{win_rate:.1f}%"
            )

            c5.metric(
                "Trades",
                total_trades
            )

            c6.metric(
                "CAGR",
                f"{cagr:.2f}%"
            )

            c7.metric(
                "Sortino",
                f"{sortino:.2f}"
            )

            c8.metric(
                "Calmar",
                f"{calmar:.2f}"
            )

            rolling_max = df["Close"].cummax()

            drawdown = (
                (df["Close"] - rolling_max)
                / rolling_max
            )

            max_drawdown = abs(drawdown.min()) * 100

            c9.metric(
                "NIFTY50",
                f"{nifty_return*100:.2f}%"
            )

            c10.metric(
                "Max DD",
                f"{max_drawdown:.2f}%"
            )

            c11.metric(
                "VaR 95%",
                f"{var_95:.2f}%"
            )

            c12.metric(
                "CVaR 95%",
                f"{cvar_95:.2f}%"
            )

            df["Equity Curve"] = (1 + df["Strategy"].fillna(0)).cumprod()
            df["Buy Hold Curve"] = df["Close"] / df["Close"].iloc[0]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df["Equity Curve"], name="Strategy"))
            fig.add_trace(go.Scatter(x=df.index, y=df["Buy Hold Curve"], name="Buy & Hold"))
            if not nifty.empty:
                    nifty["NIFTY Curve"] = nifty["Close"] / nifty["Close"].dropna().iloc[0]

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=nifty["NIFTY Curve"],
                    name="NIFTY50"
                )
            )
            fig.update_layout(template="plotly_dark", title="Backtest Equity Curve")
            st.plotly_chart(fig, use_container_width="stretch")

            st.dataframe(df[["Close", "SMA20", "SMA50", "Signal", "Return", "Strategy"]].tail(20))
with tab5:

            st.markdown("## 🎲 Monte Carlo Simulation")

            mc_symbol = st.text_input(
                "Stock Symbol",
                value="RELIANCE",
                key="mc_stock"
            )

            mc_period = st.selectbox(
                "Period",
                ["6mo", "1y", "2y", "5y"],
                index=1,
                key="mc_period"
            )

            simulations = st.slider(
                "Simulations",
                100,
                5000,
                1000,
                step=100
            )

            if st.button("Run Monte Carlo"):

                import numpy as np

                symbol = mc_symbol.upper()
                yf_symbol = symbol if symbol.endswith(".NS") else symbol + ".NS"

                df = yf.Ticker(yf_symbol).history(period=mc_period)

                if df.empty:
                    st.error("Data nahi mila")
                else:

                    returns = df["Close"].pct_change().dropna()

                    mu = returns.mean()
                    sigma = returns.std()

                    start_price = df["Close"].iloc[-1]

                    future_days = 252

                    paths = np.zeros((future_days, simulations))

                    for i in range(simulations):

                        prices = [start_price]

                        for _ in range(future_days):
                            shock = np.random.normal(mu, sigma)
                            prices.append(prices[-1] * (1 + shock))

                        paths[:, i] = prices[1:]

                    fig = go.Figure()

                    for i in range(min(100, simulations)):
                        fig.add_trace(
                            go.Scatter(
                                y=paths[:, i],
                                mode="lines",
                                line=dict(width=1),
                                showlegend=False
                            )
                        )

                    fig.update_layout(
                        template="plotly_dark",
                        title="Monte Carlo Future Price Paths"
                    )

                    st.plotly_chart(fig, use_container_width="stretch")

                    final_prices = paths[-1]

                    st.metric(
                        "Expected Price",
                        f"₹{final_prices.mean():.2f}"
                    )

                    st.metric(
                        "Best Case",
                        f"₹{final_prices.max():.2f}"
                    )

                    st.metric(
                        "Worst Case",
                        f"₹{final_prices.min():.2f}"
                    )
with tab6:

            st.markdown("## 🛡️ Risk Engine (VaR)")

            risk_symbol = st.text_input(
                "Stock Symbol",
                value="RELIANCE",
                key="risk_stock"
            ).strip().upper()

            risk_period = st.selectbox(
                "Period",
                ["6mo", "1y", "2y", "5y"],
                index=1,
                key="risk_period"
            )

            investment = st.number_input(
                "Investment Amount (₹)",
                min_value=1000,
                value=100000,
                step=1000,
                key="risk_investment"
            )

            if st.button("Calculate Risk", key="calculate_risk_btn"):
                import numpy as np

                clean_symbol = risk_symbol.replace(" ", "").replace("$", "")
                yf_symbol = clean_symbol if clean_symbol.endswith(".NS") else clean_symbol + ".NS"

                st.info(f"Fetching data for: {yf_symbol}")

                df = yf.Ticker(yf_symbol).history(period=risk_period)

                if df.empty:
                    st.error("Data nahi mila. Symbol check karo. Example: RELIANCE, TCS, INFY, SBIN")
                else:
                    returns = df["Close"].pct_change().dropna()

                    if returns.empty:
                        st.error("Enough price data nahi mila risk calculate karne ke liye.")
                    else:
                        var95 = np.percentile(returns, 5)
                        var99 = np.percentile(returns, 1)

                        loss95 = investment * abs(var95)
                        loss99 = investment * abs(var99)

                        volatility = returns.std() * np.sqrt(252) * 100

                        cumulative = (1 + returns).cumprod()
                        rolling_max = cumulative.cummax()
                        drawdown = ((cumulative - rolling_max) / rolling_max).min() * 100

                        c1, c2, c3 = st.columns(3)

                        c1.metric("VaR 95%", f"₹{loss95:,.0f}")
                        c2.metric("VaR 99%", f"₹{loss99:,.0f}")
                        c3.metric("Volatility", f"{volatility:.2f}%")

                        st.metric("Maximum Drawdown", f"{drawdown:.2f}%")

                        if volatility < 20:
                            risk_score = "LOW RISK 🟢"
                        elif volatility < 35:
                            risk_score = "MEDIUM RISK 🟡"
                        else:
                            risk_score = "HIGH RISK 🔴"

                        st.success(f"Risk Classification: {risk_score}")

                        st.info(
                            f"""
₹{investment:,.0f} investment par:

- 95% confidence par ek din me approx ₹{loss95:,.0f} se jyada loss expected nahi.  
- 99% confidence par approx ₹{loss99:,.0f} se jyada loss expected  nahi.
                            """
                        )

with tab7:
    st.markdown("## 📊 Portfolio Optimizer")

    investment = st.number_input(
        "Investment Amount (₹)",
        min_value=1000,
        value=50000
    )

    risk_profile = st.selectbox(
        "Risk Profile",
        ["Low", "Medium", "High"]
    )

    horizon = st.selectbox(
        "Investment Horizon",
        ["1 Year", "3 Years", "5 Years", "10 Years"]
    )

    stock_input = st.text_area(
        "Stocks (comma separated)",
        value="RELIANCE,TCS,HDFCBANK,INFY,SBIN"
    )

    st.button("Optimize Portfolio", key="test_btn")

    import numpy as np

    symbols = [s.strip().upper().replace("$", "")for s in stock_input.split(",")if s.strip()]
    price_data = pd.DataFrame()

    for sym in symbols:
        yf_symbol = sym if sym.endswith(".NS") else sym + ".NS"

        try:
            data = yf.Ticker(yf_symbol).history(period="1y")
            if not data.empty:
                price_data[sym] = data["Close"]
        except Exception as e:
            st.warning(f"{sym} skipped: {e}")

    price_data = price_data.dropna()
    st.write("Stocks Found:", symbols)

    if len(price_data.columns) < 2:
        st.error("Kam se kam 2 valid stocks chahiye")
    else:

        returns = price_data.pct_change().dropna()

        corr_matrix = returns.corr()

        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        n = len(price_data.columns)

        portfolio_returns = []
        portfolio_risks = []
        portfolio_sharpes = []

        best_sharpe = -999
        best_weights = None

        if risk_profile == "Low":
            max_weight = 0.30
        elif risk_profile == "Medium":
            max_weight = 0.50
        else:
            max_weight = 0.80


        max_weight = (max(max_weight, 1 / n) + 0.1)

        for _ in range(5000):

            weights = np.random.random(n)
            weights /= np.sum(weights)

            if np.max(weights) > max_weight:
                continue

            portfolio_return = np.sum(mean_returns * weights)

            portfolio_risk = np.sqrt(
                np.dot(weights.T,
                    np.dot(cov_matrix, weights))
            )

            sharpe = portfolio_return / portfolio_risk

            portfolio_returns.append(portfolio_return)

            portfolio_risks.append(portfolio_risk)

            portfolio_sharpes.append(sharpe)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = weights
                best_return = portfolio_return
                best_risk = portfolio_risk

        # Efficient Frontier

        frontier_df = pd.DataFrame({
            "Risk": portfolio_risks,
            "Return": portfolio_returns,
            "Sharpe": portfolio_sharpes
        })

        fig_frontier = go.Figure()

        fig_frontier.add_trace(
            go.Scatter(
                x=frontier_df["Risk"],
                y=frontier_df["Return"],
                mode="markers",
                marker=dict(
                    size=5,
                    color=frontier_df["Sharpe"],
                    colorscale="Viridis",
                    showscale=True
                ),
                name="Portfolios"
            )
        )

        fig_frontier.add_trace(
            go.Scatter(
                x=[max(portfolio_risks)],
                y=[max(portfolio_returns)],
                mode="markers",
                marker=dict(size=14, color="red"),
                name="Max Return"
            )
        )

        fig_frontier.update_layout(
            template="plotly_dark",
            title="Efficient Frontier",
            xaxis_title="Risk",
            yaxis_title="Return"
        )

        st.plotly_chart(fig_frontier, use_container_width="stretch")

        result_df = pd.DataFrame({
            "Stock": price_data.columns,
            "Allocation %":
                np.round(best_weights * 100, 2)
        })

        result_df["Investment Amount (₹)"] = (
            result_df["Allocation %"] / 100
        ) * investment

        st.dataframe(result_df)
        st.subheader("🏦 Sector Concentration Risk")

        sector_map = {
            "RELIANCE": "Energy",
            "TCS": "IT",
            "INFY": "IT",
            "HDFCBANK": "Banking",
            "SBIN": "Banking",
            "ICICIBANK": "Banking",
            "WIPRO": "IT",
            "MARUTI": "Auto",
            "SUNPHARMA": "Pharma",
            "ITC": "FMCG"
        }

        result_df["Sector"] = result_df["Stock"].map(sector_map).fillna("Unknown")

        sector_df = (
            result_df.groupby("Sector")["Allocation %"]
            .sum()
            .reset_index()
        )

        st.dataframe(sector_df)

        fig_sector = px.pie(
            sector_df,
            values="Allocation %",
            names="Sector",
            title="Sector Exposure"
        )

        st.plotly_chart(fig_sector, use_container_width="stretch")

        max_sector = sector_df.loc[sector_df["Allocation %"].idxmax()]

        if max_sector["Allocation %"] > 50:
            st.error(f"⚠ Overexposure detected in {max_sector['Sector']} sector ({max_sector['Allocation %']:.1f}%)")
        elif max_sector["Allocation %"] > 35:
            st.warning(f"⚠ Moderate concentration in {max_sector['Sector']} sector ({max_sector['Allocation %']:.1f}%)")
        else:
            st.success("✅ Sector diversification looks healthy")
        fig_pie = px.pie(
            result_df,
            names="Stock",
            values="Allocation %",
            title="Portfolio Allocation"
        )

        st.plotly_chart(fig_pie, use_container_width="stretch")

        st.subheader("📊 Correlation Heatmap")

        try:
            corr_matrix = returns.corr()

            fig, ax = plt.subplots(figsize=(8,6))

            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap="coolwarm",
                center=0,
                ax=ax
            )

            st.pyplot(fig)

        except Exception as e:
            st.warning("Correlation Heatmap unavailable.")

        st.subheader("🎯 Diversification Score")

        avg_corr = corr_matrix.abs().mean().mean()

        div_score = int((1 - avg_corr) * 100)

        div_score = max(0, min(100, div_score))

        st.metric("Diversification Score", f"{div_score}/100")

        if div_score >= 80:
            st.success("✅ Excellent Diversification")
        elif div_score >= 60:
            st.info("🟢 Good Diversification")
        elif div_score >= 40:
            st.warning("⚠ Moderate Diversification")
        else:
            st.error("🚨 Poor Diversification - Highly Correlated Portfolio")

        csv = result_df.to_csv(index=False)

        st.download_button(
            label="📥 Download Portfolio CSV",
            data=csv,
            file_name="bharatfin_portfolio.csv",
            mime="text/csv"
        )

        st.subheader("📉 CVaR Risk Analysis")

        portfolio_returns = returns.mean(axis=1)

        var95 = np.percentile(
            portfolio_returns,
            5
        )

        cvar95 = portfolio_returns[
            portfolio_returns <= var95
        ].mean()

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "VaR 95%",
                f"{var95:.2%}"
            )

        with col2:
            st.metric(
                "CVaR 95%",
                f"{cvar95:.2%}"
            )

        if cvar95 < -0.03:
            st.error(
                "🚨 Crash scenario risk is HIGH"
            )
        else:
            st.success(
                "✅ Crash scenario risk is acceptable"
            )

        st.plotly_chart(fig_pie, use_container_width="stretch", key="portfolio_allocation_pie_1")

        st.subheader("📉 Drawdown Risk Monitor")

        portfolio_returns = returns.mean(axis=1)

        cumulative = (
            1 + portfolio_returns
        ).cumprod()

        rolling_max = cumulative.cummax()

        drawdown = (
            cumulative - rolling_max
        ) / rolling_max

        max_drawdown = drawdown.min()

        st.metric(
            "Max Drawdown",
            f"{max_drawdown:.2%}"
        )

        fig_dd = px.line(
            drawdown,
            title="Portfolio Drawdown"
        )

        st.plotly_chart(
            fig_dd,
            use_container_width="stretch"
        )

        if max_drawdown < -0.20:
            st.error(
                "🚨 Severe drawdown risk detected"
            )
        elif max_drawdown < -0.10:
            st.warning(
                "⚠ Moderate drawdown risk"
            )
        else:
            st.success(
                "✅ Drawdown risk under control"
            )

        exp_return = np.sum(
            mean_returns * best_weights
        ) * 100

        exp_risk = np.sqrt(
            np.dot(best_weights.T,
                np.dot(cov_matrix,
                        best_weights))
        ) * 100

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Expected Return",
            f"{exp_return:.2f}%"
        )

        c2.metric(
            "Portfolio Risk",
            f"{exp_risk:.2f}%"
        )

        c3.metric(
            "Sharpe Ratio",
            f"{best_sharpe:.2f}"
        )

        st.subheader("🧪 Stress Testing Engine")

        stress_results = {
            "NIFTY Crash (-10%)": -10,
            "Banking Crash (-15%)": -15,
            "IT Crash (-20%)": -20,
            "Market Crash (-30%)": -30
        }

        for scenario, shock in stress_results.items():

            portfolio_loss = (
                result_df["Allocation %"].sum()
                * abs(shock)
                / 100
            )

            st.write(
                f"{scenario} → Portfolio Loss: -{portfolio_loss:.2f}%"
            )

        if portfolio_loss > 20:
            st.error(
                "🚨 High Stress Risk"
            )
        elif portfolio_loss > 10:
            st.warning(
                "⚠ Moderate Stress Risk"
            )
        else:
            st.success(
                "✅ Stress Test Passed"
            )

        st.success(
            "🏆 Maximum Sharpe Ratio Portfolio Found"
        )

        st.subheader("🧠 AI Risk Diagnosis")

        issues = []

        if best_sharpe < 0:
            issues.append(
                "❌ Negative Sharpe Ratio - Risk ke hisab se return weak hai"
            )

        if max_drawdown < -0.20:
            issues.append(
                "❌ High Drawdown Risk (>20%)"
            )

        for _, row in sector_df.iterrows():
            if row["Allocation %"] > 50:
                issues.append(
                    f"❌ Overexposed to {row['Sector']} sector ({row['Allocation %']:.1f}%)"
                )

        if len(issues) == 0:
            st.success(
                "✅ Portfolio looks healthy"
            )
        else:
            for item in issues:
                st.warning(item)

                st.subheader("💡 AI Recommendations")

        if best_sharpe < 0:
            st.info(
                "Increase diversification and reduce weak-performing assets."
            )

        if max_drawdown < -0.20:
            st.info(
                "Add defensive sectors like FMCG and Pharma."
            )

        for _, row in sector_df.iterrows():
            if row["Allocation %"] > 50:
                st.info(
                    f"Reduce exposure to {row['Sector']} sector."
                )

        st.subheader("🏆 Portfolio Health Score")

        st.markdown("## 🔄 AI Rebalancing Suggestions")

        top_stock = result_df.loc[
            result_df["Allocation %"].idxmax(),
            "Stock"
        ]

        top_alloc = result_df["Allocation %"].max()

        suggestions = []

        if top_alloc > 50:
            suggestions.append(
                f"⚠ Reduce {top_stock} allocation ({top_alloc:.2f}%)"
            )

        if portfolio_risk > 0.20:
            suggestions.append(
                "⚠ Portfolio risk is high. Add defensive stocks."
            )

        sharpe_ratio = best_sharpe
        if sharpe_ratio < 0.5:
            suggestions.append(
                "📈 Consider adding ITC, HINDUNILVR, ICICIBANK for diversification."
            )

        expected_return = best_return
        if expected_return > best_risk:
            suggestions.append(
                "✅ Risk-reward profile looks healthy."
            )


        score = 100

        if best_sharpe < 0:
            score -= 25

        if max_drawdown < -0.20:
            score -= 25

        if max_sector["Allocation %"] > 50:
            score -= 20

        if exp_return < 0:
            score -= 15

        score = max(0, min(100, score))

        if score >= 80:
            st.success("✅ Portfolio is already well balanced.")
        elif score >= 60:
            st.info("🟡 Portfolio needs minor optimization.")
        elif score >= 40:
            st.warning("⚠ Portfolio needs rebalancing.")
        else:
            st.error("🚨 Immediate rebalancing required.")

        if score >= 80:
            st.success("🚀 Excellent Portfolio")
        elif score >= 60:
            st.info("🟡 Good Portfolio")
        elif score >= 40:
            st.warning("⚠️ Average Portfolio")
        else:
            st.error("❌ Weak Portfolio")


        st.subheader("🤖 AI Portfolio Advisor")

        if best_sharpe > 1:
            st.success("Excellent risk-adjusted portfolio.")
        elif best_sharpe > 0.5:
            largest_stock = result_df.loc[
                result_df["Allocation %"].idxmax()
            ]

            st.info(
                f"""
            📊 Highest Allocation: {largest_stock['Stock']} ({largest_stock['Allocation %']:.2f}%)

            📈 Expected Return: {exp_return:.2f}%

            ⚠️ Portfolio Risk: {exp_risk:.2f}%

            🎯 Sharpe Ratio: {best_sharpe:.2f}
            """
            )
        else:
            st.warning("Portfolio risk jyada hai compared to expected return.")

            st.subheader("🎲 Monte Carlo Simulation")

        simulations = 1000
        days = 252

        portfolio_returns = returns.mean(axis=1)

        sim_results = []

        for _ in range(simulations):
            simulated = np.random.choice(
                portfolio_returns,
                size=days,
                replace=True
            )
            sim_results.append((1 + simulated).prod())

        fig_mc = px.histogram(
            x=sim_results,
            nbins=40,
            title="Monte Carlo Portfolio Outcomes"
        )

        st.plotly_chart(
            fig_mc,
            use_container_width="stretch",
            key="monte_carlo_sim"
        )

        st.metric(
            "Expected Portfolio Growth",
            f"{(np.mean(sim_results)-1)*100:.2f}%"
        )

        st.subheader("📌 Executive Risk Dashboard")

        risk_flags = 0

        if best_sharpe < 0:
            risk_flags += 1

        if max_drawdown < -0.20:
            risk_flags += 1

        if max_sector["Allocation %"] > 50:
            risk_flags += 1

        if score < 40:
            risk_flags += 1

        if risk_flags >= 3:
            final_risk = "HIGH RISK 🔴"
            grade = "D"
        elif risk_flags == 2:
            final_risk = "MEDIUM RISK 🟡"
            grade = "C"
        elif risk_flags == 1:
            final_risk = "LOW-MEDIUM RISK 🟠"
            grade = "B"
        else:
            final_risk = "LOW RISK 🟢"
            grade = "A"

        c1, c2, c3 = st.columns(3)

        c1.metric("Final Risk Level", final_risk)
        c2.metric("Portfolio Grade", grade)
        c3.metric("Risk Flags", risk_flags)

        if grade in ["D", "C"]:
            st.error("Portfolio needs active risk management before investing more.")
        else:
            st.success("Portfolio risk profile looks acceptable.")

        st.subheader("📈 Buy / Sell Signal Engine")
        for stock in result_df["Stock"]:

            hist = yf.Ticker(f"{stock}.NS").history(period="3mo")["Close"]
            delta = hist.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]

            if rsi < 30:
                st.success(f"🟢 {stock}: BUY Signal (RSI={rsi:.1f})")

            elif rsi > 70:
                st.error(f"🔴 {stock}: SELL Signal (RSI={rsi:.1f})")

            else:
                st.info(f"🟡 {stock}: HOLD Signal (RSI={rsi:.1f})")

        st.subheader("🏆 Stock Ranking Engine")

        ranking_df = result_df.copy()

        ranking_df["Rank"] = ranking_df["Allocation %"].rank(
            ascending=False
        )

        ranking_df = ranking_df.sort_values(
            "Rank"
        )

        st.dataframe(
            ranking_df[
                ["Stock", "Allocation %", "Rank"]
            ],
            use_container_width="stretch"
        )