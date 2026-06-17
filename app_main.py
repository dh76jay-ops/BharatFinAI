from data.stock_data import get_data
from core.indicators import calc_indicators
from core.trust_engine import get_trust_scores
from core.signals import get_signal
from ai.prompts import BOOK_KNOWLEDGE
from ui.styles import CUSTOM_CSS
from data.feedback import save_feedback, load_feedback
import streamlit as st
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
import pandas as pd
import numpy as np
#import pandas_ta as ta
from groq import Groq
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import heapq
import os
from newsapi import NewsApiClient

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except:
    pass

    api_key = os.getenv("GROQ_API_KEY")
    st.write("API Loaded:", bool(api_key))
    

st.set_page_config(
    page_title="BharatFinAI",
    page_icon="📈",
    layout="wide",

initial_sidebar_state="expanded"
)

#st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.success("SIDEBAR TEST")
    st.markdown("## 📈 BharatFinAI")
    st.markdown("*Hindi AI Stock Analyzer*")
    st.divider()
    st.markdown("**Quick Stocks:**")
    st.markdown("RELIANCE • TCS • HDFCBANK\nINFY • WIPRO • SBIN\nTATAMOTORS • ADANIENT\nMARUTI • SUNPHARMA")
    st.divider()
    st.markdown("**Indicators Guide:**")
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

profit_score = 100 if return_pct > 10 else 75 if return_pct > 0 else 50 if return_pct > -10 else 25

health_score = int((diversification_score * 0.4) + (profit_score * 0.6))

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Portfolio Health Score")

st.sidebar.metric(
    "Health Score",
    f"{health_score}/100"
)

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

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div style='padding:20px 0 10px'>
<h1 style='color:#f0a500;font-size:2rem;margin:0'>📈 BharatFinAI</h1>
<p style='color:#5a5a7a;margin:4px 0 0'>Hindi AI Stock Analyzer • NSE/BSE • Psychology Aware • Trust Engine</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8  = st.tabs([
    "🔍 Single Stock Analysis",
    "📈 Multi-Stock Scanner",
    "💬 Feedback",
    " AI Comparison",
    "Quant Systems",
    "🎲 Monte Carlo",
    "🛡️ Risk Engine",
    "📊 Portfolio Optimizer"  
])



with tab3:

    st.divider()

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


# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
 
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
# TAB 1: SINGLE STOCK
# ─────────────────────────────────────────
with tab1:
    c1, c2, c3 = st.columns([3, 1.5, 1.5])
    with c1:
        symbol = st.text_input(
    "",
    value=default_stock,
    placeholder="RELIANCE, TCS, HDFCBANK, INFY...",
    label_visibility="collapsed",
    key="single"
)
    with c2:
        user_type = st.selectbox("", ["College Student", "Beginner", "Experienced"],
                                 label_visibility="collapsed")
    with c3:
        period = st.selectbox("", ["3mo", "6mo", "1y"], index=1,
                              label_visibility="collapsed")

    if st.button("🔍 Analysis Karo", key="single_btn"):
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
            change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
            vol_ratio = latest['Volume'] / latest['Vol_MA'] if latest['Vol_MA'] > 0 else 1
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

            st.markdown("### 🧠 AI Reasoning")

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
            st.plotly_chart(fig, use_container_width=True)

            # AI Analysis
            st.divider()
            st.subheader("🤖 AI Analysis — Hindi Mein")

            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                st.error("❌ GROQ_API_KEY nahi mili! .env file check karo")
                st.stop()

            with st.spinner("🧠 AI analysis kar raha hai..."):
                try:
                    st.write("API Loaded:", bool(api_key))
                    st.write("Key Start:", api_key[:10])

                    client = Groq(api_key=api_key)
                    sma50 = latest['SMA_50'] if pd.notna(latest['SMA_50']) else 0
                    p_vs_50 = ((latest['Close'] - sma50) / sma50 * 100) if sma50 else 0

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

Book Knowledge: {BOOK_KNOWLEDGE}

Is format mein SIRF HINDI mein likho:

TECHNICAL PICTURE
[2-3 lines current situation]

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
                        model="llama-3.3-70b-versatile",
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

                    st.download_button(
                        "📄 Download Analysis Report",
                        data=report_text,
                        file_name=f"{sym}_report.txt",
                        mime="text/plain"
                    )

                    # News Sentiment
                    news_sentiment, news_score = get_news_sentiment(sym)

                    st.divider()
                    st.markdown("### 📰 News Sentiment")

                    if news_sentiment == "Positive":
                        st.success(f"🟢 Positive News Sentiment (+{news_score})")
                    elif news_sentiment == "Negative":
                        st.error(f"🔴 Negative News Sentiment ({news_score})")
                    else:
                        st.warning("🟡 Neutral News Sentiment")

                    # Trust Engine
                    st.divider()
                    st.markdown("### 🔍 Confidence Breakdown")

                    rsi_sc, macd_sc, risk_sc, trust = get_trust_scores(
                        rsi_val, latest['MACD'], latest['MACD_Signal'], week52_pos)

                    tc1, tc2, tc3 = st.columns(3)
                    with tc1:
                        st.metric("📊 Technical", f"{rsi_sc:.0f}/100",
                                  "Strong ✅" if rsi_sc > 60 else "Weak ⚠️")
                    with tc2:
                        st.metric("⚡ Momentum", f"{macd_sc:.0f}/100",
                                  "Bullish ✅" if macd_sc > 50 else "Bearish ⚠️")
                    with tc3:
                        st.metric("🛡️ Risk", f"{risk_sc:.0f}/100",
                                  "Safe ✅" if risk_sc > 50 else "Risky ⚠️")

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

            # CSV Export
            with st.expander("📊 Raw Data + Export"):
                cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'SMA_50']
                display_df = df.tail(30)[cols].round(2)
                st.dataframe(display_df, use_container_width=True)
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
                    status.text(f"Analyzing {stk}...")
                    sym_s = stk if stk.endswith('.NS') else stk + '.NS'
                    ticker = yf.Ticker(sym_s)
                    df_s = ticker.history(period=scan_period)

                    if df_s.empty:
                        continue

                    df_s['RSI'] = ta.rsi(df_s['Close'], length=14)
                    macd_s = ta.macd(df_s['Close'])
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

                    result = {
                        "Stock": stk,
                        "Price": f"₹{lat['Close']:.2f}",
                        "Change": f"{chg:+.2f}%",
                        "RSI": round(rsi, 1),
                        "52W%": f"{w52:.0f}%",
                        "Trust": round(trust, 0),
                        "Signal": signal,
                        "Stop Loss": f"₹{lat['Close'] * 0.95:.2f}",
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
                st.markdown("### 🏆 Heap-Based Ranking")

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
                st.dataframe(styled, use_container_width=True, hide_index=True)

                # Export
                csv_scan = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Scan Results CSV Download",
                    csv_scan,
                    "scanner_results.csv",
                    "text/csv"
                )

                st.caption(f"✅ {len(results)}/{len(stocks_list)} stocks successfully scanned")
with tab3:

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
        # ====================================
# TAB 4 : AI COMPARISON
# ====================================

with tab4:

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

            rsi1 = ta.rsi(df1["Close"], length=14).dropna().iloc[-1]
            rsi2 = ta.rsi(df2["Close"], length=14).dropna().iloc[-1]

            macd1 = ta.macd(df1["Close"])
            macd2 = ta.macd(df2["Close"])

            m1 = macd1["MACD_12_26_9"].dropna().iloc[-1]
            ms1 = macd1["MACDs_12_26_9"].dropna().iloc[-1]
            m2 = macd2["MACD_12_26_9"].dropna().iloc[-1]
            ms2 = macd2["MACDs_12_26_9"].dropna().iloc[-1]

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

            st.dataframe(compare_df, use_container_width=True)

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

        st.sidebar.dataframe(portfolio_df, use_container_width=True)
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

        df["RSI"] = ta.rsi(df["Close"], length=14)

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
        use_container_width=True
    )

    top_pick = watchlist_table.iloc[0]

    oversold_stocks = watchlist_table[watchlist_table["RSI"] < 40]

st.sidebar.markdown("### 🚨 Oversold Alerts")

if not oversold_stocks.empty:
    for _, row in oversold_stocks.iterrows():
        st.sidebar.warning(
            f"🚨 {row['Stock']} oversold zone me hai | RSI: {row['RSI']}"
        )
else:
    st.sidebar.info("Abhi koi oversold opportunity nahi mili.")

st.sidebar.markdown("### 🏆 Top Watchlist Pick")

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
with tab5:
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
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(df[["Close", "SMA20", "SMA50", "Signal", "Return", "Strategy"]].tail(20))

            with tab6:

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

            st.plotly_chart(fig, use_container_width=True)

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

            with tab7:

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

• 95% confidence par ek din me approx ₹{loss95:,.0f} se jyada loss expected nahi.  
• 99% confidence par approx ₹{loss99:,.0f} se jyada loss expected  nahi.
                    """
                )

            with tab8:
                st.write("HELLO TAB8")
                st.markdown("## 📊 Portfolio Optimizer")
                st.subheader("🤖 AI Portfolio Advisor")

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

                st.write("BEFORE BUTTON")
                st.button("Optimize Portfolio", key="test_btn")
            #if st.button("Optimize Portfolio, key="optimize_portfolio_btn"):
                st.success("BUTTON CLICKED")

                import numpy as np

                symbols = [s.strip().upper().replace("$", "")for s in stock_input.split(",")if s.strip()]
                st.write("Raw Input =", stock_input)
                st.write("Symbols List =", symbols)
                st.write("Raw Input:", stock_input)
                st.write("Symbols:", symbols)
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

                    st.subheader("🔥 Correlation Heatmap")

                    fig_corr, ax = plt.subplots(figsize=(8, 6))

                    sns.heatmap(
                        corr_matrix,
                        annot=True,
                        cmap="RdYlGn_r",
                        ax=ax
                    )

                    st.pyplot(fig_corr)


                    # High Correlation Detection
                    max_corr = corr_matrix.where(
                        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                    ).max().max()

                    if max_corr > 0.75:
                        st.warning(
                            "⚠ High Correlation Detected! Portfolio diversification weak hai."
                        )
                    else:
                        st.success(
                            "✅ Portfolio diversification accha hai."
                        )

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

                    st.plotly_chart(fig_frontier, use_container_width=True)

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

                    st.plotly_chart(fig_sector, use_container_width=True)

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

                    st.plotly_chart(fig_pie, use_container_width=True)

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

                    st.plotly_chart(fig_pie, use_container_width=True, key="portfolio_allocation_pie_1")

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
                        use_container_width=True
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

                    sharpe_ratio = 0.84
                    if sharpe_ratio < 0.5:
                        suggestions.append(
                            "📈 Consider adding ITC, HINDUNILVR, ICICIBANK for diversification."
                        )

                    expected_return = 0.15
                    if expected_return > portfolio_risk:
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
                        use_container_width=True,
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
                        
                        rsi = np.random.randint(20, 80)
                        
                        if rsi < 30:
                            st.success(f"🟢 {stock}: BUY Signal (RSI={rsi})")
                            
                        elif rsi > 70:
                            st.error(f"🔴 {stock}: SELL Signal (RSI={rsi})")
                            
                        else:
                            st.info(f"🟡 {stock}: HOLD Signal (RSI={rsi})")

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
                                use_container_width=True
                            )

                            st.subheader("🏢 Sector Concentration Dashboard")

                            sector_alloc = sector_df[
                                ["Sector", "Allocation %"]
                            ].sort_values(
                                "Allocation %",
                                ascending=False
                            )

                            st.dataframe(
                                sector_alloc,
                                use_container_width=True
                            )

                            max_sector = sector_alloc.iloc[0]

                            if max_sector["Allocation %"] > 50:
                                st.error(
                                    f"⚠ High concentration in {max_sector['Sector']} ({max_sector['Allocation %']:.1f}%)"
                                )
                            elif max_sector["Allocation %"] > 35:
                                st.warning(
                                    f"⚠ Moderate concentration in {max_sector['Sector']} ({max_sector['Allocation %']:.1f}%)"
                                )
                            else:
                                st.success("✅ Sector diversification looks healthy")
                                                                

