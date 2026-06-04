from data.stock_data import get_data
from core.indicators import calc_indicators
from core.trust_engine import get_trust_scores
from core.signals import get_signal
from ai.prompts import BOOK_KNOWLEDGE
from ui.styles import CUSTOM_CSS
from data.feedback import save_feedback, load_feedback
import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from groq import Groq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import heapq
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

    api_key = os.getenv("GROQ_API_KEY")
    st.write("API Loaded:", bool(api_key))
    

st.set_page_config(
    page_title="BharatFinAI",
    page_icon="📈",
    layout="wide"
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
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
tab1, tab2, tab3 = st.tabs([
    "🔍 Single Stock Analysis",
    "📈 Multi-Stock Scanner",
    "💬 Feedback"
])

tab1, tab2, tab3 = st.tabs([
    "🔍 Single Stock Analysis",
    "📊 Multi-Stock Scanner",
    "💬 Feedback"
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
 

 

 



# ─────────────────────────────────────────
# TAB 1: SINGLE STOCK
# ─────────────────────────────────────────
with tab1:
    c1, c2, c3 = st.columns([3, 1.5, 1.5])
    with c1:
        symbol = st.text_input("", placeholder="RELIANCE, TCS, HDFCBANK, INFY...",
                               label_visibility="collapsed", key="single")
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

            st.markdown(f"### {company}")

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