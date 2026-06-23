CUSTOM_CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg-base:       #0B0C0E;
  --bg-surface:    #111318;
  --bg-card:       #16181F;
  --bg-elevated:   #1C1F2A;
  --bg-hover:      #21242F;
  --border:        #252836;
  --border-light:  #2A2D3A;
  --text-primary:  #EAEDF3;
  --text-secondary:#8B90A0;
  --text-muted:    #555A6E;
  --green:         #1DB954;
  --green-dim:     #0D3320;
  --green-glow:    rgba(29,185,84,0.15);
  --red:           #F0484B;
  --red-dim:       #2D1516;
  --amber:         #F59E0B;
  --amber-dim:     #2D2008;
  --blue:          #3B82F6;
  --blue-dim:      #0D1F3C;
  --accent:        #00C9A7;
  --accent-dim:    #002921;
  --accent-glow:   rgba(0,201,167,0.12);
}

/* ── BASE ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
}

.main .block-container {
    max-width: 1280px;
    padding: 1.5rem 2rem 4rem;
    background: var(--bg-base);
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--text-muted) !important;
    margin: 1rem 0 0.4rem !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 14px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 10px !important;
    color: var(--text-secondary) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    padding: 0.4rem 0.85rem !important;
    border: none !important;
    transition: all 0.15s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: #000 !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 12px var(--accent-glow) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.2rem !important;
}

/* ── METRIC CARDS ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.85rem 1rem !important;
    transition: border-color 0.15s, transform 0.15s;
}
[data-testid="stMetric"]:hover {
    border-color: var(--accent) !important;
    transform: translateY(-1px);
}
[data-testid="stMetricLabel"] {
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--text-primary) !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #00C9A7, #00A882) !important;
    color: #000 !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.02em;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 14px var(--accent-glow) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 22px rgba(0,201,167,0.3) !important;
}

/* ── INPUTS ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--bg-card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
    font-size: 0.85rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

/* ── SELECTBOX ── */
[data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
}

/* ── ALERTS ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 0.8rem !important;
}
.stSuccess {
    background: var(--green-dim) !important;
    border: 1px solid rgba(29,185,84,0.3) !important;
    border-left: 3px solid var(--green) !important;
    color: #B8F0C8 !important;
}
.stError {
    background: var(--red-dim) !important;
    border: 1px solid rgba(240,72,75,0.3) !important;
    border-left: 3px solid var(--red) !important;
    color: #FFB3B4 !important;
}
.stWarning {
    background: var(--amber-dim) !important;
    border: 1px solid rgba(245,158,11,0.3) !important;
    border-left: 3px solid var(--amber) !important;
    color: #FFE0A3 !important;
}
.stInfo {
    background: var(--blue-dim) !important;
    border: 1px solid rgba(59,130,246,0.3) !important;
    border-left: 3px solid var(--blue) !important;
    color: #BFDBFE !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    overflow: hidden;
}
[data-testid="stDataFrame"] thead tr th {
    background: var(--bg-elevated) !important;
    color: var(--text-muted) !important;
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] tbody tr td {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border-bottom: 1px solid var(--border-light) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
}
[data-testid="stDataFrame"] tbody tr:hover td {
    background: var(--bg-hover) !important;
}

/* ── PROGRESS BAR ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--accent), #00A882) !important;
    border-radius: 99px !important;
}
.stProgress > div > div {
    background: var(--bg-elevated) !important;
    border-radius: 99px !important;
    height: 6px !important;
}

/* ── EXPANDER ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
}
.streamlit-expanderContent {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
}

/* ── DIVIDER ── */
hr {
    border-color: var(--border) !important;
    margin: 1.2rem 0 !important;
}

/* ── HEADINGS ── */
h1 { font-size: 1.5rem !important; font-weight: 800 !important; color: var(--text-primary) !important; letter-spacing: -0.02em; }
h2 { font-size: 1.1rem !important; font-weight: 700 !important; color: var(--text-primary) !important; }
h3 { font-size: 0.82rem !important; font-weight: 700 !important; color: var(--text-secondary) !important; letter-spacing: 0.04em; text-transform: uppercase; }

/* ── DOWNLOAD BUTTON ── */
.stDownloadButton > button {
    background: var(--bg-card) !important;
    border: 1.5px solid var(--border) !important;
    color: var(--text-secondary) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}
.stDownloadButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-dim) !important;
}

/* ── PLOTLY CHART ── */
[data-testid="stPlotlyChart"] {
    border-radius: 14px !important;
    overflow: hidden;
    border: 1px solid var(--border) !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── NUMBER INPUT ── */
.stNumberInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── SLIDER ── */
[data-baseweb="slider"] div[role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}

/* ── CAPTION ── */
.stCaption { color: var(--text-muted) !important; font-size: 0.68rem !important; }

/* ── SPINNER ── */
[data-testid="stSpinner"] > div {
    border-top-color: var(--accent) !important;
}

/* ── HIDE DEFAULT STREAMLIT MENU/FOOTER ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

</style>
"""