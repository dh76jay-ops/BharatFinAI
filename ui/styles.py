CUSTOM_CSS = """
<style>

.stApp {
    background-color: #050508;
    color: #f0f0ff;
}

#MainMenu, footer {
    visibility: hidden;
}

.stButton > button {
    background: linear-gradient(135deg, #f0a500, #ffc847) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    width: 100% !important;
}

[data-testid="stMetricValue"] {
    color: #f0a500 !important;
    font-weight: 800 !important;
}

.stDataFrame {
    background: #0c0c14 !important;
}

[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    min-width: 320px !important;
    max-width: 320px !important;
    width: 320px !important;
}

[data-testid="collapsedControl"] {
    display: block !important;
    visibility: visible !important;
}

</style>
"""