"""
BharatFinAI — Cached market data helpers for Risk Engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from typing import Optional, List


@st.cache_data(ttl=600, show_spinner=False)
def fetch_price_history(
    symbol: str,
    period: str = "2y",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    try:
        sym = symbol.upper().strip()
        if not sym.endswith((".NS", ".BO")) and not sym.startswith("^"):
            sym = sym + ".NS"
        ticker = yf.Ticker(sym)
        df = ticker.history(period=period, auto_adjust=auto_adjust)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_returns(symbol: str, period: str = "2y") -> pd.Series:
    df = fetch_price_history(symbol, period)
    if df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    ret = df["Close"].pct_change().dropna()
    ret.name = symbol.upper().replace(".NS", "").replace(".BO", "")
    return ret


@st.cache_data(ttl=900, show_spinner=False)
def fetch_multi_returns(symbols: List[str], period: str = "2y") -> pd.DataFrame:
    series = []
    for s in symbols:
        r = fetch_returns(s, period)
        if not r.empty:
            series.append(r)
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1).dropna(how="all")


def ensure_ns(symbol: str) -> str:
    s = symbol.upper().strip()
    if s.endswith((".NS", ".BO")) or s.startswith("^"):
        return s
    return s + ".NS"