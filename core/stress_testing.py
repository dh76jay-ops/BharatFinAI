"""
BharatFinAI Risk Engine V9 — Stress Testing Engine
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


def market_shock(portfolio_value: float, weights: pd.Series, shock_pct: float) -> Dict[str, Any]:
    loss = portfolio_value * abs(shock_pct)
    after = portfolio_value - loss
    return {
        "scenario": f"Market Shock {shock_pct*100:.0f}%",
        "shock_pct": shock_pct,
        "estimated_loss": -loss,
        "estimated_loss_pct": -abs(shock_pct) * 100,
        "portfolio_before": portfolio_value,
        "portfolio_after": after,
        "recovery_required_pct": (loss / after * 100) if after > 0 else np.inf,
    }


def volatility_shock(returns: pd.Series, portfolio_value: float, vol_multiple: float = 2.0,
                     alpha: float = 0.05, horizon_days: int = 1) -> Dict[str, Any]:
    r = returns.dropna()
    if len(r) < 20:
        return {"ok": False, "message": "Insufficient data for vol shock."}
    sigma = r.std() * vol_multiple
    from scipy import stats
    z = abs(float(stats.norm.ppf(alpha)))
    daily_var = z * sigma * np.sqrt(horizon_days)
    loss = portfolio_value * daily_var
    return {
        "ok": True, "scenario": f"Vol x{vol_multiple:.1f} Shock", "vol_multiple": vol_multiple,
        "stressed_daily_vol": float(sigma), "estimated_loss": -loss,
        "estimated_loss_pct": -daily_var * 100, "portfolio_before": portfolio_value,
        "portfolio_after": portfolio_value - loss,
        "recovery_required_pct": (loss / (portfolio_value - loss) * 100) if portfolio_value > loss else np.inf,
    }


def correlation_shock(returns_df: pd.DataFrame, weights: pd.Series, portfolio_value: float,
                      corr_floor: float = 0.85, alpha: float = 0.05) -> Dict[str, Any]:
    cols = [c for c in weights.index if c in returns_df.columns]
    if len(cols) < 2:
        return {"ok": False, "message": "Need >=2 assets for correlation shock."}
    R = returns_df[cols].dropna()
    w = weights[cols].values
    w = w / w.sum()
    cov = R.cov().values
    vol = np.sqrt(np.diag(cov))
    n = len(vol)
    stressed_corr = np.full((n, n), corr_floor)
    np.fill_diagonal(stressed_corr, 1.0)
    stressed_cov = np.outer(vol, vol) * stressed_corr
    port_var = float(w @ stressed_cov @ w)
    port_vol = np.sqrt(port_var)
    from scipy import stats
    z = abs(float(stats.norm.ppf(alpha)))
    var = z * port_vol
    loss = portfolio_value * var
    base_var = float(w @ cov @ w)
    base_vol = np.sqrt(base_var)
    return {
        "ok": True, "scenario": f"Correlation Shock (ρ→{corr_floor})",
        "baseline_port_vol_daily": base_vol, "stressed_port_vol_daily": port_vol,
        "estimated_loss": -loss, "estimated_loss_pct": -var * 100,
        "portfolio_before": portfolio_value, "portfolio_after": portfolio_value - loss,
        "recovery_required_pct": (loss / (portfolio_value - loss) * 100) if portfolio_value > loss else np.inf,
    }


def sector_shock(weights: pd.Series, sector_map: Dict[str, str], portfolio_value: float,
                 shocked_sector: str, shock_pct: float = -0.15) -> Dict[str, Any]:
    loss = 0.0
    for sym, w in weights.items():
        sec = sector_map.get(sym, "Unknown")
        if sec == shocked_sector:
            loss += portfolio_value * w * abs(shock_pct)
    after = portfolio_value - loss
    return {
        "scenario": f"Sector Shock: {shocked_sector} {shock_pct*100:.0f}%",
        "shocked_sector": shocked_sector, "shock_pct": shock_pct,
        "estimated_loss": -loss,
        "estimated_loss_pct": -(loss / portfolio_value * 100) if portfolio_value else 0,
        "portfolio_before": portfolio_value, "portfolio_after": after,
        "recovery_required_pct": (loss / after * 100) if after > 0 else np.inf,
    }


def historical_crisis_proxy(returns: pd.Series, portfolio_value: float, window: int = 20) -> Dict[str, Any]:
    r = returns.dropna()
    if len(r) < window + 5:
        return {"ok": False, "message": "Insufficient history for crisis proxy."}
    roll = (1 + r).rolling(window).apply(lambda x: x.prod() - 1, raw=True).dropna()
    worst = float(roll.min())
    loss = portfolio_value * abs(worst)
    return {
        "ok": True, "scenario": f"Historical Proxy (worst {window}D)", "window": window,
        "worst_return": worst, "estimated_loss": -loss, "estimated_loss_pct": worst * 100,
        "portfolio_before": portfolio_value, "portfolio_after": portfolio_value - loss,
        "recovery_required_pct": (loss / (portfolio_value - loss) * 100) if portfolio_value > loss else np.inf,
        "note": "Proxy using worst historical rolling return — not full crisis reconstruction.",
    }


def run_stress_suite(portfolio_value: float, weights: Optional[pd.Series] = None,
                     returns: Optional[pd.Series] = None, returns_df: Optional[pd.DataFrame] = None,
                     sector_map: Optional[Dict[str, str]] = None,
                     vol_multiples: Optional[List[float]] = None) -> pd.DataFrame:
    rows = []
    for s in [-0.05, -0.10, -0.20]:
        rows.append(market_shock(portfolio_value, weights if weights is not None else pd.Series(dtype=float), s))
    if returns is not None and len(returns.dropna()) >= 20:
        for m in (vol_multiples or [1.5, 2.0, 3.0]):
            res = volatility_shock(returns, portfolio_value, vol_multiple=m)
            if res.get("ok", True):
                rows.append(res)
    if returns_df is not None and weights is not None and len(weights) >= 2:
        res = correlation_shock(returns_df, weights, portfolio_value)
        if res.get("ok", True):
            rows.append(res)
    if weights is not None and sector_map:
        for sec in sorted(set(sector_map.values())):
            res = sector_shock(weights, sector_map, portfolio_value, sec, shock_pct=-0.15)
            if abs(res["estimated_loss"]) > 0:
                rows.append(res)
    if returns is not None:
        res = historical_crisis_proxy(returns, portfolio_value)
        if res.get("ok"):
            rows.append(res)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    cols = [c for c in ["scenario", "estimated_loss", "estimated_loss_pct",
                        "portfolio_before", "portfolio_after", "recovery_required_pct"] if c in df.columns]
    return df[cols]