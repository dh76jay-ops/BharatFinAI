"""
BharatFinAI Risk Engine V10 — Tail Risk Engine
Expected Shortfall, Student-t Monte Carlo, MDD, EVT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from scipy import stats


def expected_shortfall(returns: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
    r = returns.dropna().astype(float)
    n = len(r)
    if n < 40:
        return {"ok": False, "message": "Insufficient data for CVaR.", "n_obs": n}
    var = float(np.percentile(r, alpha * 100))
    tail = r[r <= var]
    es = float(tail.mean()) if len(tail) else var
    return {
        "ok": True, "var": var, "cvar": es, "alpha": alpha,
        "tail_obs": int(len(tail)), "n_obs": n, "method": "Historical ES",
    }


def student_t_monte_carlo(returns: pd.Series, n_sims: int = 5000, horizon: int = 1,
                          alpha: float = 0.05, seed: int = 42) -> Dict[str, Any]:
    r = returns.dropna().astype(float)
    n = len(r)
    if n < 40:
        return {"ok": False, "message": "Insufficient data for Student-t MC."}
    try:
        df_fit, loc, scale = stats.t.fit(r)
    except Exception as e:
        return {"ok": False, "message": f"Fit failed: {e}"}
    rng = np.random.default_rng(seed)
    sims = rng.standard_t(df_fit, size=(n_sims, horizon)) * scale + loc
    terminal = sims.sum(axis=1)
    var = float(np.percentile(terminal, alpha * 100))
    es = float(terminal[terminal <= var].mean())
    return {
        "ok": True, "var": var, "cvar": es, "mean": float(terminal.mean()),
        "std": float(terminal.std()), "df": float(df_fit),
        "n_sims": n_sims, "horizon": horizon, "method": "Student-t Monte Carlo",
        "terminal_samples": terminal,
    }


def max_drawdown_analysis(prices: pd.Series) -> Dict[str, Any]:
    p = prices.dropna().astype(float)
    if len(p) < 20:
        return {"ok": False, "message": "Insufficient price history."}
    cum = p / p.iloc[0]
    running_max = cum.cummax()
    dd = cum / running_max - 1.0
    mdd = float(dd.min())
    mdd_end = dd.idxmin()
    underwater = dd < 0
    duration = max_duration = 0
    for flag in underwater:
        if flag:
            duration += 1
            max_duration = max(max_duration, duration)
        else:
            duration = 0
    recovery_days = None
    try:
        trough_loc = dd.index.get_loc(mdd_end)
        post = cum.iloc[trough_loc:]
        peak_before = running_max.loc[mdd_end]
        recovered = post[post >= peak_before]
        if len(recovered):
            recovery_days = int((recovered.index[0] - mdd_end).days) if hasattr(mdd_end, "to_pydatetime") else None
    except Exception:
        pass
    return {
        "ok": True, "max_drawdown": mdd, "max_drawdown_pct": mdd * 100,
        "mdd_date": mdd_end, "max_underwater_days": int(max_duration),
        "recovery_days_after_trough": recovery_days, "drawdown_series": dd,
    }


def evt_pot_estimate(returns: pd.Series, threshold_quantile: float = 0.05, alpha: float = 0.01) -> Dict[str, Any]:
    """Peaks-Over-Threshold with GPD. ADVANCED STATISTICAL ESTIMATE — not a crash prediction."""
    r = returns.dropna().astype(float)
    n = len(r)
    if n < 100:
        return {
            "ok": False, "message": "Need \~100+ obs for reliable EVT/POT.",
            "n_obs": n, "label": "ADVANCED STATISTICAL ESTIMATE — insufficient data",
        }
    losses = -r
    u = float(np.quantile(losses, 1 - threshold_quantile))
    exceedances = losses[losses > u] - u
    n_exc = len(exceedances)
    if n_exc < 20:
        return {
            "ok": False, "message": f"Too few exceedances ({n_exc}).",
            "threshold": u, "n_exceedances": n_exc,
            "label": "ADVANCED STATISTICAL ESTIMATE — sparse exceedances",
        }
    try:
        shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
        p = alpha
        if abs(shape) > 1e-6:
            var_evt = u + (scale / shape) * (((n / n_exc) * p) ** (-shape) - 1)
        else:
            var_evt = u - scale * np.log((n / n_exc) * p)
        if shape < 1:
            es_evt = var_evt / (1 - shape) + (scale - shape * u) / (1 - shape)
        else:
            es_evt = np.nan
        return {
            "ok": True, "threshold_u": float(u), "n_exceedances": int(n_exc),
            "shape_xi": float(shape), "scale_beta": float(scale),
            "var_evt": float(-var_evt), "es_evt": float(-es_evt) if np.isfinite(es_evt) else None,
            "alpha": alpha, "threshold_quantile": threshold_quantile,
            "method": "EVT Peaks-Over-Threshold (GPD)",
            "label": "ADVANCED STATISTICAL ESTIMATE — not a crash prediction",
            "n_obs": n,
        }
    except Exception as e:
        return {
            "ok": False, "message": f"EVT fit failed: {e}",
            "label": "ADVANCED STATISTICAL ESTIMATE — fit failed",
        }


def run_tail_risk_suite(returns: pd.Series, prices: Optional[pd.Series] = None) -> Dict[str, Any]:
    es = expected_shortfall(returns)
    mc = student_t_monte_carlo(returns)
    mdd = max_drawdown_analysis(prices) if prices is not None else {"ok": False, "message": "No prices"}
    evt = evt_pot_estimate(returns)
    r = returns.dropna()
    tail_prob_5 = float((r < -0.05).mean()) if len(r) else np.nan
    return {
        "expected_shortfall": es, "student_t_mc": mc, "max_drawdown": mdd,
        "evt": evt, "tail_prob_loss_gt_5pct": tail_prob_5,
    }