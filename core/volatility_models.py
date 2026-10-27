"""
BharatFinAI Risk Engine V8 — Dynamic Volatility Models
GARCH(1,1), EWMA, Realized Vol, Regime, Dynamic VaR, Position Sizing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings("ignore")


def _reliability(n_obs: int, converged: bool = True, min_obs: int = 60, extra_flags: Optional[List[str]] = None) -> Dict[str, Any]:
    flags = list(extra_flags or [])
    if n_obs < min_obs:
        level = "INSUFFICIENT DATA"
    elif not converged:
        level = "LOW"
        flags.append("Model did not converge cleanly.")
    elif n_obs < 120:
        level = "MEDIUM"
    else:
        level = "HIGH"
    return {"level": level, "n_obs": n_obs, "converged": converged, "flags": flags, "lookback": n_obs}


def ewma_volatility(returns: pd.Series, lambda_: float = 0.94, annualize: bool = True) -> Dict[str, Any]:
    r = returns.dropna().astype(float)
    n = len(r)
    if n < 20:
        return {"ok": False, "message": "Insufficient data for EWMA.", "reliability": _reliability(n, min_obs=20)}
    lambda_ = float(np.clip(lambda_, 0.80, 0.99))
    var = r.var()
    ewma_var = []
    for x in r.values:
        var = lambda_ * var + (1.0 - lambda_) * (x ** 2)
        ewma_var.append(var)
    ewma_vol = np.sqrt(np.array(ewma_var))
    if annualize:
        ewma_vol = ewma_vol * np.sqrt(252)
    series = pd.Series(ewma_vol, index=r.index, name="EWMA_Vol")
    return {
        "ok": True, "current_vol": float(ewma_vol[-1]), "series": series,
        "lambda": lambda_, "annualized": annualize,
        "reliability": _reliability(n, min_obs=20), "method": "EWMA",
    }


def realized_volatility(returns: pd.Series, windows: Optional[List[int]] = None, annualize: bool = True) -> Dict[str, Any]:
    r = returns.dropna().astype(float)
    n = len(r)
    if windows is None:
        windows = [20, 60]
    if n < min(windows):
        return {"ok": False, "message": "Insufficient data for realized vol.", "reliability": _reliability(n, min_obs=min(windows))}
    factor = np.sqrt(252) if annualize else 1.0
    results, series_dict = {}, {}
    for w in windows:
        if n >= w:
            s = r.rolling(w).std() * factor
            results[f"{w}D"] = float(s.iloc[-1]) if pd.notna(s.iloc[-1]) else np.nan
            series_dict[f"{w}D"] = s
        else:
            results[f"{w}D"] = np.nan
    return {
        "ok": True, "windows": results, "series": series_dict,
        "historical_vol": float(r.std() * factor), "annualized": annualize,
        "reliability": _reliability(n, min_obs=min(windows)), "method": "Realized",
    }


def garch_volatility(returns: pd.Series, annualize: bool = True) -> Dict[str, Any]:
    r = returns.dropna().astype(float)
    n = len(r)
    if n < 60:
        return {"ok": False, "message": "Need ~60+ obs for GARCH.", "reliability": _reliability(n, min_obs=60)}
    try:
        from arch import arch_model
    except ImportError:
        return {"ok": False, "message": "arch library not installed.", "reliability": _reliability(n, converged=False, extra_flags=["arch missing"])}
    r_pct = r * 100.0
    try:
        am = arch_model(r_pct, vol="Garch", p=1, q=1, rescale=False)
        res = am.fit(disp="off", show_warning=False)
        cond_vol = res.conditional_volatility / 100.0
        if annualize:
            cond_vol = cond_vol * np.sqrt(252)
        fcast = res.forecast(horizon=1)
        var_f = float(fcast.variance.values[-1, 0])
        vol_f = np.sqrt(var_f) / 100.0
        if annualize:
            vol_f *= np.sqrt(252)
        params = {
            "omega": float(res.params.get("omega", np.nan)),
            "alpha": float(res.params.get("alpha[1]", np.nan)),
            "beta": float(res.params.get("beta[1]", np.nan)),
        }
        persistence = params["alpha"] + params["beta"] if pd.notna(params["alpha"]) else np.nan
        series = pd.Series(cond_vol.values, index=r.index[-len(cond_vol):], name="GARCH_Vol")
        flags = []
        raw_convergence_flag = getattr(res, "convergence_flag", None)

        try:
            convergence_flag = (
                int(raw_convergence_flag)
                if raw_convergence_flag is not None
                else None
            )
        except (TypeError, ValueError, OverflowError):
            convergence_flag = None

        converged = convergence_flag == 0

        if convergence_flag is None:
            flags.append("Optimizer convergence status unavailable.")
        elif not converged:
            flags.append(
                f"GARCH optimizer did not converge (flag={convergence_flag})."
            )

        if pd.notna(persistence) and persistence >= 1.0:
            flags.append("High persistence (alpha+beta >= 1).")
            converged = False

        return {
            "ok": True,
            "current_vol": float(series.iloc[-1]),
            "forecast_vol": float(vol_f),
            "series": series,
            "params": params,
            "persistence": (
                float(persistence) if pd.notna(persistence) else None
            ),
            "loglikelihood": float(res.loglikelihood),
            "aic": float(res.aic),
            "annualized": annualize,
            "convergence_flag": convergence_flag,
            "reliability": _reliability(
                n,
                converged=converged,
                min_obs=60,
                extra_flags=flags,
            ),
            "method": "GARCH(1,1)",
        }
    except Exception as e:
        return {"ok": False, "message": f"GARCH failed: {e}", "reliability": _reliability(n, converged=False, extra_flags=[str(e)])}


def volatility_regime(current_vol: float, historical_vol_series: Optional[pd.Series] = None,
                      low_thresh: float = 0.12, high_thresh: float = 0.25, extreme_thresh: float = 0.40) -> Dict[str, Any]:
    if current_vol is None or not np.isfinite(current_vol):
        return {"regime": "UNKNOWN", "current_vol": current_vol, "percentile": None}
    if current_vol < low_thresh:
        regime = "LOW"
    elif current_vol < high_thresh:
        regime = "NORMAL"
    elif current_vol < extreme_thresh:
        regime = "HIGH"
    else:
        regime = "EXTREME"
    pct = None
    if historical_vol_series is not None and len(historical_vol_series.dropna()) > 30:
        s = historical_vol_series.dropna()
        pct = float((s < current_vol).mean() * 100)
        if pct >= 95:
            regime = "EXTREME"
        elif pct >= 80:
            regime = "HIGH"
        elif pct <= 20:
            regime = "LOW"
    return {"regime": regime, "current_vol": float(current_vol), "percentile": pct,
            "thresholds": {"low": low_thresh, "high": high_thresh, "extreme": extreme_thresh}}


def historical_var(returns: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
    r = returns.dropna().astype(float)
    n = len(r)
    if n < 30:
        return {"ok": False, "message": "Insufficient data for Historical VaR.", "reliability": _reliability(n, min_obs=30)}
    return {"ok": True, "var": float(np.percentile(r, alpha * 100)), "alpha": alpha,
            "method": "Historical", "reliability": _reliability(n, min_obs=30)}


def parametric_var(returns: pd.Series, alpha: float = 0.05, vol: Optional[float] = None) -> Dict[str, Any]:
    from scipy import stats
    r = returns.dropna().astype(float)
    n = len(r)
    if n < 30:
        return {"ok": False, "message": "Insufficient data for Parametric VaR.", "reliability": _reliability(n, min_obs=30)}
    mu = float(r.mean())
    sigma = float(r.std()) if vol is None else float(vol) / np.sqrt(252)
    z = float(stats.norm.ppf(alpha))
    return {"ok": True, "var": float(mu + z * sigma), "mu": mu, "sigma_daily": sigma,
            "alpha": alpha, "method": "Parametric (Normal)", "reliability": _reliability(n, min_obs=30)}


def student_t_var(returns: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
    from scipy import stats
    r = returns.dropna().astype(float)
    n = len(r)
    if n < 40:
        return {"ok": False, "message": "Insufficient data for Student-t VaR.", "reliability": _reliability(n, min_obs=40)}
    try:
        df_fit, loc, scale = stats.t.fit(r)
        var = float(stats.t.ppf(alpha, df_fit, loc=loc, scale=scale))
        return {"ok": True, "var": var, "df": float(df_fit), "loc": float(loc), "scale": float(scale),
                "alpha": alpha, "method": "Student-t", "reliability": _reliability(n, min_obs=40)}
    except Exception as e:
        return {"ok": False, "message": f"Student-t fit failed: {e}", "reliability": _reliability(n, converged=False)}


def garch_var(returns: pd.Series, garch_result: Dict[str, Any], alpha: float = 0.05) -> Dict[str, Any]:
    from scipy import stats
    if not garch_result.get("ok"):
        return {"ok": False, "message": "GARCH result not available.", "reliability": garch_result.get("reliability")}
    r = returns.dropna().astype(float)
    ann_f = garch_result.get("forecast_vol")
    if ann_f is None or not np.isfinite(ann_f):
        return {"ok": False, "message": "No GARCH forecast.", "reliability": garch_result.get("reliability")}
    daily_sigma = float(ann_f) / np.sqrt(252)
    mu = float(r.mean())
    z = float(stats.norm.ppf(alpha))
    return {"ok": True, "var": float(mu + z * daily_sigma), "daily_sigma_forecast": daily_sigma,
            "alpha": alpha, "method": "GARCH-Normal", "reliability": garch_result.get("reliability")}


def compute_dynamic_var_suite(returns: pd.Series, garch_result: Optional[Dict[str, Any]] = None, alpha: float = 0.05) -> Dict[str, Any]:
    out = {
        "historical": historical_var(returns, alpha),
        "parametric": parametric_var(returns, alpha),
        "student_t": student_t_var(returns, alpha),
    }
    if garch_result is not None:
        out["garch"] = garch_var(returns, garch_result, alpha)
    return out


def dynamic_position_size(capital: float, entry_price: float, stop_loss_pct: float,
                          annual_vol: float, var_daily: float, risk_budget_pct: float = 1.0,
                          max_position_pct: float = 25.0, confidence: float = 0.7) -> Dict[str, Any]:
    if capital <= 0 or entry_price <= 0:
        return {"ok": False, "message": "Invalid capital or price."}
    stop_loss_pct = abs(float(stop_loss_pct)) or 0.02
    risk_amount = capital * (risk_budget_pct / 100.0)
    stop_distance = entry_price * stop_loss_pct
    qty_stop = risk_amount / stop_distance if stop_distance > 0 else 0
    target_port_vol = 0.10
    if annual_vol and annual_vol > 0:
        w_vol = min(max_position_pct / 100.0, target_port_vol / (annual_vol + 1e-9))
        qty_vol = (w_vol * capital) / entry_price
    else:
        qty_vol = qty_stop
    var_abs = abs(var_daily) if var_daily is not None else 0.02
    qty_var = risk_amount / (entry_price * var_abs) if var_abs > 0 else qty_stop
    raw_qty = min(qty_stop, qty_vol, qty_var)
    dampened_qty = raw_qty * float(np.clip(confidence, 0.3, 1.0))
    max_qty = (max_position_pct / 100.0 * capital) / entry_price
    final_qty = max(0.0, min(dampened_qty, max_qty))
    final_value = final_qty * entry_price
    return {
        "ok": True,
        "suggested_qty": float(round(final_qty, 2)),
        "suggested_value": float(round(final_value, 2)),
        "suggested_weight_pct": float(round(final_value / capital * 100.0, 2)),
        "components": {
            "stop_based_qty": float(round(qty_stop, 2)),
            "vol_based_qty": float(round(qty_vol, 2)),
            "var_based_qty": float(round(qty_var, 2)),
            "confidence_dampener": float(confidence),
        },
        "disclaimer": "Educational position-sizing research only. Not personalized financial advice.",
    }


def run_v8_dynamic_volatility(returns: pd.Series, ewma_lambda: float = 0.94, var_alpha: float = 0.05) -> Dict[str, Any]:
    from core.data_quality import assess_returns_quality
    quality = assess_returns_quality(returns)
    if not quality["usable"]:
        return {"ok": False, "message": "Insufficient data for reliable estimation.", "data_quality": quality}
    ewma = ewma_volatility(returns, lambda_=ewma_lambda)
    realized = realized_volatility(returns, windows=[20, 60])
    garch = garch_volatility(returns)
    cur_vol, vol_series = None, None
    if garch.get("ok"):
        cur_vol, vol_series = garch["current_vol"], garch.get("series")
    elif ewma.get("ok"):
        cur_vol, vol_series = ewma["current_vol"], ewma.get("series")
    elif realized.get("ok"):
        cur_vol = realized.get("historical_vol")
    regime = volatility_regime(cur_vol, vol_series) if cur_vol is not None else {"regime": "UNKNOWN", "current_vol": None}
    var_suite = compute_dynamic_var_suite(returns, garch if garch.get("ok") else None, alpha=var_alpha)
    return {
        "ok": True, "data_quality": quality, "ewma": ewma, "realized": realized,
        "garch": garch, "regime": regime, "var_suite": var_suite, "current_vol_used": cur_vol,
    }