"""
BharatFinAI — V8 Risk Engine Orchestrator.

Research-grade orchestration for Dynamic Volatility + Dynamic VaR.
The individual statistical models live in volatility_models.py; this module
coordinates data quality, model reliability, model selection and position-sizing
scenarios without changing legacy analytics.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from core.data_quality import assess_returns_quality
from core.volatility_models import (
    compute_dynamic_var_suite,
    dynamic_position_size,
    ewma_volatility,
    garch_volatility,
    realized_volatility,
    volatility_regime,
)


_RELIABILITY_FACTOR = {
    "HIGH": 1.00,
    "MEDIUM": 0.80,
    "LOW": 0.50,
    "INSUFFICIENT DATA": 0.00,
}


def _finite_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _model_reliability_factor(level: Optional[str]) -> float:
    return float(_RELIABILITY_FACTOR.get(str(level or "").upper(), 0.0))


def _pick_primary_var(var_suite: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    """
    Prefer Student-t VaR when available, then GARCH, then Historical,
    then Parametric. This is a selection rule for research display, not a
    claim that one model is universally superior.
    """
    for name in ("student_t", "garch", "historical", "parametric"):
        result = var_suite.get(name)
        if isinstance(result, dict) and result.get("ok") and _finite_float(result.get("var")) is not None:
            return name, result
    return "none", {}


def _risk_regime_label(regime: str) -> str:
    return {
        "LOW": "LOW RISK REGIME",
        "NORMAL": "NORMAL RISK REGIME",
        "HIGH": "HIGH RISK REGIME",
        "EXTREME": "EXTREME RISK REGIME",
    }.get(regime, "UNCLASSIFIED")


def run_v8_risk_engine(
    returns: pd.Series,
    *,
    capital: Optional[float] = None,
    entry_price: Optional[float] = None,
    stop_loss_pct: float = 5.0,
    risk_budget_pct: float = 1.0,
    max_position_pct: float = 25.0,
    ewma_lambda: float = 0.94,
    var_alpha: float = 0.05,
    run_garch: bool = True,
    benchmark_returns: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Run the V8 research-risk suite.

    Returns a structured dictionary so the UI can fail gracefully when one
    statistical model is unavailable or cannot converge.
    """
    if returns is None:
        return {
            "ok": False,
            "message": "Insufficient data for reliable estimation.",
            "data_quality": assess_returns_quality(pd.Series(dtype=float)),
        }

    r = pd.Series(returns).copy()
    quality = assess_returns_quality(r)

    # V8 contains GARCH and dynamic VaR; use a conservative 60-observation
    # publication gate even when an individual lighter-weight model could run
    # with fewer observations.
    v8_min_obs = 60
    if quality.get("n_obs", 0) < v8_min_obs or not quality["usable"]:
        return {
            "ok": False,
            "message": "Insufficient data for reliable estimation.",
            "data_quality": quality,
            "model_reliability": "INSUFFICIENT DATA",
        }

    ewma = ewma_volatility(r, lambda_=ewma_lambda)
    realized = realized_volatility(r, windows=[20, 60])

    if run_garch:
        garch = garch_volatility(r)
    else:
        garch = {
            "ok": False,
            "message": "GARCH execution was skipped for this run.",
            "reliability": {
                "level": "MEDIUM",
                "n_obs": int(r.dropna().shape[0]),
                "converged": None,
                "flags": ["Lazy execution: GARCH skipped."],
                "lookback": int(r.dropna().shape[0]),
            },
        }

    # Primary volatility source: GARCH -> EWMA -> 60D realized -> historical.
    current_vol = None
    current_vol_method = "Unavailable"
    vol_series = None

    for name, result in (
        ("GARCH(1,1)", garch),
        ("EWMA", ewma),
    ):
        if not isinstance(result, dict) or not result.get("ok"):
            continue
        if name == "GARCH(1,1)":
            rel = result.get("reliability", {})
            if isinstance(rel, dict) and rel.get("converged") is not True:
                continue
        current_vol = _finite_float(result.get("current_vol"))
        vol_series = result.get("series")
        if current_vol is not None:
            current_vol_method = name
            break

    if current_vol is None and realized.get("ok"):
        current_vol = _finite_float(realized.get("windows", {}).get("60D"))
        if current_vol is not None:
            current_vol_method = "Realized 60D"
            vol_series = realized.get("series", {}).get("60D")

    if current_vol is None:
        current_vol = _finite_float(realized.get("historical_vol")) if realized.get("ok") else None
        if current_vol is not None:
            current_vol_method = "Historical"
            vol_series = r.dropna()

    regime = (
        volatility_regime(current_vol, vol_series)
        if current_vol is not None
        else {"regime": "UNKNOWN", "current_vol": None, "percentile": None}
    )

    garch_for_var = None
    if garch.get("ok"):
        rel = garch.get("reliability", {})
        if not isinstance(rel, dict) or rel.get("converged") is True:
            garch_for_var = garch

    var_suite = compute_dynamic_var_suite(
        r,
        garch_for_var,
        alpha=float(np.clip(var_alpha, 0.001, 0.20)),
    )
    primary_var_name, primary_var = _pick_primary_var(var_suite)

    # Reliability is deliberately reported, not silently converted into
    # false precision.
    model_levels = []
    for model in (ewma, realized, garch):
        rel = model.get("reliability") if isinstance(model, dict) else None
        if isinstance(rel, dict):
            model_levels.append(str(rel.get("level", "LOW")))
    for result in var_suite.values():
        rel = result.get("reliability") if isinstance(result, dict) else None
        if isinstance(rel, dict):
            model_levels.append(str(rel.get("level", "LOW")))

    reliability_rank = {"INSUFFICIENT DATA": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
    overall_level = (
        min(model_levels, key=lambda x: reliability_rank.get(x, 0))
        if model_levels
        else "LOW"
    )

    # Position sizing is calculated only when the user has supplied the
    # required capital and entry price. It is a scenario, not advice.
    position_sizing = {
        "ok": False,
        "message": "Provide valid capital and entry price to run sizing scenario.",
    }
    if capital is not None and entry_price is not None:
        annual_vol = (
            _finite_float(garch.get("forecast_vol"))
            if garch.get("ok")
            else None
        )
        if annual_vol is None:
            annual_vol = _finite_float(ewma.get("current_vol"))
        if annual_vol is None:
            annual_vol = _finite_float(current_vol)

        var_daily = _finite_float(primary_var.get("var"))
        chosen_rel = primary_var.get("reliability", {}) if isinstance(primary_var, dict) else {}
        data_factor = float(np.clip(quality.get("score", 0) / 100.0, 0.0, 1.0))
        model_factor = _model_reliability_factor(
            chosen_rel.get("level") if isinstance(chosen_rel, dict) else None
        )
        confidence = float(min(data_factor, model_factor))

        position_sizing = dynamic_position_size(
            capital=float(capital),
            entry_price=float(entry_price),
            stop_loss_pct=float(stop_loss_pct),
            annual_vol=float(annual_vol or 0.0),
            var_daily=float(var_daily if var_daily is not None else 0.0),
            risk_budget_pct=float(risk_budget_pct),
            max_position_pct=float(max_position_pct),
            confidence=float(confidence),
        )
        if position_sizing.get("ok"):
            position_sizing["confidence_factor"] = confidence
            position_sizing["confidence_basis"] = {
                "data_quality_factor": data_factor,
                "primary_var_reliability_factor": model_factor,
                "primary_var_model": primary_var_name,
            }

    benchmark_status = {
        "available": benchmark_returns is not None and len(pd.Series(benchmark_returns).dropna()) > 0,
        "n_obs": int(len(pd.Series(benchmark_returns).dropna())) if benchmark_returns is not None else 0,
        "message": (
            "Benchmark series available for future beta/regime extensions."
            if benchmark_returns is not None and len(pd.Series(benchmark_returns).dropna()) > 0
            else "Benchmark series not supplied for this V8 run."
        ),
    }

    return {
        "ok": True,
        "data_quality": quality,
        "reliability": {
            "level": overall_level,
            "model_levels": model_levels,
            "note": "Reliability reflects data/model conditions; it is not forecast accuracy.",
        },
        "ewma": ewma,
        "realized": realized,
        "garch": garch,
        "current_volatility_method": current_vol_method,
        "regime": regime,
        "regime_label": _risk_regime_label(regime.get("regime", "UNKNOWN")),
        "var_suite": var_suite,
        "primary_var": {
            "model": primary_var_name,
            "return_var": _finite_float(primary_var.get("var")),
            "reliability": primary_var.get("reliability", {}),
        },
        "position_sizing": position_sizing,
        "benchmark": benchmark_status,
        "parameters": {
            "ewma_lambda": float(np.clip(ewma_lambda, 0.80, 0.99)),
            "var_alpha": float(np.clip(var_alpha, 0.001, 0.20)),
            "stop_loss_pct": float(abs(stop_loss_pct)),
            "risk_budget_pct": float(abs(risk_budget_pct)),
            "max_position_pct": float(abs(max_position_pct)),
        },
        "methodology": {
            "primary_var_rule": "Student-t -> GARCH -> Historical -> Parametric, based on availability/reliability.",
            "primary_volatility_rule": "GARCH forecast/current -> EWMA -> realized 60D -> historical.",
            "position_sizing_rule": (
                "Uses stop-distance, volatility and VaR constraints; the final quantity "
                "is damped by min(data-quality factor, primary-VaR reliability factor)."
            ),
            "safety": "All outputs are statistical estimates/scenario analysis for research and education.",
        },
    }
