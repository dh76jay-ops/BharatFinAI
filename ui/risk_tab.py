"""
BharatFinAI — V8 Risk Engine dashboard.

The UI layer only coordinates user inputs, cached market-data retrieval, and
presentation. Statistical models remain in core/volatility_models.py.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.risk_engine import run_v8_risk_engine
from data.market_data import fetch_price_history


def _fmt_pct(value: Any, decimals: int = 2) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(x):
        return "—"
    return f"{x * 100:.{decimals}f}%"


def _fmt_rupees(value: Any, decimals: int = 0) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(x):
        return "—"
    return f"₹{x:,.{decimals}f}"


def _reliability_label(result: Dict[str, Any]) -> str:
    return str(result.get("reliability", {}).get("level", "LOW"))


def _render_reliability(result: Dict[str, Any]) -> None:
    st.markdown("### Model Reliability")

    rows = []

    def add_model(name: str, payload: Optional[Dict[str, Any]]) -> None:
        if not isinstance(payload, dict):
            return
        rel = payload.get("reliability", {})
        if not isinstance(rel, dict):
            return
        rows.append(
            {
                "Model": name,
                "Reliability": rel.get("level", "LOW"),
                "Observations": rel.get("n_obs", "—"),
                "Converged": (
                    "Yes"
                    if rel.get("converged") is True
                    else "No"
                    if rel.get("converged") is False
                    else "N/A"
                ),
                "Warnings": "; ".join(rel.get("flags", [])) or "None",
            }
        )

    add_model("EWMA", result.get("ewma"))
    add_model("Realized Volatility", result.get("realized"))
    add_model("GARCH(1,1)", result.get("garch"))

    for key, value in result.get("var_suite", {}).items():
        add_model(key.replace("_", " ").title() + " VaR", value)

    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    overall = _reliability_label(result)
    note = result.get("reliability", {}).get("note")
    st.caption(f"Overall V8 model reliability: **{overall}**. {note or ''}")


def _render_data_quality(result: Dict[str, Any]) -> None:
    st.markdown("### Data Quality")
    q = result.get("data_quality", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("Quality Score", f"{q.get('score', 0)}/100")
    c2.metric("Observations", f"{q.get('n_obs', 0):,}")
    c3.metric("Missing", f"{q.get('missing_pct', 0):.1f}%")

    st.progress(max(0.0, min(1.0, float(q.get("score", 0)) / 100.0)))

    warnings = q.get("warnings", [])
    if warnings:
        with st.expander("Data quality warnings", expanded=False):
            for warning in warnings[:10]:
                st.warning(warning)

    st.caption(
        f"Corporate-action check: {q.get('corporate_action_check', 'NOT_VERIFIED')}. "
        "Large moves are flagged for inspection rather than silently discarded."
    )


def _render_volatility(result: Dict[str, Any]) -> None:
    st.markdown("### Volatility Analysis")

    ewma = result.get("ewma", {})
    realized = result.get("realized", {})
    garch = result.get("garch", {})

    cols = st.columns(5)
    cols[0].metric("Historical", _fmt_pct(realized.get("historical_vol")))
    cols[1].metric("EWMA", _fmt_pct(ewma.get("current_vol")))
    cols[2].metric("Realized 20D", _fmt_pct(realized.get("windows", {}).get("20D")))
    cols[3].metric("Realized 60D", _fmt_pct(realized.get("windows", {}).get("60D")))
    cols[4].metric("GARCH Current", _fmt_pct(garch.get("current_vol")))

    if garch.get("ok"):
        st.metric("GARCH Next-Period Forecast", _fmt_pct(garch.get("forecast_vol")))
        params = garch.get("params", {})
        st.caption(
            "GARCH parameters: "
            f"ω={params.get('omega', float('nan')):.4g}, "
            f"α={params.get('alpha', float('nan')):.4g}, "
            f"β={params.get('beta', float('nan')):.4g}, "
            f"α+β={garch.get('persistence', float('nan')):.4g}. "
            f"Convergence flag: {garch.get('convergence_flag', 'N/A')}."
        )
    else:
        st.info(garch.get("message", "GARCH result unavailable."))

    series_map = {}
    if ewma.get("ok") and isinstance(ewma.get("series"), pd.Series):
        series_map["EWMA"] = ewma["series"]
    if garch.get("ok") and isinstance(garch.get("series"), pd.Series):
        series_map["GARCH"] = garch["series"]
    if realized.get("ok"):
        for key in ("20D", "60D"):
            s = realized.get("series", {}).get(key)
            if isinstance(s, pd.Series):
                series_map[f"Realized {key}"] = s

    if series_map:
        chart_df = pd.concat(series_map, axis=1).dropna(how="all")
        fig = go.Figure()
        for name in chart_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=chart_df.index,
                    y=chart_df[name] * 100.0,
                    mode="lines",
                    name=name,
                )
            )
        fig.update_layout(
            template="plotly_dark",
            title="Annualized Volatility Comparison",
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            legend_title="Model",
        )
        st.plotly_chart(fig, width="stretch")

    regime = result.get("regime", {})
    regime_name = regime.get("regime", "UNKNOWN")
    st.markdown("### Volatility Regime")
    c1, c2 = st.columns(2)
    c1.metric("Current Regime", regime_name)
    percentile = regime.get("percentile")
    c2.metric(
        "Historical Volatility Percentile",
        "—" if percentile is None else f"{percentile:.1f}th",
    )

    messages = {
        "LOW": "LOW volatility regime: modeled volatility is below the configured low-risk thresholds.",
        "NORMAL": "NORMAL volatility regime: modeled volatility is within the normal range.",
        "HIGH": "HIGH volatility regime: risk estimates are elevated relative to configured thresholds/history.",
        "EXTREME": "EXTREME volatility regime: modeled volatility is unusually elevated.",
    }
    message = messages.get(regime_name, "Volatility regime is unavailable.")
    if regime_name in ("HIGH", "EXTREME"):
        st.warning(message)
    else:
        st.info(message)
    st.caption(
        "Regime labels are statistical classifications, not predictions of the next market move."
    )


def _render_var(result: Dict[str, Any], capital: float) -> None:
    st.markdown("### Dynamic VaR")

    rows = []
    for key, payload in result.get("var_suite", {}).items():
        if not isinstance(payload, dict):
            continue
        value = payload.get("var")
        if value is None or not np.isfinite(float(value)):
            continue
        rows.append(
            {
                "Model": key.replace("_", " ").title(),
                "VaR Return": _fmt_pct(value, 3),
                "Estimated Loss": _fmt_rupees(capital * abs(float(value))),
                "Reliability": payload.get("reliability", {}).get("level", "LOW"),
            }
        )

    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    primary = result.get("primary_var", {})
    st.caption(
        f"Primary display model: **{primary.get('model', 'none')}**. "
        "VaR is a quantile estimate under a model/history assumption; it is not a guarantee."
    )


def _render_position_sizing(result: Dict[str, Any]) -> None:
    st.markdown("### Dynamic Position Sizing")

    sizing = result.get("position_sizing", {})
    if not sizing.get("ok"):
        st.info(sizing.get("message", "Sizing scenario unavailable."))
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Scenario Quantity", f"{sizing.get('suggested_qty', 0):,.2f}")
    c2.metric("Scenario Value", _fmt_rupees(sizing.get("suggested_value")))
    c3.metric("Scenario Weight", f"{sizing.get('suggested_weight_pct', 0):.2f}%")

    comp = sizing.get("components", {})
    st.dataframe(
        pd.DataFrame(
            [
                ["Stop-distance constraint", comp.get("stop_based_qty")],
                ["Volatility constraint", comp.get("vol_based_qty")],
                ["VaR constraint", comp.get("var_based_qty")],
                ["Reliability dampener", comp.get("confidence_dampener")],
            ],
            columns=["Sizing Component", "Value"],
        ),
        width="stretch",
        hide_index=True,
    )

    basis = sizing.get("confidence_basis", {})
    if basis:
        st.caption(
            f"Reliability factor = min(data quality {basis.get('data_quality_factor', 0):.2f}, "
            f"primary-VaR reliability {basis.get('primary_var_reliability_factor', 0):.2f})."
        )

    st.warning(
        "This is an educational research-sizing scenario. It is not personalized "
        "financial advice and does not guarantee a risk outcome."
    )


def render_risk_tab() -> None:
    st.markdown("## 🛡️ Risk Engine — V8 Dynamic Risk")
    st.caption(
        "GARCH + EWMA + realized volatility + volatility regime + dynamic VaR + "
        "research position-sizing."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        symbol = st.text_input(
            "Stock Symbol",
            value="RELIANCE",
            key="risk_stock",
        ).strip().upper()
    with c2:
        period = st.selectbox(
            "Lookback Period",
            ["6mo", "1y", "2y", "5y"],
            index=1,
            key="risk_period",
        )
    with c3:
        investment = st.number_input(
            "Capital / Investment (₹)",
            min_value=1000.0,
            value=100000.0,
            step=1000.0,
            key="risk_investment",
        )

    c4, c5, c6, c7 = st.columns(4)
    with c4:
        confidence_label = st.selectbox(
            "VaR Confidence",
            ["95%", "99%"],
            index=0,
            key="risk_var_confidence",
        )
    with c5:
        ewma_lambda = st.number_input(
            "EWMA Lambda",
            min_value=0.80,
            max_value=0.99,
            value=0.94,
            step=0.01,
            format="%.2f",
            key="risk_ewma_lambda",
        )
    with c6:
        stop_loss_pct = st.number_input(
            "Stop Distance (%)",
            min_value=0.5,
            max_value=50.0,
            value=5.0,
            step=0.5,
            key="risk_stop_distance",
        )
    with c7:
        risk_budget_pct = st.number_input(
            "Risk Budget (% capital)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            key="risk_budget",
        )

    c8, c9, c10 = st.columns(3)
    with c8:
        max_position_pct = st.number_input(
            "Maximum Position (%)",
            min_value=1.0,
            max_value=100.0,
            value=25.0,
            step=1.0,
            key="risk_max_position",
        )
    with c9:
        entry_price = st.number_input(
            "Entry Price (₹, 0 = latest close)",
            min_value=0.0,
            value=0.0,
            step=0.05,
            key="risk_entry_price",
        )
    with c10:
        run_garch = st.checkbox(
            "Run GARCH(1,1)",
            value=True,
            key="risk_run_garch",
            help="Disable for a faster EWMA/realized/VaR-only calculation.",
        )

    clicked = st.button("Calculate V8 Risk", key="calculate_risk_v8_btn", type="primary")

    if clicked:
        with st.spinner("Fetching market data and running V8 risk models..."):
            df = fetch_price_history(symbol, period=period, auto_adjust=True)

            if df.empty or "Close" not in df.columns:
                st.session_state["risk_v8_result"] = {
                    "ok": False,
                    "message": "Market data unavailable. Check the symbol and data source.",
                }
            else:
                close = pd.to_numeric(df["Close"], errors="coerce").dropna()
                returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

                if len(returns) == 0:
                    st.session_state["risk_v8_result"] = {
                        "ok": False,
                        "message": "Insufficient price history to calculate returns.",
                    }
                else:
                    effective_entry = float(entry_price) if entry_price > 0 else float(close.iloc[-1])
                    alpha = 0.05 if confidence_label == "95%" else 0.01

                    result = run_v8_risk_engine(
                        returns,
                        capital=float(investment),
                        entry_price=effective_entry,
                        stop_loss_pct=float(stop_loss_pct),
                        risk_budget_pct=float(risk_budget_pct),
                        max_position_pct=float(max_position_pct),
                        ewma_lambda=float(ewma_lambda),
                        var_alpha=alpha,
                        run_garch=bool(run_garch),
                    )

                    result["_ui"] = {
                        "symbol": symbol,
                        "period": period,
                        "latest_price": float(close.iloc[-1]),
                        "entry_price_used": effective_entry,
                        "capital": float(investment),
                    }
                    st.session_state["risk_v8_result"] = result

    result = st.session_state.get("risk_v8_result")
    if not result:
        st.info("Choose your settings and click **Calculate V8 Risk**.")
        return

    if not result.get("ok"):
        st.error(result.get("message", "Risk calculation failed safely."))
        if result.get("data_quality"):
            _render_data_quality(result)
        return

    meta = result.get("_ui", {})
    st.caption(
        f"{meta.get('symbol', symbol)} · {meta.get('period', period)} · "
        f"latest close {_fmt_rupees(meta.get('latest_price'))} · "
        f"entry price used {_fmt_rupees(meta.get('entry_price_used'))}"
    )

    # V8 summary
    st.markdown("### V8 Risk Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Volatility Regime", result.get("regime", {}).get("regime", "UNKNOWN"))
    c2.metric("Primary VaR Model", result.get("primary_var", {}).get("model", "—"))
    c3.metric("Primary VaR", _fmt_pct(result.get("primary_var", {}).get("return_var")))
    c4.metric("Data Quality", f"{result.get('data_quality', {}).get('score', 0)}/100")

    if result.get("regime", {}).get("regime") in ("HIGH", "EXTREME"):
        st.warning(result.get("regime_label", "Elevated volatility regime."))
    else:
        st.info(result.get("regime_label", "Volatility regime calculated."))

    _render_data_quality(result)
    _render_volatility(result)
    _render_var(result, capital=float(meta.get("capital", investment)))
    _render_position_sizing(result)

    with st.expander("V8 Methodology & Assumptions", expanded=False):
        method = result.get("methodology", {})
        st.write("**Primary VaR selection:**", method.get("primary_var_rule"))
        st.write("**Primary volatility selection:**", method.get("primary_volatility_rule"))
        st.write("**Position-sizing rule:**", method.get("position_sizing_rule"))
        st.write(
            "**Safety statement:**",
            method.get("safety"),
        )

    st.caption(
        "Research/educational use only. Historical observations, statistical estimates "
        "and scenario calculations do not determine future returns."
    )
