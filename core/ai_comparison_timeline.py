"""Data validation, performance metrics, and Plotly charting for dual-stock comparison.

This module deliberately does not forward-fill missing observations. All returns are
computed from each instrument's own observed trading sessions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

TRADING_DAYS_PER_YEAR = 252


def normalize_comparison_symbol(symbol: str, exchange: str = "NSE") -> str:
    """Normalize a user-entered ticker to a Yahoo Finance NSE/BSE symbol."""
    raw = (symbol or "").strip().upper().replace(" ", "")
    if not raw:
        raise ValueError("Stock symbol is required.")
    if raw.endswith((".NS", ".BO")) or raw.startswith("^"):
        return raw
    exchange = (exchange or "NSE").strip().upper()
    if exchange not in {"NSE", "BSE"}:
        raise ValueError("Exchange must be NSE or BSE.")
    return raw + (".NS" if exchange == "NSE" else ".BO")


def clean_history(
    history: pd.DataFrame,
    *,
    today: Any = None,
    stale_after_days: int = 7,
) -> Tuple[pd.DataFrame, List[str]]:
    """Return a sorted, date-normalized Close series and validation warnings."""
    warnings: List[str] = []
    if history is None or not isinstance(history, pd.DataFrame) or history.empty:
        return pd.DataFrame(columns=["Close"]), ["No historical price data was returned."]
    if "Close" not in history.columns:
        return pd.DataFrame(columns=["Close"]), ["The returned history has no Close column."]

    frame = history.copy()
    try:
        parsed_index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce"))
    except Exception:
        return pd.DataFrame(columns=["Close"]), ["Historical dates could not be parsed."]

    valid_dates = ~parsed_index.isna()
    if not bool(np.all(valid_dates)):
        warnings.append(f"Dropped {int((~valid_dates).sum())} row(s) with invalid dates.")
        frame = frame.loc[valid_dates].copy()
        parsed_index = parsed_index[valid_dates]
    if len(parsed_index) == 0:
        return pd.DataFrame(columns=["Close"]), warnings + ["No valid historical dates remain."]

    # Yahoo daily bars are exchange-local. Remove timezone metadata without shifting
    # the local trading date, then normalize to one date per daily bar.
    if parsed_index.tz is not None:
        parsed_index = parsed_index.tz_localize(None)
    frame.index = parsed_index.normalize()

    duplicate_count = int(frame.index.duplicated(keep="last").sum())
    if duplicate_count:
        warnings.append(f"Found {duplicate_count} duplicate date row(s); kept the last row per date.")
        frame = frame.loc[~frame.index.duplicated(keep="last")]

    frame = frame.sort_index()
    close = pd.to_numeric(frame["Close"], errors="coerce")
    close = close.replace([np.inf, -np.inf], np.nan)
    invalid_close_count = int(close.isna().sum() + (close <= 0).sum())
    if invalid_close_count:
        warnings.append(f"Dropped {invalid_close_count} row(s) with missing, non-numeric, or non-positive Close values.")
    frame["Close"] = close
    frame = frame.loc[frame["Close"].notna() & (frame["Close"] > 0)].copy()

    if frame.empty:
        return frame, warnings + ["No valid closing prices remain after cleaning."]

    if len(frame) > 1:
        gap_days = frame.index.to_series().diff().dt.days
        long_gap_count = int((gap_days > 7).sum())
        if long_gap_count:
            warnings.append(
                f"Detected {long_gap_count} internal gap(s) longer than 7 calendar days; charts break across these gaps and volatility excludes gap-spanning returns."
            )

    current_day = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.now().normalize()
    last_date = pd.Timestamp(frame.index[-1]).normalize()
    age_days = int((current_day - last_date).days)
    if age_days > stale_after_days:
        warnings.append(
            f"Latest observation is {last_date:%Y-%m-%d} ({age_days} calendar days old); data may be stale."
        )

    return frame, warnings


def calculate_price_metrics(history: pd.DataFrame) -> Dict[str, Any]:
    """Calculate returns and risks from the first through the last valid Close."""
    if history is None or history.empty or "Close" not in history.columns:
        return {"valid": False, "reason": "Insufficient valid history."}

    close = pd.to_numeric(history["Close"], errors="coerce")
    close = close.replace([np.inf, -np.inf], np.nan).dropna()
    close = close[close > 0].sort_index()
    if len(close) < 2:
        return {"valid": False, "reason": "At least two valid closing prices are required."}

    close_change = close.pct_change().replace([np.inf, -np.inf], np.nan)
    gap_days = close.index.to_series().diff().dt.days
    # Weekends/normal holiday closures remain valid session-to-session returns;
    # larger gaps may hide multiple missing bars, so do not treat them as one day.
    daily_returns = close_change.where(gap_days <= 7).dropna()
    first_date = pd.Timestamp(close.index[0])
    last_date = pd.Timestamp(close.index[-1])
    first_price = float(close.iloc[0])
    last_price = float(close.iloc[-1])
    total_return = last_price / first_price - 1.0
    elapsed_days = max((last_date - first_date).days, 0)
    annualized_return = None
    if elapsed_days >= 365 and first_price > 0 and last_price > 0:
        annualized_return = float((last_price / first_price) ** (365.25 / elapsed_days) - 1.0)

    drawdown = close / close.cummax() - 1.0
    trough_date = pd.Timestamp(drawdown.idxmin())
    peak_candidates = close.loc[:trough_date]
    peak_date = pd.Timestamp(peak_candidates.idxmax()) if not peak_candidates.empty else first_date
    max_drawdown = float(drawdown.min())
    annualized_volatility = None
    if len(daily_returns) >= 2:
        annualized_volatility = float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

    return {
        "valid": True,
        "first_date": first_date.strftime("%Y-%m-%d"),
        "last_date": last_date.strftime("%Y-%m-%d"),
        "observations": int(len(close)),
        "first_close": first_price,
        "last_close": last_price,
        "total_return": float(total_return),
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "drawdown_peak_date": peak_date.strftime("%Y-%m-%d"),
        "drawdown_trough_date": trough_date.strftime("%Y-%m-%d"),
        "annualized_volatility": annualized_volatility,
        "daily_returns": daily_returns,
    }


def divergence_evidence(history1: pd.DataFrame, history2: pd.DataFrame) -> Dict[str, Any]:
    """Summarize normalized divergence only on dates observed for both stocks."""
    if history1 is None or history2 is None or history1.empty or history2.empty:
        return {"common_observations": 0, "note": "No common valid dates are available."}
    common = history1.index.intersection(history2.index).sort_values()
    if len(common) < 2:
        return {"common_observations": int(len(common)), "note": "Fewer than two common valid dates are available."}

    s1 = history1.loc[common, "Close"].astype(float)
    s2 = history2.loc[common, "Close"].astype(float)
    norm1 = s1 / float(s1.iloc[0]) * 100.0
    norm2 = s2 / float(s2.iloc[0]) * 100.0
    spread = norm1 - norm2
    max_positive_date = pd.Timestamp(spread.idxmax())
    max_negative_date = pd.Timestamp(spread.idxmin())
    crossings = []
    signs = np.sign(spread.to_numpy())
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1] and signs[i] != 0 and signs[i - 1] != 0:
            crossings.append(pd.Timestamp(common[i]).strftime("%Y-%m-%d"))

    # Group consecutive common-date observations by which normalized series is ahead.
    lead_regimes = []
    start_i = 0
    lead_labels = ["Stock 1 leads" if v > 0 else "Stock 2 leads" if v < 0 else "Equal" for v in spread.to_numpy()]
    for i in range(1, len(lead_labels) + 1):
        if i == len(lead_labels) or lead_labels[i] != lead_labels[start_i]:
            segment = spread.iloc[start_i:i]
            if lead_labels[start_i] != "Equal":
                peak_date = pd.Timestamp(segment.abs().idxmax())
                lead_regimes.append({
                    "leader": lead_labels[start_i],
                    "start_date": pd.Timestamp(common[start_i]).strftime("%Y-%m-%d"),
                    "end_date": pd.Timestamp(common[i - 1]).strftime("%Y-%m-%d"),
                    "common_observations": int(len(segment)),
                    "largest_gap_points": float(segment.loc[peak_date] if lead_labels[start_i] == "Stock 1 leads" else -segment.loc[peak_date]),
                    "largest_gap_date": peak_date.strftime("%Y-%m-%d"),
                })
            start_i = i
    lead_regimes = sorted(lead_regimes, key=lambda item: item["common_observations"], reverse=True)[:12]
    return {
        "common_observations": int(len(common)),
        "common_start_date": pd.Timestamp(common[0]).strftime("%Y-%m-%d"),
        "common_end_date": pd.Timestamp(common[-1]).strftime("%Y-%m-%d"),
        "first_stock_max_lead_date": max_positive_date.strftime("%Y-%m-%d"),
        "first_stock_max_lead_points": float(spread.loc[max_positive_date]),
        "second_stock_max_lead_date": max_negative_date.strftime("%Y-%m-%d"),
        "second_stock_max_lead_points": float(-spread.loc[max_negative_date]),
        "observed_relative_performance_crossings": crossings[:12],
        "largest_lead_regimes_by_observation_count": lead_regimes,
        "method": "Both series rebased to 100 on the first common valid trading date; no missing prices were filled.",
    }


def build_timeline_figure(
    history1: pd.DataFrame,
    history2: pd.DataFrame,
    label1: str,
    label2: str,
    mode: str = "Normalized performance (start = 100)",
) -> Tuple[go.Figure, List[str]]:
    """Build an interactive dual-stock date-axis chart and return chart warnings."""
    figure = go.Figure()
    warnings: List[str] = []
    normalized = mode.startswith("Normalized")

    if normalized:
        common = history1.index.intersection(history2.index).sort_values()
        if len(common) < 2:
            raise ValueError("Normalized comparison needs at least two shared valid trading dates.")
        series_list = [
            (label1, history1.loc[common, "Close"].astype(float), history1),
            (label2, history2.loc[common, "Close"].astype(float), history2),
        ]
        if len(common) < min(len(history1), len(history2)):
            warnings.append(
                f"Normalized chart uses {len(common)} common valid trading dates only; unshared dates were not filled or fabricated."
            )
    else:
        series_list = [(label1, history1["Close"].astype(float), history1),
                       (label2, history2["Close"].astype(float), history2)]

    for label, close_series, source_history in series_list:
        source_close = source_history["Close"].astype(float)
        source_dates = list(source_close.index)
        source_values = source_close.to_numpy(dtype=float)
        source_position = {pd.Timestamp(d): i for i, d in enumerate(source_dates)}

        if normalized:
            chart_y = close_series / float(close_series.iloc[0]) * 100.0
        else:
            chart_y = close_series

        x_points = []
        y_points = []
        custom_rows = []
        previous_plot_date = None
        for date, y_value, original_close in zip(close_series.index, chart_y.to_numpy(dtype=float), close_series.to_numpy(dtype=float)):
            date = pd.Timestamp(date)
            if previous_plot_date is not None and (date - previous_plot_date).days > 7:
                # Insert a null point between unusually long gaps so the line cannot
                # imply an unobserved price path across missing data.
                x_points.append(previous_plot_date + (date - previous_plot_date) / 2)
                y_points.append(None)
                custom_rows.append(["No data: line intentionally broken across a gap", None])

            source_i = source_position.get(date)
            if source_i is None or source_i == 0:
                change_text = "Daily change unavailable (first observation)"
            else:
                prior_date = pd.Timestamp(source_dates[source_i - 1])
                gap_length = (date - prior_date).days
                if gap_length > 7:
                    change_text = "Daily change unavailable (data gap > 7 calendar days)"
                else:
                    pct_change = (source_values[source_i] / source_values[source_i - 1] - 1.0) * 100.0
                    change_text = f"Daily change: {pct_change:+.2f}%"

            x_points.append(date)
            y_points.append(float(y_value))
            custom_rows.append([change_text, float(original_close)])
            previous_plot_date = date

        chart_label = f"{label} — normalized" if normalized else f"{label} — Close"
        custom = np.asarray(custom_rows, dtype=object)
        if normalized:
            hover = (
                "<b>%{fullData.name}</b><br>Trading date: %{x|%Y-%m-%d}"
                "<br>Normalized value: %{y:.2f}"
                "<br>Original Close: ₹%{customdata[1]:,.2f}"
                "<br>%{customdata[0]}<extra></extra>"
            )
        else:
            hover = (
                "<b>%{fullData.name}</b><br>Trading date: %{x|%Y-%m-%d}"
                "<br>Close: ₹%{y:,.2f}"
                "<br>%{customdata[0]}<extra></extra>"
            )
        figure.add_trace(go.Scatter(
            x=x_points,
            y=y_points,
            customdata=custom,
            mode="lines",
            name=chart_label,
            connectgaps=False,
            hovertemplate=hover,
        ))

    figure.update_layout(
        title=("Normalized Historical Performance" if normalized else "Historical Closing Prices"),
        template="plotly_dark",
        height=520,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=65, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(
            title="Actual trading date",
            type="date",
            rangeslider=dict(visible=True),
            rangeselector=dict(buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ]),
        ),
        yaxis=dict(title="Rebased value (100 = common start)" if normalized else "Closing price (₹)"),
    )
    return figure, warnings


def ai_evidence_payload(
    label1: str,
    metrics1: Dict[str, Any],
    label2: str,
    metrics2: Dict[str, Any],
    divergence: Dict[str, Any],
    price_basis: str,
    requested_period: str,
    data_warnings: List[str] | None = None,
) -> Dict[str, Any]:
    """Return JSON-safe facts only; exclude raw series and non-serializable pandas objects."""
    def facts(metrics: Dict[str, Any]) -> Dict[str, Any]:
        if not metrics.get("valid"):
            return {"valid": False, "reason": metrics.get("reason", "Insufficient history")}
        return {k: v for k, v in metrics.items() if k != "daily_returns"}

    return {
        "requested_period": requested_period,
        "price_basis": price_basis,
        "stocks": {label1: facts(metrics1), label2: facts(metrics2)},
        "normalized_divergence": divergence,
        "data_quality_warnings": list(data_warnings or []),
        "limitations": [
            "Statistics are computed from the first to last valid closing price available for each stock in the selected period.",
            "Annualized return is supplied only when the observed span is at least 365 calendar days.",
            "Volatility is sample standard deviation of observed session-to-session close returns multiplied by sqrt(252); returns spanning more than 7 calendar days are excluded.",
            "Normalized divergence uses shared valid dates only and no missing observations are forward-filled.",
            "Historical results do not forecast future prices or guarantee future returns.",
        ],
    }
