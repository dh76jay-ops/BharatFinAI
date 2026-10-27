"""
BharatFinAI — AI Comparison Research Engine

Purpose:
- Calculate evidence-based stock metrics.
- Produce transparent research factor scores.
- Handle missing and insufficient data.
- Generate Bull, Base and Bear research scenarios.
- Calculate illustrative, risk-budget-based position sizing.

Scores are research heuristics, not probabilities of profit.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


FACTOR_WEIGHTS = {
    "Trend & Momentum": 20,
    "Business Quality": 15,
    "Growth": 15,
    "Valuation": 15,
    "Financial Strength": 15,
    "Risk Control": 10,
    "Data Confidence": 10,
}


def safe_float(value):
    """Return a finite float, otherwise None."""

    try:
        result = float(value)

        if np.isfinite(result):
            return result

    except (TypeError, ValueError, OverflowError):
        pass

    return None


def normalize_symbol(symbol):
    """Normalize common NSE symbols for Yahoo Finance."""

    value = str(symbol or "").strip().upper()

    if not value:
        raise ValueError("Stock symbol is required.")

    if value.startswith("^") or "." in value:
        return value

    return value + ".NS"


def _mean_available(values):
    """Average available numeric values without inventing missing values."""

    valid = [
        safe_float(value)
        for value in values
    ]

    valid = [
        value
        for value in valid
        if value is not None
    ]

    if not valid:
        return None

    return float(np.mean(valid))


def _growth_score(value):
    """Heuristic growth score from 0 to 100."""

    value = safe_float(value)

    if value is None:
        return None

    if value >= 15:
        return 90
    if value >= 5:
        return 75
    if value >= 0:
        return 60
    if value >= -10:
        return 35

    return 15


def _roe_score(value):
    """Heuristic profitability score from 0 to 100."""

    value = safe_float(value)

    if value is None:
        return None

    if value >= 20:
        return 90
    if value >= 15:
        return 80
    if value >= 10:
        return 65
    if value >= 5:
        return 45
    if value >= 0:
        return 30

    return 10


def _margin_score(value):
    """Heuristic operating-margin score."""

    value = safe_float(value)

    if value is None:
        return None

    if value >= 20:
        return 90
    if value >= 12:
        return 75
    if value >= 5:
        return 60
    if value >= 0:
        return 40

    return 15


def _pe_score(value):
    """
    Broad valuation heuristic.

    Lower positive P/E receives a higher score.
    This is not a sector-adjusted fair-value estimate.
    """

    value = safe_float(value)

    if value is None or value <= 0:
        return None

    if value <= 15:
        return 85
    if value <= 25:
        return 70
    if value <= 40:
        return 50

    return 30


def _pb_score(value):
    """Broad P/B valuation heuristic, not a peer comparison."""

    value = safe_float(value)

    if value is None or value <= 0:
        return None

    if value <= 2:
        return 85
    if value <= 4:
        return 65
    if value <= 7:
        return 45

    return 25


def _debt_score(debt_equity):
    """Higher score means lower debt/equity under this rule."""

    value = safe_float(debt_equity)

    if value is None or value < 0:
        return None

    if value <= 0.3:
        return 90
    if value <= 0.7:
        return 75
    if value <= 1.5:
        return 55

    return 30


def _growth_from_statement(statement, row_aliases):
    """
    Estimate year-over-year growth from the two latest statement columns.

    Returns None if suitable data is unavailable.
    """

    if statement is None:
        return None

    if not isinstance(statement, pd.DataFrame):
        return None

    if statement.empty or len(statement.columns) < 2:
        return None

    try:
        columns = sorted(
            statement.columns,
            key=lambda column: str(column),
            reverse=True,
        )

        row_lookup = {
            str(row).strip().lower(): row
            for row in statement.index
        }

        selected_row = None

        for alias in row_aliases:
            selected_row = row_lookup.get(alias.lower())

            if selected_row is not None:
                break

        if selected_row is None:
            return None

        latest = safe_float(
            statement.loc[selected_row, columns[0]]
        )

        previous = safe_float(
            statement.loc[selected_row, columns[1]]
        )

        if latest is None or previous is None or previous <= 0:
            return None

        growth = ((latest / previous) - 1) * 100

        return safe_float(growth)

    except Exception:
        return None


def _last_value(series):
    """Return the latest finite value from a series."""

    if series is None or len(series) == 0:
        return None

    valid = pd.to_numeric(
        series,
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if valid.empty:
        return None

    return safe_float(valid.iloc[-1])


def analyze_stock(
    symbol,
    history,
    info=None,
    financials=None,
    benchmark=None,
):
    """
    Analyze one stock using supplied historical prices and fundamentals.

    Parameters
    ----------
    symbol:
        Display symbol.
    history:
        Historical OHLCV DataFrame.
    info:
        Optional Yahoo Finance ticker.info dictionary.
    financials:
        Optional annual income statement DataFrame.
    benchmark:
        Optional benchmark price history, for example NIFTY 50.

    Returns
    -------
    dict
        Metrics, factor scores, evidence, warnings and scenarios.
    """

    info = info or {}
    history = (
        history.copy()
        if isinstance(history, pd.DataFrame)
        else pd.DataFrame()
    )

    result = {
        "symbol": str(symbol),
        "metrics": {},
        "factors": {},
        "warnings": [],
        "scenarios": {},
        "observations": 0,
        "latest_date": None,
        "source": "Yahoo Finance, when supplied by the caller",
    }

    if history.empty or "Close" not in history.columns:
        result["warnings"].append(
            "Historical price data is unavailable."
        )
        return result

    # Basic price-data cleanup.
    history = history.sort_index()
    history = history.loc[
        ~history.index.duplicated(keep="last")
    ]

    close = pd.to_numeric(
        history["Close"],
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if close.empty:
        result["warnings"].append(
            "No valid closing prices were available."
        )
        return result

    price = safe_float(close.iloc[-1])

    returns = close.pct_change().replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    result["observations"] = int(len(returns))
    result["latest_date"] = str(close.index[-1])

    metrics = {
        "Price": price,
        "Observations": int(len(returns)),
    }

    # Returns and volatility.
    metrics["Return 20D (%)"] = (
        safe_float((close.iloc[-1] / close.iloc[-21] - 1) * 100)
        if len(close) >= 21
        else None
    )

    metrics["Return 60D (%)"] = (
        safe_float((close.iloc[-1] / close.iloc[-61] - 1) * 100)
        if len(close) >= 61
        else None
    )

    metrics["Annualized Volatility (%)"] = (
        safe_float(returns.std(ddof=1) * math.sqrt(252) * 100)
        if len(returns) >= 20
        else None
    )

    running_peak = close.cummax()

    drawdowns = (
        (close / running_peak) - 1
    ) * 100

    metrics["Maximum Drawdown (%)"] = (
        safe_float(drawdowns.min())
        if len(drawdowns)
        else None
    )

    metrics["Current Drawdown (%)"] = (
        safe_float(drawdowns.iloc[-1])
        if len(drawdowns)
        else None
    )

    # Trend indicators.
    sma20 = close.rolling(20, min_periods=20).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()

    metrics["SMA20"] = _last_value(sma20)
    metrics["SMA50"] = _last_value(sma50)
    metrics["SMA200"] = _last_value(sma200)

    ema12 = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    macd = ema12 - ema26

    macd_signal = macd.ewm(
        span=9,
        adjust=False,
    ).mean()

    metrics["MACD"] = _last_value(macd)
    metrics["MACD Signal"] = _last_value(macd_signal)
    metrics["MACD Histogram"] = (
        safe_float(macd.iloc[-1] - macd_signal.iloc[-1])
        if len(macd)
        else None
    )

    # RSI: Wilder-style exponential smoothing.
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    relative_strength = (
        avg_gain / avg_loss.replace(0, np.nan)
    )

    rsi = 100 - (100 / (1 + relative_strength))

    # Continuous gains can legitimately produce RSI = 100.
    rsi = rsi.where(
        avg_loss != 0,
        100.0,
    )

    metrics["RSI14"] = _last_value(rsi)

    # Bollinger Bands.
    bb_mid = close.rolling(
        20,
        min_periods=20,
    ).mean()

    bb_std = close.rolling(
        20,
        min_periods=20,
    ).std()

    metrics["Bollinger Upper"] = _last_value(
        bb_mid + 2 * bb_std
    )

    metrics["Bollinger Middle"] = _last_value(bb_mid)

    metrics["Bollinger Lower"] = _last_value(
        bb_mid - 2 * bb_std
    )

    # Volume relative to its 20-session average.
    if "Volume" in history.columns:
        volume = pd.to_numeric(
            history["Volume"],
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        avg_volume = volume.rolling(
            20,
            min_periods=20,
        ).mean()

        current_volume = _last_value(volume)
        average_volume = _last_value(avg_volume)

        metrics["Volume Ratio"] = (
            safe_float(current_volume / average_volume)
            if current_volume is not None
            and average_volume is not None
            and average_volume > 0
            else None
        )
    else:
        metrics["Volume Ratio"] = None

    # ATR from actual High, Low and previous Close.
    metrics["ATR14"] = None

    if all(
        column in history.columns
        for column in ["High", "Low", "Close"]
    ):
        high = pd.to_numeric(
            history["High"],
            errors="coerce",
        )

        low = pd.to_numeric(
            history["Low"],
            errors="coerce",
        )

        previous_close = close.shift(1)

        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = true_range.rolling(
            14,
            min_periods=14,
        ).mean()

        metrics["ATR14"] = _last_value(atr)

    metrics["ATR (%)"] = (
        safe_float(metrics["ATR14"] / price * 100)
        if price and metrics["ATR14"] is not None
        else None
    )

    # Relative strength versus a supplied benchmark.
    metrics["Relative Strength vs Benchmark (%)"] = None

    if (
        isinstance(benchmark, pd.DataFrame)
        and "Close" in benchmark.columns
    ):
        benchmark_close = pd.to_numeric(
            benchmark["Close"],
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        stock_returns = close.pct_change()
        benchmark_returns = benchmark_close.pct_change()

        aligned = pd.concat(
            [
                stock_returns.rename("stock"),
                benchmark_returns.rename("benchmark"),
            ],
            axis=1,
        ).dropna()

        if len(aligned) >= 20:
            aligned = aligned.tail(252)

            stock_return = (
                (1 + aligned["stock"]).prod() - 1
            ) * 100

            benchmark_return = (
                (1 + aligned["benchmark"]).prod() - 1
            ) * 100

            metrics["Relative Strength vs Benchmark (%)"] = (
                safe_float(stock_return - benchmark_return)
            )

    # Fundamental inputs. Missing values remain None.
    roe = safe_float(info.get("returnOnEquity"))
    operating_margin = safe_float(info.get("operatingMargins"))

    revenue_growth = safe_float(info.get("revenueGrowth"))
    earnings_growth = safe_float(info.get("earningsGrowth"))

    # Yahoo Finance commonly supplies these growth ratios as decimals.
    revenue_growth = (
        revenue_growth * 100
        if revenue_growth is not None
        else _growth_from_statement(
            financials,
            ["Total Revenue", "Operating Revenue"],
        )
    )

    earnings_growth = (
        earnings_growth * 100
        if earnings_growth is not None
        else _growth_from_statement(
            financials,
            ["Net Income", "Net Income Common Stockholders"],
        )
    )

    metrics["EPS"] = safe_float(info.get("trailingEps"))
    metrics["P/E"] = safe_float(info.get("trailingPE"))
    metrics["P/B"] = safe_float(info.get("priceToBook"))

    metrics["ROE (%)"] = (
        safe_float(roe * 100)
        if roe is not None
        else None
    )

    metrics["Operating Margin (%)"] = (
        safe_float(operating_margin * 100)
        if operating_margin is not None
        else None
    )

    metrics["Revenue Growth (%)"] = revenue_growth
    metrics["Earnings Growth (%)"] = earnings_growth

    raw_debt_equity = safe_float(
        info.get("debtToEquity")
    )

    # Yahoo Finance commonly expresses debtToEquity in percentage points.
    metrics["Debt/Equity"] = (
        safe_float(raw_debt_equity / 100)
        if raw_debt_equity is not None
        else None
    )

    metrics["Current Ratio"] = safe_float(
        info.get("currentRatio")
    )

    metrics["Operating Cash Flow"] = safe_float(
        info.get("operatingCashflow")
    )

    metrics["Free Cash Flow"] = safe_float(
        info.get("freeCashflow")
    )

    # ---------------------------------------------------------
    # Transparent factor scoring
    # Higher scores mean more favorable evidence under the
    # listed heuristic rules, not a probability of profit.
    # ---------------------------------------------------------

    trend_inputs = []

    if metrics["SMA50"] is not None:
        trend_inputs.append(
            80 if price > metrics["SMA50"] else 30
        )

    if metrics["SMA200"] is not None:
        trend_inputs.append(
            80 if price > metrics["SMA200"] else 30
        )

    rsi_value = metrics["RSI14"]

    if rsi_value is not None:
        if 50 <= rsi_value <= 65:
            trend_inputs.append(80)
        elif 40 <= rsi_value < 50 or 65 < rsi_value <= 70:
            trend_inputs.append(60)
        elif 30 <= rsi_value < 40:
            trend_inputs.append(40)
        elif rsi_value < 30:
            trend_inputs.append(45)
        else:
            trend_inputs.append(30)

    if metrics["MACD Histogram"] is not None:
        trend_inputs.append(
            75 if metrics["MACD Histogram"] > 0 else 35
        )

    if metrics["Return 20D (%)"] is not None:
        trend_inputs.append(
            75 if metrics["Return 20D (%)"] > 0 else 35
        )

    trend_score = _mean_available(trend_inputs)

    business_score = _mean_available(
        [
            _roe_score(metrics["ROE (%)"]),
            _margin_score(metrics["Operating Margin (%)"]),
        ]
    )

    growth_score = _mean_available(
        [
            _growth_score(metrics["Revenue Growth (%)"]),
            _growth_score(metrics["Earnings Growth (%)"]),
        ]
    )

    valuation_score = _mean_available(
        [
            _pe_score(metrics["P/E"]),
            _pb_score(metrics["P/B"]),
        ]
    )

    current_ratio = metrics["Current Ratio"]

    current_ratio_score = None

    if current_ratio is not None:
        if current_ratio >= 1.5:
            current_ratio_score = 85
        elif current_ratio >= 1:
            current_ratio_score = 65
        elif current_ratio >= 0.7:
            current_ratio_score = 40
        else:
            current_ratio_score = 20

    cash_flow_score = None

    if metrics["Operating Cash Flow"] is not None:
        cash_flow_score = (
            75
            if metrics["Operating Cash Flow"] > 0
            else 25
        )

    financial_strength_score = _mean_available(
        [
            _debt_score(metrics["Debt/Equity"]),
            current_ratio_score,
            cash_flow_score,
        ]
    )

    risk_inputs = []

    volatility = metrics["Annualized Volatility (%)"]

    if volatility is not None:
        if volatility <= 15:
            risk_inputs.append(90)
        elif volatility <= 25:
            risk_inputs.append(75)
        elif volatility <= 40:
            risk_inputs.append(50)
        else:
            risk_inputs.append(25)

    max_drawdown = metrics["Maximum Drawdown (%)"]

    if max_drawdown is not None:
        drawdown_size = abs(max_drawdown)

        if drawdown_size <= 10:
            risk_inputs.append(90)
        elif drawdown_size <= 20:
            risk_inputs.append(75)
        elif drawdown_size <= 35:
            risk_inputs.append(50)
        else:
            risk_inputs.append(25)

    atr_pct = metrics["ATR (%)"]

    if atr_pct is not None:
        if atr_pct <= 2:
            risk_inputs.append(90)
        elif atr_pct <= 4:
            risk_inputs.append(70)
        elif atr_pct <= 6:
            risk_inputs.append(50)
        else:
            risk_inputs.append(25)

    risk_score = _mean_available(risk_inputs)

    # Data confidence depends on observations and metric coverage.
    observation_count = len(returns)

    if observation_count >= 250:
        observation_score = 90
    elif observation_count >= 120:
        observation_score = 75
    elif observation_count >= 60:
        observation_score = 55
    else:
        observation_score = 25

    coverage_fields = [
        metrics["RSI14"],
        metrics["MACD Histogram"],
        metrics["SMA50"],
        metrics["SMA200"],
        metrics["ATR14"],
        metrics["Annualized Volatility (%)"],
        metrics["ROE (%)"],
        metrics["Revenue Growth (%)"],
        metrics["P/E"],
        metrics["Debt/Equity"],
    ]

    coverage = (
        sum(value is not None for value in coverage_fields)
        / len(coverage_fields)
    ) * 100

    data_confidence_score = (
        0.6 * observation_score
        + 0.4 * coverage
    )

    factors = {
        "Trend & Momentum": safe_float(trend_score),
        "Business Quality": safe_float(business_score),
        "Growth": safe_float(growth_score),
        "Valuation": safe_float(valuation_score),
        "Financial Strength": safe_float(financial_strength_score),
        "Risk Control": safe_float(risk_score),
        "Data Confidence": safe_float(data_confidence_score),
    }

    result["metrics"] = metrics
    result["factors"] = factors
    result["data_coverage_pct"] = round(coverage, 1)

    if observation_count < 60:
        result["warnings"].append(
            "Fewer than 60 return observations: "
            "some estimates and scores may be unreliable."
        )

    if valuation_score is not None:
        result["warnings"].append(
            "Valuation scoring uses broad P/E and P/B thresholds. "
            "It is not adjusted for sector, peers or accounting differences."
        )

    # Scenario conditions. No fabricated price targets.
    above_sma50 = (
        metrics["SMA50"] is not None
        and price > metrics["SMA50"]
    )

    above_sma200 = (
        metrics["SMA200"] is not None
        and price > metrics["SMA200"]
    )

    positive_macd = (
        metrics["MACD Histogram"] is not None
        and metrics["MACD Histogram"] > 0
    )

    positive_return20 = (
        metrics["Return 20D (%)"] is not None
        and metrics["Return 20D (%)"] > 0
    )

    bullish_conditions = sum(
        [
            above_sma50,
            above_sma200,
            positive_macd,
            positive_return20,
        ]
    )

    if bullish_conditions >= 3:
        base_outlook = (
            "Observed trend indicators are mostly positive, "
            "but continuation is not guaranteed."
        )
    elif bullish_conditions <= 1:
        base_outlook = (
            "Observed trend indicators are mostly weak or mixed."
        )
    else:
        base_outlook = (
            "Observed trend indicators are mixed."
        )

    result["scenarios"] = {
        "Bull": {
            "Assumptions": (
                "Price remains above relevant moving averages "
                "and momentum continues to improve."
            ),
            "Evidence": (
                f"Above SMA50: {above_sma50}; "
                f"Above SMA200: {above_sma200}; "
                f"Positive MACD histogram: {positive_macd}; "
                f"Positive 20D return: {positive_return20}."
            ),
            "Invalidation": (
                "A sustained trend breakdown or deterioration "
                "in momentum would weaken this scenario."
            ),
        },
        "Base": {
            "Assumptions": (
                "Recent observed conditions persist without "
                "a material positive or negative catalyst."
            ),
            "Evidence": base_outlook,
            "Invalidation": (
                "A sustained breakout, breakdown or material "
                "fundamental change would weaken this scenario."
            ),
        },
        "Bear": {
            "Assumptions": (
                "Price weakens, momentum deteriorates or "
                "business/market risks increase."
            ),
            "Evidence": (
                f"Historical maximum drawdown: "
                f"{metrics['Maximum Drawdown (%)']}%; "
                f"current drawdown: {metrics['Current Drawdown (%)']}%."
            ),
            "Invalidation": (
                "A sustained recovery above relevant trend levels "
                "with improving momentum would weaken this scenario."
            ),
        },
    }

    return result


def weighted_research_score(
    factors,
    weights=None,
):
    """
    Calculate a weighted research score.

    Missing factor scores are excluded. Remaining weights are
    renormalized and returned explicitly. The score is not a
    calibrated probability of profit.
    """

    weights = weights or FACTOR_WEIGHTS

    available = {}

    for name, score in factors.items():
        value = safe_float(score)
        weight = safe_float(weights.get(name, 0))

        if (
            value is not None
            and weight is not None
            and weight > 0
        ):
            available[name] = {
                "score": max(0.0, min(100.0, value)),
                "weight": weight,
            }

    total_weight = sum(
        item["weight"]
        for item in available.values()
    )

    if total_weight <= 0:
        return {
            "score": None,
            "contributions": {},
            "available_weight_pct": 0,
        }

    contributions = {}

    for name, item in available.items():
        normalized_weight = item["weight"] / total_weight

        contributions[name] = {
            "score": item["score"],
            "effective_weight_pct": normalized_weight * 100,
            "points_contributed": (
                item["score"] * normalized_weight
            ),
        }

    score = sum(
        item["points_contributed"]
        for item in contributions.values()
    )

    originally_configured_weight = sum(
        max(0.0, safe_float(weights.get(name, 0)) or 0)
        for name in factors
    )

    available_weight = (
        total_weight / originally_configured_weight * 100
        if originally_configured_weight > 0
        else 0
    )

    return {
        "score": round(score, 2),
        "contributions": contributions,
        "available_weight_pct": round(
            available_weight,
            2,
        ),
    }


def calculate_position_size(
    price,
    atr,
    capital,
    risk_budget_pct=1.0,
    max_exposure_pct=20.0,
    atr_multiple=2.0,
):
    """
    Illustrative risk-budget position sizing.

    Assumes a hypothetical stop distance of ATR * atr_multiple.
    Excludes gaps, slippage, brokerage and taxes.
    """

    price = safe_float(price)
    atr = safe_float(atr)
    capital = safe_float(capital)
    risk_budget_pct = safe_float(risk_budget_pct)
    max_exposure_pct = safe_float(max_exposure_pct)
    atr_multiple = safe_float(atr_multiple)

    if (
        price is None
        or atr is None
        or capital is None
        or risk_budget_pct is None
        or max_exposure_pct is None
        or atr_multiple is None
        or price <= 0
        or atr <= 0
        or capital <= 0
        or risk_budget_pct <= 0
        or max_exposure_pct <= 0
        or atr_multiple <= 0
    ):
        return {
            "available": False,
            "reason": "Valid price, ATR and risk-budget inputs are required.",
        }

    if risk_budget_pct > 100 or max_exposure_pct > 100:
        return {
            "available": False,
            "reason": "Percentages must not exceed 100.",
        }

    risk_budget = capital * risk_budget_pct / 100
    max_exposure = capital * max_exposure_pct / 100

    stop_distance = atr * atr_multiple

    shares_by_risk = math.floor(
        risk_budget / stop_distance
    )

    shares_by_exposure = math.floor(
        max_exposure / price
    )

    shares = max(
        0,
        min(shares_by_risk, shares_by_exposure),
    )

    estimated_exposure = shares * price
    illustrative_loss = shares * stop_distance

    return {
        "available": True,
        "risk_budget": round(risk_budget, 2),
        "stop_distance": round(stop_distance, 2),
        "illustrative_stop_reference": round(
            price - stop_distance,
            2,
        ),
        "shares_by_risk": shares_by_risk,
        "shares_by_exposure": shares_by_exposure,
        "illustrative_quantity": shares,
        "estimated_exposure": round(
            estimated_exposure,
            2,
        ),
        "estimated_stop_scenario_loss": round(
            illustrative_loss,
            2,
        ),
        "warning": (
            "Illustrative scenario only. Stop execution is not guaranteed. "
            "Gaps, slippage, liquidity, brokerage and taxes can increase losses."
        ),
    }