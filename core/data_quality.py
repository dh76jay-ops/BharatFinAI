"""
BharatFinAI — Data Quality Layer for Risk Engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any


def assess_returns_quality(returns: pd.Series, min_obs: int = 60) -> Dict[str, Any]:
    report = {
        "n_obs": 0,
        "missing_pct": 100.0,
        "duplicate_dates": 0,
        "zero_returns_pct": 0.0,
        "extreme_outliers": 0,
        "stale_flag": False,
        "score": 0,
        "level": "INSUFFICIENT DATA",
        "warnings": [],
        "usable": False,
    }

    if returns is None or len(returns) == 0:
        report["warnings"].append("No return series provided.")
        return report

    r = returns.dropna()
    report["n_obs"] = int(len(r))
    report["missing_pct"] = float((len(returns) - len(r)) / max(len(returns), 1) * 100)

    if hasattr(returns.index, "duplicated"):
        report["duplicate_dates"] = int(returns.index.duplicated().sum())

    if len(r) == 0:
        report["warnings"].append("All values are NaN.")
        return report

    zero_pct = float((r.abs() < 1e-12).mean() * 100)
    report["zero_returns_pct"] = zero_pct
    extremes = int((r.abs() > 0.20).sum())
    report["extreme_outliers"] = extremes

    try:
        last_ts = r.index[-1]
        if hasattr(last_ts, "to_pydatetime"):
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            last = last_ts.to_pydatetime()
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            age_days = (now - last).days
            if age_days > 7:
                report["stale_flag"] = True
                report["warnings"].append(f"Last observation is {age_days} days old.")
    except Exception:
        pass

    score = 100
    if report["n_obs"] < min_obs:
        score -= 40
        report["warnings"].append(f"Only {report['n_obs']} observations (min {min_obs}).")
    elif report["n_obs"] < 120:
        score -= 15

    if report["missing_pct"] > 5:
        score -= 15
        report["warnings"].append(f"Missing values: {report['missing_pct']:.1f}%.")
    if report["duplicate_dates"] > 0:
        score -= 10
        report["warnings"].append(f"{report['duplicate_dates']} duplicate dates.")
    if extremes > 5:
        score -= 10
        report["warnings"].append(f"{extremes} extreme daily moves (>20%).")
    if zero_pct > 15:
        score -= 10
        report["warnings"].append(f"High zero-return share: {zero_pct:.1f}%.")
    if report["stale_flag"]:
        score -= 15

    score = int(max(0, min(100, score)))
    report["score"] = score

    if score >= 75 and report["n_obs"] >= min_obs:
        report["level"] = "HIGH"
        report["usable"] = True
    elif score >= 50 and report["n_obs"] >= max(30, min_obs // 2):
        report["level"] = "MEDIUM"
        report["usable"] = True
    elif report["n_obs"] >= 20:
        report["level"] = "LOW"
        report["usable"] = True
    else:
        report["level"] = "INSUFFICIENT DATA"
        report["usable"] = False

    return report