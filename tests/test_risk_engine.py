"""
V8 Risk Engine tests.

The tests use deterministic synthetic returns so the mathematical/reporting
contracts can be checked without a live market-data connection.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.data_quality import assess_returns_quality
from core.risk_engine import run_v8_risk_engine
from core.volatility_models import (
    dynamic_position_size,
    ewma_volatility,
    garch_volatility,
    historical_var,
    realized_volatility,
    student_t_var,
    volatility_regime,
)


def _returns(n: int = 260) -> pd.Series:
    rng = np.random.default_rng(123)
    values = rng.normal(0.0002, 0.012, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    return pd.Series(values, index=idx, name="TEST")


class TestDataQuality(unittest.TestCase):
    def test_quality_report_is_usable_with_clean_data(self):
        report = assess_returns_quality(_returns())
        self.assertTrue(report["usable"])
        self.assertGreaterEqual(report["score"], 75)
        self.assertEqual(report["n_obs"], 260)

    def test_insufficient_data_is_not_published(self):
        report = assess_returns_quality(_returns(15))
        self.assertFalse(report["usable"])
        self.assertEqual(report["level"], "INSUFFICIENT DATA")


class TestV8Models(unittest.TestCase):
    def test_ewma(self):
        result = ewma_volatility(_returns(), lambda_=0.94)
        self.assertTrue(result["ok"])
        self.assertTrue(np.isfinite(result["current_vol"]))
        self.assertEqual(result["lambda"], 0.94)

    def test_realized_volatility(self):
        result = realized_volatility(_returns(), windows=[20, 60])
        self.assertTrue(result["ok"])
        self.assertTrue(np.isfinite(result["windows"]["20D"]))
        self.assertTrue(np.isfinite(result["windows"]["60D"]))

    def test_regime(self):
        result = volatility_regime(0.10)
        self.assertEqual(result["regime"], "LOW")

    def test_historical_var(self):
        r = pd.Series([-0.10, -0.04, -0.02, 0.01, 0.02, 0.03] * 10)
        result = historical_var(r, alpha=0.05)
        self.assertTrue(result["ok"])
        self.assertLess(result["var"], 0.0)

    def test_student_t_var(self):
        result = student_t_var(_returns())
        self.assertIsInstance(result, dict)
        if result["ok"]:
            self.assertLess(result["var"], 0.0)
        else:
            self.assertIn("message", result)

    def test_garch_failure_is_graceful(self):
        result = garch_volatility(_returns())
        self.assertIsInstance(result, dict)
        self.assertIn("reliability", result)
        if result["ok"]:
            self.assertIn("convergence_flag", result)
            self.assertIn("forecast_vol", result)
        else:
            self.assertIn("message", result)


class TestPositionSizing(unittest.TestCase):
    def test_position_sizing_is_capped_and_dampened(self):
        result = dynamic_position_size(
            capital=100000,
            entry_price=1000,
            stop_loss_pct=5,
            annual_vol=0.25,
            var_daily=-0.02,
            risk_budget_pct=1,
            max_position_pct=25,
            confidence=0.5,
        )
        self.assertTrue(result["ok"])
        self.assertGreaterEqual(result["suggested_qty"], 0)
        self.assertLessEqual(result["suggested_weight_pct"], 25.0)
        self.assertEqual(result["components"]["confidence_dampener"], 0.5)


class TestRiskEngine(unittest.TestCase):
    def test_v8_suite_runs_without_garch(self):
        result = run_v8_risk_engine(
            _returns(),
            capital=100000,
            entry_price=1000,
            stop_loss_pct=5,
            risk_budget_pct=1,
            max_position_pct=25,
            run_garch=False,
        )
        self.assertTrue(result["ok"])
        self.assertIn("ewma", result)
        self.assertIn("realized", result)
        self.assertIn("var_suite", result)
        self.assertIn("position_sizing", result)
        self.assertIn("methodology", result)

    def test_v8_blocks_insufficient_data(self):
        result = run_v8_risk_engine(_returns(20), run_garch=False)
        self.assertFalse(result["ok"])
        self.assertEqual(result["message"], "Insufficient data for reliable estimation.")


if __name__ == "__main__":
    unittest.main()
