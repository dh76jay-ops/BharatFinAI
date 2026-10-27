import unittest

import numpy as np
import pandas as pd

from core.ai_comparison_engine import (
    FACTOR_WEIGHTS,
    analyze_stock,
    calculate_position_size,
    normalize_symbol,
    weighted_research_score,
)


def make_sample_history(n=260):
    """Generate deterministic synthetic OHLCV data for testing."""

    dates = pd.bdate_range(
        end="2026-10-16",
        periods=n,
    )

    close = np.linspace(100, 180, n)

    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(n, 100000),
        },
        index=dates,
    )


class TestAIComparisonEngine(unittest.TestCase):

    def test_normalize_nse_symbol(self):
        self.assertEqual(
            normalize_symbol("reliance"),
            "RELIANCE.NS",
        )

    def test_existing_exchange_suffix_is_preserved(self):
        self.assertEqual(
            normalize_symbol("RELIANCE.NS"),
            "RELIANCE.NS",
        )

    def test_analyze_stock_calculates_metrics(self):
        result = analyze_stock(
            symbol="TEST",
            history=make_sample_history(),
            info={
                "returnOnEquity": 0.15,
                "trailingPE": 25,
                "priceToBook": 3,
                "revenueGrowth": 0.10,
                "earningsGrowth": 0.12,
                "debtToEquity": 50,
                "operatingMargins": 0.18,
            },
        )

        metrics = result["metrics"]

        self.assertGreater(metrics["Price"], 0)
        self.assertIsNotNone(metrics["RSI14"])
        self.assertIsNotNone(metrics["SMA50"])
        self.assertIsNotNone(metrics["MACD"])
        self.assertIsNotNone(
            metrics["Annualized Volatility (%)"]
        )
        self.assertIsNotNone(metrics["ATR14"])

    def test_factor_scores_are_bounded(self):
        result = analyze_stock(
            symbol="TEST",
            history=make_sample_history(),
            info={
                "returnOnEquity": 0.15,
                "trailingPE": 25,
                "priceToBook": 3,
            },
        )

        for score in result["factors"].values():
            if score is not None:
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 100)

    def test_missing_factors_are_not_given_fake_scores(self):
        factors = {
            name: None
            for name in FACTOR_WEIGHTS
        }

        result = weighted_research_score(
            factors,
            FACTOR_WEIGHTS,
        )

        self.assertIsNone(result["score"])
        self.assertEqual(
            result["available_weight_pct"],
            0,
        )

    def test_available_factor_weights_are_renormalized(self):
        factors = {
            name: None
            for name in FACTOR_WEIGHTS
        }

        factors["Trend & Momentum"] = 80

        result = weighted_research_score(
            factors,
            FACTOR_WEIGHTS,
        )

        self.assertEqual(result["score"], 80)
        self.assertEqual(
            result["available_weight_pct"],
            20,
        )

    def test_invalid_position_sizing_inputs_are_rejected(self):
        result = calculate_position_size(
            price=None,
            atr=2,
            capital=100000,
        )

        self.assertFalse(result["available"])

    def test_position_size_respects_risk_budget(self):
        result = calculate_position_size(
            price=100,
            atr=2,
            capital=100000,
            risk_budget_pct=1,
            max_exposure_pct=20,
            atr_multiple=2,
        )

        self.assertTrue(result["available"])

        self.assertLessEqual(
            result["estimated_stop_scenario_loss"],
            result["risk_budget"],
        )

        self.assertLessEqual(
            result["estimated_exposure"],
            20000,
        )

    def test_empty_history_returns_warning(self):
        result = analyze_stock(
            symbol="TEST",
            history=pd.DataFrame(),
        )

        self.assertTrue(result["warnings"])
        self.assertEqual(result["metrics"], {})

    def test_scenarios_are_generated(self):
        result = analyze_stock(
            symbol="TEST",
            history=make_sample_history(),
        )

        self.assertEqual(
            set(result["scenarios"].keys()),
            {"Bull", "Base", "Bear"},
        )


if __name__ == "__main__":
    unittest.main()