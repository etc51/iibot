from __future__ import annotations

import unittest
from datetime import datetime, timezone

from samosbor.domain import BacktestResult, EquityPoint, PortfolioState
from samosbor.research.monte_carlo import MonteCarloSimulator


class MonteCarloTest(unittest.TestCase):
    def test_monte_carlo_summary_has_expected_keys(self):
        points = [
            EquityPoint(datetime(2025, 1, 1, tzinfo=timezone.utc), 100.0, 100.0, 0.0),
            EquityPoint(datetime(2025, 1, 31, tzinfo=timezone.utc), 103.0, 103.0, 0.0),
            EquityPoint(datetime(2025, 2, 1, tzinfo=timezone.utc), 103.0, 103.0, 0.0),
            EquityPoint(datetime(2025, 2, 28, tzinfo=timezone.utc), 101.0, 101.0, 0.0),
            EquityPoint(datetime(2025, 3, 1, tzinfo=timezone.utc), 101.0, 101.0, 0.0),
            EquityPoint(datetime(2025, 3, 31, tzinfo=timezone.utc), 107.0, 107.0, 0.0),
        ]
        result = BacktestResult(
            portfolio=PortfolioState(cash=107.0),
            trades=[],
            equity_curve=points,
            events=[],
        )
        simulator = MonteCarloSimulator(
            iterations=200,
            horizon_months=6,
            target_monthly_return_pct=5.0,
            seed=7,
        )
        payload = simulator.run(result)
        summary = payload["summary"]

        self.assertEqual(summary["iterations"], 200)
        self.assertEqual(summary["horizon_months"], 6)
        self.assertGreaterEqual(summary["probability_positive_pct"], 0.0)
        self.assertLessEqual(summary["probability_positive_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
