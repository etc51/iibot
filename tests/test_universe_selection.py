from __future__ import annotations

import unittest

from samosbor.autonomy.universe_selection import (
    build_universe_selection_tuning_payload,
)


class UniverseSelectionTuningTest(unittest.TestCase):
    def test_universe_tuning_selects_consensus_subset(self):
        payload = build_universe_selection_tuning_payload(
            configured_symbols=[
                "SBER",
                "GAZP",
                "LKOH",
                "ROSN",
                "NVTK",
                "TATN",
                "GMKN",
                "PLZL",
            ],
            current_allowed_symbols=[],
            optimizer_payload={
                "evaluated_candidates": 24,
                "best_candidate": {
                    "symbols": ["SBER", "GAZP", "LKOH", "ROSN", "NVTK"],
                    "score": 12.0,
                    "summary": {
                        "avg_monthly_return_pct": 1.2,
                        "max_drawdown_pct": 2.1,
                        "profit_factor": 1.5,
                        "trades": 14,
                    },
                },
            },
            walk_forward_payload={
                "summary": {
                    "folds_evaluated": 4,
                    "average_test_normalized_monthly_return_pct": 0.8,
                    "probability_positive_pct": 66.7,
                },
                "folds": [
                    {
                        "best_candidate": {"symbols": ["NVTK", "ROSN", "LKOH", "GAZP", "SBER"]},
                        "test_summary": {
                            "normalized_monthly_return_pct": 0.9,
                            "trades": 8,
                        },
                    }
                ],
            },
            max_allowed_symbols=5,
        )

        self.assertTrue(payload["changed"])
        self.assertEqual(payload["consensus_symbols"], ["GAZP", "LKOH", "NVTK", "ROSN", "SBER"])
        self.assertEqual(payload["proposed_allowed_symbols"], ["GAZP", "LKOH", "NVTK", "ROSN", "SBER"])
        self.assertEqual(payload["proposed_effective_symbols"], ["GAZP", "LKOH", "NVTK", "ROSN", "SBER"])

    def test_universe_tuning_can_clear_filter_when_full_configured_set_is_strong(self):
        payload = build_universe_selection_tuning_payload(
            configured_symbols=["USDRUBF", "CNYRUBF"],
            current_allowed_symbols=["USDRUBF"],
            optimizer_payload={
                "evaluated_candidates": 12,
                "best_candidate": {
                    "symbols": ["USDRUBF", "CNYRUBF"],
                    "summary": {},
                },
            },
            walk_forward_payload={
                "summary": {
                    "folds_evaluated": 3,
                    "probability_positive_pct": 60.0,
                },
                "folds": [
                    {
                        "best_candidate": {"symbols": ["USDRUBF", "CNYRUBF"]},
                        "test_summary": {
                            "normalized_monthly_return_pct": 0.3,
                            "trades": 7,
                        },
                    }
                ],
            },
            max_allowed_symbols=2,
        )

        self.assertTrue(payload["changed"])
        self.assertEqual(payload["proposed_allowed_symbols"], [])
        self.assertEqual(payload["proposed_effective_symbols"], ["CNYRUBF", "USDRUBF"])
        self.assertIn("updated", payload["reason"])

    def test_universe_tuning_rejects_weak_walk_forward_regime(self):
        payload = build_universe_selection_tuning_payload(
            configured_symbols=["USDRUBF", "CNYRUBF"],
            current_allowed_symbols=[],
            optimizer_payload={
                "evaluated_candidates": 12,
                "best_candidate": {
                    "symbols": ["USDRUBF"],
                    "summary": {},
                },
            },
            walk_forward_payload={
                "summary": {
                    "folds_evaluated": 3,
                    "probability_positive_pct": 42.0,
                },
                "folds": [
                    {
                        "best_candidate": {"symbols": ["USDRUBF"]},
                        "test_summary": {
                            "normalized_monthly_return_pct": 0.4,
                            "trades": 6,
                        },
                    }
                ],
            },
            max_allowed_symbols=1,
            min_walk_forward_positive_probability_pct=55.0,
        )

        self.assertFalse(payload["changed"])
        self.assertEqual(payload["proposed_allowed_symbols"], [])
        self.assertIn("positive-fold probability", payload["reason"])

    def test_universe_tuning_rejects_thin_latest_fold_evidence(self):
        payload = build_universe_selection_tuning_payload(
            configured_symbols=["USDRUBF", "CNYRUBF"],
            current_allowed_symbols=[],
            optimizer_payload={
                "evaluated_candidates": 12,
                "best_candidate": {
                    "symbols": ["USDRUBF"],
                    "summary": {},
                },
            },
            walk_forward_payload={
                "summary": {
                    "folds_evaluated": 1,
                    "probability_positive_pct": 100.0,
                },
                "folds": [
                    {
                        "best_candidate": {"symbols": ["USDRUBF"]},
                        "test_summary": {
                            "normalized_monthly_return_pct": 0.2,
                            "trades": 2,
                        },
                    }
                ],
            },
            max_allowed_symbols=1,
        )

        self.assertFalse(payload["changed"])
        self.assertEqual(payload["proposed_allowed_symbols"], [])
        self.assertIn("history is too short", payload["reason"])

    def test_universe_tuning_rejects_too_narrow_consensus_for_broad_runtime(self):
        payload = build_universe_selection_tuning_payload(
            configured_symbols=["SBER", "GAZP", "LKOH", "ROSN", "NVTK", "TATN", "GMKN", "PLZL"],
            current_allowed_symbols=[],
            optimizer_payload={
                "evaluated_candidates": 12,
                "best_candidate": {
                    "symbols": ["SBER", "GAZP"],
                    "summary": {},
                },
            },
            walk_forward_payload={
                "summary": {
                    "folds_evaluated": 4,
                    "probability_positive_pct": 62.0,
                },
                "folds": [
                    {
                        "best_candidate": {"symbols": ["GAZP", "SBER"]},
                        "test_summary": {
                            "normalized_monthly_return_pct": 0.5,
                            "trades": 9,
                        },
                    }
                ],
            },
            max_allowed_symbols=2,
        )

        self.assertFalse(payload["changed"])
        self.assertEqual(payload["proposed_allowed_symbols"], [])
        self.assertIn("too narrow", payload["reason"])


if __name__ == "__main__":
    unittest.main()
