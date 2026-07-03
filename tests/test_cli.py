from __future__ import annotations

import unittest

from samosbor.cli import build_parser


class CliParserTest(unittest.TestCase):
    def test_tune_entry_hours_defaults_to_three_trades_per_hour(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "--config",
                "configs/demo.toml",
                "tune-entry-hours",
            ]
        )

        self.assertEqual(args.command, "tune-entry-hours")
        self.assertEqual(args.min_trades_per_hour, 3)

    def test_walk_forward_accepts_adaptive_history_flag(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "--config",
                "configs/demo.toml",
                "walk-forward",
                "--adaptive-history",
            ]
        )

        self.assertEqual(args.command, "walk-forward")
        self.assertTrue(args.adaptive_history)

    def test_walk_forward_adaptive_history_defaults_to_false(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "--config",
                "configs/demo.toml",
                "walk-forward",
            ]
        )

        self.assertEqual(args.command, "walk-forward")
        self.assertFalse(args.adaptive_history)

    def test_collect_microstructure_accepts_interval_depth_and_once(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "--config",
                "configs/demo.toml",
                "collect-microstructure",
                "--interval-sec",
                "15",
                "--depth",
                "10",
                "--once",
            ]
        )

        self.assertEqual(args.command, "collect-microstructure")
        self.assertEqual(args.interval_sec, 15)
        self.assertEqual(args.depth, 10)
        self.assertTrue(args.once)


if __name__ == "__main__":
    unittest.main()
