from __future__ import annotations

import unittest

from samosbor.domain import TradeMode
from samosbor.safety import LiveTradingDisabledError, assert_paper_only_mode


class SafetyTest(unittest.TestCase):
    def test_local_paper_mode_is_allowed(self):
        assert_paper_only_mode(
            TradeMode.LOCAL_PAPER,
            allow_live_trading=False,
            live_flag=False,
        )

    def test_live_mode_is_blocked(self):
        with self.assertRaises(LiveTradingDisabledError):
            assert_paper_only_mode(
                TradeMode.LIVE,
                allow_live_trading=False,
                live_flag=False,
            )


if __name__ == "__main__":
    unittest.main()
