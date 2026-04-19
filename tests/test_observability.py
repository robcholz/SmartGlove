from __future__ import annotations

import unittest

from server.observability import capture_collapsed_output


class ObservabilityTest(unittest.TestCase):
    def test_collapses_consecutive_duplicate_messages_while_refreshing_timestamp(
        self,
    ) -> None:
        output = capture_collapsed_output(
            [
                (
                    ("INFO", "server.app", "device aca704299de8 is online"),
                    "2026-04-19 10:00:00,000 INFO server.app device aca704299de8 is online",
                ),
                (
                    ("INFO", "server.app", "device aca704299de8 is online"),
                    "2026-04-19 10:00:01,000 INFO server.app device aca704299de8 is online",
                ),
                (
                    ("INFO", "server.app", "device aca704299de8 is online"),
                    "2026-04-19 10:00:02,000 INFO server.app device aca704299de8 is online",
                ),
                (
                    ("INFO", "server.app", "device aca704299de8 is offline"),
                    "2026-04-19 10:00:03,000 INFO server.app device aca704299de8 is offline",
                ),
            ]
        )

        self.assertIn(
            "\r2026-04-19 10:00:00,000 INFO server.app device aca704299de8 is online",
            output,
        )
        self.assertIn(
            "\r2026-04-19 10:00:01,000 INFO server.app device aca704299de8 is online (x2)",
            output,
        )
        self.assertIn(
            "\r2026-04-19 10:00:02,000 INFO server.app device aca704299de8 is online (x3)",
            output,
        )
        self.assertTrue(
            output.endswith(
                "2026-04-19 10:00:03,000 INFO server.app device aca704299de8 is offline\n"
            )
        )


if __name__ == "__main__":
    unittest.main()
