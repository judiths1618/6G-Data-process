import datetime as dt
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dq_local_beam import _format_event_time_value, parse_event_time


class ParseEventTimeTest(unittest.TestCase):
    def test_deepsense_scen1_parses_bracketed_value(self) -> None:
        timestamp = parse_event_time("['00-42-15-125']", "deepsense_scen1")
        expected = dt.datetime(1970, 1, 1, 0, 42, 15, 125000, tzinfo=dt.timezone.utc)
        self.assertEqual(timestamp, expected)

    def test_deepsense_scen1_supports_missing_fraction(self) -> None:
        timestamp = parse_event_time("['00-42-15-0']", "deepsense_scen1")
        expected = dt.datetime(1970, 1, 1, 0, 42, 15, tzinfo=dt.timezone.utc)
        self.assertEqual(timestamp, expected)

    def test_deepsense_scen42_parses_plain_value(self) -> None:
        timestamp = parse_event_time("12-00-06.900", "deepsense_scen42")
        expected = dt.datetime(1970, 1, 1, 12, 0, 6, 900000, tzinfo=dt.timezone.utc)
        self.assertEqual(timestamp, expected)

    def test_formatting_uses_iso_output(self) -> None:
        instant = dt.datetime(2023, 5, 1, 12, 34, 56, 789000, tzinfo=dt.timezone.utc)
        formatted = _format_event_time_value(instant, "deepsense_scen1", "['00-00-00-0']")
        self.assertEqual(formatted, "2023-05-01T12:34:56.789000Z")


if __name__ == "__main__":
    unittest.main()
