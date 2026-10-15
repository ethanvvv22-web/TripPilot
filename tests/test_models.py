import unittest
from datetime import date

from trippilot.models import OpeningPeriod, google_day

from .fixtures import CATHEDRAL, DAY, MUSEUM, NO_HOURS, PARK


class GoogleDayTest(unittest.TestCase):
    def test_sunday_is_zero(self):
        self.assertEqual(google_day(date(2026, 10, 18)), 0)

    def test_saturday_is_six(self):
        self.assertEqual(google_day(date(2026, 10, 17)), 6)


class IsOpenBetweenTest(unittest.TestCase):
    def test_visit_inside_hours(self):
        self.assertTrue(CATHEDRAL.is_open_between(6, 9 * 60, 10 * 60))

    def test_visit_running_past_closing(self):
        self.assertFalse(CATHEDRAL.is_open_between(6, 17 * 60, 19 * 60))

    def test_closed_day(self):
        self.assertFalse(MUSEUM.is_open_between(1, 11 * 60, 12 * 60))

    def test_open_24_7(self):
        self.assertTrue(PARK.is_open_between(3, 2 * 60, 4 * 60))

    def test_unknown_hours(self):
        self.assertIsNone(NO_HOURS.is_open_between(6, 9 * 60, 10 * 60))

    def test_period_wrapping_past_saturday_midnight(self):
        # Saturday 22:00 -> Sunday 02:00
        bar = PARK.model_copy(update={"opening_periods": [
            OpeningPeriod(start=6 * DAY + 22 * 60, end=7 * DAY + 2 * 60),
        ]})
        self.assertTrue(bar.is_open_between(0, 0, 60))
        self.assertTrue(bar.is_open_between(6, 23 * 60, 23 * 60 + 30))
        self.assertFalse(bar.is_open_between(0, 3 * 60, 4 * 60))


class HoursOnTest(unittest.TestCase):
    def test_regular_day(self):
        self.assertEqual(MUSEUM.hours_on(6), "10:00-16:00")

    def test_closed_day(self):
        self.assertEqual(MUSEUM.hours_on(1), "closed")

    def test_always_open(self):
        self.assertEqual(PARK.hours_on(1), "open 24 hours")


if __name__ == "__main__":
    unittest.main()
