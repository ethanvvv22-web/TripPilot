import unittest
from datetime import time

from trippilot.models import Itinerary, Stop
from trippilot.validation import ViolationType, validate_itinerary

from .fixtures import MONDAY, PLACES, SATURDAY, fixed_travel, request


def stop(place_id: str, arrive: str, depart: str, cost: float = 0) -> Stop:
    return Stop(
        place_id=place_id,
        name=PLACES[place_id].name if place_id in PLACES else place_id,
        arrival=time.fromisoformat(arrive),
        departure=time.fromisoformat(depart),
        estimated_cost=cost,
    )


def types(violations):
    return [v.type for v in violations]


class ValidateItineraryTest(unittest.TestCase):
    def check(self, stops, travel=fixed_travel(10), **request_overrides):
        req = request(**request_overrides)
        itinerary = Itinerary(date=req.date, stops=stops)
        return validate_itinerary(itinerary, req, PLACES, travel)

    def test_feasible_itinerary_has_no_violations(self):
        violations = self.check([
            stop("p_cathedral", "09:15", "10:30"),
            stop("p_museum", "10:45", "12:00", cost=120),
            stop("p_park", "12:15", "13:00"),
        ])
        self.assertEqual(violations, [])

    def test_unknown_hours_are_not_a_violation(self):
        self.assertEqual(self.check([stop("p_unknown_hours", "09:30", "10:00")]), [])

    def test_hallucinated_place(self):
        violations = self.check([stop("p_made_up", "09:30", "10:00")])
        self.assertEqual(types(violations), [ViolationType.UNKNOWN_PLACE])

    def test_departure_not_after_arrival(self):
        violations = self.check([stop("p_cathedral", "10:00", "10:00")])
        self.assertIn(ViolationType.INVALID_TIMES, types(violations))

    def test_visit_when_closed(self):
        violations = self.check([stop("p_museum", "15:00", "17:00")])
        self.assertEqual(types(violations), [ViolationType.CLOSED])
        self.assertIn("10:00-16:00", violations[0].message)

    def test_closed_all_day(self):
        req = request(date=MONDAY)
        itinerary = Itinerary(date=MONDAY, stops=[stop("p_museum", "11:00", "12:00")])
        violations = validate_itinerary(itinerary, req, PLACES, fixed_travel(10))
        self.assertEqual(types(violations), [ViolationType.CLOSED])
        self.assertIn("closed", violations[0].message)

    def test_overlapping_stops(self):
        violations = self.check([
            stop("p_cathedral", "09:15", "10:30"),
            stop("p_park", "10:00", "11:00"),
        ])
        self.assertEqual(types(violations), [ViolationType.OVERLAP])

    def test_not_enough_travel_time(self):
        violations = self.check([
            stop("p_cathedral", "09:30", "10:30"),
            stop("p_park", "10:35", "11:00"),
        ], travel=fixed_travel(20))
        self.assertEqual(types(violations), [ViolationType.INSUFFICIENT_TRAVEL_TIME])
        self.assertEqual(violations[0].stop_index, 1)

    def test_first_stop_too_soon_after_start(self):
        violations = self.check([stop("p_cathedral", "09:05", "10:00")], travel=fixed_travel(15))
        self.assertEqual(types(violations), [ViolationType.INSUFFICIENT_TRAVEL_TIME])
        self.assertIn("start location", violations[0].message)

    def test_no_route(self):
        violations = self.check([stop("p_cathedral", "09:30", "10:00")],
                                travel=lambda origin, destination, mode: None)
        self.assertEqual(types(violations), [ViolationType.NO_ROUTE])

    def test_outside_trip_window(self):
        violations = self.check([
            stop("p_cathedral", "08:30", "09:30"),
            stop("p_park", "17:30", "19:00"),
        ])
        self.assertEqual(types(violations), [ViolationType.OUTSIDE_TRIP_WINDOW] * 2)

    def test_wrong_date(self):
        req = request()
        itinerary = Itinerary(date=MONDAY, stops=[stop("p_park", "09:30", "10:00")])
        violations = validate_itinerary(itinerary, req, PLACES, fixed_travel(10))
        self.assertIn(ViolationType.OUTSIDE_TRIP_WINDOW, types(violations))

    def test_over_budget(self):
        violations = self.check([
            stop("p_cathedral", "09:15", "10:00", cost=200),
            stop("p_museum", "10:15", "11:00", cost=150),
        ])
        self.assertEqual(types(violations), [ViolationType.OVER_BUDGET])

    def test_no_budget_means_no_budget_check(self):
        violations = self.check([stop("p_museum", "10:15", "11:00", cost=10_000)], budget=None)
        self.assertEqual(violations, [])

    def test_reports_every_violation_at_once(self):
        violations = self.check([
            stop("p_made_up", "09:30", "10:00"),
            stop("p_museum", "15:30", "16:30", cost=500),
        ])
        self.assertEqual(
            set(types(violations)),
            {ViolationType.UNKNOWN_PLACE, ViolationType.CLOSED, ViolationType.OVER_BUDGET},
        )

    def test_duplicate_place(self):
        violations = self.check([
            stop("p_cathedral", "09:15", "10:00"),
            stop("p_park", "10:15", "11:00"),
            stop("p_cathedral", "11:15", "12:00"),
        ])
        self.assertEqual(types(violations), [ViolationType.DUPLICATE_PLACE])
        self.assertEqual(violations[0].stop_index, 2)

    def test_already_visited_place(self):
        req = request()
        itinerary = Itinerary(date=req.date, stops=[stop("p_park", "09:30", "10:00")])
        violations = validate_itinerary(itinerary, req, PLACES, fixed_travel(10), already_visited={"p_park"})
        self.assertEqual(types(violations), [ViolationType.DUPLICATE_PLACE])

    def test_empty_itinerary(self):
        self.assertEqual(self.check([]), [])


if __name__ == "__main__":
    unittest.main()
