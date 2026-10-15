"""Replanning and recommendation runs against a scripted model."""

import unittest
from datetime import time

from trippilot.agent import ModelTurn, TripPlanner
from trippilot.models import Itinerary, Stop
from trippilot.offline import OfflinePlaces

from .fixtures import PLACES, SATURDAY, fixed_travel, request
from .test_planner import ScriptedBackend, stop, tool


def current_plan() -> Itinerary:
    return Itinerary(date=SATURDAY, stops=[
        Stop(place_id="p_cathedral", name="Uppsala Cathedral", arrival=time(9, 30), departure=time(10, 30), estimated_cost=50),
        Stop(place_id="p_park", name="City Park", arrival=time(11, 0), departure=time(12, 0)),
        Stop(place_id="p_museum", name="Gustavianum", arrival=time(13, 0), departure=time(14, 0), estimated_cost=100),
    ])


def planner(turns):
    backend = ScriptedBackend(turns)
    return TripPlanner(backend, OfflinePlaces(PLACES.values()), fixed_travel(10)), backend.session


class ReplanTest(unittest.TestCase):
    def test_started_stops_are_locked_and_rest_is_replanned(self):
        p, session = planner([
            # Rain: drop the park, keep the museum without searching again.
            ModelTurn(tool_calls=[tool("submit_itinerary", stops=[stop("p_museum", "Gustavianum", "11:00", "12:30", 100)])]),
            ModelTurn(text="Swapped the park for more museum time."),
        ])
        result = p.replan(request(budget=300), current_plan(), now=time(10, 0),
                          change="It is raining; avoid outdoor places.", known_places=PLACES)

        self.assertEqual(result.status, "accepted")
        self.assertEqual([s.name for s in result.locked_stops], ["Uppsala Cathedral"])
        self.assertEqual([s.name for s in result.itinerary.stops], ["Uppsala Cathedral", "Gustavianum"])

        message = session.received[0]
        self.assertIn("Resume at 10:30", message)         # after the locked stop ends
        self.assertIn("raining", message)
        self.assertIn("City Park", message)                # remaining stops are listed
        self.assertIn("do not revisit): Uppsala Cathedral", message)
        self.assertIn("Budget: 250 left", message)         # 300 minus the cathedral's 50

    def test_replan_starts_from_last_visited_location(self):
        travelled = []

        def travel(origin, destination, mode):
            travelled.append(origin)
            return 10

        backend = ScriptedBackend([
            ModelTurn(tool_calls=[tool("submit_itinerary", stops=[stop("p_museum", "Gustavianum", "11:00", "12:00")])]),
            ModelTurn(text="ok"),
        ])
        TripPlanner(backend, OfflinePlaces(PLACES.values()), travel).replan(
            request(), current_plan(), now=time(10, 0), change="Running late.", known_places=PLACES)
        self.assertEqual(travelled[0], PLACES["p_cathedral"].location)

    def test_replanned_part_is_still_validated(self):
        p, _ = planner([
            # Museum closes at 16:00.
            ModelTurn(tool_calls=[tool("submit_itinerary", stops=[stop("p_museum", "Gustavianum", "15:30", "16:30")])]),
            ModelTurn(text="done"),
        ])
        result = p.replan(request(), current_plan(), now=time(10, 0), change="Later start.", known_places=PLACES)
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.violations[0].type.value, "closed")

    def test_revisiting_a_visited_place_is_rejected(self):
        p, _ = planner([
            ModelTurn(tool_calls=[tool("submit_itinerary", stops=[
                stop("p_museum", "Gustavianum", "11:00", "12:00"),
                stop("p_cathedral", "Uppsala Cathedral", "12:15", "13:00"),
            ])]),
            ModelTurn(text="done"),
        ])
        result = p.replan(request(), current_plan(), now=time(10, 0), change="Rain.", known_places=PLACES)
        self.assertEqual(result.status, "rejected")
        self.assertEqual([v.type.value for v in result.violations], ["duplicate_place"])
        self.assertIn("already visited", result.violations[0].message)

    def test_before_any_stop_nothing_is_locked(self):
        p, session = planner([ModelTurn(text="no changes")])
        result = p.replan(request(), current_plan(), now=time(9, 0), change="Skip the park.", known_places=PLACES)
        self.assertEqual(result.locked_stops, [])
        self.assertIn("from the start location", session.received[0])

    def test_day_already_over(self):
        p, _ = planner([])
        with self.assertRaises(ValueError):
            p.replan(request(), current_plan(), now=time(18, 0), change="x", known_places=PLACES)


class RecommendTest(unittest.TestCase):
    def test_repair_then_accept(self):
        p, session = planner([
            ModelTurn(tool_calls=[tool("search_places", queries=["museum"], max_results=3)]),
            ModelTurn(tool_calls=[tool("submit_recommendations", recommendations=[
                {"place_id": "p_fake", "reason": "Made up."}])]),
            ModelTurn(tool_calls=[tool("submit_recommendations", recommendations=[
                {"place_id": "p_museum", "reason": "University history museum."}])]),
            ModelTurn(text="Try Gustavianum."),
        ])
        result = p.recommend(request(start_time=time(10, 0)), "Where can I learn about local history?")

        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.submissions, 2)
        self.assertEqual([r.place.name for r in result.recommendations], ["Gustavianum"])
        self.assertIn("local history", session.received[0])

    def test_no_recommendations(self):
        p, _ = planner([ModelTurn(text="Sorry.")])
        self.assertEqual(p.recommend(request(), "anything").status, "no_submission")


if __name__ == "__main__":
    unittest.main()
