import json
import unittest
from datetime import time

from trippilot.agent.planner import instructions
from trippilot.agent.toolbox import TOOL_SPECS, ToolBox, tool_specs
from trippilot.models import Itinerary, Stop
from trippilot.offline import UPPSALA_PLACES, UPPSALA_START, OfflinePlaces, straight_line_minutes
from trippilot.validation import ViolationType, validate_itinerary

from .fixtures import PLACES, fixed_travel, request


def toolbox(travel=fixed_travel(10)):
    return ToolBox(request(), OfflinePlaces(PLACES.values()), travel)


def call(box, name, **arguments):
    return box.execute(name, json.dumps(arguments))


class ToolSpecsTest(unittest.TestCase):
    def test_every_spec_has_a_handler(self):
        for batched in (True, False):
            box = ToolBox(request(), OfflinePlaces(PLACES.values()), fixed_travel(10), batched=batched)
            for spec in tool_specs(batched):
                self.assertIn(spec["name"], box._handlers())

    def test_instructions_match_tool_style(self):
        self.assertIn("all queries in one call", instructions(batched=True))
        self.assertNotIn("all queries in one call", instructions(batched=False))

    def test_strict_mode_requires_every_property(self):
        # Foundry strict function tools need all properties listed as required.
        def check(schema):
            if schema.get("type") == "object":
                self.assertEqual(set(schema["required"]), set(schema["properties"]))
                self.assertFalse(schema["additionalProperties"])
                for prop in schema["properties"].values():
                    check(prop)
            elif schema.get("type") == "array":
                check(schema["items"])
        for spec in tool_specs(True) + tool_specs(False):
            check(spec["parameters"])


class ToolBoxTest(unittest.TestCase):
    def test_search_registers_known_places(self):
        box = toolbox()
        out = call(box, "search_places", queries=["museum"], max_results=5)
        self.assertEqual([p["place_id"] for p in out["places"]], ["p_museum"])
        self.assertEqual(out["places"][0]["hours_on_trip_date"], "10:00-16:00")
        self.assertIn("p_museum", box.known_places)

    def test_place_details_for_unknown_id(self):
        out = call(toolbox(), "get_place_details", place_id="p_nope")
        self.assertIn("error", out)

    def test_multi_query_search_describes_each_place_once(self):
        box = toolbox()
        out = call(box, "search_places", queries=["museum", "church", "museum"], max_results=3)
        self.assertEqual(sorted(p["place_id"] for p in out["places"]), ["p_cathedral", "p_museum"])
        self.assertEqual(out["matches"], {"museum": ["p_museum"], "church": ["p_cathedral"]})

    def test_too_many_queries_are_capped(self):
        out = call(toolbox(), "search_places", queries=[f"q{i}" for i in range(7)], max_results=1)
        self.assertEqual(len(out["matches"]), 5)
        self.assertIn("first 5", out["note"])

    def test_travel_time_matrix(self):
        box = toolbox(fixed_travel(12))
        call(box, "search_places", queries=["museum", "church"], max_results=3)
        out = call(box, "get_travel_times", place_ids=["p_museum", "p_cathedral", "p_nope", "start"])
        self.assertEqual(out["minutes"], {
            "start": {"p_museum": 12, "p_cathedral": 12},
            "p_museum": {"p_cathedral": 12},
            "p_cathedral": {"p_museum": 12},
        })
        self.assertEqual(out["unknown_place_ids"], ["p_nope"])

    def test_legacy_tool_style(self):
        box = ToolBox(request(), OfflinePlaces(PLACES.values()), fixed_travel(12), batched=False)
        out = call(box, "search_places", query="museum", max_results=3)
        self.assertEqual([p["place_id"] for p in out["places"]], ["p_museum"])
        out = call(box, "get_travel_times", from_place_id="start", to_place_ids=["p_museum", "p_nope"])
        self.assertEqual(out["minutes"], {"p_museum": 12, "p_nope": "unknown place_id"})

    def test_shown_travel_time_is_always_enough(self):
        # Regression: minutes were rounded to nearest, so a gap of the shown 9 min
        # failed against an exact 9.1 min and the validator's message read "only 9, needs about 9".
        req = request(end_time=time(23, 0))
        places = {p.place_id: p for p in UPPSALA_PLACES}
        box = ToolBox(req, OfflinePlaces(), straight_line_minutes, seed_places=places)
        matrix = box.get_travel_times(list(places)[:10])["minutes"]
        for a, row in matrix.items():
            if a == "start":
                continue
            for b, shown in row.items():
                arrive = 10 * 60 + shown
                itinerary = Itinerary(date=req.date, stops=[
                    Stop(place_id=a, name=a, arrival=time(9, 0), departure=time(10, 0)),
                    Stop(place_id=b, name=b, arrival=time(arrive // 60, arrive % 60), departure=time(22, 0)),
                ])
                violations = validate_itinerary(itinerary, req.model_copy(update={"start_time": time(8, 0)}),
                                                places, straight_line_minutes)
                travel = [v for v in violations if v.type == ViolationType.INSUFFICIENT_TRAVEL_TIME and v.stop_index == 1]
                self.assertEqual(travel, [], f"{a} -> {b}: shown {shown} min")

    def test_travel_times_without_route(self):
        box = toolbox(lambda origin, destination, mode: None)
        call(box, "search_places", queries=["museum"], max_results=3)
        out = call(box, "get_travel_times", place_ids=["p_museum"])
        self.assertEqual(out["minutes"]["start"]["p_museum"], "no route")

    def test_routes_client_matrix_is_fetched_in_one_request(self):
        class FakeRoutes:
            def __init__(self):
                self.matrix_calls = []

            def compute_matrix(self, origins, destinations, mode):
                self.matrix_calls.append(len(origins) * len(destinations))

            def travel_minutes(self, origin, destination, mode):
                return 5

        routes = FakeRoutes()
        box = toolbox(routes.travel_minutes)
        call(box, "search_places", queries=["museum", "church", "park"], max_results=3)
        call(box, "get_travel_times", place_ids=["p_museum", "p_cathedral", "p_park"])
        self.assertEqual(routes.matrix_calls, [16])  # start + 3 places, all pairs

    def test_submit_accepted(self):
        box = toolbox()
        call(box, "search_places", queries=["museum"], max_results=3)
        out = call(box, "submit_itinerary", stops=[
            {"place_id": "p_museum", "name": "Gustavianum", "arrival": "10:00", "departure": "11:00", "estimated_cost": 100},
        ])
        self.assertEqual(out, {"accepted": True})
        self.assertTrue(box.submissions[-1].accepted)

    def test_submit_uses_provider_place_name(self):
        box = toolbox()
        call(box, "search_places", queries=["museum"], max_results=3)
        call(box, "submit_itinerary", stops=[
            {"place_id": "p_museum", "name": "Gustavianun (typo)", "arrival": "10:00", "departure": "11:00", "estimated_cost": 0},
        ])
        self.assertEqual(box.submissions[-1].itinerary.stops[0].name, "Gustavianum")

    def test_submit_place_not_from_search_is_rejected(self):
        # p_cathedral exists in the provider but was never surfaced by a tool call.
        box = toolbox()
        out = call(box, "submit_itinerary", stops=[
            {"place_id": "p_cathedral", "name": "Uppsala Cathedral", "arrival": "10:00", "departure": "11:00", "estimated_cost": 0},
        ])
        self.assertFalse(out["accepted"])
        self.assertEqual(out["violations"][0]["type"], "unknown_place")

    def test_submit_malformed_time(self):
        box = toolbox()
        out = call(box, "submit_itinerary", stops=[
            {"place_id": "p_museum", "name": "Gustavianum", "arrival": "ten", "departure": "11:00", "estimated_cost": 0},
        ])
        self.assertFalse(out["accepted"])
        self.assertIn("Malformed", out["error"])
        self.assertEqual(len(box.submissions), 1)

    def test_bad_json_and_unknown_tool_are_reported(self):
        box = toolbox()
        self.assertIn("not valid JSON", box.execute("search_places", "{oops")["error"])
        self.assertIn("Unknown tool", call(box, "book_hotel")["error"])
        self.assertIn("Invalid arguments", call(box, "search_places", city="x")["error"])

    def test_backend_errors_are_returned_not_raised(self):
        class Broken:
            def search_text(self, *a, **k):
                raise RuntimeError("HTTP 503")
        box = ToolBox(request(), Broken(), fixed_travel(10))
        self.assertIn("HTTP 503", call(box, "search_places", queries=["museum"], max_results=3)["error"])

    def test_recommendations_accepted(self):
        box = ToolBox(request(start_time=time(10, 0)), OfflinePlaces(PLACES.values()), fixed_travel(10))
        call(box, "search_places", queries=["museum"], max_results=3)
        out = call(box, "submit_recommendations", recommendations=[
            {"place_id": "p_museum", "reason": "Great for history."},
        ])
        self.assertEqual(out, {"accepted": True})
        self.assertEqual(box.recommendation_submissions[-1].recommendations[0].place.name, "Gustavianum")

    def test_recommendations_reject_unsearched_and_closing_places(self):
        # The museum closes at 16:00, so it is not open for an hour from 15:30.
        box = ToolBox(request(start_time=time(15, 30)), OfflinePlaces(PLACES.values()), fixed_travel(10))
        call(box, "search_places", queries=["museum"], max_results=3)
        out = call(box, "submit_recommendations", recommendations=[
            {"place_id": "p_museum", "reason": "History."},
            {"place_id": "p_made_up", "reason": "Invented."},
        ])
        self.assertFalse(out["accepted"])
        self.assertEqual(len(out["problems"]), 2)
        self.assertIn("16:00", out["problems"][0])
        self.assertIn("p_made_up", out["problems"][1])

    def test_empty_recommendations_rejected(self):
        out = call(toolbox(), "submit_recommendations", recommendations=[])
        self.assertFalse(out["accepted"])

    def test_seed_places_are_usable_without_search(self):
        box = ToolBox(request(), OfflinePlaces(PLACES.values()), fixed_travel(10),
                      seed_places={"p_museum": PLACES["p_museum"]})
        out = call(box, "submit_itinerary", stops=[
            {"place_id": "p_museum", "name": "Gustavianum", "arrival": "10:00", "departure": "11:00", "estimated_cost": 0},
        ])
        self.assertEqual(out, {"accepted": True})

    def test_every_call_is_traced(self):
        box = toolbox()
        call(box, "search_places", queries=["museum"], max_results=3)
        call(box, "get_place_details", place_id="p_museum")
        self.assertEqual([r.name for r in box.trace], ["search_places", "get_place_details"])
        self.assertGreaterEqual(box.trace[0].duration_ms, 0)


if __name__ == "__main__":
    unittest.main()
