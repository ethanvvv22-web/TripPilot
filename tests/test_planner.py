"""Planner loop tests against a scripted model, no network needed."""

import json
import unittest

from trippilot.agent import ModelTurn, ToolCall, TripPlanner
from trippilot.offline import OfflinePlaces

from .fixtures import PLACES, fixed_travel, request


def tool(name, **arguments):
    return ToolCall(call_id=f"call_{name}", name=name, arguments=json.dumps(arguments))


def stop(place_id, name, arrival, departure, cost=0):
    return {"place_id": place_id, "name": name, "arrival": arrival, "departure": departure, "estimated_cost": cost}


class ScriptedSession:
    """Replays a fixed list of model turns and records what it was sent."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.received = []
        self.closed = False

    def send_user_message(self, text):
        self.received.append(text)
        return self.turns.pop(0)

    def send_tool_results(self, results):
        self.received.append([json.loads(output) for _, output in results])
        return self.turns.pop(0) if self.turns else ModelTurn(text="(script ended)")

    def close(self):
        self.closed = True


class ScriptedBackend:
    def __init__(self, turns):
        self.session = ScriptedSession(turns)

    def new_session(self):
        return self.session


def planner(turns, **kwargs):
    backend = ScriptedBackend(turns)
    return TripPlanner(backend, OfflinePlaces(PLACES.values()), fixed_travel(10), **kwargs), backend.session


class TripPlannerTest(unittest.TestCase):
    def test_repair_loop_ends_accepted(self):
        p, session = planner([
            ModelTurn(tool_calls=[tool("search_places", queries=["museum"], max_results=3)]),
            # First attempt visits the museum after it closes at 16:00.
            ModelTurn(tool_calls=[tool("submit_itinerary", stops=[stop("p_museum", "Gustavianum", "15:30", "16:30")])]),
            ModelTurn(tool_calls=[tool("submit_itinerary", stops=[stop("p_museum", "Gustavianum", "14:30", "15:30")])]),
            ModelTurn(text="Visit Gustavianum in the afternoon."),
        ])
        result = p.plan(request())

        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.submissions, 2)
        self.assertEqual(result.violations, [])
        self.assertEqual(result.summary, "Visit Gustavianum in the afternoon.")
        self.assertEqual([r.name for r in result.trace], ["search_places", "submit_itinerary", "submit_itinerary"])
        # The model saw the validator's violation before repairing.
        self.assertEqual(session.received[2][0]["violations"][0]["type"], "closed")
        self.assertTrue(session.closed)

    def test_request_is_sent_as_first_message(self):
        p, session = planner([ModelTurn(text="Hello")])
        p.plan(request(preferences=["history"]))
        self.assertIn("Uppsala", session.received[0])
        self.assertIn("history", session.received[0])

    def test_no_submission(self):
        p, _ = planner([ModelTurn(text="I cannot plan this.")])
        result = p.plan(request())
        self.assertEqual(result.status, "no_submission")
        self.assertIsNone(result.itinerary)

    def test_final_submission_decides_status(self):
        p, _ = planner([
            ModelTurn(tool_calls=[tool("search_places", queries=["museum"], max_results=3)]),
            ModelTurn(tool_calls=[tool("submit_itinerary", stops=[stop("p_museum", "Gustavianum", "10:00", "11:00")])]),
            # A later, worse submission overrides the accepted one.
            ModelTurn(tool_calls=[tool("submit_itinerary", stops=[stop("p_made_up", "Fake Place", "10:00", "11:00")])]),
            ModelTurn(text="Done."),
        ])
        result = p.plan(request())
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.violations[0].type.value, "unknown_place")

    def test_submission_limit(self):
        bad = tool("submit_itinerary", stops=[stop("p_made_up", "Fake", "10:00", "11:00")])
        p, session = planner([ModelTurn(tool_calls=[bad])] * 3 + [ModelTurn(text="Gave up.")], max_submissions=2)
        result = p.plan(request())
        self.assertEqual(result.submissions, 2)
        self.assertEqual(result.status, "rejected")
        self.assertIn("Submission limit", session.received[3][0]["error"])

    def test_turn_limit(self):
        search = tool("search_places", queries=["museum"], max_results=3)
        p, _ = planner([ModelTurn(tool_calls=[search])] * 10, max_turns=4)
        result = p.plan(request())
        self.assertTrue(result.hit_turn_limit)
        self.assertEqual(result.turns, 4)
        self.assertEqual(result.status, "no_submission")

    def test_tool_call_callback_and_places(self):
        seen = []
        backend = ScriptedBackend([
            ModelTurn(tool_calls=[tool("search_places", queries=["museum"], max_results=3)]),
            ModelTurn(text="ok"),
        ])
        p = TripPlanner(backend, OfflinePlaces(PLACES.values()), fixed_travel(10),
                        on_tool_call=lambda record: seen.append(record.name))
        result = p.plan(request())
        self.assertEqual(seen, ["search_places"])
        self.assertIn("p_museum", result.places)

    def test_parallel_tool_calls_answered_together(self):
        p, session = planner([
            ModelTurn(tool_calls=[
                tool("search_places", queries=["museum"], max_results=3),
                tool("search_places", queries=["park"], max_results=3),
            ]),
            ModelTurn(text="ok"),
        ])
        p.plan(request())
        self.assertEqual(len(session.received[1]), 2)


if __name__ == "__main__":
    unittest.main()
