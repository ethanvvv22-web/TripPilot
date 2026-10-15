"""The benchmark data is sound and scoring works, without calling a model."""

import os
import tempfile
import unittest
from datetime import time

from trippilot.agent import ModelTurn, TripPlanner
from trippilot.evaluation import (Expectations, check_expectations, load_benchmark, merge_results, preference_coverage,
                                render_markdown, run_benchmark, run_scenario, summarize)
from trippilot.models import Itinerary, Stop
from trippilot.offline import UPPSALA_PLACES, OfflinePlaces, straight_line_minutes
from trippilot.validation import validate_itinerary

from .test_planner import ScriptedBackend, stop, tool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK = load_benchmark(os.path.join(ROOT, "eval", "scenarios.json"))
PLACES = {p.place_id: p for p in UPPSALA_PLACES}


class ScenarioDataTest(unittest.TestCase):
    def test_ids_are_unique(self):
        ids = [s.id for s in BENCHMARK.scenarios]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(ids), 30)

    def test_expected_place_ids_exist(self):
        for s in BENCHMARK.scenarios:
            for pid in s.expect.include_place_ids + s.expect.exclude_place_ids:
                self.assertIn(pid, PLACES, s.id)

    def test_expected_types_exist_in_dataset(self):
        all_types = {t for p in UPPSALA_PLACES for t in p.types}
        for s in BENCHMARK.scenarios:
            for t in s.expect.include_types + s.expect.exclude_types:
                self.assertIn(t, all_types, s.id)

    def test_base_itineraries_are_valid(self):
        # A replan scenario is only meaningful if the plan it starts from is feasible.
        for s in BENCHMARK.scenarios:
            if s.task != "replan":
                continue
            violations = validate_itinerary(BENCHMARK.base_itinerary(s), BENCHMARK.request(s), PLACES, straight_line_minutes)
            self.assertEqual(violations, [], s.id)

    def test_replan_times_are_inside_the_day(self):
        for s in BENCHMARK.scenarios:
            if s.replan:
                self.assertLess(s.replan.now, s.end_time, s.id)


SNAPSHOT = os.path.join(ROOT, "data", "uppsala_snapshot.json")


@unittest.skipUnless(os.path.exists(SNAPSHOT), "no Google snapshot; record one with python -m trippilot.snapshot")
class RealScenarioDataTest(unittest.TestCase):
    """The benchmark over the recorded Google snapshot (not in the repo; see .gitignore)."""

    @classmethod
    def setUpClass(cls):
        cls.benchmark = load_benchmark(os.path.join(ROOT, "eval", "scenarios_real.json"))
        cls.places = cls.benchmark.dataset.by_id()

    def test_uses_the_snapshot(self):
        self.assertTrue(self.benchmark.dataset.name.startswith("snapshot"))
        self.assertEqual(len({s.id for s in self.benchmark.scenarios}), len(self.benchmark.scenarios))

    def test_referenced_places_exist(self):
        for s in self.benchmark.scenarios:
            for pid in s.expect.include_place_ids + s.expect.exclude_place_ids:
                self.assertIn(pid, self.places, s.id)

    def test_base_itinerary_is_feasible(self):
        for s in self.benchmark.scenarios:
            if s.replan:
                violations = validate_itinerary(self.benchmark.base_itinerary(s), self.benchmark.request(s),
                                                self.places, self.benchmark.dataset.travel_minutes)
                self.assertEqual(violations, [], s.id)


def itinerary(*stops):
    return Itinerary(date=BENCHMARK.scenarios[0].date, stops=list(stops))


def s(place_id, arrive="10:00", depart="11:00"):
    return Stop(place_id=place_id, name=PLACES[place_id].name,
                arrival=time.fromisoformat(arrive), departure=time.fromisoformat(depart))


class ExpectationsTest(unittest.TestCase):
    def test_all_met(self):
        it = itinerary(s("off_castle"), s("off_cafe", "11:15", "12:00"))
        expect = Expectations(include_place_ids=["off_castle"], include_types=["fika"],
                              exclude_types=["garden"], end_by=time(12, 0), min_stops=2)
        self.assertEqual(check_expectations(expect, it, it.stops, PLACES), [])

    def test_each_failure_is_reported(self):
        it = itinerary(s("off_linnaeus", "10:00", "16:30"))
        expect = Expectations(include_place_ids=["off_castle"], exclude_place_ids=["off_linnaeus"],
                              include_types=["restaurant"], exclude_types=["garden"],
                              end_by=time(16, 0), min_stops=2)
        self.assertEqual(len(check_expectations(expect, it, it.stops, PLACES)), 6)

    def test_place_checks_only_look_at_replanned_part(self):
        it = itinerary(s("off_linnaeus"), s("off_castle", "11:30", "12:30"))
        expect = Expectations(exclude_types=["garden"])
        self.assertEqual(check_expectations(expect, it, it.stops[1:], PLACES), [])

    def test_no_itinerary(self):
        self.assertEqual(check_expectations(Expectations(), None, [], PLACES), ["no itinerary"])

    def test_preference_coverage(self):
        stops = [s("off_cafe"), s("off_cathedral", "11:30", "12:00")]
        self.assertEqual(preference_coverage(["fika", "history", "swimming"], stops, PLACES), 2 / 3)
        self.assertIsNone(preference_coverage([], stops, PLACES))


def scenario(scenario_id):
    return next(x for x in BENCHMARK.scenarios if x.id == scenario_id)


def planner(turns):
    return TripPlanner(ScriptedBackend(turns), OfflinePlaces(), straight_line_minutes)


class RunScenarioTest(unittest.TestCase):
    def test_plan_repaired_on_second_attempt(self):
        p = planner([
            ModelTurn(tool_calls=[tool("search_places", queries=["viking"], max_results=3)]),
            # Gamla Uppsala opens at 10:00; the walk from the station takes over an hour.
            ModelTurn(tool_calls=[tool("submit_itinerary", stops=[stop("off_gamla", "Gamla Uppsala Museum", "10:05", "12:00")])],
                      input_tokens=1000, output_tokens=100),
            ModelTurn(tool_calls=[tool("submit_itinerary", stops=[stop("off_gamla", "Gamla Uppsala Museum", "11:30", "13:00")])]),
            ModelTurn(text="ok"),
        ])
        row = run_scenario(p, BENCHMARK, scenario("plan-sat-walk-viking"))
        self.assertFalse(row["first_valid"])
        self.assertEqual(row["first_violation_types"], ["insufficient_travel_time"])
        self.assertTrue(row["final_valid"])
        self.assertTrue(row["task_success"])
        self.assertEqual(row["preference_coverage"], 1.0)
        self.assertEqual(row["input_tokens"], 1000)

    def test_replan_expectation_failure(self):
        # Valid replan that ignores "skip the castle".
        p = planner([
            ModelTurn(tool_calls=[tool("submit_itinerary", stops=[stop("off_castle", "Uppsala Castle", "11:15", "12:15")])]),
            ModelTurn(text="ok"),
        ])
        row = run_scenario(p, BENCHMARK, scenario("replan-skip-castle-lunch"))
        self.assertTrue(row["final_valid"])
        self.assertFalse(row["task_success"])
        self.assertIn("contains off_castle", row["failed_expectations"])

    def test_exceptions_become_error_rows(self):
        class Broken:
            def new_session(self):
                raise RuntimeError("429 Too Many Requests")
        row = run_scenario(TripPlanner(Broken(), OfflinePlaces(), straight_line_minutes), BENCHMARK,
                           scenario("plan-sat-gardens"))
        self.assertIn("429", row["error"])
        self.assertFalse(row["task_success"])


class SummaryTest(unittest.TestCase):
    def test_rates_and_markdown(self):
        rows = [
            {"id": "a", "task": "plan", "tags": ["x"], "first_valid": False, "final_valid": True,
             "expectations_met": True, "task_success": True, "first_violation_types": ["closed"],
             "submissions": 2, "turns": 5, "input_tokens": 100, "output_tokens": 10, "wall_seconds": 3,
             "preference_coverage": 1.0, "status": "accepted", "failed_expectations": [], "final_violation_types": []},
            {"id": "b", "task": "replan", "tags": ["x"], "first_valid": True, "final_valid": True,
             "expectations_met": False, "task_success": False, "first_violation_types": [],
             "submissions": 1, "turns": 3, "input_tokens": 300, "output_tokens": 30, "wall_seconds": 5,
             "preference_coverage": None, "status": "accepted", "failed_expectations": ["contains off_castle"],
             "final_violation_types": []},
            {"id": "c", "task": "plan", "tags": [], "error": "boom", "first_valid": False, "final_valid": False,
             "expectations_met": False, "task_success": False},
        ]
        summary = summarize(rows)
        self.assertAlmostEqual(summary["overall"]["first_attempt_valid"], 1 / 3)
        self.assertAlmostEqual(summary["overall"]["final_valid"], 2 / 3)
        self.assertAlmostEqual(summary["by_task"]["replan"]["task_success"], 0)
        self.assertEqual(summary["first_attempt_violation_types"], {"closed": 1})
        self.assertEqual([f["id"] for f in summary["failures"]], ["b"])
        self.assertEqual(summary["errors"], ["c: boom"])
        md = render_markdown(summary)
        self.assertIn("| overall | 3 | 33% | 67% |", md)

    def test_merge_prefers_successful_rows(self):
        import json
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            rows_first = [{"id": "a", "repeat": 0, "task_success": True},
                          {"id": "b", "repeat": 0, "error": "network"}]
            rows_second = [{"id": "b", "repeat": 0, "task_success": False},
                           {"id": "a", "repeat": 0, "error": "network"}]
            for d, rows in ((first, rows_first), (second, rows_second)):
                with open(os.path.join(d, "results.jsonl"), "w", encoding="utf-8") as f:
                    f.writelines(json.dumps(r) + "\n" for r in rows)
            merged = {r["id"]: r for r in merge_results([first, second])}
        self.assertEqual(set(merged), {"a", "b"})
        self.assertNotIn("error", merged["a"])   # a later error doesn't replace a success
        self.assertNotIn("error", merged["b"])   # a later success replaces an error

    def test_run_benchmark_writes_files(self):
        p = planner([ModelTurn(text="no plan")])
        with tempfile.TemporaryDirectory() as out:
            summary = run_benchmark(p, BENCHMARK, out, workers=1, only=["plan-sat-gardens"])
            self.assertEqual(summary["overall"]["runs"], 1)
            for name in ("results.jsonl", "summary.json", "summary.md"):
                self.assertTrue(os.path.exists(os.path.join(out, name)))


if __name__ == "__main__":
    unittest.main()
