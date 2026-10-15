"""Spans emitted during an agent run."""

import json
import os
import tempfile
import unittest

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from trippilot.agent import ModelTurn, TripPlanner
from trippilot.offline import OfflinePlaces
from trippilot.telemetry import JsonLinesSpanExporter

from .fixtures import PLACES, fixed_travel, request
from .test_planner import ScriptedBackend, stop, tool

EXPORTER = InMemorySpanExporter()


def setUpModule():
    # The global provider can only be set once per process.
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
    trace.set_tracer_provider(provider)


class TracingTest(unittest.TestCase):
    def setUp(self):
        EXPORTER.clear()
        backend = ScriptedBackend([
            ModelTurn(tool_calls=[tool("search_places", queries=["museum"], max_results=3)], input_tokens=100, output_tokens=20),
            ModelTurn(tool_calls=[tool("submit_itinerary", stops=[stop("p_museum", "Gustavianum", "15:30", "16:30")])],
                      input_tokens=150, output_tokens=40),
            ModelTurn(tool_calls=[tool("submit_itinerary", stops=[stop("p_museum", "Gustavianum", "14:30", "15:30")])],
                      input_tokens=200, output_tokens=40),
            ModelTurn(text="Done.", input_tokens=250, output_tokens=30),
        ])
        self.result = TripPlanner(backend, OfflinePlaces(PLACES.values()), fixed_travel(10)).plan(request())
        self.spans = EXPORTER.get_finished_spans()

    def by_name(self, prefix):
        return [s for s in self.spans if s.name.startswith(prefix)]

    def test_span_tree(self):
        run = self.by_name("invoke_agent")[0]
        self.assertEqual(run.name, "invoke_agent trippilot.plan")
        self.assertEqual(len(self.by_name("chat")), 4)
        self.assertEqual([s.name for s in self.by_name("execute_tool")],
                         ["execute_tool search_places", "execute_tool submit_itinerary", "execute_tool submit_itinerary"])
        for span in self.by_name("chat") + self.by_name("execute_tool"):
            self.assertEqual(span.parent.span_id, run.context.span_id)

    def test_token_usage_is_summed(self):
        run = self.by_name("invoke_agent")[0]
        self.assertEqual(run.attributes["gen_ai.usage.input_tokens"], 700)
        self.assertEqual(run.attributes["gen_ai.usage.output_tokens"], 130)
        self.assertEqual((self.result.input_tokens, self.result.output_tokens), (700, 130))

    def test_submissions_are_state_change_events(self):
        first, second = self.by_name("execute_tool submit_itinerary")
        self.assertFalse(first.attributes["trippilot.accepted"])
        self.assertEqual(list(first.attributes["trippilot.violation_types"]), ["closed"])
        self.assertEqual(first.events[0].name, "submission.rejected")
        self.assertTrue(second.attributes["trippilot.accepted"])
        self.assertEqual(second.events[0].name, "submission.accepted")


class JsonLinesExporterTest(unittest.TestCase):
    def test_writes_one_json_object_per_span(self):
        EXPORTER.clear()
        backend = ScriptedBackend([ModelTurn(text="hi")])
        TripPlanner(backend, OfflinePlaces(PLACES.values()), fixed_travel(10)).plan(request())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "spans.jsonl")
            JsonLinesSpanExporter(path).export(EXPORTER.get_finished_spans())
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
        self.assertEqual({r["name"] for r in rows}, {"chat", "invoke_agent trippilot.plan"})


if __name__ == "__main__":
    unittest.main()
