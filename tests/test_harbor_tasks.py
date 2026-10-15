"""Harbor task export: structure, and the verifier script run on the host."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

from trippilot.harbor_tasks import parse_spec, write_task

from .test_evaluation import BENCHMARK, scenario

GOOD_REPLAN = [
    {"place_id": "off_cathedral", "name": "Uppsala Cathedral", "arrival": "09:30", "departure": "10:30", "estimated_cost": 0},
    {"place_id": "off_gustavianum", "name": "Gustavianum", "arrival": "10:45", "departure": "11:45", "estimated_cost": 100},
    {"place_id": "off_cafe", "name": "Café Linné", "arrival": "12:00", "departure": "12:45", "estimated_cost": 80},
    {"place_id": "off_castle", "name": "Uppsala Castle", "arrival": "13:15", "departure": "14:15", "estimated_cost": 90},
    {"place_id": "off_carolina", "name": "Carolina Rediviva", "arrival": "14:45", "departure": "16:00", "estimated_cost": 0},
]


class HarborTaskTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.task_dir = write_task(BENCHMARK, scenario("replan-rain-1100"), self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def verify(self, stops):
        """Run tests/verify.py the way test.sh does, against the task's bundled package."""
        output = os.path.join(self.tmp.name, "itinerary.json")
        reward = os.path.join(self.tmp.name, "reward.json")
        with open(output, "w", encoding="utf-8") as f:
            json.dump({"stops": stops}, f)
        env = {**os.environ, "PYTHONPATH": os.path.join(self.task_dir, "environment"), "PYTHONIOENCODING": "utf-8"}
        subprocess.run([sys.executable, os.path.join(self.task_dir, "tests", "verify.py"), output,
                        os.path.join(self.task_dir, "tests", "scenario.json"), reward],
                       check=True, env=env, capture_output=True)
        with open(reward, encoding="utf-8") as f:
            return json.load(f)

    def test_layout(self):
        for path in ("task.toml", "instruction.md", "environment/Dockerfile", "environment/trippilot/validation.py",
                     "tests/test.sh", "tests/verify.py", "tests/scenario.json"):
            self.assertTrue(os.path.exists(os.path.join(self.task_dir, path)), path)
        with open(os.path.join(self.task_dir, "tests", "test.sh"), "rb") as f:
            self.assertNotIn(b"\r\n", f.read())  # runs in a Linux container

    def test_instruction_carries_spec_and_resume_point(self):
        with open(os.path.join(self.task_dir, "instruction.md"), encoding="utf-8") as f:
            text = f.read()
        self.assertIn("Resume at 11:45", text)
        self.assertEqual(parse_spec(text)["scenario_id"], "replan-rain-1100")

    def test_spec_survives_a_legacy_code_page(self):
        # Harbor on a Chinese Windows locale read instruction.md as GBK, mangling "Café".
        with open(os.path.join(self.task_dir, "instruction.md"), "rb") as f:
            text = f.read().decode("gbk", errors="replace")
        names = [s["name"] for s in parse_spec(text)["replan"]["base_stops"]]
        self.assertIn("Café Linné", names)

    def test_locked_stop_names_do_not_matter(self):
        stops = [dict(s, name="renamed") for s in GOOD_REPLAN]
        self.assertEqual(self.verify(stops)["reward"], 1)

    def test_good_replan_scores_one(self):
        self.assertEqual(self.verify(GOOD_REPLAN)["reward"], 1)

    def test_outdoor_stop_after_rain_scores_zero(self):
        stops = GOOD_REPLAN[:-1] + [{"place_id": "off_botanical", "name": "Uppsala Botanical Garden",
                                     "arrival": "14:45", "departure": "16:00", "estimated_cost": 0}]
        result = self.verify(stops)
        self.assertEqual((result["feasible"], result["expectations_met"], result["reward"]), (1, 0, 0))

    def test_changing_started_stops_scores_zero(self):
        stops = [dict(GOOD_REPLAN[0], arrival="09:45")] + GOOD_REPLAN[1:]
        self.assertEqual(self.verify(stops)["feasible"], 0)

    def test_missing_output_scores_zero(self):
        reward = os.path.join(self.tmp.name, "reward.json")
        env = {**os.environ, "PYTHONPATH": os.path.join(self.task_dir, "environment")}
        subprocess.run([sys.executable, os.path.join(self.task_dir, "tests", "verify.py"), "/nonexistent.json",
                        os.path.join(self.task_dir, "tests", "scenario.json"), reward],
                       check=True, env=env, capture_output=True)
        with open(reward, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["reward"], 0)


if __name__ == "__main__":
    unittest.main()
