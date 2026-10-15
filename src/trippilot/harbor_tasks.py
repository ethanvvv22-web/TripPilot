"""Export the benchmark scenarios as Harbor tasks.

    python -m trippilot.harbor_tasks --out eval/harbor/tasks

Each task carries a minimal copy of the data model, validator and offline data
in its container image, and its verifier scores /app/itinerary.json with the
same deterministic checks as trippilot.evaluation.
"""

import argparse
import json
import os
import re
import shutil
from typing import Any, Dict, Optional

from .agent.planner import render_replan, render_request, split_for_replan
from .evaluation import Benchmark, Scenario, load_benchmark

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
# Modules the verifier needs; they depend only on pydantic.
VERIFIER_MODULES = ["models.py", "validation.py", "offline.py", "scoring.py", "dataset.py"]
# A recorded dataset is bundled next to the modules as trippilot/dataset.json.
DATASET_FILE = "dataset.json"
OUTPUT_PATH = "/app/itinerary.json"

DOCKERFILE = """\
FROM python:3.12-slim
RUN pip install --no-cache-dir "pydantic>=2.5,<3"
COPY trippilot /opt/trippilot/trippilot
ENV PYTHONPATH=/opt/trippilot
WORKDIR /app
"""

TASK_TOML = """\
schema_version = "1.3"

[task]
name = "trippilot/{id}"

[metadata]
category = "trip-planning"
task_kind = "{task}"
tags = {tags}

[agent]
timeout_sec = 600.0

[verifier]
timeout_sec = 120.0

[environment]
cpus = 1
memory_mb = 1024
"""

TEST_SH = """\
#!/bin/bash
mkdir -p /logs/verifier
python /tests/verify.py {output} /tests/scenario.json /logs/verifier/reward.json
"""

VERIFY_PY = '''\
"""Score the agent's itinerary with TripPilot's deterministic checks; writes reward.json."""

import json
import sys
from datetime import time

from trippilot.models import Itinerary, Stop, TripRequest
import os

import trippilot
from trippilot.dataset import load
from trippilot.scoring import Expectations, check_expectations, locked_stops, preference_coverage
from trippilot.validation import validate_itinerary

output_path, scenario_path, reward_path = sys.argv[1:4]
with open(scenario_path, encoding="utf-8") as f:
    spec = json.load(f)
request = TripRequest(**spec["request"])
bundled = os.path.join(os.path.dirname(trippilot.__file__), "dataset.json")
dataset = load(bundled if os.path.exists(bundled) else None)
places = dataset.by_id()

report = {"feasible": 0, "expectations_met": 0, "reward": 0}
try:
    with open(output_path, encoding="utf-8") as f:
        produced = json.load(f)
    itinerary = Itinerary(date=request.date, travel_mode=request.travel_mode,
                          stops=[Stop(**s) for s in produced["stops"]])
except Exception as e:  # missing or malformed output scores zero
    report["error"] = f"{type(e).__name__}: {e}"
    itinerary = None

if itinerary is not None:
    # Locked stops are recomputed here rather than trusted from the agent.
    locked = []
    if spec.get("replan"):
        base = [Stop(**s) for s in spec["replan"]["base_stops"]]
        locked = locked_stops(base, time.fromisoformat(spec["replan"]["now"]))
        # Compare what matters (place and times); names are display text.
        key = lambda s: (s.place_id, s.arrival, s.departure)  # noqa: E731
        if [key(s) for s in itinerary.stops[:len(locked)]] != [key(s) for s in locked]:
            report["error"] = "replan changed stops that had already started"
    violations = validate_itinerary(itinerary, request, places, dataset.travel_minutes)
    failed = check_expectations(Expectations(**spec["expect"]), itinerary, itinerary.stops[len(locked):], places)
    feasible = not violations and "error" not in report
    report.update({
        "feasible": int(feasible),
        "expectations_met": int(not failed),
        "reward": int(feasible and not failed),
        "violations": [v.message for v in violations],
        "failed_expectations": failed,
    })
    coverage = preference_coverage(request.preferences, itinerary.stops, places)
    if coverage is not None:
        report["preference_coverage"] = coverage

# reward.json must hold numeric metrics only; details go to a separate file.
numeric = {k: v for k, v in report.items() if isinstance(v, (int, float))}
with open(reward_path, "w", encoding="utf-8") as f:
    json.dump(numeric, f)
with open(reward_path.replace("reward.json", "details.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(json.dumps(report, indent=2, ensure_ascii=False))
'''


def parse_spec(instruction: str) -> Dict[str, Any]:
    """Read back the task spec embedded by `instruction` (its last ```json block)."""
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", instruction, flags=re.DOTALL)
    if not blocks:
        raise ValueError("instruction has no ```json task spec")
    return json.loads(blocks[-1])


REPO_ROOT = os.path.dirname(os.path.dirname(PACKAGE_DIR))


def _repo_relative(path: Optional[str]) -> Optional[str]:
    return os.path.relpath(path, REPO_ROOT).replace(os.sep, "/") if path else None


def task_spec(benchmark: Benchmark, scenario: Scenario) -> Dict[str, Any]:
    """Everything the agent and verifier need, as plain JSON."""
    spec: Dict[str, Any] = {
        "scenario_id": scenario.id,
        "task": scenario.task,
        "request": json.loads(benchmark.request(scenario).model_dump_json()),
        # Repo-relative path the agent loads; the verifier uses its bundled copy.
        "dataset": _repo_relative(benchmark.dataset_path),
        "expect": json.loads(scenario.expect.model_dump_json()),
    }
    if scenario.replan:
        base = benchmark.base_itinerary(scenario)
        spec["replan"] = {
            "base_stops": [json.loads(s.model_dump_json()) for s in base.stops],
            "now": scenario.replan.now.isoformat(timespec="minutes"),
            "change": scenario.replan.change,
        }
    return spec


def instruction(benchmark: Benchmark, scenario: Scenario, spec: Dict[str, Any]) -> str:
    request = benchmark.request(scenario)
    if scenario.replan:
        locked, remaining, rest = split_for_replan(request, benchmark.base_itinerary(scenario), scenario.replan.now,
                                                   benchmark.dataset.by_id())
        text = render_replan(rest, locked, remaining, scenario.replan.change)
    else:
        text = render_request(request)
    return (
        f"{text}\n\n"
        f"Write the final itinerary to `{OUTPUT_PATH}` as JSON: "
        '`{"stops": [{"place_id", "name", "arrival", "departure", "estimated_cost"}, ...]}`. '
        "For a replan, include the stops that had already started, unchanged, first.\n\n"
        "Machine-readable task spec:\n\n"
        # ASCII-only so the spec survives however the harness decodes the file (Harbor on a
        # Chinese Windows locale read it as GBK, turning "Café" into mojibake).
        f"```json\n{json.dumps(spec, indent=2, ensure_ascii=True)}\n```\n"
    )


def write_task(benchmark: Benchmark, scenario: Scenario, out_dir: str) -> str:
    task_dir = os.path.join(out_dir, scenario.id)
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)
    package_dir = os.path.join(task_dir, "environment", "trippilot")
    os.makedirs(package_dir)
    os.makedirs(os.path.join(task_dir, "tests"))
    spec = task_spec(benchmark, scenario)

    files = {
        "task.toml": TASK_TOML.format(id=scenario.id, task=scenario.task, tags=json.dumps(scenario.tags)),
        "instruction.md": instruction(benchmark, scenario, spec),
        os.path.join("environment", "Dockerfile"): DOCKERFILE,
        os.path.join("environment", "trippilot", "__init__.py"): "",
        os.path.join("tests", "test.sh"): TEST_SH.format(output=OUTPUT_PATH),
        os.path.join("tests", "verify.py"): VERIFY_PY,
        os.path.join("tests", "scenario.json"): json.dumps(spec, indent=2, ensure_ascii=False) + "\n",
    }
    for name, content in files.items():
        # newline="\n": the scripts run in a Linux container.
        with open(os.path.join(task_dir, name), "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
    for module in VERIFIER_MODULES:
        shutil.copy(os.path.join(PACKAGE_DIR, module), package_dir)
    if benchmark.dataset_path:
        shutil.copy(benchmark.dataset_path, os.path.join(package_dir, DATASET_FILE))
    return task_dir


def main() -> None:
    parser = argparse.ArgumentParser(prog="trippilot.harbor_tasks")
    parser.add_argument("--scenarios", default=os.path.join("eval", "scenarios_real.json"))
    parser.add_argument("--out", default=os.path.join("eval", "harbor", "tasks"))
    parser.add_argument("--only", nargs="*", help="scenario ids to export")
    args = parser.parse_args()

    benchmark = load_benchmark(args.scenarios)
    scenarios = [s for s in benchmark.scenarios if not args.only or s.id in args.only]
    for scenario in scenarios:
        write_task(benchmark, scenario, args.out)
    print(f"Wrote {len(scenarios)} Harbor tasks to {args.out}")


if __name__ == "__main__":
    main()
