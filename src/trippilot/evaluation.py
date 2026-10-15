"""Benchmark the agent on fixed scenarios over the offline dataset.

    python -m trippilot.evaluation --out eval/results/run1 --workers 2

Every score is computed by code: the validator decides feasibility and each
scenario's expectations are deterministic checks on the final itinerary.
"""

import argparse
import json
import os
import statistics
import sys
import time as clock
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any, Dict, List, Literal, Mapping, Optional

from pydantic import BaseModel, Field

from .agent.planner import PlanResult, TripPlanner
from .models import Itinerary, LatLng, Stop, TravelMode, TripRequest
from .dataset import Dataset, load as load_dataset
from .scoring import Expectations, check_expectations, preference_coverage  # noqa: F401 (re-exported)
from .telemetry import tracer


class ReplanSpec(BaseModel):
    base: str  # key into the file's base_itineraries
    now: time
    change: str


class Scenario(BaseModel):
    id: str
    task: Literal["plan", "replan"]
    tags: List[str] = Field(default_factory=list)
    city: str = "Uppsala"
    date: date
    start_time: time = time(9, 0)
    end_time: time = time(17, 0)
    travel_mode: TravelMode = TravelMode.WALK
    budget: Optional[float] = None
    preferences: List[str] = Field(default_factory=list)
    replan: Optional[ReplanSpec] = None
    expect: Expectations = Field(default_factory=Expectations)

    def request(self, start: LatLng) -> TripRequest:
        return TripRequest(
            city=self.city, date=self.date, start_time=self.start_time, end_time=self.end_time,
            start_location=start, travel_mode=self.travel_mode, budget=self.budget,
            preferences=self.preferences,
        )


@dataclass
class Benchmark:
    scenarios: List[Scenario]
    base_itineraries: Dict[str, List[Stop]]
    dataset: Dataset
    # Path of the dataset file relative to the repo (None for the synthetic sample).
    dataset_path: Optional[str] = None

    def request(self, scenario: Scenario) -> TripRequest:
        return scenario.request(self.dataset.start)

    def base_itinerary(self, scenario: Scenario) -> Itinerary:
        stops = self.base_itineraries[scenario.replan.base]
        return Itinerary(date=scenario.date, travel_mode=scenario.travel_mode, stops=stops)


def load_benchmark(path: str) -> Benchmark:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    # "dataset" is relative to the scenarios file; omitted means the synthetic sample.
    dataset_path = raw.get("dataset")
    if dataset_path:
        dataset_path = os.path.normpath(os.path.join(os.path.dirname(path), dataset_path))
    return Benchmark(
        scenarios=[Scenario(**s) for s in raw["scenarios"]],
        base_itineraries={k: [Stop(**s) for s in v] for k, v in raw.get("base_itineraries", {}).items()},
        dataset=load_dataset(dataset_path),
        dataset_path=dataset_path,
    )


# ---------------------------------------------------------------- scoring

def _first_submission_accepted(result: PlanResult) -> bool:
    for record in result.trace:
        if record.name == "submit_itinerary":
            return bool(record.output.get("accepted"))
    return False


def _first_violation_types(result: PlanResult) -> List[str]:
    for record in result.trace:
        if record.name == "submit_itinerary":
            return [v["type"] for v in record.output.get("violations", [])] or (
                ["malformed"] if "error" in record.output else [])
    return []


def score(scenario: Scenario, result: PlanResult, wall_seconds: float) -> Dict[str, Any]:
    stops = result.itinerary.stops if result.itinerary else []
    new_stops = stops[len(result.locked_stops):]
    failed = check_expectations(scenario.expect, result.itinerary, new_stops, result.places)
    final_valid = result.status == "accepted"
    return {
        "id": scenario.id,
        "task": scenario.task,
        "tags": scenario.tags,
        "status": result.status,
        "first_valid": _first_submission_accepted(result),
        "final_valid": final_valid,
        "expectations_met": not failed,
        "failed_expectations": failed,
        "task_success": final_valid and not failed,
        "preference_coverage": preference_coverage(scenario.preferences, stops, result.places),
        "first_violation_types": _first_violation_types(result),
        "final_violation_types": [v.type.value for v in result.violations],
        "submissions": result.submissions,
        "turns": result.turns,
        "hit_turn_limit": result.hit_turn_limit,
        "tool_calls": len(result.trace),
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "model_seconds": round(result.model_seconds, 1),
        "wall_seconds": round(wall_seconds, 1),
        "stops": [f"{s.arrival:%H:%M}-{s.departure:%H:%M} {s.name}" for s in stops],
    }


def run_scenario(planner: TripPlanner, benchmark: Benchmark, scenario: Scenario, repeat: int = 0) -> Dict[str, Any]:
    with tracer.start_as_current_span(f"eval_scenario {scenario.id}") as span:
        span.set_attribute("trippilot.scenario", scenario.id)
        span.set_attribute("trippilot.repeat", repeat)
        started = clock.perf_counter()
        try:
            if scenario.task == "replan":
                result = planner.replan(benchmark.request(scenario), benchmark.base_itinerary(scenario),
                                        scenario.replan.now, scenario.replan.change, benchmark.dataset.by_id())
            else:
                result = planner.plan(benchmark.request(scenario))
        except Exception as e:  # one failing run must not stop the benchmark
            span.set_attribute("error.type", type(e).__name__)
            return {"id": scenario.id, "task": scenario.task, "tags": scenario.tags, "repeat": repeat,
                    "error": f"{type(e).__name__}: {e}", "first_valid": False, "final_valid": False,
                    "expectations_met": False, "task_success": False}
        row = score(scenario, result, clock.perf_counter() - started)
        row["repeat"] = repeat
        span.set_attribute("trippilot.task_success", row["task_success"])
        return row


# ---------------------------------------------------------------- summary

def _rate(rows, key) -> float:
    return sum(1 for r in rows if r.get(key)) / len(rows) if rows else 0.0


def _mean(values) -> Optional[float]:
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if "error" not in r]

    def block(subset):
        return {
            "runs": len(subset),
            "first_attempt_valid": _rate(subset, "first_valid"),
            "final_valid": _rate(subset, "final_valid"),
            "expectations_met": _rate(subset, "expectations_met"),
            "task_success": _rate(subset, "task_success"),
            "mean_preference_coverage": _mean(r.get("preference_coverage") for r in subset),
            "mean_submissions": _mean(r.get("submissions") for r in subset),
            "mean_turns": _mean(r.get("turns") for r in subset),
            "mean_input_tokens": _mean(r.get("input_tokens") for r in subset),
            "mean_output_tokens": _mean(r.get("output_tokens") for r in subset),
            "mean_wall_seconds": _mean(r.get("wall_seconds") for r in subset),
        }

    tags = sorted({t for r in rows for t in r.get("tags", [])})
    return {
        "overall": block(rows),
        "by_task": {task: block([r for r in rows if r["task"] == task]) for task in ("plan", "replan")},
        "by_tag": {tag: block([r for r in rows if tag in r.get("tags", [])]) for tag in tags},
        "first_attempt_violation_types": dict(Counter(t for r in ok for t in r.get("first_violation_types", [])).most_common()),
        "errors": [f"{r['id']}: {r['error']}" for r in rows if "error" in r],
        "failures": [
            {"id": r["id"], "status": r.get("status"), "failed_expectations": r.get("failed_expectations"),
             "final_violation_types": r.get("final_violation_types")}
            for r in ok if not r["task_success"]
        ],
    }


def render_markdown(summary: Dict[str, Any]) -> str:
    def pct(x):
        return "—" if x is None else f"{x * 100:.0f}%"

    def num(x, digits=1):
        return "—" if x is None else f"{x:,.{digits}f}"

    lines = ["| Group | Runs | First-attempt valid | Final valid | Expectations met | Task success | Pref. coverage | Submissions | Turns | Tokens in | Tokens out | Seconds |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    groups = [("overall", summary["overall"])] + list(summary["by_task"].items()) + [
        (f"tag: {k}", v) for k, v in summary["by_tag"].items()]
    for name, b in groups:
        if not b["runs"]:
            continue
        lines.append(f"| {name} | {b['runs']} | {pct(b['first_attempt_valid'])} | {pct(b['final_valid'])} | "
                     f"{pct(b['expectations_met'])} | {pct(b['task_success'])} | {pct(b['mean_preference_coverage'])} | "
                     f"{num(b['mean_submissions'])} | {num(b['mean_turns'])} | {num(b['mean_input_tokens'], 0)} | "
                     f"{num(b['mean_output_tokens'], 0)} | {num(b['mean_wall_seconds'], 0)} |")
    lines += ["", "**First-attempt violation types:** " +
              (", ".join(f"{k} ×{v}" for k, v in summary["first_attempt_violation_types"].items()) or "none")]
    if summary["failures"]:
        lines += ["", "**Not successful:**"]
        lines += [f"- `{f['id']}` status={f['status']} expectations={f['failed_expectations']} "
                  f"violations={f['final_violation_types']}" for f in summary["failures"]]
    if summary["errors"]:
        lines += ["", "**Errors:**"] + [f"- {e}" for e in summary["errors"]]
    return "\n".join(lines)


def merge_results(run_dirs: List[str]) -> List[Dict[str, Any]]:
    """Combine results.jsonl files; for each (scenario, repeat) a later successful row wins.

    Used to rerun only the scenarios that hit infrastructure errors (network, quota).
    """
    merged: Dict[tuple, Dict[str, Any]] = {}
    for run_dir in run_dirs:
        with open(os.path.join(run_dir, "results.jsonl"), encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                key = (row["id"], row.get("repeat", 0))
                if key not in merged or "error" not in row:
                    merged[key] = row
    return list(merged.values())


def write_summary(rows: List[Dict[str, Any]], out_dir: str) -> Dict[str, Any]:
    summary = summarize(rows)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(render_markdown(summary) + "\n")
    return summary


def run_benchmark(planner: TripPlanner, benchmark: Benchmark, out_dir: str,
                  workers: int = 2, repeats: int = 1, only: Optional[List[str]] = None) -> Dict[str, Any]:
    scenarios = [s for s in benchmark.scenarios if not only or s.id in only]
    jobs = [(s, r) for r in range(repeats) for s in scenarios]
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.jsonl")
    rows = []

    def job(item):
        scenario, repeat = item
        row = run_scenario(planner, benchmark, scenario, repeat)
        flag = "ERROR" if "error" in row else ("ok  " if row["task_success"] else "FAIL")
        print(f"[{flag}] {scenario.id} (repeat {repeat})", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool, open(results_path, "w", encoding="utf-8") as f:
        for row in pool.map(job, jobs):
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return write_summary(rows, out_dir)


def main() -> None:
    from dotenv import load_dotenv

    from .agent.foundry import FoundryAgent
    from .agent.planner import instructions, tool_specs
    from .telemetry import setup_tracing

    parser = argparse.ArgumentParser(prog="trippilot.evaluation")
    parser.add_argument("--scenarios", default=os.path.join("eval", "scenarios_real.json"))
    parser.add_argument("--out", default=os.path.join("eval", "results", datetime.now().strftime("%Y%m%d-%H%M%S")))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--only", nargs="*", help="scenario ids to run")
    parser.add_argument("--tools", choices=["batched", "legacy"], default="batched",
                        help="tool style: batched (multi-query search, travel matrix) or legacy (one per call)")
    parser.add_argument("--merge", nargs="+", metavar="RUN_DIR",
                        help="don't run anything; merge these result dirs (later wins) into --out")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.merge:
        os.makedirs(args.out, exist_ok=True)
        rows = merge_results(args.merge)
        with open(os.path.join(args.out, "results.jsonl"), "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        print(render_markdown(write_summary(rows, args.out)))
        return

    load_dotenv()
    os.makedirs(args.out, exist_ok=True)
    setup_tracing(jsonl_path=os.path.join(args.out, "spans.jsonl"))

    benchmark = load_benchmark(args.scenarios)
    batched = args.tools == "batched"
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"tools": args.tools, "model": os.environ["MODEL_DEPLOYMENT_NAME"], "workers": args.workers,
                   "dataset": benchmark.dataset.name,
                   "repeats": args.repeats, "scenarios": args.scenarios, "only": args.only,
                   "started": datetime.now().isoformat(timespec="seconds")}, f, indent=2)
    agent = FoundryAgent(
        endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        instructions=instructions(batched),
        tool_specs=tool_specs(batched),
    )
    try:
        dataset = benchmark.dataset
        planner = TripPlanner(agent, dataset.provider(), dataset.travel_minutes, batched=batched)
        summary = run_benchmark(planner, benchmark, args.out, args.workers, args.repeats, args.only)
    finally:
        agent.delete()
    print("\n" + render_markdown(summary))
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
