"""Drives agent runs: the tool-call loop plus validate-and-repair.

Three tasks share one loop: planning a day, replanning the rest of a day
after something changes, and recommending places.
"""

import json
import time as clock
from dataclasses import dataclass, field
from datetime import time
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from ..models import Itinerary, Place, Stop, TripRequest, minutes
from ..telemetry import tracer
from ..validation import TravelTimeFn, Violation
from .backend import AgentBackend, ModelTurn
from .toolbox import SUBMIT_TOOLS, TOOL_SPECS, PlacesProvider, Recommendation, ToolBox, ToolCallRecord, tool_specs

_INSTRUCTIONS = """\
You are TripPilot, a trip-planning agent. The user message says which task to do.

Plan a day:
{plan_steps}
3. Build the itinerary: stops in visiting order, arrival and departure as 24-hour HH:MM,
   leaving enough time to travel between stops and staying inside each place's opening hours.
4. Call submit_itinerary with the complete itinerary.
5. If it is not accepted, fix every reported violation and submit the whole itinerary again.

Replan the rest of a day:
- You get the remaining stops of an existing itinerary and what changed.
- Change as little as the user's change allows: keep remaining stops that still fit, adjust times,
  and replace only what must go. Remaining stops' place_ids can be used without searching again.
- Submit only the new remaining part with submit_itinerary, starting from the given time and location.

Recommend places:
- {recommend_search}, then call submit_recommendations with 3-5 places and a one-sentence reason each.
- If it is not accepted, replace the problem places and submit the full list again.

Rules:
{batch_rule}- Only use place_ids returned by search_places or listed in the request. Never invent places.
- Keep the total estimated cost within the budget when one is given.
- After a submission is accepted, reply with a short summary for the user.
"""


def instructions(batched: bool = True) -> str:
    """Agent instructions matching `tool_specs(batched)`."""
    if batched:
        return _INSTRUCTIONS.format(
            plan_steps="1. Use search_places to find candidates that match the user's preferences; put all queries in one call.\n"
                       "2. Check opening hours on the trip date and call get_travel_times once with all candidate place_ids\n"
                       "   to get the travel time between every pair.",
            recommend_search="Search (all queries in one call)",
            batch_rule="- Make independent tool calls in the same turn rather than one call per turn.\n",
        )
    return _INSTRUCTIONS.format(
        plan_steps="1. Use search_places to find candidates that match the user's preferences.\n"
                   "2. Check opening hours on the trip date and use get_travel_times between consecutive stops.",
        recommend_search="Search",
        batch_rule="",
    )


INSTRUCTIONS = instructions(batched=True)

MAX_TURNS = 15
MAX_SUBMISSIONS = 4


@dataclass
class PlanResult:
    status: str  # "accepted" | "rejected" | "no_submission"
    itinerary: Optional[Itinerary]
    violations: List[Violation]
    submissions: int
    turns: int
    hit_turn_limit: bool
    summary: str
    trace: List[ToolCallRecord] = field(default_factory=list)
    # Every place a tool surfaced during the run, keyed by place_id.
    places: Dict[str, Place] = field(default_factory=dict)
    # For replans: stops before the change, kept unchanged at the start of `itinerary`.
    locked_stops: List[Stop] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model_seconds: float = 0.0


@dataclass
class RecommendResult:
    status: str  # "accepted" | "rejected" | "no_submission"
    recommendations: List[Recommendation]
    problems: List[str]
    submissions: int
    turns: int
    hit_turn_limit: bool
    summary: str
    trace: List[ToolCallRecord] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model_seconds: float = 0.0


@dataclass
class RunStats:
    last_turn: ModelTurn
    turns: int
    hit_turn_limit: bool
    input_tokens: int
    output_tokens: int
    model_seconds: float


class TripPlanner:
    def __init__(
        self,
        backend: AgentBackend,
        places: PlacesProvider,
        travel_minutes: TravelTimeFn,
        max_turns: int = MAX_TURNS,
        max_submissions: int = MAX_SUBMISSIONS,
        on_tool_call: Optional[Callable[[ToolCallRecord], None]] = None,
        batched: bool = True,
    ):
        self.backend = backend
        # Must match the tool_specs/instructions the backend's agent was created with.
        self.batched = batched
        self.places = places
        self.travel_minutes = travel_minutes
        self.max_turns = max_turns
        self.max_submissions = max_submissions
        # Called after each executed tool call, e.g. to stream progress to a UI.
        self.on_tool_call = on_tool_call

    def plan(self, request: TripRequest) -> PlanResult:
        toolbox = ToolBox(request, self.places, self.travel_minutes, batched=self.batched)
        stats = self._run("plan", toolbox, render_request(request))
        return _plan_result(toolbox, stats)

    def replan(
        self,
        request: TripRequest,
        itinerary: Itinerary,
        now: time,
        change: str,
        known_places: Mapping[str, Place],
    ) -> PlanResult:
        """Replan the part of `itinerary` not yet started at `now`.

        Stops already started stay as they are; the agent plans from the end of
        the last started stop, at its location, with the budget that is left.
        """
        locked, remaining, rest_request = split_for_replan(request, itinerary, now, known_places)

        # Visited places stay known so a revisit is reported as a duplicate, not an unknown place.
        seed = {s.place_id: known_places[s.place_id] for s in locked + remaining if s.place_id in known_places}
        toolbox = ToolBox(rest_request, self.places, self.travel_minutes, seed_places=seed,
                          visited_place_ids=[s.place_id for s in locked], batched=self.batched)
        stats = self._run("replan", toolbox, render_replan(rest_request, locked, remaining, change),
                          {"trippilot.locked_stops": len(locked), "trippilot.remaining_stops": len(remaining)})

        result = _plan_result(toolbox, stats)
        if result.itinerary is not None:
            result.itinerary = result.itinerary.model_copy(update={"stops": locked + result.itinerary.stops})
        result.places = {**known_places, **result.places}
        result.locked_stops = locked
        return result

    def recommend(self, request: TripRequest, query: str) -> RecommendResult:
        toolbox = ToolBox(request, self.places, self.travel_minutes, batched=self.batched)
        stats = self._run("recommend", toolbox, render_recommend(request, query))
        last = toolbox.recommendation_submissions[-1] if toolbox.recommendation_submissions else None
        return RecommendResult(
            status=_status(last),
            recommendations=last.recommendations if last else [],
            problems=last.problems if last else [],
            submissions=len(toolbox.recommendation_submissions),
            turns=stats.turns,
            hit_turn_limit=stats.hit_turn_limit,
            summary=stats.last_turn.text,
            trace=toolbox.trace,
            input_tokens=stats.input_tokens,
            output_tokens=stats.output_tokens,
            model_seconds=stats.model_seconds,
        )

    def _run(self, task: str, toolbox: ToolBox, message: str, attributes: Optional[dict] = None) -> RunStats:
        request = toolbox.request
        with tracer.start_as_current_span(f"invoke_agent trippilot.{task}") as run_span:
            run_span.set_attribute("gen_ai.operation.name", "invoke_agent")
            run_span.set_attribute("trippilot.task", task)
            run_span.set_attribute("trippilot.city", request.city)
            run_span.set_attribute("trippilot.travel_mode", request.travel_mode.value)
            for key, value in (attributes or {}).items():
                run_span.set_attribute(key, value)

            usage = {"input": 0, "output": 0, "seconds": 0.0}

            def model_call(send, payload, turn_number: int) -> ModelTurn:
                with tracer.start_as_current_span("chat") as span:
                    span.set_attribute("gen_ai.operation.name", "chat")
                    span.set_attribute("trippilot.turn", turn_number)
                    started = clock.perf_counter()
                    turn = send(payload)
                    usage["seconds"] += clock.perf_counter() - started
                    usage["input"] += turn.input_tokens
                    usage["output"] += turn.output_tokens
                    span.set_attribute("gen_ai.usage.input_tokens", turn.input_tokens)
                    span.set_attribute("gen_ai.usage.output_tokens", turn.output_tokens)
                    span.set_attribute("trippilot.tool_calls", len(turn.tool_calls))
                    return turn

            session = self.backend.new_session()
            hit_turn_limit = False
            try:
                turn = model_call(session.send_user_message, message, 1)
                turns = 1
                while turn.tool_calls:
                    if turns >= self.max_turns:
                        hit_turn_limit = True
                        break
                    results = []
                    for call in turn.tool_calls:
                        if call.name in SUBMIT_TOOLS and toolbox.submission_count(call.name) >= self.max_submissions:
                            output = {"accepted": False, "error": "Submission limit reached; stop and explain the conflicts to the user."}
                        else:
                            output = toolbox.execute(call.name, call.arguments, call.call_id)
                            if self.on_tool_call:
                                self.on_tool_call(toolbox.trace[-1])
                        results.append((call.call_id, json.dumps(output)))
                    turns += 1
                    turn = model_call(session.send_tool_results, results, turns)
            finally:
                session.close()

            submit_tool = "submit_recommendations" if task == "recommend" else "submit_itinerary"
            run_span.set_attribute("trippilot.turns", turns)
            run_span.set_attribute("trippilot.hit_turn_limit", hit_turn_limit)
            run_span.set_attribute("trippilot.submissions", toolbox.submission_count(submit_tool))
            run_span.set_attribute("gen_ai.usage.input_tokens", usage["input"])
            run_span.set_attribute("gen_ai.usage.output_tokens", usage["output"])
            return RunStats(turn, turns, hit_turn_limit, usage["input"], usage["output"], usage["seconds"])


def split_for_replan(
    request: TripRequest,
    itinerary: Itinerary,
    now: time,
    known_places: Mapping[str, Place],
) -> Tuple[List[Stop], List[Stop], TripRequest]:
    """Split a plan at `now` into (locked, remaining) stops and the request for the rest of the day.

    The rest of the day starts when the last started stop ends, at its location,
    with whatever budget those stops left.
    """
    locked = [s for s in itinerary.stops if minutes(s.arrival) < minutes(now)]
    remaining = [s for s in itinerary.stops if minutes(s.arrival) >= minutes(now)]
    resume_at = max(now, locked[-1].departure) if locked else now
    if resume_at >= request.end_time:
        raise ValueError("Nothing left to replan: the day is already over at that time.")

    location = request.start_location
    if locked and locked[-1].place_id in known_places:
        location = known_places[locked[-1].place_id].location
    spent = sum(s.estimated_cost for s in locked)
    rest_request = request.model_copy(update={
        "start_time": resume_at,
        "start_location": location,
        "budget": None if request.budget is None else max(request.budget - spent, 0),
    })
    return locked, remaining, rest_request


def _status(last_submission) -> str:
    # The last submission is the agent's final answer; the deterministic checks,
    # not the model, decide whether it is accepted.
    if last_submission is None:
        return "no_submission"
    return "accepted" if last_submission.accepted else "rejected"


def _plan_result(toolbox: ToolBox, stats: RunStats) -> PlanResult:
    last = toolbox.submissions[-1] if toolbox.submissions else None
    return PlanResult(
        status=_status(last),
        itinerary=last.itinerary if last else None,
        violations=last.violations if last else [],
        submissions=len(toolbox.submissions),
        turns=stats.turns,
        hit_turn_limit=stats.hit_turn_limit,
        summary=stats.last_turn.text,
        trace=toolbox.trace,
        places=toolbox.known_places,
        input_tokens=stats.input_tokens,
        output_tokens=stats.output_tokens,
        model_seconds=stats.model_seconds,
    )


def _constraints(request: TripRequest, budget_scope: str = "in total") -> List[str]:
    lines = [f"Travel mode: {request.travel_mode.value.lower()}."]
    if request.budget is not None:
        lines.append(f"Budget: {request.budget:g} {budget_scope}.")
    if request.preferences:
        lines.append("Preferences: " + ", ".join(request.preferences) + ".")
    return lines


def render_request(request: TripRequest) -> str:
    return "\n".join([
        f"Task: plan a day in {request.city} on {request.date:%A %Y-%m-%d}.",
        f"Time window: {request.start_time:%H:%M}-{request.end_time:%H:%M}, starting from the start location.",
        *_constraints(request),
    ])


def render_replan(request: TripRequest, locked: List[Stop], remaining: List[Stop], change: str) -> str:
    where = "the last visited stop" if locked else "the start location"
    lines = [
        f"Task: replan the rest of the day in {request.city} on {request.date:%A %Y-%m-%d}.",
        f"Resume at {request.start_time:%H:%M} from {where}; the day ends at {request.end_time:%H:%M}.",
        f"What changed: {change}",
    ]
    if locked:
        lines.append("Already visited today (do not revisit): " + ", ".join(s.name for s in locked) + ".")
    lines.append("Remaining stops in the current plan:" if remaining else "There are no remaining stops in the current plan.")
    lines += [f"- {s.arrival:%H:%M}-{s.departure:%H:%M} {s.name} (place_id {s.place_id}, cost {s.estimated_cost:g})"
              for s in remaining]
    lines += _constraints(request, budget_scope="left for the rest of the day")
    return "\n".join(lines)


def render_recommend(request: TripRequest, query: str) -> str:
    return "\n".join([
        f"Task: recommend places in {request.city} on {request.date:%A %Y-%m-%d}, from {request.start_time:%H:%M}.",
        f"The user asks: {query}",
        *_constraints(request),
    ])


__all__ = ["INSTRUCTIONS", "TOOL_SPECS", "instructions", "tool_specs", "PlanResult", "RecommendResult", "TripPlanner",
           "render_recommend", "render_replan", "render_request"]
