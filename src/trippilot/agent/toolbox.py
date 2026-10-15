"""Tools the agent can call, and the code that executes them.

`submit_itinerary` and `submit_recommendations` are how deterministic checks
reach the model: the agent submits, the code answers with concrete problems,
and the agent fixes them and resubmits.
"""

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol

from pydantic import ValidationError

from ..models import Itinerary, LatLng, Place, Stop, TravelMode, TripRequest, google_day, minutes
from ..telemetry import tracer
from ..validation import TravelTimeFn, Violation, validate_itinerary

START = "start"
MAX_QUERIES = 5
MAX_MATRIX_PLACES = 10
# A recommended place must stay open at least this long after the request's start time.
MIN_OPEN_MINUTES = 60


class PlacesProvider(Protocol):
    def search_text(self, query: str, near: Optional[LatLng] = None,
                    radius_m: float = 5000, max_results: int = 10) -> List[Place]: ...

    def get_place(self, place_id: str) -> Optional[Place]: ...


_STOP_SCHEMA = {
    "type": "object",
    "properties": {
        "place_id": {"type": "string", "description": "A place_id returned by search_places."},
        "name": {"type": "string"},
        "arrival": {"type": "string", "description": "Arrival time, 24-hour HH:MM."},
        "departure": {"type": "string", "description": "Departure time, 24-hour HH:MM."},
        "estimated_cost": {"type": "number", "description": "Estimated spend at this stop, in the trip's currency."},
    },
    "required": ["place_id", "name", "arrival", "departure", "estimated_cost"],
    "additionalProperties": False,
}

_SEARCH_BATCHED = {
    "name": "search_places",
    "description": "Search for places near the trip's start location. Pass every query you need in one call. "
                   "Returns place_ids you must use in the itinerary, with rating, price level (0-4) and "
                   "opening hours on the trip date.",
    "parameters": {
        "type": "object",
        "properties": {
            "queries": {"type": "array", "items": {"type": "string"},
                        "description": "Up to 5 searches, e.g. ['history museum', 'cafe for fika']."},
            "max_results": {"type": "integer", "description": "Results per query, 1-10."},
        },
        "required": ["queries", "max_results"],
        "additionalProperties": False,
    },
}

_SEARCH_LEGACY = {
    "name": "search_places",
    "description": "Search for places near the trip's start location. Returns place_ids you must use "
                   "in the itinerary, with rating, price level (0-4) and opening hours on the trip date.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for, e.g. 'history museum' or 'vegetarian lunch'."},
            "max_results": {"type": "integer", "description": "Number of results, 1-10."},
        },
        "required": ["query", "max_results"],
        "additionalProperties": False,
    },
}

_DETAILS = {
    "name": "get_place_details",
    "description": "Get details for one place_id, including opening hours on the trip date.",
    "parameters": {
        "type": "object",
        "properties": {"place_id": {"type": "string"}},
        "required": ["place_id"],
        "additionalProperties": False,
    },
}

_TRAVEL_BATCHED = {
    "name": "get_travel_times",
    "description": "Travel times in minutes, using the trip's travel mode, between every pair of the given "
                   "places and from the trip's start location ('start'). Call once with all candidates.",
    "parameters": {
        "type": "object",
        "properties": {
            "place_ids": {"type": "array", "items": {"type": "string"},
                          "description": "Up to 10 candidate place_ids."},
        },
        "required": ["place_ids"],
        "additionalProperties": False,
    },
}

_TRAVEL_LEGACY = {
    "name": "get_travel_times",
    "description": "Travel times in minutes, using the trip's travel mode, from one point to several places. "
                   "Use 'start' as from_place_id for the trip's start location.",
    "parameters": {
        "type": "object",
        "properties": {
            "from_place_id": {"type": "string"},
            "to_place_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["from_place_id", "to_place_ids"],
        "additionalProperties": False,
    },
}

_SUBMIT_ITINERARY = {
    "name": "submit_itinerary",
    "description": "Submit the complete itinerary for validation. Returns accepted=true, or the violations "
                   "to fix before resubmitting the whole itinerary.",
    "parameters": {
        "type": "object",
        "properties": {"stops": {"type": "array", "items": _STOP_SCHEMA}},
        "required": ["stops"],
        "additionalProperties": False,
    },
}

_SUBMIT_RECOMMENDATIONS = {
    "name": "submit_recommendations",
    "description": "Submit place recommendations (instead of an itinerary) when the user asks for suggestions. "
                   "Every place must come from search_places and be open for at least an hour from the start time.",
    "parameters": {
        "type": "object",
        "properties": {
            "recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "place_id": {"type": "string", "description": "A place_id returned by search_places."},
                        "reason": {"type": "string", "description": "One sentence on why it fits the user."},
                    },
                    "required": ["place_id", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["recommendations"],
        "additionalProperties": False,
    },
}


def tool_specs(batched: bool = True) -> List[Dict[str, Any]]:
    """Function-tool definitions. `batched` selects multi-query search and the travel-time matrix;
    the legacy style (one query, one origin per call) is kept for A/B benchmarks."""
    return [
        _SEARCH_BATCHED if batched else _SEARCH_LEGACY,
        _DETAILS,
        _TRAVEL_BATCHED if batched else _TRAVEL_LEGACY,
        _SUBMIT_ITINERARY,
        _SUBMIT_RECOMMENDATIONS,
    ]


TOOL_SPECS = tool_specs(batched=True)

SUBMIT_TOOLS = ("submit_itinerary", "submit_recommendations")


@dataclass
class ToolCallRecord:
    name: str
    arguments: Dict[str, Any]
    output: Dict[str, Any]
    duration_ms: float


@dataclass
class Submission:
    itinerary: Optional[Itinerary]
    violations: List[Violation] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def accepted(self) -> bool:
        return self.itinerary is not None and self.error is None and not self.violations


@dataclass
class Recommendation:
    place: Place
    reason: str


@dataclass
class RecommendationSubmission:
    recommendations: List[Recommendation]
    problems: List[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return bool(self.recommendations) and not self.problems


class ToolBox:
    """Executes tool calls for one planning session and records what happened."""

    def __init__(self, request: TripRequest, places: PlacesProvider, travel_minutes: TravelTimeFn,
                 seed_places: Optional[Dict[str, Place]] = None, visited_place_ids: Iterable[str] = (),
                 batched: bool = True):
        self.request = request
        self.batched = batched
        self.places = places
        self.travel_minutes = travel_minutes
        # Places visited earlier the same day; itineraries must not return to them.
        self.visited_place_ids = set(visited_place_ids)
        # Only places surfaced by a tool (or carried over from a previous plan) may be used.
        self.known_places: Dict[str, Place] = dict(seed_places or {})
        self.submissions: List[Submission] = []
        self.recommendation_submissions: List[RecommendationSubmission] = []
        self.trace: List[ToolCallRecord] = []

    def submission_count(self, tool_name: str) -> int:
        if tool_name == "submit_recommendations":
            return len(self.recommendation_submissions)
        return len(self.submissions)

    def execute(self, name: str, raw_arguments: str, call_id: str = "") -> Dict[str, Any]:
        """Run one tool call; errors are returned to the model instead of raised."""
        with tracer.start_as_current_span(f"execute_tool {name}") as span:
            span.set_attribute("gen_ai.operation.name", "execute_tool")
            span.set_attribute("gen_ai.tool.name", name)
            span.set_attribute("gen_ai.tool.call.id", call_id)
            started = time.perf_counter()
            arguments: Dict[str, Any] = {}
            try:
                arguments = json.loads(raw_arguments or "{}")
                handler = self._handlers().get(name)
                output = handler(**arguments) if handler else {"error": f"Unknown tool '{name}'."}
            except json.JSONDecodeError as e:
                output = {"error": f"Arguments are not valid JSON: {e}"}
            except TypeError as e:
                output = {"error": f"Invalid arguments for {name}: {e}"}
            except Exception as e:  # tool backends (HTTP etc.) must not crash the loop
                output = {"error": f"{name} failed: {e}"}
            self.trace.append(ToolCallRecord(name, arguments, output, (time.perf_counter() - started) * 1000))
            _annotate(span, name, output)
            return output

    def _handlers(self):
        return {
            "search_places": self.search_places if self.batched else self.search_place,
            "get_place_details": self.get_place_details,
            "get_travel_times": self.get_travel_times if self.batched else self.get_travel_times_from,
            "submit_itinerary": self.submit_itinerary,
            "submit_recommendations": self.submit_recommendations,
        }

    def search_places(self, queries: List[str], max_results: int = 5) -> Dict[str, Any]:
        max_results = max(1, min(int(max_results), 10))
        matches: Dict[str, List[str]] = {}
        found: Dict[str, Place] = {}
        for query in queries[:MAX_QUERIES]:
            places = self.places.search_text(query, near=self.request.start_location, max_results=max_results)
            matches[query] = [p.place_id for p in places]
            for place in places:
                found[place.place_id] = place
                self.known_places[place.place_id] = place
        # Each place is described once, however many queries matched it.
        output: Dict[str, Any] = {"places": [self._summary(p) for p in found.values()], "matches": matches}
        if len(queries) > MAX_QUERIES:
            output["note"] = f"Only the first {MAX_QUERIES} queries were run."
        return output

    def search_place(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Legacy style: one query per call."""
        return {"places": self.search_places([query], max_results)["places"]}

    def get_place_details(self, place_id: str) -> Dict[str, Any]:
        place = self.known_places.get(place_id) or self.places.get_place(place_id)
        if place is None:
            return {"error": f"No place with place_id '{place_id}'."}
        self.known_places[place.place_id] = place
        return self._summary(place)

    def get_travel_times(self, place_ids: List[str]) -> Dict[str, Any]:
        requested = list(dict.fromkeys(p for p in place_ids if p != START))
        unknown = [p for p in requested if self._location(p) is None]
        ids = [p for p in requested if p not in unknown][:MAX_MATRIX_PLACES]
        points = [START] + ids
        locations = [self._location(p) for p in points]
        mode = self.request.travel_mode
        _warm_travel_cache(self.travel_minutes, locations, mode)

        minutes: Dict[str, Dict[str, Any]] = {}
        for a, origin in zip(points, locations):
            row = {}
            for b, destination in zip(points, locations):
                if a == b or b == START:  # nobody needs to travel back to the start
                    continue
                m = self.travel_minutes(origin, destination, mode)
                # Round up: a gap of the shown minutes must always pass the validator.
                row[b] = "no route" if m is None else math.ceil(m)
            minutes[a] = row
        output: Dict[str, Any] = {"mode": mode.value, "minutes": minutes}
        if unknown:
            output["unknown_place_ids"] = unknown
        if len(requested) - len(unknown) > MAX_MATRIX_PLACES:
            output["note"] = f"Only the first {MAX_MATRIX_PLACES} places were included."
        return output

    def get_travel_times_from(self, from_place_id: str, to_place_ids: List[str]) -> Dict[str, Any]:
        """Legacy style: one origin per call."""
        origin = self._location(from_place_id)
        if origin is None:
            return {"error": f"Unknown from_place_id '{from_place_id}'. Use 'start' or a place_id from search_places."}
        mode = self.request.travel_mode
        times: Dict[str, Any] = {}
        for place_id in to_place_ids:
            destination = self._location(place_id)
            if destination is None:
                times[place_id] = "unknown place_id"
                continue
            m = self.travel_minutes(origin, destination, mode)
            times[place_id] = "no route" if m is None else math.ceil(m)
        return {"mode": mode.value, "minutes": times}

    def submit_itinerary(self, stops: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            itinerary = Itinerary(
                date=self.request.date,
                travel_mode=self.request.travel_mode,
                stops=[Stop(**s) for s in stops],
            )
        except (ValidationError, TypeError) as e:
            self.submissions.append(Submission(itinerary=None, error=str(e)))
            return {"accepted": False, "error": f"Malformed itinerary: {e}"}

        # Show the provider's name, not the model's retyping of it (which can mangle accents).
        for stop in itinerary.stops:
            known = self.known_places.get(stop.place_id)
            if known is not None:
                stop.name = known.name

        violations = validate_itinerary(itinerary, self.request, self.known_places, self.travel_minutes,
                                        already_visited=self.visited_place_ids)
        self.submissions.append(Submission(itinerary=itinerary, violations=violations))
        if not violations:
            return {"accepted": True}
        return {
            "accepted": False,
            "violations": [v.model_dump(mode="json") for v in violations],
            "instruction": "Fix every violation and submit the whole itinerary again.",
        }

    def submit_recommendations(self, recommendations: List[Dict[str, str]]) -> Dict[str, Any]:
        day = google_day(self.request.date)
        start = minutes(self.request.start_time)
        accepted, problems = [], []
        for item in recommendations:
            place = self.known_places.get(item["place_id"])
            if place is None:
                problems.append(f"place_id '{item['place_id']}' was not returned by any search. Only recommend searched places.")
            elif place.is_open_between(day, start, start + MIN_OPEN_MINUTES) is False:
                problems.append(f"{place.name} is not open for an hour from {self.request.start_time:%H:%M} "
                                f"(hours that day: {place.hours_on(day)}).")
            else:
                accepted.append(Recommendation(place=place, reason=item["reason"]))
        if not recommendations:
            problems.append("No recommendations were given.")
        self.recommendation_submissions.append(RecommendationSubmission(accepted, problems))
        if problems:
            return {"accepted": False, "problems": problems,
                    "instruction": "Replace or drop the problem places and submit the full list again."}
        return {"accepted": True}

    def _location(self, place_id: str) -> Optional[LatLng]:
        if place_id == START:
            return self.request.start_location
        place = self.known_places.get(place_id)
        return place.location if place else None

    def _summary(self, place: Place) -> Dict[str, Any]:
        return _place_summary(place, self.request)


def _place_summary(place: Place, request: TripRequest) -> Dict[str, Any]:
    return {
        "place_id": place.place_id,
        "name": place.name,
        "address": place.address,
        "types": place.types[:4],
        "rating": place.rating,
        "price_level": place.price_level,
        "hours_on_trip_date": place.hours_on(google_day(request.date)),
    }


def _warm_travel_cache(travel_minutes: TravelTimeFn, locations: List[LatLng], mode: TravelMode) -> None:
    """Fill a RoutesClient's cache with one route-matrix request instead of one per pair."""
    client = getattr(travel_minutes, "__self__", None)
    if not hasattr(client, "compute_matrix"):
        return
    # Routes API caps a matrix at 625 elements (100 for transit).
    if len(locations) ** 2 <= (100 if mode is TravelMode.TRANSIT else 625):
        client.compute_matrix(locations, locations, mode)


def _annotate(span, name: str, output: Dict[str, Any]) -> None:
    """Record the outcome of a tool call; submissions become state-change events."""
    if "error" in output:
        span.set_attribute("error.type", "tool_error")
    if name == "search_places":
        span.set_attribute("trippilot.results", len(output.get("places", [])))
    if name in SUBMIT_TOOLS:
        accepted = bool(output.get("accepted"))
        problems = [v["type"] for v in output.get("violations", [])] + ["problem"] * len(output.get("problems", []))
        span.set_attribute("trippilot.accepted", accepted)
        span.set_attribute("trippilot.violation_types", problems)
        span.add_event("submission.accepted" if accepted else "submission.rejected",
                       {"trippilot.violation_count": len(problems)})
