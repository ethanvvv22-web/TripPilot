"""Deterministic itinerary validation.

The LLM proposes itineraries; this module decides whether they are feasible.
Every check is plain code so results are reproducible and explainable, and
each violation message is written so the agent can act on it when repairing.
"""

import math
from enum import Enum
from typing import Callable, Collection, List, Mapping, Optional

from pydantic import BaseModel

from .models import Itinerary, LatLng, Place, TravelMode, TripRequest, google_day, minutes

# (origin, destination, mode) -> travel time in minutes, or None if no route exists.
TravelTimeFn = Callable[[LatLng, LatLng, TravelMode], Optional[float]]


class ViolationType(str, Enum):
    UNKNOWN_PLACE = "unknown_place"
    DUPLICATE_PLACE = "duplicate_place"
    INVALID_TIMES = "invalid_times"
    OUTSIDE_TRIP_WINDOW = "outside_trip_window"
    OVERLAP = "overlap"
    NO_ROUTE = "no_route"
    INSUFFICIENT_TRAVEL_TIME = "insufficient_travel_time"
    CLOSED = "closed"
    OVER_BUDGET = "over_budget"


class Violation(BaseModel):
    type: ViolationType
    message: str
    stop_index: Optional[int] = None


def validate_itinerary(
    itinerary: Itinerary,
    request: TripRequest,
    places: Mapping[str, Place],
    travel_minutes: TravelTimeFn,
    already_visited: Collection[str] = (),
) -> List[Violation]:
    """Return every rule the itinerary breaks; an empty list means it is feasible.

    `already_visited` holds place_ids visited earlier the same day (e.g. before a
    replan); the itinerary must not return to them.
    """
    violations: List[Violation] = []
    seen = set(already_visited)
    day = google_day(itinerary.date)
    stops = itinerary.stops

    if itinerary.date != request.date:
        violations.append(Violation(
            type=ViolationType.OUTSIDE_TRIP_WINDOW,
            message=f"Itinerary is for {itinerary.date}, but the trip is on {request.date}.",
        ))

    if stops:
        if stops[0].arrival < request.start_time:
            violations.append(Violation(
                type=ViolationType.OUTSIDE_TRIP_WINDOW, stop_index=0,
                message=f"First stop arrives at {stops[0].arrival:%H:%M}, before the trip starts at {request.start_time:%H:%M}.",
            ))
        if stops[-1].departure > request.end_time:
            violations.append(Violation(
                type=ViolationType.OUTSIDE_TRIP_WINDOW, stop_index=len(stops) - 1,
                message=f"Last stop ends at {stops[-1].departure:%H:%M}, after the trip ends at {request.end_time:%H:%M}.",
            ))

    # The trip starts at the user's location at start_time.
    prev_location: Optional[LatLng] = request.start_location
    prev_departure = minutes(request.start_time)
    prev_label = "the start location"

    for i, stop in enumerate(stops):
        label = f"Stop {i + 1} ({stop.name})"
        arrival, departure = minutes(stop.arrival), minutes(stop.departure)

        if departure <= arrival:
            violations.append(Violation(
                type=ViolationType.INVALID_TIMES, stop_index=i,
                message=f"{label} departs at {stop.departure:%H:%M}, not after its arrival at {stop.arrival:%H:%M}.",
            ))

        if stop.place_id in seen:
            earlier = "was already visited earlier today" if stop.place_id in already_visited else "appears earlier in the itinerary"
            violations.append(Violation(
                type=ViolationType.DUPLICATE_PLACE, stop_index=i,
                message=f"{label} {earlier}. Choose a different place.",
            ))
        seen.add(stop.place_id)

        place = places.get(stop.place_id)
        if place is None:
            violations.append(Violation(
                type=ViolationType.UNKNOWN_PLACE, stop_index=i,
                message=f"{label} has place_id '{stop.place_id}', which was not returned by any place search. Only use places from tool results.",
            ))
        elif departure > arrival and place.is_open_between(day, arrival, departure) is False:
            violations.append(Violation(
                type=ViolationType.CLOSED, stop_index=i,
                message=f"{label} is not open for the whole visit {stop.arrival:%H:%M}-{stop.departure:%H:%M} (hours that day: {place.hours_on(day)}).",
            ))

        gap = arrival - prev_departure
        if gap < 0:
            # For the first stop this is already reported as OUTSIDE_TRIP_WINDOW.
            if i > 0:
                violations.append(Violation(
                    type=ViolationType.OVERLAP, stop_index=i,
                    message=f"{label} arrives at {stop.arrival:%H:%M}, before leaving {prev_label}.",
                ))
        elif place is not None and prev_location is not None:
            needed = travel_minutes(prev_location, place.location, itinerary.travel_mode)
            if needed is None:
                violations.append(Violation(
                    type=ViolationType.NO_ROUTE, stop_index=i,
                    message=f"No {itinerary.travel_mode.value.lower()} route from {prev_label} to {label}.",
                ))
            elif gap < needed:
                violations.append(Violation(
                    type=ViolationType.INSUFFICIENT_TRAVEL_TIME, stop_index=i,
                    message=f"{label}: only {gap} min after leaving {prev_label}, but travel needs at least {math.ceil(needed)} min by {itinerary.travel_mode.value.lower()}.",
                ))

        prev_location = place.location if place is not None else None
        prev_departure = departure
        prev_label = label

    if request.budget is not None:
        total = sum(stop.estimated_cost for stop in stops)
        if total > request.budget:
            violations.append(Violation(
                type=ViolationType.OVER_BUDGET,
                message=f"Estimated total cost {total:g} exceeds the budget of {request.budget:g}.",
            ))

    return violations
