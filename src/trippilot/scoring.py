"""Scenario scoring that only needs the data model.

Kept free of agent, Azure and telemetry imports so the Harbor verifier can run
it inside a minimal container.
"""

from datetime import time
from typing import List, Mapping, Optional

from pydantic import BaseModel, Field

from .models import Itinerary, Place, Stop, minutes


class Expectations(BaseModel):
    """Scenario-specific checks. For replans, place and type checks apply to the replanned part."""

    include_place_ids: List[str] = Field(default_factory=list)
    exclude_place_ids: List[str] = Field(default_factory=list)
    include_types: List[str] = Field(default_factory=list)   # each type must appear on some stop
    exclude_types: List[str] = Field(default_factory=list)   # no stop may have any of these types
    end_by: Optional[time] = None
    min_stops: int = 1


def locked_stops(base: List[Stop], now: time) -> List[Stop]:
    """Stops already started at `now`; a replan keeps them unchanged (see TripPlanner.replan)."""
    return [s for s in base if minutes(s.arrival) < minutes(now)]


def check_expectations(expect: Expectations, itinerary: Optional[Itinerary], new_stops: List[Stop],
                       places: Mapping[str, Place]) -> List[str]:
    """Return a description of every unmet expectation."""
    if itinerary is None:
        return ["no itinerary"]
    failed = []
    if len(itinerary.stops) < expect.min_stops:
        failed.append(f"fewer than {expect.min_stops} stops")
    new_ids = [s.place_id for s in new_stops]
    new_types = [set(places[s.place_id].types) if s.place_id in places else set() for s in new_stops]
    for pid in expect.include_place_ids:
        if pid not in new_ids:
            failed.append(f"missing {pid}")
    for pid in expect.exclude_place_ids:
        if pid in new_ids:
            failed.append(f"contains {pid}")
    for t in expect.include_types:
        if not any(t in types for types in new_types):
            failed.append(f"no stop of type {t}")
    for t in expect.exclude_types:
        if any(t in types for types in new_types):
            failed.append(f"contains a stop of type {t}")
    if expect.end_by and itinerary.stops and itinerary.stops[-1].departure > expect.end_by:
        failed.append(f"ends after {expect.end_by:%H:%M}")
    return failed


def preference_coverage(preferences: List[str], stops: List[Stop], places: Mapping[str, Place]) -> Optional[float]:
    """Share of preferences matched by the name or a type of at least one stop."""
    if not preferences:
        return None
    haystacks = []
    for s in stops:
        place = places.get(s.place_id)
        haystacks.append(" ".join([s.name.lower(), *(place.types if place else [])]))
    hits = sum(1 for p in preferences if any(p.lower() in h for h in haystacks))
    return hits / len(preferences)
