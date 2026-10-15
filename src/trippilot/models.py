"""Core data types shared by the tools, the validator and the agent."""

from datetime import date, time
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

MINUTES_PER_DAY = 24 * 60
MINUTES_PER_WEEK = 7 * MINUTES_PER_DAY


class TravelMode(str, Enum):
    """Travel modes, named as the Google Routes API expects them."""

    WALK = "WALK"
    DRIVE = "DRIVE"
    TRANSIT = "TRANSIT"
    BICYCLE = "BICYCLE"


class LatLng(BaseModel):
    lat: float
    lng: float


class OpeningPeriod(BaseModel):
    """One opening interval in minutes since Sunday 00:00 (Google numbers days 0 = Sunday).

    `end` may exceed MINUTES_PER_WEEK when a period wraps past Saturday midnight.
    """

    start: int = Field(ge=0, lt=MINUTES_PER_WEEK)
    end: int = Field(gt=0)


class Place(BaseModel):
    place_id: str
    name: str
    location: LatLng
    address: Optional[str] = None
    types: List[str] = Field(default_factory=list)
    rating: Optional[float] = None
    price_level: Optional[int] = Field(default=None, ge=0, le=4)
    # None means the opening hours are unknown, which is not the same as "never open".
    opening_periods: Optional[List[OpeningPeriod]] = None

    def is_open_between(self, google_day: int, start_minute: int, end_minute: int) -> Optional[bool]:
        """Whether the place is open for the whole interval on the given day.

        Returns None when opening hours are unknown.
        """
        if self.opening_periods is None:
            return None
        start = google_day * MINUTES_PER_DAY + start_minute
        end = google_day * MINUTES_PER_DAY + end_minute
        for period in self.opening_periods:
            # Check the visit both as-is and shifted one week, so periods that
            # wrap past Saturday midnight still cover early-Sunday visits.
            for shift in (0, MINUTES_PER_WEEK):
                if period.start <= start + shift and end + shift <= period.end:
                    return True
        return False

    def hours_on(self, google_day: int) -> str:
        """Human-readable opening hours for one day, e.g. '10:00-17:00'."""
        if self.opening_periods is None:
            return "unknown"
        day_start = google_day * MINUTES_PER_DAY
        spans = []
        for period in self.opening_periods:
            if period.end - period.start >= MINUTES_PER_WEEK:
                return "open 24 hours"
            if day_start <= period.start < day_start + MINUTES_PER_DAY:
                spans.append(f"{_clock(period.start)}-{_clock(period.end)}")
        return ", ".join(spans) if spans else "closed"


class TripRequest(BaseModel):
    """What the user asked for; the validator checks itineraries against it."""

    city: str
    date: date
    start_time: time
    end_time: time
    start_location: LatLng
    travel_mode: TravelMode = TravelMode.WALK
    budget: Optional[float] = Field(default=None, ge=0)
    preferences: List[str] = Field(default_factory=list)


class Stop(BaseModel):
    place_id: str
    name: str
    arrival: time
    departure: time
    estimated_cost: float = Field(default=0, ge=0)


class Itinerary(BaseModel):
    date: date
    travel_mode: TravelMode = TravelMode.WALK
    stops: List[Stop]


def google_day(d: date) -> int:
    """Convert a date to Google's day numbering (0 = Sunday)."""
    return (d.weekday() + 1) % 7


def minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def _clock(week_minute: int) -> str:
    m = week_minute % MINUTES_PER_DAY
    return f"{m // 60:02d}:{m % 60:02d}"
