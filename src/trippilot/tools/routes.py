"""Google Routes API client: travel times between places."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from pydantic import BaseModel

from ..models import LatLng, TravelMode

ROUTE_MATRIX_URL = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
MATRIX_FIELDS = "originIndex,destinationIndex,duration,distanceMeters,status,condition"


class RouteLeg(BaseModel):
    duration_minutes: float
    distance_meters: int


def parse_route_matrix(elements: List[Dict[str, Any]]) -> Dict[Tuple[int, int], RouteLeg]:
    """Map (origin index, destination index) to a leg; pairs without a route are omitted."""
    legs = {}
    for element in elements:
        if element.get("condition") != "ROUTE_EXISTS":
            continue
        seconds = float(element.get("duration", "0s").rstrip("s"))
        legs[(element["originIndex"], element["destinationIndex"])] = RouteLeg(
            duration_minutes=seconds / 60,
            distance_meters=element.get("distanceMeters", 0),
        )
    return legs


class RoutesClient:
    def __init__(self, api_key: str, session: Optional[requests.Session] = None, timeout: float = 15):
        self.api_key = api_key
        self.session = session or requests.Session()
        self.timeout = timeout
        self._cache: Dict[Tuple, Optional[float]] = {}

    def compute_matrix(
        self,
        origins: Sequence[LatLng],
        destinations: Sequence[LatLng],
        mode: TravelMode,
        departure_time: Optional[datetime] = None,
    ) -> Dict[Tuple[int, int], RouteLeg]:
        body: Dict[str, Any] = {
            "origins": [{"waypoint": _waypoint(o)} for o in origins],
            "destinations": [{"waypoint": _waypoint(d)} for d in destinations],
            "travelMode": mode.value,
        }
        if departure_time is not None:
            # Transit times depend on the timetable; other modes default to "now".
            body["departureTime"] = departure_time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        response = self.session.post(
            ROUTE_MATRIX_URL,
            json=body,
            headers={"X-Goog-Api-Key": self.api_key, "X-Goog-FieldMask": MATRIX_FIELDS},
            timeout=self.timeout,
        )
        response.raise_for_status()
        legs = parse_route_matrix(response.json())
        for i, origin in enumerate(origins):
            for j, destination in enumerate(destinations):
                leg = legs.get((i, j))
                self._cache[_key(origin, destination, mode)] = leg.duration_minutes if leg else None
        return legs

    def travel_minutes(self, origin: LatLng, destination: LatLng, mode: TravelMode) -> Optional[float]:
        """Travel time for one pair; matches validation.TravelTimeFn."""
        key = _key(origin, destination, mode)
        if key not in self._cache:
            self.compute_matrix([origin], [destination], mode)
        return self._cache[key]


def _waypoint(point: LatLng) -> Dict[str, Any]:
    return {"location": {"latLng": {"latitude": point.lat, "longitude": point.lng}}}


def _key(origin: LatLng, destination: LatLng, mode: TravelMode) -> Tuple:
    # ~1 m precision: nearby duplicate coordinates share a cache entry.
    return (round(origin.lat, 5), round(origin.lng, 5),
            round(destination.lat, 5), round(destination.lng, 5), mode)
