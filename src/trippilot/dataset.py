"""Place datasets the agent can run against without live APIs.

Two kinds: the small hand-written Uppsala sample in `offline.py` (synthetic
hours and straight-line travel times) and recorded snapshots of real Google
data made by `snapshot.py`.
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .models import LatLng, Place, TravelMode
from .offline import UPPSALA_PLACES, UPPSALA_START, OfflinePlaces, straight_line_minutes
from .validation import TravelTimeFn


@dataclass
class Dataset:
    name: str
    start: LatLng
    places: List[Place]
    travel_minutes: TravelTimeFn

    def provider(self) -> OfflinePlaces:
        return OfflinePlaces(self.places)

    def by_id(self) -> Dict[str, Place]:
        return {p.place_id: p for p in self.places}


def synthetic() -> Dataset:
    return Dataset("synthetic-uppsala", UPPSALA_START, list(UPPSALA_PLACES), straight_line_minutes)


def load_snapshot(path: str) -> Dataset:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    places = [Place(**p) for p in raw["places"]]
    start = LatLng(lat=raw["start"]["lat"], lng=raw["start"]["lng"])
    return Dataset(f"snapshot-{raw['city'].lower()}", start, places,
                   SnapshotTravel(raw["ids"], [start] + [p.location for p in places], raw["travel_minutes"]))


def load(path: Optional[str]) -> Dataset:
    return load_snapshot(path) if path else synthetic()


def _key(point: LatLng) -> Tuple[float, float]:
    return round(point.lat, 5), round(point.lng, 5)


class SnapshotTravel:
    """Travel-time lookup over a recorded matrix; matches validation.TravelTimeFn."""

    def __init__(self, ids: List[str], points: List[LatLng], minutes: Dict[str, List[List[Optional[float]]]]):
        self.index = {_key(p): i for i, p in enumerate(points)}
        self.minutes = minutes

    def __call__(self, origin: LatLng, destination: LatLng, mode: TravelMode) -> Optional[float]:
        i, j = self.index.get(_key(origin)), self.index.get(_key(destination))
        if i is None or j is None:
            # Only snapshot places and the start can be routed; anything else is a bug upstream.
            raise KeyError(f"location not in snapshot: {origin if i is None else destination}")
        return self.minutes[mode.value][i][j]
