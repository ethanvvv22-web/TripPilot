"""Offline stand-ins for Google Places and Routes.

Lets the agent run end to end without a Google API key. Place names and
coordinates are approximate; opening hours and prices are made up.
"""

import math
from typing import Iterable, List, Optional

from .models import LatLng, OpeningPeriod, Place, TravelMode

_DAY = 24 * 60

# Straight-line speed (km/h) per mode; distances get a detour factor below.
_SPEED_KMH = {TravelMode.WALK: 4.5, TravelMode.BICYCLE: 14, TravelMode.TRANSIT: 18, TravelMode.DRIVE: 30}
_DETOUR = 1.3


def _hours(open_h: float, close_h: float, days: Iterable[int] = range(7)) -> List[OpeningPeriod]:
    return [OpeningPeriod(start=d * _DAY + int(open_h * 60), end=d * _DAY + int(close_h * 60)) for d in days]


UPPSALA_START = LatLng(lat=59.8586, lng=17.6460)  # Uppsala Central Station

UPPSALA_PLACES = [
    Place(place_id="off_cathedral", name="Uppsala Cathedral", location=LatLng(lat=59.8582, lng=17.6337),
          types=["church", "tourist_attraction", "history"], rating=4.7, price_level=0,
          opening_periods=_hours(8, 18)),
    Place(place_id="off_gustavianum", name="Gustavianum", location=LatLng(lat=59.8579, lng=17.6352),
          types=["museum", "history", "science"], rating=4.5, price_level=1,
          opening_periods=_hours(10, 16, days=[0, 2, 3, 4, 5, 6])),
    Place(place_id="off_castle", name="Uppsala Castle", location=LatLng(lat=59.8535, lng=17.6346),
          types=["castle", "museum", "history", "art"], rating=4.4, price_level=1,
          opening_periods=_hours(11, 16, days=[0, 3, 4, 5, 6])),
    Place(place_id="off_linnaeus", name="Linnaeus Garden", location=LatLng(lat=59.8617, lng=17.6337),
          types=["garden", "park", "museum", "history"], rating=4.5, price_level=1,
          opening_periods=_hours(11, 17)),
    Place(place_id="off_botanical", name="Uppsala Botanical Garden", location=LatLng(lat=59.8517, lng=17.6292),
          types=["garden", "park", "nature"], rating=4.6, price_level=0,
          opening_periods=_hours(7, 19)),
    Place(place_id="off_gamla", name="Gamla Uppsala Museum", location=LatLng(lat=59.8977, lng=17.6290),
          types=["museum", "history", "viking", "tourist_attraction"], rating=4.6, price_level=1,
          opening_periods=_hours(10, 16)),
    Place(place_id="off_cafe", name="Café Linné", location=LatLng(lat=59.8611, lng=17.6340),
          types=["cafe", "coffee", "fika", "food"], rating=4.3, price_level=1,
          opening_periods=_hours(9, 18)),
    Place(place_id="off_restaurant", name="Domtrappkällaren", location=LatLng(lat=59.8584, lng=17.6346),
          types=["restaurant", "swedish", "food", "lunch", "dinner"], rating=4.4, price_level=2,
          opening_periods=_hours(11.5, 22, days=[1, 2, 3, 4, 5, 6])),
    Place(place_id="off_upplandsmuseet", name="Upplandsmuseet", location=LatLng(lat=59.8589, lng=17.6372),
          types=["museum", "history", "culture", "indoor"], rating=4.3, price_level=0,
          opening_periods=_hours(12, 17, days=[0, 2, 3, 4, 5, 6])),
    Place(place_id="off_ofvandahls", name="Ofvandahls Hovkonditori", location=LatLng(lat=59.8601, lng=17.6407),
          types=["cafe", "coffee", "fika", "pastry", "food"], rating=4.4, price_level=1,
          opening_periods=_hours(8, 18)),
    Place(place_id="off_carolina", name="Carolina Rediviva", location=LatLng(lat=59.8560, lng=17.6310),
          types=["library", "history", "exhibition", "indoor"], rating=4.6, price_level=0,
          opening_periods=_hours(10, 17, days=[1, 2, 3, 4, 5, 6])),
    Place(place_id="off_hambergs", name="Hambergs Fisk", location=LatLng(lat=59.8590, lng=17.6400),
          types=["restaurant", "seafood", "food", "lunch", "dinner"], rating=4.5, price_level=3,
          opening_periods=_hours(11, 21, days=[2, 3, 4, 5, 6])),
    Place(place_id="off_fyrishov", name="Fyrishov", location=LatLng(lat=59.8710, lng=17.6200),
          types=["swimming", "sports", "family", "kids", "indoor"], rating=4.2, price_level=2,
          opening_periods=_hours(9, 20)),
    Place(place_id="off_stadstradgarden", name="Stadsträdgården", location=LatLng(lat=59.8530, lng=17.6460),
          types=["park", "garden", "nature", "family", "kids"], rating=4.4, price_level=0,
          opening_periods=[OpeningPeriod(start=0, end=7 * _DAY)]),
]


class OfflinePlaces:
    """Keyword search over a fixed list of places; same interface as PlacesClient."""

    def __init__(self, places: Iterable[Place] = UPPSALA_PLACES):
        self.places = {p.place_id: p for p in places}

    def search_text(self, query: str, near: Optional[LatLng] = None,
                    radius_m: float = 5000, max_results: int = 10) -> List[Place]:
        words = [w for w in query.lower().split() if len(w) > 2]
        scored = []
        for place in self.places.values():
            haystack = " ".join([place.name.lower(), *place.types])
            score = sum(1 for w in words if w in haystack)
            if score:
                scored.append((score, place.rating or 0, place))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [p for _, _, p in scored[:max_results]]

    def get_place(self, place_id: str) -> Optional[Place]:
        return self.places.get(place_id)


def straight_line_minutes(origin: LatLng, destination: LatLng, mode: TravelMode) -> Optional[float]:
    """Estimate travel time from great-circle distance; matches validation.TravelTimeFn."""
    km = _haversine_km(origin, destination) * _DETOUR
    return km / _SPEED_KMH[mode] * 60


def _haversine_km(a: LatLng, b: LatLng) -> float:
    lat1, lat2 = math.radians(a.lat), math.radians(b.lat)
    dlat, dlng = lat2 - lat1, math.radians(b.lng - a.lng)
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))
