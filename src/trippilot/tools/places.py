"""Google Places API (New) client: text search and place details."""

from typing import Any, Dict, List, Optional

import requests

from ..models import MINUTES_PER_DAY, MINUTES_PER_WEEK, LatLng, OpeningPeriod, Place

PLACES_BASE_URL = "https://places.googleapis.com/v1"

# Only request the fields we use; Google bills Places calls by field mask.
PLACE_FIELDS = [
    "id",
    "displayName",
    "formattedAddress",
    "location",
    "types",
    "rating",
    "priceLevel",
    "regularOpeningHours",
]

PRICE_LEVELS = {
    "PRICE_LEVEL_FREE": 0,
    "PRICE_LEVEL_INEXPENSIVE": 1,
    "PRICE_LEVEL_MODERATE": 2,
    "PRICE_LEVEL_EXPENSIVE": 3,
    "PRICE_LEVEL_VERY_EXPENSIVE": 4,
}


def parse_opening_periods(hours: Optional[Dict[str, Any]]) -> Optional[List[OpeningPeriod]]:
    """Convert Places `regularOpeningHours` into week-minute periods."""
    if not hours or "periods" not in hours:
        return None
    periods = []
    for raw in hours["periods"]:
        start = _week_minute(raw["open"])
        if "close" not in raw:
            # Google encodes "open 24/7" as a single period with no close.
            periods.append(OpeningPeriod(start=start, end=start + MINUTES_PER_WEEK))
            continue
        end = _week_minute(raw["close"])
        if end <= start:
            end += MINUTES_PER_WEEK
        periods.append(OpeningPeriod(start=start, end=end))
    return periods


def parse_place(data: Dict[str, Any]) -> Place:
    location = data.get("location", {})
    return Place(
        place_id=data["id"],
        name=data.get("displayName", {}).get("text", data["id"]),
        location=LatLng(lat=location.get("latitude", 0.0), lng=location.get("longitude", 0.0)),
        address=data.get("formattedAddress"),
        types=data.get("types", []),
        rating=data.get("rating"),
        price_level=PRICE_LEVELS.get(data.get("priceLevel")),
        opening_periods=parse_opening_periods(data.get("regularOpeningHours")),
    )


class PlacesClient:
    def __init__(self, api_key: str, session: Optional[requests.Session] = None, timeout: float = 10):
        self.api_key = api_key
        self.session = session or requests.Session()
        self.timeout = timeout
        self._details_cache: Dict[str, Optional[Place]] = {}

    def search_text(
        self,
        query: str,
        near: Optional[LatLng] = None,
        radius_m: float = 5000,
        max_results: int = 10,
    ) -> List[Place]:
        body: Dict[str, Any] = {"textQuery": query, "maxResultCount": max_results}
        if near is not None:
            body["locationBias"] = {
                "circle": {
                    "center": {"latitude": near.lat, "longitude": near.lng},
                    "radius": radius_m,
                }
            }
        response = self.session.post(
            f"{PLACES_BASE_URL}/places:searchText",
            json=body,
            headers=self._headers(",".join(f"places.{f}" for f in PLACE_FIELDS)),
            timeout=self.timeout,
        )
        response.raise_for_status()
        places = [parse_place(p) for p in response.json().get("places", [])]
        for place in places:
            self._details_cache[place.place_id] = place
        return places

    def get_place(self, place_id: str) -> Optional[Place]:
        """Fetch one place, or None if Google does not know the id."""
        if place_id in self._details_cache:
            return self._details_cache[place_id]
        response = self.session.get(
            f"{PLACES_BASE_URL}/places/{place_id}",
            headers=self._headers(",".join(PLACE_FIELDS)),
            timeout=self.timeout,
        )
        if response.status_code in (400, 404):
            place = None
        else:
            response.raise_for_status()
            place = parse_place(response.json())
        self._details_cache[place_id] = place
        return place

    def _headers(self, field_mask: str) -> Dict[str, str]:
        return {"X-Goog-Api-Key": self.api_key, "X-Goog-FieldMask": field_mask}


def _week_minute(point: Dict[str, int]) -> int:
    return point["day"] * MINUTES_PER_DAY + point.get("hour", 0) * 60 + point.get("minute", 0)
