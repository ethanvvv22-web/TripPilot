"""Record a reproducible snapshot of real Google data for one city.

    python -m trippilot.snapshot --out data/uppsala_snapshot.json

Places come from Places API (New) text search; travel times between every
pair come from the Routes API route matrix, for each travel mode. Evaluation
and Harbor tasks read the snapshot, so they are reproducible, cost nothing to
rerun and need no API key.
"""

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

from .models import LatLng, Place, TravelMode
from .offline import _haversine_km

# (search query, tags implied by the query itself, on top of tags derived from Google's types)
CATEGORIES = [
    ("museum", ["museum"]),
    ("art museum", ["art", "museum"]),
    ("church", ["church", "history"]),
    ("historic site", ["history"]),
    ("Gamla Uppsala viking", ["viking", "history"]),
    ("cafe", ["cafe", "fika"]),
    ("konditori bakery", ["cafe", "fika", "pastry"]),
    ("lunch restaurant", ["restaurant", "lunch"]),
    ("dinner restaurant", ["restaurant", "dinner"]),
    ("seafood restaurant", ["restaurant", "seafood"]),
    ("park", ["park", "outdoor"]),
    ("botanical garden", ["garden", "outdoor"]),
    ("library", ["library", "indoor"]),
    ("tourist attraction", ["attraction"]),
    ("swimming pool", ["swimming", "indoor"]),
]

# Google place types -> tags our scenarios and the agent's searches use.
TYPE_TAGS = {
    "museum": ["museum", "indoor"],
    "art_gallery": ["art", "indoor"],
    "art_museum": ["art", "museum", "indoor"],
    "history_museum": ["history", "museum", "indoor"],
    "church": ["church", "history"],
    "place_of_worship": ["church"],
    "historical_landmark": ["history"],
    "historical_place": ["history"],
    "cafe": ["cafe", "fika", "coffee", "food"],
    "coffee_shop": ["cafe", "fika", "coffee", "food"],
    "bakery": ["cafe", "fika", "pastry", "food"],
    "restaurant": ["restaurant", "food"],
    "seafood_restaurant": ["restaurant", "seafood", "food"],
    "park": ["park", "outdoor", "nature"],
    "city_park": ["park", "outdoor", "nature"],
    "botanical_garden": ["garden", "outdoor", "nature"],
    "garden": ["garden", "outdoor", "nature"],
    "library": ["library", "indoor"],
    "tourist_attraction": ["attraction"],
    "swimming_pool": ["swimming", "sports", "indoor"],
    "sports_complex": ["sports"],
    "playground": ["family", "kids", "outdoor"],
}

MAX_PER_QUERY = 6
# New places taken per category, so every category is represented.
NEW_PER_CATEGORY = 3
MAX_DISTANCE_KM = 7
# Routes API limits per matrix request (elements = origins x destinations).
MATRIX_BLOCK = {TravelMode.TRANSIT: 10}  # 10 x 10 = 100
DEFAULT_BLOCK = 25                        # 25 x 25 = 625
# Stay under the per-minute element quota; back off further on HTTP 429.
ELEMENTS_PER_MINUTE = 2500
RETRY_WAITS = [15, 30, 60, 120]


def apply_overrides(places: List[Place], overrides: Dict[str, Dict[str, List[str]]], log=print) -> List[Place]:
    """Hand corrections for tags inferred wrongly, keyed by exact place name."""
    by_name = {p.name: p for p in places}
    for name in overrides:
        if name not in by_name:
            log(f"  override for {name!r} matched no place")
    result = []
    for p in places:
        o = overrides.get(p.name)
        if o:
            types = [t for t in p.types if t not in o.get("remove", [])] + o.get("add", [])
            p = p.model_copy(update={"types": list(dict.fromkeys(types))})
        result.append(p)
    return result


def tag(place: Place, extra: Sequence[str]) -> Place:
    tags = [t for g in place.types for t in TYPE_TAGS.get(g, [])] + list(extra)
    return place.model_copy(update={"types": list(dict.fromkeys(place.types + tags))})


def collect_places(places_client, start: LatLng, log=print) -> List[Place]:
    found: Dict[str, Place] = {}
    for query, extra in CATEGORIES:
        results = places_client.search_text(f"{query} Uppsala", near=start, radius_m=MAX_DISTANCE_KM * 1000,
                                             max_results=MAX_PER_QUERY)
        added = 0
        for place in results:
            if _haversine_km(start, place.location) > MAX_DISTANCE_KM:
                continue
            if place.place_id in found:
                found[place.place_id] = tag(found[place.place_id], extra)
                continue
            if added >= NEW_PER_CATEGORY:
                continue
            found[place.place_id] = tag(place, extra)
            added += 1
        log(f"  {query!r}: {len(results)} results, {added} new (total {len(found)})")
    return list(found.values())


def _with_retry(request):
    for wait in RETRY_WAITS + [None]:
        try:
            return request()
        except Exception as e:  # requests.HTTPError; only rate limits are retried
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status != 429 or wait is None:
                raise
            print(f"    rate limited; retrying in {wait}s", flush=True)
            pause(wait)


def pause(seconds: float) -> None:
    time.sleep(seconds)


def travel_matrix(routes_client, points: List[LatLng], mode: TravelMode,
                  departure: Optional[datetime]) -> List[List[Optional[float]]]:
    """Minutes from every point to every point, requested in blocks the API accepts."""
    n = len(points)
    block = MATRIX_BLOCK.get(mode, DEFAULT_BLOCK)
    minutes: List[List[Optional[float]]] = [[None] * n for _ in range(n)]
    for i0 in range(0, n, block):
        for j0 in range(0, n, block):
            origins, destinations = points[i0:i0 + block], points[j0:j0 + block]
            legs = _with_retry(lambda: routes_client.compute_matrix(
                origins, destinations, mode, departure_time=departure if mode is TravelMode.TRANSIT else None))
            pause(len(origins) * len(destinations) / ELEMENTS_PER_MINUTE * 60)
            for (i, j), leg in legs.items():
                minutes[i0 + i][j0 + j] = round(leg.duration_minutes, 1)
    for i in range(n):
        minutes[i][i] = 0.0
    return minutes


def next_saturday_10am(tz: str = "Europe/Stockholm") -> datetime:
    today = date.today()
    saturday = today + timedelta(days=(5 - today.weekday()) % 7 or 7)
    return datetime(saturday.year, saturday.month, saturday.day, 10, 0, tzinfo=ZoneInfo(tz))


def record(places_client, routes_client, city: str, start_query: str, log=print,
           overrides: Optional[Dict[str, Dict[str, List[str]]]] = None) -> dict:
    start_hits = places_client.search_text(start_query, max_results=1)
    if not start_hits:
        raise SystemExit(f"Start location not found: {start_query}")
    start = start_hits[0]
    log(f"Start: {start.name} ({start.location.lat:.5f}, {start.location.lng:.5f})")

    log("Searching places:")
    places = apply_overrides(collect_places(places_client, start.location, log), overrides or {}, log)
    points = [start.location] + [p.location for p in places]

    departure = next_saturday_10am()
    travel = {}
    for mode in TravelMode:
        log(f"Route matrix {mode.value} ({len(points)} x {len(points)})")
        travel[mode.value] = travel_matrix(routes_client, points, mode, departure)

    return {
        "city": city,
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": "Google Places API (New) text search + Routes API computeRouteMatrix",
        "transit_departure": departure.isoformat(),
        "note": "Tags beyond Google's own types are derived from those types and the search query, "
                "then corrected by hand (tag_overrides).",
        "tag_overrides": overrides or {},
        "start": {"name": start.name, "lat": start.location.lat, "lng": start.location.lng},
        "places": [json.loads(p.model_dump_json()) for p in places],
        "ids": ["start"] + [p.place_id for p in places],
        "travel_minutes": travel,
    }


def main() -> None:
    from dotenv import load_dotenv

    from .tools import PlacesClient, RoutesClient

    parser = argparse.ArgumentParser(prog="trippilot.snapshot")
    parser.add_argument("--city", default="Uppsala")
    parser.add_argument("--start", default="Uppsala Centralstation")
    parser.add_argument("--out", default=os.path.join("data", "uppsala_snapshot.json"))
    parser.add_argument("--overrides", default=os.path.join("data", "tag_overrides.json"))
    parser.add_argument("--retag", action="store_true",
                        help="re-apply tag overrides to the existing snapshot; no API calls")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    overrides = {}
    if os.path.exists(args.overrides):
        with open(args.overrides, encoding="utf-8") as f:
            overrides = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    if args.retag:
        with open(args.out, encoding="utf-8") as f:
            snapshot = json.load(f)
        places = apply_overrides([Place(**p) for p in snapshot["places"]], overrides)
        snapshot["places"] = [json.loads(p.model_dump_json()) for p in places]
        snapshot["tag_overrides"] = overrides
        snapshot["retagged"] = datetime.now().isoformat(timespec="seconds")
    else:
        load_dotenv()
        key = os.environ["GOOGLE_MAPS_API_KEY"]
        snapshot = record(PlacesClient(key), RoutesClient(key), args.city, args.start, overrides=overrides)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=1)
    missing = sum(v is None for mode in snapshot["travel_minutes"].values() for row in mode for v in row)
    print(f"Wrote {len(snapshot['places'])} places to {args.out} ({missing} pairs without a route)")


if __name__ == "__main__":
    main()
