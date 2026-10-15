"""Build eval/scenarios_real.json from readable definitions over the recorded Uppsala snapshot.

    PYTHONPATH=src python eval/build_scenarios_real.py

Scenarios name places; this script resolves Google place ids and checks that the
base itinerary used by replans is itself feasible on the snapshot.
"""

import json
import os
import sys

from trippilot.dataset import load_snapshot
from trippilot.models import Itinerary, LatLng, Stop, TravelMode, TripRequest
from trippilot.validation import validate_itinerary

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = os.path.join(HERE, "..", "data", "uppsala_snapshot.json")
OUT = os.path.join(HERE, "scenarios_real.json")

SAT, SUN, MON, TUE, WED, THU, FRI = (f"2026-10-{d}" for d in range(17, 24))

# Saturday plan the replan scenarios start from.
BASE = [
    ("Uppsala Cathedral", "09:30", "10:30", 0),
    ("Gustavianum - Uppsala University Museum", "11:00", "12:00", 120),
    ("UNO fika & butik", "12:20", "13:00", 80),
    ("Uppsala Castle", "13:30", "14:45", 120),
    ("Botanical Garden", "15:00", "16:00", 0),
    ("Carolina park - 'Engelska Parken'", "16:20", "17:00", 0),
]


def plan(id, date, prefs, tags, start="09:00", end="17:00", mode="WALK", budget=None, **expect):
    return {"id": id, "task": "plan", "tags": tags, "date": date, "start_time": start, "end_time": end,
            "travel_mode": mode, "budget": budget, "preferences": prefs, "expect": expect}


def replan(id, now, change, tags, **expect):
    return {"id": id, "task": "replan", "tags": ["replan"] + tags, "date": SAT, "budget": 600,
            "preferences": ["history", "fika"],
            "replan": {"base": "saturday_real", "now": now, "change": change}, "expect": expect}


SCENARIOS = [
    # --- day plans -------------------------------------------------------------------------
    plan("real-sat-classic", SAT, ["history", "fika"], ["baseline"], budget=600,
         include_types=["museum", "fika"], min_stops=4),
    plan("real-sat-museums-art", SAT, ["museum", "art"], ["museums"], start="11:00",
         include_types=["museum", "art"], min_stops=3),
    plan("real-mon-museums", MON, ["museum", "history"], ["closures"], start="10:00", end="16:00",
         include_types=["museum"]),
    plan("real-mon-castle-wish", MON, ["castle", "art"], ["closures", "conflict"], start="10:00", end="15:00",
         exclude_places=["Uppsala Castle", "Uppsala Art Museum"], min_stops=2),
    plan("real-sun-seafood-lunch", SUN, ["seafood", "lunch"], ["closures"], start="11:00", end="15:00", budget=500,
         include_types=["seafood"]),
    plan("real-sun-history-fika", SUN, ["history", "fika"], ["closures", "budget"], start="10:00", end="16:00",
         budget=400, include_types=["museum", "fika"]),
    plan("real-sat-early-churches", SAT, ["museum", "church"], ["short-window", "closures"], start="08:00",
         end="11:00", include_types=["church"], min_stops=2),
    plan("real-sat-late-afternoon", SAT, ["museum", "fika"], ["short-window"], start="15:00", end="18:00",
         include_types=["museum", "fika"]),
    plan("real-tue-lunch-museum-2h", TUE, ["lunch", "museum"], ["short-window"], start="11:30", end="13:30",
         include_types=["restaurant", "museum"]),
    plan("real-sat-bike-viking", SAT, ["viking", "history"], ["travel-mode", "distance"], start="10:00",
         end="16:00", mode="BICYCLE", include_places=["Old Uppsala museum"], min_stops=3),
    plan("real-sat-walk-viking-fika", SAT, ["viking", "fika"], ["distance"], start="10:00",
         include_types=["viking", "fika"]),
    plan("real-thu-transit-viking-swim", THU, ["viking", "swimming"], ["travel-mode", "distance"], end="18:00",
         mode="TRANSIT", include_types=["viking", "swimming"]),
    plan("real-fri-evening-dinner-park", FRI, ["dinner", "park"], ["evening"], start="17:00", end="22:00",
         include_types=["restaurant", "park"]),
    plan("real-wed-swim-lunch-midday", WED, ["swimming", "lunch"], ["split-hours"], start="11:00", end="15:00",
         include_types=["swimming", "restaurant"]),
    plan("real-sat-family", SAT, ["swimming", "park", "fika"], ["dense"], budget=500,
         include_types=["swimming", "park", "fika"]),
    plan("real-sat-packed-day", SAT, ["history", "museum", "fika", "garden", "restaurant"], ["dense"], end="18:00",
         include_types=["museum", "fika", "garden", "restaurant"], min_stops=6),
    plan("real-sun-zero-budget", SUN, ["history", "park"], ["budget"], start="10:00", end="16:00", budget=0,
         include_types=["history"], min_stops=2),
    plan("real-sat-tight-budget-lunch", SAT, ["lunch", "museum", "fika"], ["budget", "dense"], start="11:00",
         end="16:00", budget=150, include_types=["restaurant", "museum", "fika"]),
    plan("real-tue-rain-indoor", TUE, ["indoor", "museum", "fika"], ["weather"], start="10:00",
         exclude_types=["outdoor"], include_types=["museum"]),
    plan("real-sat-garden-early", SAT, ["garden", "fika"], ["short-window"], start="07:00", end="10:00",
         include_types=["garden", "fika"]),
    plan("real-mon-library-swim-bike", MON, ["library", "swimming"], ["split-hours", "travel-mode"], end="15:00",
         mode="BICYCLE", include_types=["library", "swimming"]),
    plan("real-fri-drive-viking-church-lunch", FRI, ["viking", "church", "lunch"], ["travel-mode", "dense"],
         start="10:00", end="16:00", mode="DRIVE", include_types=["viking", "church", "restaurant"]),
    # --- replans of the Saturday base plan --------------------------------------------------
    replan("real-replan-rain-1230", "12:30", "It started raining; swap outdoor stops for indoor ones.",
           ["weather"], exclude_types=["outdoor"]),
    replan("real-replan-late-keep-castle", "12:10", "I'm 40 minutes behind schedule; push things back but keep the castle.",
           ["delay"], include_places=["Uppsala Castle"]),
    replan("real-replan-skip-castle-lunch", "10:00", "Skip the castle; I'd rather have a proper sit-down lunch.",
           ["preference-change"], exclude_places=["Uppsala Castle"], include_types=["restaurant"]),
    replan("real-replan-tired-end-1600", "14:00", "I'm tired. Finish by 16:00 and end the day with a fika.",
           ["time-change"], end_by="16:00", include_types=["fika"]),
    replan("real-replan-add-viking", "09:00", "Change of plans: I want to see Gamla Uppsala's Viking history today.",
           ["preference-change", "distance"], include_types=["viking"]),
    replan("real-replan-pouring-lunch", "12:30", "It's pouring now. Indoor places only, and we need a proper lunch.",
           ["weather"], exclude_types=["outdoor"], include_types=["restaurant"]),
    replan("real-replan-gustavianum-closed", "10:45", "Gustavianum is closed today for an event; find another museum.",
           ["closures"], exclude_places=["Gustavianum - Uppsala University Museum"], include_types=["museum"]),
    replan("real-replan-kids-swim", "13:10", "The kids want to go swimming this afternoon.",
           ["preference-change"], include_types=["swimming"]),
]


def main() -> None:
    dataset = load_snapshot(SNAPSHOT)
    ids = {p.name: p.place_id for p in dataset.places}
    tag_set = {t for p in dataset.places for t in p.types}

    def pid(name):
        if name not in ids:
            sys.exit(f"Unknown place in scenario definitions: {name!r}")
        return ids[name]

    base = [{"place_id": pid(n), "name": n, "arrival": a, "departure": d, "estimated_cost": c} for n, a, d, c in BASE]
    request = TripRequest(city="Uppsala", date=SAT, start_time="09:00", end_time="17:00",
                          start_location=dataset.start, travel_mode=TravelMode.WALK, budget=600)
    violations = validate_itinerary(Itinerary(date=SAT, stops=[Stop(**s) for s in base]), request,
                                    dataset.by_id(), dataset.travel_minutes)
    if violations:
        sys.exit("Base itinerary is not feasible:\n" + "\n".join(v.message for v in violations))

    scenarios = []
    for s in SCENARIOS:
        expect = dict(s.pop("expect"))
        expect["include_place_ids"] = [pid(n) for n in expect.pop("include_places", [])]
        expect["exclude_place_ids"] = [pid(n) for n in expect.pop("exclude_places", [])]
        for t in expect.get("include_types", []) + expect.get("exclude_types", []):
            if t not in tag_set:
                sys.exit(f"{s['id']}: tag {t!r} does not occur in the snapshot")
        scenarios.append({**{k: v for k, v in s.items() if v is not None}, "expect": expect})

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "description": "TripPilot benchmark on recorded real Google data for Uppsala (2026-10-17 is a Saturday). "
                           "Generated by eval/build_scenarios_real.py.",
            "dataset": "../data/uppsala_snapshot.json",
            "base_itineraries": {"saturday_real": base},
            "scenarios": scenarios,
        }, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(scenarios)} scenarios to {OUT}")


if __name__ == "__main__":
    main()
