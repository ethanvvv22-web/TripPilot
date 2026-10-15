"""Plan a trip from the command line.

    python -m trippilot --offline --preferences history fika

--offline plans on the recorded Google snapshot of Uppsala (data/uppsala_snapshot.json),
starting at its start point.
"""

import argparse
import os
import sys
from datetime import date, time, timedelta

from dotenv import load_dotenv

from .agent import INSTRUCTIONS, TOOL_SPECS, TripPlanner
from .agent.foundry import FoundryAgent
from .dataset import load_snapshot
from .models import LatLng, TravelMode, TripRequest
from .telemetry import setup_tracing


def next_saturday() -> date:
    today = date.today()
    return today + timedelta(days=(5 - today.weekday()) % 7 or 7)


def main() -> None:
    parser = argparse.ArgumentParser(prog="trippilot")
    parser.add_argument("--offline", action="store_true",
                        help="use the recorded Google snapshot of Uppsala instead of live APIs")
    parser.add_argument("--snapshot", default=os.path.join("data", "uppsala_snapshot.json"))
    parser.add_argument("--city", default="Uppsala")
    parser.add_argument("--date", type=date.fromisoformat, default=next_saturday())
    parser.add_argument("--start", type=time.fromisoformat, default=time(9, 0))
    parser.add_argument("--end", type=time.fromisoformat, default=time(17, 0))
    parser.add_argument("--lat", type=float, default=59.8582, help="start latitude (live mode)")
    parser.add_argument("--lng", type=float, default=17.6465, help="start longitude (live mode)")
    parser.add_argument("--mode", choices=[m.value for m in TravelMode], default=TravelMode.WALK.value)
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--preferences", nargs="*", default=[])
    parser.add_argument("--trace", metavar="FILE", help="append OpenTelemetry spans to FILE as JSON lines")
    args = parser.parse_args()
    if args.trace:
        setup_tracing(jsonl_path=args.trace)

    # Windows consoles default to a legacy code page that can't print names like "Café Linné".
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv()
    start = LatLng(lat=args.lat, lng=args.lng)
    if args.offline:
        dataset = load_snapshot(args.snapshot)
        places, travel, start = dataset.provider(), dataset.travel_minutes, dataset.start
    else:
        from .tools import PlacesClient, RoutesClient
        key = os.environ["GOOGLE_MAPS_API_KEY"]
        places, travel = PlacesClient(key), RoutesClient(key).travel_minutes

    request = TripRequest(
        city=args.city, date=args.date, start_time=args.start, end_time=args.end,
        start_location=start, travel_mode=TravelMode(args.mode),
        budget=args.budget, preferences=args.preferences,
    )

    agent = FoundryAgent(
        endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        instructions=INSTRUCTIONS,
        tool_specs=TOOL_SPECS,
    )
    try:
        result = TripPlanner(agent, places, travel).plan(request)
    finally:
        agent.delete()

    print(f"Status: {result.status}  (submissions: {result.submissions}, turns: {result.turns}, "
          f"tokens: {result.input_tokens} in / {result.output_tokens} out, model time: {result.model_seconds:.0f}s)")
    if result.itinerary:
        for stop in result.itinerary.stops:
            print(f"  {stop.arrival:%H:%M}-{stop.departure:%H:%M}  {stop.name}  (~{stop.estimated_cost:g})")
    for v in result.violations:
        print(f"  ! {v.message}")
    print("\nTool calls:")
    for record in result.trace:
        print(f"  {record.name:<18} {record.duration_ms:7.1f} ms")
    print(f"\n{result.summary}")


if __name__ == "__main__":
    main()
