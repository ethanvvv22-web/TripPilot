"""Hand-written places and helpers shared by the tests. Opening hours are made up."""

from datetime import date, time

from trippilot.models import LatLng, OpeningPeriod, Place, TravelMode, TripRequest

SATURDAY = date(2026, 10, 17)
MONDAY = date(2026, 10, 19)

DAY = 24 * 60


def daily(open_h: int, close_h: int, days=range(7)):
    return [OpeningPeriod(start=d * DAY + open_h * 60, end=d * DAY + close_h * 60) for d in days]


CATHEDRAL = Place(
    place_id="p_cathedral", name="Uppsala Cathedral", types=["church"],
    location=LatLng(lat=59.8582, lng=17.6337),
    opening_periods=daily(8, 18),
)
# Closed on Mondays (Google day 1).
MUSEUM = Place(
    place_id="p_museum", name="Gustavianum", types=["museum"],
    location=LatLng(lat=59.8579, lng=17.6352),
    opening_periods=daily(10, 16, days=[0, 2, 3, 4, 5, 6]),
)
PARK = Place(
    place_id="p_park", name="City Park", types=["park"],
    location=LatLng(lat=59.8600, lng=17.6400),
    opening_periods=[OpeningPeriod(start=0, end=7 * DAY)],
)
NO_HOURS = Place(
    place_id="p_unknown_hours", name="Street Art Walk",
    location=LatLng(lat=59.8550, lng=17.6300),
)

PLACES = {p.place_id: p for p in (CATHEDRAL, MUSEUM, PARK, NO_HOURS)}

START = LatLng(lat=59.8586, lng=17.6389)


def request(**overrides) -> TripRequest:
    fields = dict(
        city="Uppsala", date=SATURDAY,
        start_time=time(9, 0), end_time=time(18, 0),
        start_location=START, travel_mode=TravelMode.WALK, budget=300,
    )
    fields.update(overrides)
    return TripRequest(**fields)


def fixed_travel(minutes_between: float = 10):
    """Travel-time function that returns the same duration for every pair."""
    return lambda origin, destination, mode: minutes_between
