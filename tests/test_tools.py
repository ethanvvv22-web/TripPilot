"""Tool clients tested against recorded-style JSON, without network access."""

import unittest

from trippilot.models import LatLng, TravelMode
from trippilot.tools.places import PlacesClient, parse_opening_periods, parse_place
from trippilot.tools.routes import RoutesClient, parse_route_matrix

PLACE_JSON = {
    "id": "ChIJ_test",
    "displayName": {"text": "Gustavianum", "languageCode": "en"},
    "formattedAddress": "Akademigatan 3, Uppsala",
    "location": {"latitude": 59.8579, "longitude": 17.6352},
    "types": ["museum", "tourist_attraction"],
    "rating": 4.5,
    "priceLevel": "PRICE_LEVEL_INEXPENSIVE",
    "regularOpeningHours": {"periods": [
        {"open": {"day": 2, "hour": 10, "minute": 0}, "close": {"day": 2, "hour": 16, "minute": 0}},
        {"open": {"day": 6, "hour": 22, "minute": 0}, "close": {"day": 0, "hour": 2, "minute": 0}},
    ]},
}


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Records requests and replies with queued responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(("POST", url, json, headers))
        return self.responses.pop(0)

    def get(self, url, headers=None, timeout=None):
        self.calls.append(("GET", url, None, headers))
        return self.responses.pop(0)


class ParsePlaceTest(unittest.TestCase):
    def test_fields(self):
        place = parse_place(PLACE_JSON)
        self.assertEqual(place.place_id, "ChIJ_test")
        self.assertEqual(place.name, "Gustavianum")
        self.assertEqual(place.location.lat, 59.8579)
        self.assertEqual(place.price_level, 1)
        self.assertTrue(place.is_open_between(2, 11 * 60, 12 * 60))
        self.assertFalse(place.is_open_between(3, 11 * 60, 12 * 60))

    def test_period_wrapping_into_sunday(self):
        place = parse_place(PLACE_JSON)
        self.assertTrue(place.is_open_between(0, 0, 90))

    def test_open_24_7_has_no_close(self):
        periods = parse_opening_periods({"periods": [{"open": {"day": 0, "hour": 0, "minute": 0}}]})
        self.assertEqual(periods[0].end - periods[0].start, 7 * 24 * 60)

    def test_missing_hours_are_unknown(self):
        data = {k: v for k, v in PLACE_JSON.items() if k != "regularOpeningHours"}
        self.assertIsNone(parse_place(data).opening_periods)


class PlacesClientTest(unittest.TestCase):
    def test_search_sends_field_mask_and_location_bias(self):
        session = FakeSession(FakeResponse({"places": [PLACE_JSON]}))
        client = PlacesClient("key", session=session)
        places = client.search_text("museum", near=LatLng(lat=59.86, lng=17.64), radius_m=2000)

        self.assertEqual([p.name for p in places], ["Gustavianum"])
        _, url, body, headers = session.calls[0]
        self.assertTrue(url.endswith("/places:searchText"))
        self.assertEqual(headers["X-Goog-Api-Key"], "key")
        self.assertIn("places.regularOpeningHours", headers["X-Goog-FieldMask"])
        self.assertEqual(body["locationBias"]["circle"]["radius"], 2000)

    def test_search_results_are_reused_by_get_place(self):
        session = FakeSession(FakeResponse({"places": [PLACE_JSON]}))
        client = PlacesClient("key", session=session)
        client.search_text("museum")
        self.assertEqual(client.get_place("ChIJ_test").name, "Gustavianum")
        self.assertEqual(len(session.calls), 1)

    def test_unknown_place_id_returns_none(self):
        client = PlacesClient("key", session=FakeSession(FakeResponse({}, status_code=404)))
        self.assertIsNone(client.get_place("ChIJ_missing"))


MATRIX_JSON = [
    {"originIndex": 0, "destinationIndex": 0, "condition": "ROUTE_EXISTS",
     "duration": "600s", "distanceMeters": 800, "status": {}},
    {"originIndex": 0, "destinationIndex": 1, "condition": "ROUTE_NOT_FOUND", "status": {}},
]


class RoutesTest(unittest.TestCase):
    A = LatLng(lat=59.8586, lng=17.6389)
    B = LatLng(lat=59.8582, lng=17.6337)
    C = LatLng(lat=60.0, lng=18.0)

    def test_parse_matrix_skips_missing_routes(self):
        legs = parse_route_matrix(MATRIX_JSON)
        self.assertEqual(legs[(0, 0)].duration_minutes, 10)
        self.assertNotIn((0, 1), legs)

    def test_travel_minutes_is_cached_from_matrix(self):
        session = FakeSession(FakeResponse(MATRIX_JSON))
        client = RoutesClient("key", session=session)
        client.compute_matrix([self.A], [self.B, self.C], TravelMode.WALK)

        self.assertEqual(client.travel_minutes(self.A, self.B, TravelMode.WALK), 10)
        self.assertIsNone(client.travel_minutes(self.A, self.C, TravelMode.WALK))
        self.assertEqual(len(session.calls), 1)
        _, _, body, _ = session.calls[0]
        self.assertEqual(body["travelMode"], "WALK")

    def test_cache_is_per_travel_mode(self):
        session = FakeSession(FakeResponse(MATRIX_JSON), FakeResponse(MATRIX_JSON))
        client = RoutesClient("key", session=session)
        client.travel_minutes(self.A, self.B, TravelMode.WALK)
        client.travel_minutes(self.A, self.B, TravelMode.DRIVE)
        self.assertEqual(len(session.calls), 2)


if __name__ == "__main__":
    unittest.main()
