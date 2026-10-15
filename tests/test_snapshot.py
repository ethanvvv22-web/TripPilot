"""Snapshot recording (with fake Google clients) and snapshot datasets."""

import json
import os
import tempfile
import unittest

from trippilot.dataset import load_snapshot
from trippilot.models import LatLng, Place, TravelMode
from trippilot import snapshot
from trippilot.snapshot import record, tag, travel_matrix
from trippilot.tools.routes import RouteLeg

START = LatLng(lat=59.858, lng=17.646)


def place(i, types=("museum",), km_east=0.5):
    # ~0.0175 degrees of longitude per km at Uppsala's latitude
    return Place(place_id=f"p{i}", name=f"Place {i}", location=LatLng(lat=59.858, lng=17.646 + km_east * 0.0175 + i * 1e-4),
                 types=list(types))


class FakePlaces:
    def __init__(self, by_query):
        self.by_query = by_query

    def search_text(self, query, near=None, radius_m=5000, max_results=10):
        if query == "Uppsala Centralstation":
            return [Place(place_id="station", name="Uppsala Centralstation", location=START)]
        for key, places in self.by_query.items():
            if query.startswith(key):
                return places[:max_results]
        return []


class FakeRoutes:
    """Minutes = 1 + |i - j| by point order; records request sizes."""

    def __init__(self, points):
        self.order = {(p.lat, p.lng): i for i, p in enumerate(points)}
        self.requests = []

    def compute_matrix(self, origins, destinations, mode, departure_time=None):
        self.requests.append((mode, len(origins), len(destinations), departure_time))
        legs = {}
        for a, o in enumerate(origins):
            for b, d in enumerate(destinations):
                i, j = self.order[(o.lat, o.lng)], self.order[(d.lat, d.lng)]
                legs[(a, b)] = RouteLeg(duration_minutes=1 + abs(i - j), distance_meters=100)
        return legs


def setUpModule():
    snapshot.pause = lambda seconds: None  # no real waiting in tests


class TagTest(unittest.TestCase):
    def test_google_types_and_query_tags(self):
        tagged = tag(place(1, types=("cafe", "food")), ["viking"])
        for t in ("cafe", "fika", "coffee", "viking"):
            self.assertIn(t, tagged.types)
        self.assertEqual(len(tagged.types), len(set(tagged.types)))


class OverridesTest(unittest.TestCase):
    def test_add_and_remove(self):
        ruins = place(1, types=("museum", "indoor", "history"))
        ruins = ruins.model_copy(update={"name": "Castle ruins"})
        warnings = []
        fixed = snapshot.apply_overrides(
            [ruins, place(2)],
            {"Castle ruins": {"remove": ["indoor", "museum"], "add": ["outdoor"]}, "Nowhere": {"add": ["x"]}},
            log=warnings.append)
        self.assertEqual(fixed[0].types, ["history", "outdoor"])
        self.assertEqual(fixed[1].types, ["museum"])
        self.assertEqual(len(warnings), 1)  # unmatched override is reported


class TravelMatrixTest(unittest.TestCase):
    def test_blocks_respect_api_limits_and_reassemble(self):
        points = [START] + [place(i).location for i in range(26)]  # 27 points
        routes = FakeRoutes(points)
        walk = travel_matrix(routes, points, TravelMode.WALK, departure=None)
        transit = travel_matrix(routes, points, TravelMode.TRANSIT, departure="dep")
        for mode, o, d, _ in routes.requests:
            self.assertLessEqual(o * d, 100 if mode is TravelMode.TRANSIT else 625)
        for matrix in (walk, transit):
            self.assertEqual(matrix[0][26], 27)
            self.assertEqual(matrix[26][0], 27)
            self.assertEqual(matrix[13][13], 0.0)
        self.assertTrue(all(dep == "dep" for mode, _, _, dep in routes.requests if mode is TravelMode.TRANSIT))
        self.assertTrue(all(dep is None for mode, _, _, dep in routes.requests if mode is not TravelMode.TRANSIT))


class RetryTest(unittest.TestCase):
    def test_rate_limited_requests_are_retried(self):
        class RateLimited(Exception):
            response = type("R", (), {"status_code": 429})()

        points = [START, place(1).location]
        routes = FakeRoutes(points)
        real = routes.compute_matrix
        failures = [RateLimited()]

        def flaky(*args, **kwargs):
            if failures:
                raise failures.pop()
            return real(*args, **kwargs)

        routes.compute_matrix = flaky
        self.assertEqual(travel_matrix(routes, points, TravelMode.WALK, None)[0][1], 2)


class RecordTest(unittest.TestCase):
    def setUp(self):
        self.museums = [place(i) for i in range(3)]
        self.cafes = [place(10, types=("cafe",)), self.museums[0]]   # a duplicate across queries
        self.far = [place(20, km_east=12)]                            # beyond the radius
        places = FakePlaces({"museum": self.museums, "cafe": self.cafes, "park": self.far})
        points = [START] + [p.location for p in self.museums + self.cafes[:1]]
        self.snapshot = record(places, FakeRoutes(points), "Uppsala", "Uppsala Centralstation", log=lambda *a: None)

    def test_places_deduplicated_and_filtered(self):
        ids = [p["place_id"] for p in self.snapshot["places"]]
        self.assertEqual(ids, ["p0", "p1", "p2", "p10"])
        self.assertEqual(self.snapshot["ids"][0], "start")

    def test_dataset_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "snap.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.snapshot, f)
            dataset = load_snapshot(path)
        by_id = dataset.by_id()
        self.assertEqual(dataset.travel_minutes(dataset.start, by_id["p10"].location, TravelMode.WALK), 5)
        self.assertEqual(dataset.travel_minutes(by_id["p1"].location, by_id["p0"].location, TravelMode.BICYCLE), 2)
        # Place 0 was also returned by the cafe search, so it carries the cafe tags too.
        self.assertEqual(sorted(p.name for p in dataset.provider().search_text("fika")), ["Place 0", "Place 10"])
        with self.assertRaises(KeyError):
            dataset.travel_minutes(LatLng(lat=0, lng=0), dataset.start, TravelMode.WALK)


if __name__ == "__main__":
    unittest.main()
