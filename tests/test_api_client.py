import unittest
import asyncio
from starlette.requests import Request
from geosituation.api.main import (
    serve_dashboard,
    get_system_status,
    get_executive_summary,
    get_incidents,
    get_sectors,
    get_hotspots,
    get_heatmap,
    get_trajectories,
    ingest_article,
    ArticleIngestRequest
)

class TestAPIEndpointsDirect(unittest.TestCase):

    def test_dashboard_endpoint(self):
        req = Request({"type": "http", "method": "GET", "path": "/", "headers": []})
        resp = asyncio.run(serve_dashboard(req))
        self.assertEqual(resp.status_code, 200)
        self.assertIn("GeoSituation-NLP", resp.body.decode("utf-8"))

    def test_status_endpoint(self):
        data = asyncio.run(get_system_status())
        self.assertEqual(data["status"], "OPERATIONAL")
        self.assertGreater(data["total_canonical_events"], 0)

    def test_summary_endpoint(self):
        data = asyncio.run(get_executive_summary())
        self.assertIn("level_1_perception", data)
        self.assertIn("level_2_comprehension", data)
        self.assertIn("level_3_projection", data)

    def test_incidents_endpoint(self):
        data = asyncio.run(get_incidents(category="ALL", sector="ALL"))
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_sectors_endpoint(self):
        data = asyncio.run(get_sectors())
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_hotspots_endpoint(self):
        data = asyncio.run(get_hotspots())
        self.assertIsInstance(data, list)

    def test_heatmap_endpoint(self):
        data = asyncio.run(get_heatmap())
        self.assertIn("heatmap_points", data)

    def test_trajectories_endpoint(self):
        data = asyncio.run(get_trajectories())
        self.assertIsInstance(data, list)

    def test_live_ingest_endpoint(self):
        req = ArticleIngestRequest(
            headline="Delivery van caught fire near Sector 62 intersection",
            body="Noida: An electrical delivery van caught fire near Sector 62 Fortis hospital crossing. Fire fighters doused the vehicle.",
            source_url="https://test.com/fire"
        )
        data = asyncio.run(ingest_article(req))
        self.assertTrue(data["success"])

if __name__ == "__main__":
    unittest.main()
