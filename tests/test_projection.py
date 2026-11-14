import unittest
from datetime import datetime, timedelta
from geosituation.projection.spatial_kde import SpatialKDEProjector
from geosituation.projection.hotspot_dbscan import STHotspotDetector
from geosituation.projection.momentum import SituationalMomentumProjector

class TestProjectionEngine(unittest.TestCase):

    def setUp(self):
        self.kde = SpatialKDEProjector(grid_res=30)
        self.dbscan = STHotspotDetector(spatial_eps_km=2.5, temporal_eps_days=5.0, min_samples=2)
        self.momentum = SituationalMomentumProjector(forecast_horizon_days=7)

    def test_spatial_kde_grid_generation(self):
        now = datetime.now().isoformat()
        events = [
            {
                "latitude": 28.5700,
                "longitude": 77.3220,
                "severity_weight": 5.0,
                "verification_confidence": 0.8
            },
            {
                "latitude": 28.5720,
                "longitude": 77.3250,
                "severity_weight": 2.5,
                "verification_confidence": 0.7
            }
        ]
        grid = self.kde.compute_density_grid(events)
        self.assertIn("heatmap_points", grid)
        self.assertGreater(grid["max_density"], 0.0)
        self.assertGreater(len(grid["heatmap_points"]), 0)

    def test_st_dbscan_hotspot_drift(self):
        now = datetime.now()
        # Create cluster of 3 incidents separated in space and time
        events = [
            {
                "event_id": "EVT_1",
                "sector_id": "Sector 137",
                "category": "PROPERTY_CRIME",
                "latitude": 28.5080,
                "longitude": 77.4080,
                "incident_timestamp": (now - timedelta(days=3)).isoformat()
            },
            {
                "event_id": "EVT_2",
                "sector_id": "Sector 142",
                "category": "PROPERTY_CRIME",
                "latitude": 28.4910,
                "longitude": 77.4190,
                "incident_timestamp": (now - timedelta(days=2)).isoformat()
            },
            {
                "event_id": "EVT_3",
                "sector_id": "Sector 142",
                "category": "PROPERTY_CRIME",
                "latitude": 28.4900,
                "longitude": 77.4200,
                "incident_timestamp": (now - timedelta(days=1)).isoformat()
            }
        ]

        hotspots = self.dbscan.detect_hotspots(events)
        self.assertGreaterEqual(len(hotspots), 1)
        h = hotspots[0]
        self.assertIn("Sector 142", h["sectors_involved"])
        self.assertEqual(h["dominant_category"], "PROPERTY_CRIME")
        self.assertIn("Drift:", h["trajectory_summary"])

    def test_momentum_projection(self):
        sector_map = {
            "Sector 142": {
                "sector_id": "Sector 142",
                "sector_name": "Sector 142, Noida",
                "raw_incident_count": 4,
                "sai_score": 9.5,
                "decayed_incident_count": 3.8
            }
        }
        anomalies = [
            {
                "sector_id": "Sector 142",
                "z_score": 2.4,
                "is_anomaly": True
            }
        ]
        trajectories = self.momentum.project_sector_trajectories(sector_map, anomalies)
        self.assertEqual(len(trajectories), 1)
        self.assertEqual(trajectories[0]["momentum_trajectory"], "ACCELERATING_ACTIVITY")
        self.assertGreater(trajectories[0]["projected_7d_incidents"], 1.0)

if __name__ == "__main__":
    unittest.main()
