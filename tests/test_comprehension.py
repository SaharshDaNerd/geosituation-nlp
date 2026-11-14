import unittest
from datetime import datetime, timedelta
from geosituation.comprehension.coreference import CrossDocumentDeduplicator, haversine_distance_km
from geosituation.comprehension.situational_index import SituationalIndexCalculator
from geosituation.comprehension.anomaly_detector import SectorAnomalyDetector

class TestComprehensionEngine(unittest.TestCase):

    def test_haversine_distance(self):
        # Distance between Sector 18 (28.5700, 77.3220) and Sector 16 (28.5780, 77.3180) ~ 0.96 km
        d = haversine_distance_km(28.5700, 77.3220, 28.5780, 77.3180)
        self.assertAlmostEqual(d, 0.96, delta=0.2)

    def test_coreference_deduplication(self):
        now = datetime.now().isoformat()
        reports = [
            {
                "report_id": "REP_01",
                "headline": "Armed robbery at jewelry showroom in Sector 18",
                "full_text": "Robbers looted gold showroom in Sector 18 Atta market at gunpoint.",
                "category": "VIOLENT_CRIME",
                "category_name": "Violent Crime",
                "severity_weight": 5.0,
                "model_confidence": 0.40,
                "sector_id": "Sector 18",
                "sector_name": "Sector 18, Noida",
                "zone": "Commercial",
                "latitude": 28.5700,
                "longitude": 77.3220,
                "incident_timestamp": now,
                "published_at": now,
                "source_url": "url1",
                "entities": {"roles_and_actors": ["Robbers"], "vehicles": ["Motorcycle"]}
            },
            {
                "report_id": "REP_02",
                "headline": "Gunmen storm Atta Market jewelry shop in Sector 18",
                "full_text": "Gunmen entered jewelry store in Sector 18 market and looted ornaments.",
                "category": "VIOLENT_CRIME",
                "category_name": "Violent Crime",
                "severity_weight": 5.0,
                "model_confidence": 0.35,
                "sector_id": "Sector 18",
                "sector_name": "Sector 18, Noida",
                "zone": "Commercial",
                "latitude": 28.5700,
                "longitude": 77.3220,
                "incident_timestamp": now,
                "published_at": now,
                "source_url": "url2",
                "entities": {"roles_and_actors": ["Gunmen"], "vehicles": ["Motorcycle"]}
            }
        ]

        dedup = CrossDocumentDeduplicator()
        canonical = dedup.deduplicate(reports)
        # Should be fused into 1 canonical event
        self.assertEqual(len(canonical), 1)
        self.assertTrue(canonical[0]["is_multi_source_verified"])
        self.assertEqual(canonical[0]["report_count"], 2)
        # Verification confidence should be higher than individual reports: 1 - (1 - 0.4)*(1 - 0.35) = 0.61
        self.assertAlmostEqual(canonical[0]["verification_confidence"], 0.61, delta=0.05)

    def test_temporal_decay_half_life(self):
        calc = SituationalIndexCalculator(half_life_days=7.0)
        now = datetime.now()
        
        # Incident exactly 7 days ago should have weight ~ 0.50
        seven_days_ago = (now - timedelta(days=7)).isoformat()
        w = calc.compute_temporal_weight(seven_days_ago, now)
        self.assertAlmostEqual(w, 0.50, delta=0.02)

        # Incident 14 days ago should have weight ~ 0.25
        fourteen_days_ago = (now - timedelta(days=14)).isoformat()
        w2 = calc.compute_temporal_weight(fourteen_days_ago, now)
        self.assertAlmostEqual(w2, 0.25, delta=0.02)

    def test_anomaly_detector_poisson_zscore(self):
        now = datetime.now()
        detector = SectorAnomalyDetector(baseline_window_days=30, recent_window_days=7, z_elevated=1.5)

        # Sector 18 has a surge: 3 incidents in the last 2 days
        events = [
            {
                "event_id": f"EVT_{i}",
                "sector_id": "Sector 18",
                "category": "VIOLENT_CRIME",
                "incident_timestamp": (now - timedelta(days=1)).isoformat()
            }
            for i in range(3)
        ]

        anomalies = detector.detect_anomalies(events, now)
        s18_anom = next((a for a in anomalies if a["sector_id"] == "Sector 18"), None)
        self.assertIsNotNone(s18_anom)
        self.assertTrue(s18_anom["is_anomaly"])
        self.assertGreater(s18_anom["z_score"], 1.5)

if __name__ == "__main__":
    unittest.main()
