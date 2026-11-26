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

    def test_distinct_same_sector_events_not_overmerged(self):
        now = datetime.now().isoformat()
        reports = [
            {
                "report_id": "REP_A",
                "headline": "Man dies after his motorcycle crashes into tree in Noida Sector 63",
                "full_text": "Man dies after his motorcycle crashes into tree in Noida Sector 63 near factory gate.",
                "category": "TRAFFIC_INCIDENT",
                "category_name": "Traffic & Road Hazard",
                "severity_weight": 2.0,
                "model_confidence": 0.90,
                "sector_id": "Sector 63",
                "sector_name": "Sector 63, Noida",
                "zone": "Industrial",
                "latitude": 28.625,
                "longitude": 77.382,
                "incident_timestamp": now,
                "published_at": now,
                "source_url": "Hindustan Times",
                "entities": {"organizations": ["Noida Police"], "vehicles": ["Motorcycle"], "instruments_or_hazards": [], "roles_and_actors": []}
            },
            {
                "report_id": "REP_B",
                "headline": "Speeding bus rams stationary tanker causing morning jam in Sector 63",
                "full_text": "A speeding private bus rammed a stationary water tanker in Sector 63, while Noida Police cleared the pileup with a motorcycle patrol.",
                "category": "TRAFFIC_INCIDENT",
                "category_name": "Traffic & Road Hazard",
                "severity_weight": 2.0,
                "model_confidence": 0.85,
                "sector_id": "Sector 63",
                "sector_name": "Sector 63, Noida",
                "zone": "Industrial",
                "latitude": 28.625,
                "longitude": 77.382,
                "incident_timestamp": now,
                "published_at": now,
                "source_url": "Times of India",
                "entities": {"organizations": ["Noida Police"], "vehicles": ["Motorcycle", "Bus"], "instruments_or_hazards": [], "roles_and_actors": []}
            }
        ]

        dedup = CrossDocumentDeduplicator()
        canonical = dedup.deduplicate(reports)
        self.assertEqual(len(canonical), 2)
        self.assertEqual(dedup.last_merge_stats["merged_reports"], 0)
        self.assertEqual(dedup.last_merge_stats["output_events"], 2)

    def test_cross_lingual_hindi_english_and_long_short_merge(self):
        now = datetime.now().isoformat()
        reports = [
            {
                "report_id": "REP_EN_FIRE",
                "headline": "Massive Fire Guts Plastic Decorative Items Factory in Noida's Sector 6",
                "full_text": "Massive Fire Guts Plastic Decorative Items Factory in Noida's Sector 6.",
                "category": "CIVIC_HAZARD",
                "category_name": "Civic & Industrial Hazard",
                "severity_weight": 3.5,
                "model_confidence": 0.90,
                "sector_id": "Sector 6",
                "sector_name": "Sector 6, Noida",
                "zone": "Urban Sector",
                "latitude": 28.5938,
                "longitude": 77.3205,
                "incident_timestamp": now,
                "published_at": now,
                "source_url": "tennews.in",
                "entities": {"organizations": [], "vehicles": [], "instruments_or_hazards": ["Fire"], "roles_and_actors": []}
            },
            {
                "report_id": "REP_HI_FIRE",
                "headline": "नोएडा सेक्टर-6 के प्लास्टिक फैक्ट्री में लगी भीषण आग, दूसरी इमारत भी आई चपेट में",
                "full_text": "नोएडा सेक्टर-6 के प्लास्टिक फैक्ट्री में लगी भीषण आग, दूसरी इमारत भी आई चपेट में",
                "category": "CIVIC_HAZARD",
                "category_name": "Civic & Industrial Hazard",
                "severity_weight": 3.5,
                "model_confidence": 0.90,
                "sector_id": "Sector 6",
                "sector_name": "Sector 6, Noida",
                "zone": "Urban Sector",
                "latitude": 28.5938,
                "longitude": 77.3205,
                "incident_timestamp": now,
                "published_at": now,
                "source_url": "ETV Bharat",
                "entities": {"organizations": [], "vehicles": [], "instruments_or_hazards": ["Fire"], "roles_and_actors": []}
            },
            {
                "report_id": "REP_TOI_CRYPTO",
                "headline": "Promised profits, man loses ₹24L in crypto fraud",
                "full_text": "Promised profits, man loses ₹24L in crypto fraud. Noida: A 44-year-old man from Sector 21 lost Rs 24 lakh in a cryptocurrency investment scam. Cybercrime branch registered FIR under BNS Section 318(4) and IT Act Section 66D after Binance wallet transfers could not be withdrawn.",
                "category": "PROPERTY_CRIME",
                "category_name": "Property / Financial Crime",
                "severity_weight": 2.5,
                "model_confidence": 0.80,
                "sector_id": "Sector 21",
                "sector_name": "Sector 21, Noida",
                "zone": "Residential",
                "latitude": 28.5885,
                "longitude": 77.3340,
                "incident_timestamp": now,
                "published_at": now,
                "source_url": "Times of India",
                "entities": {"organizations": [], "vehicles": [], "instruments_or_hazards": [], "roles_and_actors": []}
            },
            {
                "report_id": "REP_HT_CRYPTO",
                "headline": "Noida man duped of ₹24L in cryptocurrency fraud",
                "full_text": "Noida man duped of ₹24L in cryptocurrency fraud. Noida: A 44-year-old man in Sector 21 was duped.",
                "category": "PROPERTY_CRIME",
                "category_name": "Property / Financial Crime",
                "severity_weight": 2.5,
                "model_confidence": 0.75,
                "sector_id": "Sector 21",
                "sector_name": "Sector 21, Noida",
                "zone": "Residential",
                "latitude": 28.5885,
                "longitude": 77.3340,
                "incident_timestamp": now,
                "published_at": now,
                "source_url": "Hindustan Times",
                "entities": {"organizations": [], "vehicles": [], "instruments_or_hazards": [], "roles_and_actors": []}
            }
        ]

        dedup = CrossDocumentDeduplicator()
        canonical = dedup.deduplicate(reports)
        self.assertEqual(len(canonical), 2)
        fire_evt = next(e for e in canonical if e["sector_id"] == "Sector 6")
        self.assertEqual(fire_evt["report_count"], 2)
        self.assertIn("Massive Fire", fire_evt["headline"])  # English preferred over Hindi
        crypto_evt = next(e for e in canonical if e["sector_id"] == "Sector 21")
        self.assertEqual(crypto_evt["report_count"], 2)

if __name__ == "__main__":
    unittest.main()
