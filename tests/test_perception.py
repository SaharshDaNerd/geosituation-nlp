import unittest
from datetime import datetime
from geosituation.perception.gazetteer import NoidaGazetteer
from geosituation.perception.classifier import IncidentClassifier
from geosituation.perception.preprocessor import TextPreprocessor
from geosituation.perception.ner_extractor import IncidentEntityExtractor

class TestPerceptionEngine(unittest.TestCase):

    def setUp(self):
        self.gazetteer = NoidaGazetteer()
        self.classifier = IncidentClassifier()
        self.preprocessor = TextPreprocessor()
        self.ner = IncidentEntityExtractor()

    def test_gazetteer_sector_regex(self):
        res = self.gazetteer.resolve_location("Chain snatching reported in Sector 18 near Atta Market")
        self.assertIsNotNone(res)
        self.assertEqual(res["sector_id"], "Sector 18")
        self.assertGreaterEqual(res["confidence"], 0.90)

    def test_gazetteer_landmark_alias(self):
        res = self.gazetteer.resolve_location("Two vehicles collided near Fortis Hospital electronic city")
        self.assertIsNotNone(res)
        self.assertEqual(res["sector_id"], "Sector 62")

    def test_classifier_violent_crime(self):
        text = "Armed robbers stormed a jewelry showroom at gunpoint and stabbed the security guard"
        res = self.classifier.predict(text)
        self.assertEqual(res["predicted_category"], "VIOLENT_CRIME")
        self.assertGreater(res["confidence"], 0.20)

    def test_classifier_traffic_hazard(self):
        text = "Speeding truck collided with a bus on the expressway causing heavy pileup and injuries"
        res = self.classifier.predict(text)
        self.assertEqual(res["predicted_category"], "TRAFFIC_INCIDENT")

    def test_classifier_civic_hazard(self):
        text = "Massive fire broke out in a garment packaging factory warehouse ten fire tenders rushed"
        res = self.classifier.predict(text)
        self.assertEqual(res["predicted_category"], "CIVIC_HAZARD")

    def test_classifier_non_incident_filtering(self):
        text = "Noida Authority inaugurated a new tree plantation drive and public library park"
        res = self.classifier.predict(text)
        self.assertEqual(res["predicted_category"], "NON_INCIDENT")

    def test_ner_entity_extraction(self):
        text = "Two masked assailants on a motorcycle attacked a delivery boy with an iron rod near Fortis Hospital"
        entities = self.ner.extract_entities(text)
        self.assertIn("Motorcycle", entities["vehicles"])
        self.assertIn("Iron Rod", entities["instruments_or_hazards"])
        self.assertIn("Assailants", entities["roles_and_actors"])
        self.assertIn("Fortis Hospital", entities["organizations"])

    def test_temporal_anchor_extraction(self):
        ref = datetime(2026, 11, 14, 15, 0)
        res = self.preprocessor.extract_temporal_anchor("The robbery occurred yesterday late night at 10:30 pm", ref)
        dt = datetime.fromisoformat(res["incident_timestamp"])
        self.assertEqual(dt.day, 13)
        self.assertEqual(dt.hour, 22)
        self.assertEqual(dt.minute, 30)

if __name__ == "__main__":
    unittest.main()
