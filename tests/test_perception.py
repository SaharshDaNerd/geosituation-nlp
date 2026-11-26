import os
import unittest
from datetime import datetime
from unittest.mock import patch

from geosituation.config import build_pipeline_config
from geosituation.perception.gazetteer import NoidaGazetteer
from geosituation.perception.classifier import IncidentClassifier
from geosituation.perception.preprocessor import TextPreprocessor
from geosituation.perception.ner_extractor import IncidentEntityExtractor
from geosituation.perception.collector import NewsIncidentCollector


class TestPerceptionEngine(unittest.TestCase):

    def setUp(self):
        self.gazetteer = NoidaGazetteer()
        self.classifier = IncidentClassifier()
        self.preprocessor = TextPreprocessor()
        self.ner = IncidentEntityExtractor()
        self.collector = NewsIncidentCollector()

    def test_gazetteer_sector_regex(self):
        res = self.gazetteer.resolve_location("Chain snatching reported in Sector 18 near Atta Market")
        self.assertIsNotNone(res)
        self.assertEqual(res["sector_id"], "Sector 18")
        self.assertGreaterEqual(res["confidence"], 0.90)

    def test_gazetteer_landmark_alias(self):
        res = self.gazetteer.resolve_location("Two vehicles collided near Fortis Hospital electronic city")
        self.assertIsNotNone(res)
        self.assertEqual(res["sector_id"], "Sector 62")

    def test_gazetteer_avoids_false_atta_and_authority_hijack(self):
        # "Atta Gujran" in Greater Noida must not falsely map to Sector 18
        res_gn = self.gazetteer.resolve_location("24-year-old arrested after firing outside Greater Noida college, resident of Atta Gujran village")
        self.assertIsNone(res_gn)

        # "Noida Authority" must not override actual location "Barola" (Sector 49)
        res_barola = self.gazetteer.resolve_location("Body found in drain in Noida Barola, locals question Noida Authority negligence")
        self.assertIsNotNone(res_barola)
        self.assertEqual(res_barola["sector_id"], "Sector 49")

    def test_gazetteer_multi_sector_and_hindi(self):
        locs = self.gazetteer.resolve_all_locations("Snatching incidents reported across Sectors 15 and 16 near metro stations")
        sector_ids = [loc["sector_id"] for loc in locs]
        self.assertIn("Sector 15", sector_ids)
        self.assertIn("Sector 16", sector_ids)

        hi_res = self.gazetteer.resolve_location("नोएडा सेक्टर 62 के पास सड़क हादसा, दो घायल")
        self.assertIsNotNone(hi_res)
        self.assertEqual(hi_res["sector_id"], "Sector 62")

        hi_barola = self.gazetteer.resolve_location("नोएडा बरौला में नाले में मिला शव")
        self.assertIsNotNone(hi_barola)
        self.assertEqual(hi_barola["sector_id"], "Sector 49")

        hi_entities = self.ner.extract_entities("नोएडा सेक्टर-6 की फैक्ट्री में लगी भीषण आग, बाइक सवार घायल")
        self.assertIn("Fire", hi_entities["instruments_or_hazards"])
        self.assertIn("Bike", hi_entities["vehicles"])

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

    def test_bs4_layered_extraction_jsonld_and_fallbacks(self):
        # 1. JSON-LD extraction (no <p> tags, like Times of India)
        html_jsonld = """
        <html><head>
        <script type="application/ld+json">
        {"@type": "NewsArticle", "headline": "Fire at Sector 63 unit",
         "datePublished": "2026-11-25T10:15:00+05:30",
         "articleBody": "A major blaze erupted at a factory in Noida Sector 63 early Thursday morning."}
        </script></head><body><div>No p tags here</div></body></html>
        """
        res1 = self.collector.extract_article_from_html(html_jsonld)
        self.assertEqual(res1["extraction_layer"], "jsonld")
        self.assertIn("Sector 63", res1["body"])
        self.assertIsNotNone(res1["pub_date"])

        # 2. Semantic <article> container with nav/footer stripped
        html_article = """
        <html><body>
          <nav><p>Subscribe to breaking news alerts across India today for free.</p></nav>
          <article><p>Two bike-borne snatchers targeted a commuter near Sector 18 market late on Friday evening.</p></article>
          <footer><p>Copyright 2026 All rights reserved by the publisher network.</p></footer>
        </body></html>
        """
        res2 = self.collector.extract_article_from_html(html_article)
        self.assertEqual(res2["extraction_layer"], "semantic_container")
        self.assertIn("Sector 18", res2["body"])
        self.assertNotIn("Subscribe", res2["body"])
        self.assertNotIn("Copyright", res2["body"])

        # 3. Meta og:description fallback when DOM is JS-only
        html_meta = """
        <html><head>
        <meta property="og:description" content="A speeding truck hit a divider in Noida Sector 62 causing traffic snarl.">
        </head><body><div id="app"></div></body></html>
        """
        res3 = self.collector.extract_article_from_html(html_meta)
        self.assertEqual(res3["extraction_layer"], "meta_fallback")
        self.assertIn("Sector 62", res3["body"])

        # 4. Empty/malformed HTML handling
        res4 = self.collector.extract_article_from_html("")
        self.assertEqual(res4["extraction_layer"], "empty")
        self.assertEqual(res4["body"], "")

    def test_config_strictness_presets_and_overrides(self):
        strict_cfg = build_pipeline_config("strict")
        recall_cfg = build_pipeline_config("high_recall")
        self.assertFalse(strict_cfg["grounding"]["allow_rss_summary_only"])
        self.assertTrue(recall_cfg["grounding"]["allow_rss_summary_only"])
        self.assertGreater(strict_cfg["grounding"]["min_spatial_confidence"], recall_cfg["grounding"]["min_spatial_confidence"])

        with patch.dict(os.environ, {"PIPELINE_MODE": "strict", "MIN_TEXT_LENGTH": "42", "DEDUP_SIM_THRESHOLD": "0.85"}):
            override_cfg = build_pipeline_config()
            self.assertEqual(override_cfg["grounding"]["min_text_length"], 42)
            self.assertEqual(override_cfg["deduplication"]["sim_threshold"], 0.85)

    def test_strict_mode_drops_bodyless_rss_and_multi_sector_expansion(self):
        strict_cfg = build_pipeline_config("strict")
        strict_collector = NewsIncidentCollector(grounding_config=strict_cfg["grounding"], ingest_config=strict_cfg["ingest"])
        dropped = strict_collector._evaluate_and_build_records(
            headline="Robbery reported in Noida Sector 18 market",
            body="",
            outlet="TestOutlet",
            link="https://example.com/1",
            pub_date=datetime.now(),
            ingest_channel="google_news_rss",
        )
        self.assertEqual(dropped, [])
        self.assertGreaterEqual(strict_collector.get_diagnostics()["dropped_rss_summary_disallowed"], 1)

        # Balanced mode expands explicitly listed multi-sector stories up to max_sectors_per_article=2
        records = self.collector.process_raw_article_multi(
            headline="Chain snatchers target commuters in Noida Sectors 15 and 16",
            body="Police launched a probe after two snatching incidents near Sectors 15 and 16.",
            source_url="TestOutlet",
            pub_date=datetime.now(),
        )
        self.assertEqual(len(records), 2)
        self.assertEqual({r["sector_id"] for r in records}, {"Sector 15", "Sector 16"})


if __name__ == "__main__":
    unittest.main()
