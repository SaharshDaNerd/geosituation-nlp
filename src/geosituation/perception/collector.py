import re
import json
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Optional
import requests

from geosituation.perception.preprocessor import TextPreprocessor
from geosituation.perception.classifier import IncidentClassifier
from geosituation.perception.gazetteer import NoidaGazetteer
from geosituation.perception.ner_extractor import IncidentEntityExtractor
from geosituation.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

class NewsIncidentCollector:
    """
    Level 1 Pipeline Coordinator:
    Ingests text news reports, applies preprocessing, categorization,
    geocoding, and entity extraction to create canonical Level 1 Incident Records.
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.classifier = IncidentClassifier()
        self.gazetteer = NoidaGazetteer()
        self.ner = IncidentEntityExtractor()

    def process_raw_article(self, headline: str, body: str, source_url: str, pub_date: Optional[datetime] = None) -> Optional[Dict]:
        """
        Processes a single raw text article into a structured Level 1 Incident Record.
        Returns None if text is classified as NON_INCIDENT or cannot be geocoded to Noida.
        """
        full_text = f"{headline}. {body}"
        cleaned_text = self.preprocessor.clean_text(full_text)
        
        # 1. Classification
        cls_result = self.classifier.predict(cleaned_text)
        if cls_result["predicted_category"] == "NON_INCIDENT":
            # Discard non-incidents from situational awareness map
            return None

        # 2. Geocoding / Spatial Resolution
        location_result = self.gazetteer.resolve_location(cleaned_text)
        if not location_result:
            # Cannot locate inside Noida: do not hallucinate coordinates
            return None

        # 3. Temporal Resolution
        temporal_result = self.preprocessor.extract_temporal_anchor(cleaned_text, ref_date=pub_date)

        # 4. Entity Extraction
        entities = self.ner.extract_entities(cleaned_text)

        # Generate unique report ID
        report_id = f"REP_{uuid.uuid4().hex[:10].upper()}"

        record = {
            "report_id": report_id,
            "source_url": source_url,
            "headline": headline,
            "full_text": cleaned_text,
            "published_at": pub_date.isoformat() if pub_date else datetime.now().isoformat(),
            "incident_timestamp": temporal_result["incident_timestamp"],
            "temporal_indicator": temporal_result["temporal_indicator"],
            "temporal_confidence": temporal_result["temporal_confidence"],
            "category": cls_result["predicted_category"],
            "category_name": cls_result["category_name"],
            "model_confidence": cls_result["confidence"],
            "severity_weight": cls_result["severity_weight"],
            "sector_id": location_result["sector_id"],
            "sector_name": location_result["sector_name"],
            "zone": location_result["zone"],
            "latitude": location_result["lat"],
            "longitude": location_result["lon"],
            "spatial_match_type": location_result["match_type"],
            "spatial_confidence": location_result["confidence"],
            "entities": entities
        }
        return record

    def fetch_google_news_rss(self, query: str = "Noida crime OR accident OR fire OR protest") -> List[Dict]:
        """
        Fetches live news RSS feed. Gracefully handles network outages.
        """
        encoded_query = requests.utils.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        articles = []
        try:
            resp = requests.get(rss_url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall(".//item")[:20]:
                    title = item.findtext("title", "")
                    link = item.findtext("link", "")
                    pub_date_str = item.findtext("pubDate", "")
                    
                    pub_date = None
                    if pub_date_str:
                        try:
                            pub_date = datetime.strptime(pub_date_str[:25], "%a, %d %b %Y %H:%M:%S")
                        except Exception:
                            pub_date = datetime.now()
                            
                    # Clean title
                    headline = title.split(" - ")[0] if " - " in title else title
                    record = self.process_raw_article(headline=headline, body="", source_url=link, pub_date=pub_date)
                    if record:
                        articles.append(record)
        except Exception:
            # Fallback to local offline mode
            pass
            
        return articles
