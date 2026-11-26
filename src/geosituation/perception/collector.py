import re
import json
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Optional
import requests
from email.utils import parsedate_to_datetime

from geosituation.perception.preprocessor import TextPreprocessor
from geosituation.perception.classifier import IncidentClassifier
from geosituation.perception.gazetteer import NoidaGazetteer
from geosituation.perception.ner_extractor import IncidentEntityExtractor
from geosituation.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, INCIDENT_CATEGORIES

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

    # Words that indicate an actual incident. Real-news feeds are full of
    # property launches, events and policy stories; those never reach the map.
    INCIDENT_TERMS = re.compile(
        r"\b(kill(ed|ing|s)?|murder(ed)?|found dead|(one|two|three|four|five|\d+) dead|death|died|dies|"
        r"body found|shot|shoot(ing|out)?|stabb(ed|ing)|"
        r"assault(ed)?|attack(ed)?|rap(e|ed)|molest(ed|ation)?|kidnap(ped|ping)?|abduct(ed|ion)?|"
        r"rob(bed|bery|bers?)|loot(ed)?|thie(f|ves)|theft|stole(n)?|snatch(ed|ing|ers?)?|burglar(y|s|ies)?|"
        r"fraud|scam|cheat(ed|ing)?|duped|extort(ion|ed)?|arrest(ed|s)?|held|nabbed|booked|encounter|"
        r"accident|crash(ed|es)?|collid(e|ed)|collision|mow(ed)? down|hit[- ]and[- ]run|overturn(ed|s)?|"
        r"rams?|rammed|run over|pile-?up|injur(ed|ies|y)|hurt|fire|blaze|gutted|explosion|blast|"
        r"collapse(d|s)?|caved?[- ]in|electrocut(ed|ion)|drown(ed|s)?|"
        r"waterlog(ged|ging)|flood(ed|ing)?|gas leak|"
        r"protest(s|ers|ed)?|agitation|clash(ed|es)?|ruckus|brawl|scuffle|lathi|blockade|dharna)\b",
        re.IGNORECASE,
    )

    # Stories that mention incident words but are not a new, located incident:
    # retrospectives, media, court stages of old cases, self-harm.
    EXCLUDE_TERMS = re.compile(
        r"\b(suicide|film|films|documentar(y|ies)|web series|anniversary|years after|verdict|"
        r"acquit(ted|tal)?|convicted|sentenced|bail|hearing|chargesheet|mock drill|drill)\b",
        re.IGNORECASE,
    )

    # Other cities that also number their neighbourhoods as "Sector N".
    OTHER_CITIES = re.compile(
        r"\b(gurugram|gurgaon|chandigarh|mohali|panchkula|faridabad|ghaziabad|dwarka|rohini|"
        r"navi mumbai|kharghar|vashi|bhilai|bokaro|lucknow|jaipur|bhubaneswar|zirakpur|kurukshetra|"
        r"karnal|sonipat|rewari|manesar|bahadurgarh|panipat|hisar|ambala|yamunanagar)\b",
        re.IGNORECASE,
    )

    # Headline-only text is too short for the statistical classifier to be
    # reliable, so unambiguous cue words decide the category first. Order
    # matters: a deadly fire is a civic hazard, a fatal crash is traffic.
    HEADLINE_CATEGORY_RULES = [
        ("CIVIC_HAZARD", re.compile(r"\b(fire|blaze|gutted|explosion|blast|collapse(d|s)?|caved?[- ]in|electrocut(ed|ion)|gas leak|waterlog(ged|ging)|flood(ed|ing)?|drown(ed|s)?)\b", re.I)),
        ("TRAFFIC_INCIDENT", re.compile(r"\b(accident|crash(ed|es)?|collid(e|ed)|collision|mow(ed)? down|hit[- ]and[- ]run|overturn(ed|s)?|rams?|rammed|run over|pile-?up)\b", re.I)),
        ("PROPERTY_CRIME", re.compile(r"\b(burglar(y|s|ies)?|theft|thie(f|ves)|stole(n)?|snatch(ed|ing|ers?)?|loot(ed)?|fraud|scam|cheat(ed|ing)?|duped|extort(ion|ed)?)\b", re.I)),
        ("VIOLENT_CRIME", re.compile(r"\b(murder(ed)?|kill(ed|ing)?|stabb(ed|ing)|shot|shoot(ing|out)?|firing|assault(ed)?|attack(ed)?|rap(e|ed)|molest(ed|ation)?|kidnap(ped|ping)?|abduct(ed|ion)?|rob(bed|bery|bers?)|encounter|found dead|body found|beat(en)? to death|thrash(ed)?)\b", re.I)),
        ("PUBLIC_DISRUPTION", re.compile(r"\b(protest(s|ers|ed)?|agitation|clash(ed|es)?|ruckus|brawl|scuffle|lathi|blockade|dharna)\b", re.I)),
    ]

    @staticmethod
    def _clean_feed_title(title: str, outlet: str) -> str:
        """Strip the outlet suffixes Google News and publishers append to titles."""
        t = title.strip()
        if outlet and t.endswith(" - " + outlet):
            t = t[: -len(" - " + outlet)].strip()
        elif " - " in t:
            t = t.rpartition(" - ")[0].strip()
        if outlet and t.endswith(" | " + outlet):
            t = t[: -len(" | " + outlet)].strip()
        # Generic " | Outlet" suffix (e.g. "... | Hindustan Times")
        t = re.sub(r"\s+\|\s+[A-Z][\w .&']{2,40}$", "", t).strip()
        # Leading "Noida News | " style prefixes
        t = re.sub(r"^[\w .']{2,25}\s+\|\s+", "", t).strip()
        return t

    def fetch_google_news_rss(self, query: str = "Noida crime OR accident OR fire OR protest",
                              max_items: int = 100, timeout: int = 15) -> List[Dict]:
        """
        Fetches a live Google News RSS search and returns Level 1 records for
        real articles that (a) name Noida in the headline, (b) read as a new
        incident and (c) resolve to a known sector or landmark.
        Network errors return [].

        Each record's `source_url` holds the outlet name as plain text and the
        original link is kept in `article_url`.
        """
        rss_url = "https://news.google.com/rss/search"
        params = {"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}

        articles = []
        try:
            resp = requests.get(rss_url, params=params, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception:
            return articles

        for item in root.findall(".//item")[:max_items]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            outlet = (item.findtext("source") or "").strip()
            if not outlet and " - " in title:
                outlet = title.rpartition(" - ")[2].strip()
            headline = self._clean_feed_title(title, outlet)
            if not headline:
                continue

            # pubDate is RFC 822 in GMT -> convert to local naive time like the rest of the pipeline
            try:
                pub_date = parsedate_to_datetime((item.findtext("pubDate") or "").strip()).astimezone().replace(tzinfo=None)
            except Exception:
                continue  # undated items cannot be placed on the timeline

            low = headline.lower()
            # Must explicitly be Noida. Greater Noida has its own sector numbering,
            # so it is excluded except for the shared Noida-Greater Noida Expressway.
            if "noida" not in low:
                continue
            if "greater noida" in low.replace("noida-greater noida", "").replace("noida greater noida", ""):
                continue
            if self.OTHER_CITIES.search(low):
                continue
            if not self.INCIDENT_TERMS.search(headline) or self.EXCLUDE_TERMS.search(headline):
                continue

            record = self.process_raw_article(headline=headline, body="", source_url=outlet or "Google News", pub_date=pub_date)
            if not record:
                continue

            for cat, pattern in self.HEADLINE_CATEGORY_RULES:
                if pattern.search(headline):
                    if cat != record["category"]:
                        meta = INCIDENT_CATEGORIES[cat]
                        record["category"] = cat
                        record["category_name"] = meta["name"]
                        record["severity_weight"] = meta["severity_weight"]
                    record["category_source"] = "headline_cue_rule"
                    break
            else:
                record["category_source"] = "classifier"

            record["article_url"] = link
            record["ingest_channel"] = "google_news_rss"
            articles.append(record)

        return articles
