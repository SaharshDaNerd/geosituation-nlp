import base64
import json
import logging
import re
import uuid
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from geosituation.config import (
    GROUNDING_CONFIG,
    INCIDENT_CATEGORIES,
    INGEST_CONFIG,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from geosituation.perception.classifier import IncidentClassifier
from geosituation.perception.gazetteer import NoidaGazetteer
from geosituation.perception.ner_extractor import IncidentEntityExtractor
from geosituation.perception.preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class NewsIncidentCollector:
    """
    Level 1 Pipeline Coordinator:
    Ingests text news reports across multiple channels, performs layered BeautifulSoup
    article extraction, applies configurable grounding/anti-hallucination checks,
    categorizes incidents, resolves Noida sectors, and tracks full funnel diagnostics.
    """

    BROWSER_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en;q=0.9,hi;q=0.8",
    }

    def __init__(
        self,
        grounding_config: Optional[Dict] = None,
        ingest_config: Optional[Dict] = None,
    ):
        self.preprocessor = TextPreprocessor()
        self.classifier = IncidentClassifier()
        self.gazetteer = NoidaGazetteer()
        self.ner = IncidentEntityExtractor()
        self.grounding_config = dict(grounding_config or GROUNDING_CONFIG)
        self.ingest_config = dict(ingest_config or INGEST_CONFIG)
        self.session = self._build_http_session()
        self._url_cache_path = RAW_DATA_DIR / "decoded_urls_cache.json"
        self._decoded_url_cache: Dict[str, str] = self._load_url_cache()
        self.stats: Counter = Counter()
        self.reset_stats()

    def _load_url_cache(self) -> Dict[str, str]:
        if self._url_cache_path.exists():
            try:
                return json.loads(self._url_cache_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_url_cache(self) -> None:
        try:
            self._url_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._url_cache_path.write_text(json.dumps(self._decoded_url_cache, indent=2), encoding="utf-8")
        except Exception:
            pass

    @classmethod
    def _build_http_session(cls) -> requests.Session:
        session = requests.Session()
        session.headers.update(cls.BROWSER_HEADERS)
        retry = Retry(
            total=2,
            backoff_factor=0.4,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def reset_stats(self) -> None:
        self.stats = Counter()

    def get_diagnostics(self) -> Dict[str, int]:
        return dict(self.stats)

    def _http_get(self, url: str, params: Optional[Dict] = None, timeout: int = 8) -> Optional[requests.Response]:
        """Performs an HTTP GET with UTF-8 encoding normalization and structured counters."""
        self.stats["http_requests"] += 1
        try:
            resp = self.session.get(url, params=params, timeout=timeout, allow_redirects=True)
            if resp.status_code == 200:
                self.stats["http_200"] += 1
                # Prevent requests from defaulting charset-less text/html to ISO-8859-1 (fixes ₹ -> â‚¹ mojibake)
                ctype = resp.headers.get("Content-Type", "").lower()
                if "text/html" in ctype and (not resp.encoding or resp.encoding.lower() in ("iso-8859-1", "latin-1")):
                    resp.encoding = "utf-8"
                return resp
            if 400 <= resp.status_code < 500:
                self.stats["http_4xx"] += 1
                logger.warning("HTTP %s for %s", resp.status_code, url)
            else:
                self.stats["http_5xx"] += 1
                logger.warning("HTTP %s for %s", resp.status_code, url)
            return None
        except requests.Timeout:
            self.stats["http_timeouts"] += 1
            logger.debug("Timeout fetching %s", url)
            return None
        except requests.RequestException as exc:
            self.stats["http_errors"] += 1
            logger.debug("Request error fetching %s: %s", url, exc)
            return None

    # ------------------------------------------------------------------ Layered BS4 Extraction

    @classmethod
    def _extract_jsonld_article(cls, soup: BeautifulSoup) -> Dict[str, str]:
        """Extracts headline, articleBody, and datePublished from JSON-LD scripts."""
        result: Dict[str, str] = {}
        for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
            raw = script.string or script.get_text() or ""
            if not raw.strip():
                continue
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue

            nodes = data if isinstance(data, list) else [data]
            expanded = []
            for node in nodes:
                if isinstance(node, dict) and isinstance(node.get("@graph"), list):
                    expanded.extend(node["@graph"])
                elif isinstance(node, dict):
                    expanded.append(node)

            for item in expanded:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("@type", ""))
                if any(t in item_type for t in ("Article", "NewsArticle", "ReportageNewsArticle")) or item.get("articleBody"):
                    if not result.get("headline") and item.get("headline"):
                        result["headline"] = str(item["headline"]).strip()
                    if not result.get("body") and item.get("articleBody"):
                        result["body"] = re.sub(r"\s+", " ", str(item["articleBody"])).strip()
                    if not result.get("datePublished") and item.get("datePublished"):
                        result["datePublished"] = str(item["datePublished"]).strip()
                    if result.get("body"):
                        return result
        return result

    @staticmethod
    def _extract_meta_fields(soup: BeautifulSoup) -> Dict[str, str]:
        """Extracts OpenGraph/standard meta description and publish date."""
        meta_info: Dict[str, str] = {}
        for attr, val in [("property", "og:description"), ("name", "description")]:
            tag = soup.find("meta", attrs={attr: val})
            if tag and tag.get("content"):
                meta_info["description"] = re.sub(r"\s+", " ", tag["content"]).strip()
                break

        for attr, val in [
            ("property", "article:published_time"),
            ("name", "publish-date"),
            ("itemprop", "datePublished"),
            ("name", "pubdate"),
        ]:
            tag = soup.find("meta", attrs={attr: val})
            if tag and tag.get("content"):
                meta_info["datePublished"] = tag["content"].strip()
                break

        if not meta_info.get("datePublished"):
            time_tag = soup.find("time", attrs={"datetime": True})
            if time_tag:
                meta_info["datePublished"] = time_tag["datetime"].strip()

        return meta_info

    @staticmethod
    def _strip_dom_boilerplate(soup: BeautifulSoup) -> None:
        """Removes navigation, scripts, ads, sidebars, and cookie banners in place."""
        for tag in soup(["script", "style", "noscript", "nav", "footer", "aside", "header", "form"]):
            tag.decompose()
        for attr_match in ("cookie", "banner", "sidebar", "related", "newsletter", "promo", "ad-"):
            for elem in soup.find_all(attrs={"class": re.compile(attr_match, re.I)}):
                elem.decompose()

    @staticmethod
    def _extract_container_text(container) -> str:
        """Extracts paragraphs or <br>-separated text lines from a semantic container."""
        if container is None:
            return ""
        paragraphs = [
            p.get_text(" ", strip=True)
            for p in container.find_all("p")
            if len(p.get_text(strip=True)) >= 35
        ]
        if paragraphs:
            return " ".join(paragraphs[:5])

        # Times of India and several Indian CMS layouts place article prose directly
        # inside <div class="_s30J"> / <div data-articlebody> separated by <br> tags.
        raw_text = container.get_text(separator="\n", strip=True)
        lines = [ln.strip() for ln in raw_text.splitlines() if len(ln.strip()) >= 35]
        return " ".join(lines[:5])

    def extract_article_from_html(self, html: str, url: str = "", outlet: str = "") -> Dict:
        """
        Layered BeautifulSoup article extractor:
          1. JSON-LD (`articleBody`, `headline`, `datePublished`)
          2. Semantic containers (`<article>`, `<main>`, TOI/HT/IE body selectors)
          3. Generic `<p>` fallback across cleaned DOM
          4. Meta description (`og:description` / `description`) fallback
        """
        if not html or not html.strip():
            self.stats["bs4_parse_failures"] += 1
            return {"headline": "", "body": "", "pub_date": None, "extraction_layer": "empty"}

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            self.stats["bs4_parse_failures"] += 1
            logger.debug("BS4 failed to parse HTML from %s: %s", url, exc)
            return {"headline": "", "body": "", "pub_date": None, "extraction_layer": "parse_error"}

        # Layer 1: Structured JSON-LD & Meta (read BEFORE stripping <script> tags)
        jsonld = self._extract_jsonld_article(soup)
        meta = self._extract_meta_fields(soup)

        h1 = soup.find("h1")
        headline = jsonld.get("headline") or (h1.get_text(" ", strip=True) if h1 else "")
        if not headline and soup.title:
            headline = self._clean_feed_title(soup.title.get_text(" ", strip=True), outlet)

        pub_date = None
        raw_dt = jsonld.get("datePublished") or meta.get("datePublished")
        if raw_dt:
            try:
                iso_str = raw_dt.replace("Z", "+00:00")
                pub_date = datetime.fromisoformat(iso_str).astimezone().replace(tzinfo=None)
            except Exception:
                pub_date = None

        body = ""
        layer = "empty"

        if jsonld.get("body") and len(jsonld["body"]) >= 45:
            body = jsonld["body"][:1500]
            layer = "jsonld"
            self.stats["bs4_jsonld_hits"] += 1
        else:
            self._strip_dom_boilerplate(soup)

            # Layer 2: Semantic / Indian news site containers
            selectors = [
                ("div", {"data-articlebody": True}),
                ("div", {"class": re.compile(r"(_s30J|storyDetails|story_details|article_content|Normal|full-details)", re.I)}),
                ("article", {}),
                ("main", {}),
            ]
            for tag_name, attrs in selectors:
                container = soup.find(tag_name, attrs=attrs) if attrs else soup.find(tag_name)
                candidate = self._extract_container_text(container)
                if len(candidate) >= 45:
                    body = candidate[:1500]
                    layer = "semantic_container"
                    self.stats["bs4_article_hits"] += 1
                    break

            # Layer 3: Generic <p> fallback across remaining DOM
            if not body:
                paragraphs = [
                    p.get_text(" ", strip=True)
                    for p in soup.find_all("p")
                    if len(p.get_text(strip=True)) >= 35
                ]
                if paragraphs:
                    body = " ".join(paragraphs[:4])[:1500]
                    layer = "paragraph_fallback"
                    self.stats["bs4_paragraph_hits"] += 1

        # Layer 4: Meta description fallback if DOM body was empty/paywalled
        if not body and meta.get("description"):
            body = meta["description"]
            layer = "meta_fallback"
            self.stats["bs4_meta_fallbacks"] += 1

        if not body:
            self.stats["bs4_parse_failures"] += 1

        return {
            "headline": headline.strip(),
            "body": body.strip(),
            "pub_date": pub_date,
            "extraction_layer": layer,
        }

    def _decode_google_news_url(self, gnews_url: str) -> str:
        """
        Attempts to resolve a `news.google.com/rss/articles/...` wrapper URL to the
        canonical publisher URL using protobuf base64 decoding or Google's batchexecute RPC.
        Returns original URL if decoding fails.
        """
        if not gnews_url or "news.google.com" not in gnews_url:
            return gnews_url
        if gnews_url in self._decoded_url_cache:
            return self._decoded_url_cache[gnews_url]

        path = urlparse(gnews_url).path
        article_id = path.rstrip("/").split("/")[-1]
        if not article_id:
            return gnews_url

        # Fast path: legacy base64-encoded protobuf URLs without AU_yqL signature
        try:
            padded = article_id + "=" * (-len(article_id) % 4)
            raw_bytes = base64.urlsafe_b64decode(padded)
            match = re.search(rb'https?://[^\x00-\x20\x7f-\xff"<>]+', raw_bytes)
            if match:
                decoded = match.group(0).decode("utf-8", errors="ignore")
                if "news.google.com" not in decoded:
                    self._decoded_url_cache[gnews_url] = decoded
                    self.stats["google_url_decoded"] += 1
                    return decoded
        except Exception:
            pass

        # Second path: batchexecute resolution for newer AU_yqL tokens
        try:
            resp = self._http_get(f"https://news.google.com/rss/articles/{article_id}", timeout=5)
            if resp is not None:
                soup = BeautifulSoup(resp.text, "html.parser")
                cwiz = soup.find("c-wiz", attrs={"data-n-a-ts": True, "data-n-a-sg": True})
                if cwiz:
                    ts = cwiz["data-n-a-ts"]
                    sg = cwiz["data-n-a-sg"]
                    inner = json.dumps([
                        "garturlreq",
                        [["X", "X", ["X", "X"], None, None, 1, 1, "US:en", None, 1, None, None, None, None, None, 0, 1],
                         "X", "X", 1, [1, 1, 1], 1, 1, None, 0, 0, None, 0],
                        article_id,
                        int(ts),
                        sg,
                    ])
                    payload = {"f.req": json.dumps([[["Fbv4je", inner]]])}
                    rpc_resp = self.session.post(
                        "https://news.google.com/_/DotsSplashUi/data/batchexecute",
                        data=payload,
                        headers={"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"},
                        timeout=5,
                    )
                    if rpc_resp.status_code == 200:
                        url_match = re.search(r'"(https?://[^"\\]+)"', rpc_resp.text)
                        if url_match and "news.google.com" not in url_match.group(1):
                            decoded = url_match.group(1)
                            self._decoded_url_cache[gnews_url] = decoded
                            self.stats["google_url_decoded"] += 1
                            return decoded
        except Exception as exc:
            logger.debug("Google News URL decode failed for %s: %s", gnews_url, exc)

        self.stats["google_url_decode_failed"] += 1
        self._decoded_url_cache[gnews_url] = gnews_url
        return gnews_url

    def _fetch_article_lead(self, url: str, timeout: int = 5) -> str:
        """Fetches an article URL (decoding Google News links if enabled) and extracts body text."""
        if not url:
            return ""
        target_url = url
        if "news.google.com" in url:
            if not self.ingest_config.get("decode_google_news_urls", True):
                return ""
            target_url = self._decode_google_news_url(url)
            if "news.google.com" in target_url:
                return ""

        resp = self._http_get(target_url, timeout=timeout)
        if resp is None or "text/html" not in resp.headers.get("Content-Type", ""):
            return ""
        parsed = self.extract_article_from_html(resp.text, url=target_url)
        return parsed.get("body", "")

    # ------------------------------------------------------------------ Core Level 1 Processing

    def _build_record_for_location(
        self,
        headline: str,
        cleaned_text: str,
        source_url: str,
        pub_date: Optional[datetime],
        cls_result: Dict,
        location_result: Dict,
        temporal_result: Dict,
        entities: Dict,
        related_sectors: List[str],
    ) -> Dict:
        report_id = f"REP_{uuid.uuid4().hex[:10].upper()}"
        return {
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
            "related_sectors": related_sectors,
            "entities": entities,
        }

    def process_raw_article_multi(
        self,
        headline: str,
        body: str,
        source_url: str,
        pub_date: Optional[datetime] = None,
        allow_rule_override: bool = False,
    ) -> List[Dict]:
        """
        Processes a raw article into one or more Level 1 records (for explicitly named
        multi-sector stories like 'Sectors 15 and 16'), enforcing grounding thresholds.
        """
        full_text = f"{headline}. {body}" if body else headline
        cleaned_text = self.preprocessor.clean_text(full_text)

        min_len = self.grounding_config.get("min_text_length", 15)
        if len(cleaned_text) < min_len:
            self.stats["dropped_min_length"] += 1
            return []

        # 1. Classification (with optional cue override check for short headlines/Hindi)
        cls_result = self.classifier.predict(cleaned_text)
        cue_cat = self._match_category_rule(headline, body)

        if cls_result["predicted_category"] == "NON_INCIDENT" and not (allow_rule_override and cue_cat):
            self.stats["dropped_classifier_non_incident"] += 1
            return []

        min_cls_conf = self.grounding_config.get("min_classifier_confidence", 0.15)
        if cls_result["confidence"] < min_cls_conf and not (allow_rule_override and cue_cat):
            self.stats["dropped_low_classifier_conf"] += 1
            return []

        if cue_cat and cue_cat != cls_result["predicted_category"]:
            meta = INCIDENT_CATEGORIES[cue_cat]
            cls_result = {
                **cls_result,
                "predicted_category": cue_cat,
                "category_name": meta["name"],
                "severity_weight": meta["severity_weight"],
            }

        # 2. Spatial Resolution (Zero-Hallucination Gate)
        locations = self.gazetteer.resolve_all_locations(
            cleaned_text,
            allow_landmark=self.grounding_config.get("allow_landmark_match", True),
            min_confidence=self.grounding_config.get("min_spatial_confidence", 0.75),
            allow_multi_sector=self.grounding_config.get("allow_multi_sector_expansion", True),
        )
        if not locations:
            self.stats["dropped_unresolved_location"] += 1
            return []

        max_sectors = self.grounding_config.get("max_sectors_per_article", 2)
        if not self.grounding_config.get("allow_multi_sector_expansion", True):
            max_sectors = 1
        chosen_locations = locations[: max(1, max_sectors)]
        related_sectors = [loc["sector_id"] for loc in locations]

        # 3. Temporal & Entity Extraction
        temporal_result = self.preprocessor.extract_temporal_anchor(cleaned_text, ref_date=pub_date)
        entities = self.ner.extract_entities(cleaned_text)

        records = []
        for idx, loc in enumerate(chosen_locations):
            rec = self._build_record_for_location(
                headline=headline,
                cleaned_text=cleaned_text,
                source_url=source_url,
                pub_date=pub_date,
                cls_result=cls_result,
                location_result=loc,
                temporal_result=temporal_result,
                entities=entities,
                related_sectors=related_sectors,
            )
            rec["is_multi_sector_secondary"] = idx > 0
            records.append(rec)
            self.stats["accepted_level1"] += 1
            if idx > 0:
                self.stats["multi_sector_expanded"] += 1

        return records

    def process_raw_article(self, headline: str, body: str, source_url: str, pub_date: Optional[datetime] = None) -> Optional[Dict]:
        """Backwards-compatible single-record wrapper returning primary resolved incident."""
        records = self.process_raw_article_multi(headline=headline, body=body, source_url=source_url, pub_date=pub_date)
        return records[0] if records else None

    # ------------------------------------------------------------------ Lexical & Domain Rules

    INCIDENT_TERMS = re.compile(
        r"\b(kill(ed|ing|s)?|murder(ed)?|found dead|dead body|(one|two|three|four|five|\d+) dead|death|died|dies|"
        r"body found|shot|shots|fired|gunshot|shoot(ing|out)?|stabb(ed|ing)|knife|"
        r"assault(ed)?|attack(ed)?|rap(e|ed)|molest(ed|ation)?|eve[- ]teasing|harass(ed|ment)?|kidnap(ped|ping)?|abduct(ed|ion)?|"
        r"rob(bed|bery|bers?)|loot(ed)?|thie(f|ves)|theft|stole(n)?|snatch(ed|ing|ers?)?|burglar(y|s|ies)?|"
        r"fraud|scam|cheat(ed|ing)?|duped|conman|cybercrime|cyber fraud|ransom|extort(ion|ed)?|"
        r"arrest(ed|s)?|held|nabbed|detained|custody|booked|fir|case registered|gang|busted|encounter|miscreants?|"
        r"accident|mishap|crash(ed|es)?|collid(e|ed)|collision|mow(ed)? down|hit[- ]and[- ]run|overturn(ed|s)?|"
        r"rams?|rammed|run over|pile-?up|snarl|gridlock|jammed|injur(ed|ies|y)|hurt|"
        r"fire|fireball|blaze|gutted|engulfed|short[- ]circuit|explosion|blast|evacuated|"
        r"collapse(d|s)?|caved?[- ]in|electrocut(ed|ion)|drown(ed|s)?|"
        r"waterlog(ged|ging)|flood(ed|ing)?|gas leak|"
        r"protest(s|ers|ed)?|agitation|agitate|demonstration|clash(ed|es)?|ruckus|brawl|scuffle|lathi|blockade|dharna)\b",
        re.IGNORECASE,
    )

    HINDI_INCIDENT_TERMS = re.compile(
        r"(हत्या|मर्डर|मौत|शव|लाश|गोली|फायरिंग|चाकू|हमला|मारपीट|लूट|डकैती|चोरी|चोर|झपटमारी|छिनैती|"
        r"धोखाधड़ी|ठगी|साइबर|गिरफ्तार|दबोचा|मुठभेड़|एनकाउंटर|हादसा|दुर्घटना|टक्कर|पलटी|रौंदा|घायल|"
        r"आग|भीषण आग|धमाका|ब्लास्ट|धंसा|गिरा|करंट|जलभराव|प्रदर्शन|जाम|हंगामा|बवाल|बदमाश)"
    )

    EXCLUDE_TERMS = re.compile(
        r"\b(suicide|film|films|documentar(y|ies)|web series|anniversary|years after|verdict|"
        r"acquit(ted|tal)?|convicted|sentenced|bail|hearing|chargesheet|mock drill|drill)\b",
        re.IGNORECASE,
    )

    OTHER_CITIES = re.compile(
        r"\b(gurugram|gurgaon|chandigarh|mohali|panchkula|faridabad|ghaziabad|dwarka|rohini|"
        r"navi mumbai|kharghar|vashi|bhilai|bokaro|lucknow|jaipur|bhubaneswar|zirakpur|kurukshetra|"
        r"karnal|sonipat|rewari|manesar|bahadurgarh|panipat|hisar|ambala|yamunanagar)\b",
        re.IGNORECASE,
    )

    GREATER_NOIDA_ONLY = re.compile(
        r"\b(pari chowk|knowledge park|greater noida west|noida extension|bisrakh|ecotech|kasna|surajpur|jewar|"
        r"atta gujran|alpha\s+[12i]|beta\s+[12i]|gamma\s+[12i]|delta\s+[12i])\b",
        re.IGNORECASE,
    )

    HEADLINE_CATEGORY_RULES = [
        ("CIVIC_HAZARD", re.compile(r"\b(fire|fireball|blaze|gutted|engulfed|explosion|blast|short[- ]circuit|collapse(d|s)?|caved?[- ]in|electrocut(ed|ion)|gas leak|waterlog(ged|ging)|flood(ed|ing)?|drown(ed|s)?)\b|(आग|धमाका|ब्लास्ट|करंट|जलभराव)", re.I)),
        ("TRAFFIC_INCIDENT", re.compile(r"\b(accident|mishap|crash(ed|es)?|collid(e|ed)|collision|mow(ed)? down|hit[- ]and[- ]run|overturn(ed|s)?|rams?|rammed|run over|pile-?up|snarl|gridlock)\b|(हादसा|दुर्घटना|टक्कर|पलटी|रौंदा)", re.I)),
        ("PROPERTY_CRIME", re.compile(r"\b(burglar(y|s|ies)?|theft|thie(f|ves)|stole(n)?|snatch(ed|ing|ers?)?|loot(ed)?|fraud|scam|cheat(ed|ing)?|duped|cybercrime|cyber fraud|extort(ion|ed)?)\b|(चोरी|लूट|झपटमारी|छिनैती|धोखाधड़ी|ठगी)", re.I)),
        ("VIOLENT_CRIME", re.compile(r"\b(murder(ed)?|kill(ed|ing)?|stabb(ed|ing)|shot|shots|fired|gunshot|shoot(ing|out)?|firing|assault(ed)?|attack(ed)?|rap(e|ed)|molest(ed|ation)?|kidnap(ped|ping)?|abduct(ed|ion)?|rob(bed|bery|bers?)|encounter|found dead|dead body|body found|beat(en)? to death|thrash(ed)?)\b|(हत्या|मर्डर|गोली|फायरिंग|चाकू|मारपीट|मुठभेड़|एनकाउंटर|शव|लाश)", re.I)),
        ("PUBLIC_DISRUPTION", re.compile(r"\b(protest(s|ers|ed)?|agitation|agitate|demonstration|clash(ed|es)?|ruckus|brawl|scuffle|lathi|blockade|dharna)\b|(प्रदर्शन|जाम|हंगामा|बवाल)", re.I)),
    ]

    @classmethod
    def _match_category_rule(cls, headline: str, body: str) -> Optional[str]:
        for cat, pattern in cls.HEADLINE_CATEGORY_RULES:
            if pattern.search(headline) or (body and pattern.search(body)):
                return cat
        return None

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
        t = re.sub(r"\s+\|\s+[A-Z][\w .&']{2,40}$", "", t).strip()
        t = re.sub(r"^[\w .']{2,25}\s+\|\s+", "", t).strip()
        return t

    @staticmethod
    def _clean_rss_description(raw_desc: str) -> str:
        """Strips HTML markup from RSS <description> tags to recover summary text."""
        if not raw_desc:
            return ""
        text = BeautifulSoup(raw_desc, "html.parser").get_text(separator=" ")
        return re.sub(r"\s+", " ", text).strip()

    def _is_valid_noida_incident_text(
        self,
        text: str,
        require_noida_keyword: Optional[bool] = None,
        is_hindi: bool = False,
    ) -> bool:
        """Validates Noida scope, non-Greater-Noida locality, and incident cues with rejection counters."""
        if require_noida_keyword is None:
            require_noida_keyword = self.grounding_config.get("require_noida_keyword", True)

        low = text.lower()
        if require_noida_keyword and ("noida" not in low and "नोएडा" not in text):
            self.stats["dropped_non_noida"] += 1
            return False

        # Reject Greater Noida stories unless an explicit Noida Sector N (or Expressway) is present
        has_explicit_noida_sector = bool(self.gazetteer.regex_sector_pattern.search(low))
        cleaned_gn = low.replace("noida-greater noida", "").replace("noida greater noida", "").replace("greater noida expressway", "")
        if (self.GREATER_NOIDA_ONLY.search(low) or "greater noida" in cleaned_gn or "ग्रेटर नोएडा" in text) and not has_explicit_noida_sector:
            self.stats["dropped_greater_noida"] += 1
            return False

        if self.OTHER_CITIES.search(low):
            self.stats["dropped_other_city"] += 1
            return False

        has_cue = bool(self.INCIDENT_TERMS.search(text)) or (is_hindi and bool(self.HINDI_INCIDENT_TERMS.search(text)))
        if not has_cue:
            self.stats["dropped_no_incident_cue"] += 1
            return False

        if self.EXCLUDE_TERMS.search(text):
            self.stats["dropped_excluded_term"] += 1
            return False

        return True

    def _evaluate_and_build_records(
        self,
        headline: str,
        body: str,
        outlet: str,
        link: str,
        pub_date: datetime,
        ingest_channel: str,
        require_noida_keyword: Optional[bool] = None,
        is_hindi: bool = False,
    ) -> List[Dict]:
        """Shared Level 1 validation, rule-assisted categorization, and multi-sector builder."""
        if not body.strip() and not self.grounding_config.get("allow_rss_summary_only", True):
            self.stats["dropped_rss_summary_disallowed"] += 1
            return []

        combined = f"{headline}. {body}".strip() if body else headline.strip()
        if not self._is_valid_noida_incident_text(
            combined,
            require_noida_keyword=require_noida_keyword,
            is_hindi=is_hindi,
        ):
            return []

        records = self.process_raw_article_multi(
            headline=headline,
            body=body,
            source_url=outlet or "News Feed",
            pub_date=pub_date,
            allow_rule_override=True,
        )
        if not records:
            return []

        cue_cat = self._match_category_rule(headline, body)
        for record in records:
            record["category_source"] = "headline_cue_rule" if cue_cat else "classifier"
            record["article_url"] = link
            record["ingest_channel"] = ingest_channel
        return records

    def _evaluate_and_build_record(
        self,
        headline: str,
        body: str,
        outlet: str,
        link: str,
        pub_date: datetime,
        ingest_channel: str,
        require_noida_keyword: Optional[bool] = None,
        is_hindi: bool = False,
    ) -> Optional[Dict]:
        records = self._evaluate_and_build_records(
            headline=headline,
            body=body,
            outlet=outlet,
            link=link,
            pub_date=pub_date,
            ingest_channel=ingest_channel,
            require_noida_keyword=require_noida_keyword,
            is_hindi=is_hindi,
        )
        return records[0] if records else None

    # ------------------------------------------------------------------ Channels

    def _parse_rss_xml(self, content: bytes, source_label: str) -> List[ET.Element]:
        try:
            root = ET.fromstring(content)
            return root.findall(".//item")
        except ET.ParseError as exc:
            self.stats["bs4_parse_failures"] += 1
            logger.warning("XML parse failed for %s: %s", source_label, exc)
            return []

    def fetch_google_news_rss(
        self,
        query: str = "Noida crime OR accident OR fire OR protest",
        max_items: Optional[int] = None,
        timeout: int = 15,
        hl: str = "en-IN",
        ceid: str = "IN:en",
        is_hindi: bool = False,
        fetch_unresolved_bodies: bool = False,
        max_body_fetches: int = 8,
    ) -> List[Dict]:
        """
        Fetches a live Google News RSS search and returns Level 1 records.
        Optionally decodes Google News redirects to fetch lead paragraphs when headlines
        omit sector names.
        """
        if max_items is None:
            max_items = self.ingest_config.get("max_rss_items", 100)

        rss_url = "https://news.google.com/rss/search"
        params = {"q": query, "hl": hl, "gl": "IN", "ceid": ceid}

        resp = self._http_get(rss_url, params=params, timeout=timeout)
        if resp is None:
            return []

        items = self._parse_rss_xml(resp.content, f"google_news:{query[:30]}")
        self.stats["raw_google_news_items"] += len(items[:max_items])

        articles: List[Dict] = []
        body_fetches_done = 0

        for item in items[:max_items]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            outlet = (item.findtext("source") or "").strip()
            if not outlet and " - " in title:
                outlet = title.rpartition(" - ")[2].strip()
            headline = self._clean_feed_title(title, outlet)
            if not headline:
                continue

            try:
                pub_date = parsedate_to_datetime((item.findtext("pubDate") or "").strip()).astimezone().replace(tzinfo=None)
            except Exception:
                continue

            desc = self._clean_rss_description(item.findtext("description") or "")
            body = "" if desc.lower().startswith(headline[:40].lower()) else desc
            if not body:
                self.stats["bs4_rss_fallbacks"] += 1

            # If headline + description lacks a resolved Noida location, optionally decode Google URL and scrape lead
            if (
                fetch_unresolved_bodies
                and body_fetches_done < max_body_fetches
                and not self.gazetteer.resolve_location(f"{headline} {body}")
                and self._is_valid_noida_incident_text(headline, require_noida_keyword=True, is_hindi=is_hindi)
            ):
                lead = self._fetch_article_lead(link, timeout=5)
                body_fetches_done += 1
                if lead:
                    body = f"{body} {lead}".strip()

            records = self._evaluate_and_build_records(
                headline=headline,
                body=body,
                outlet=outlet or "Google News",
                link=link,
                pub_date=pub_date,
                ingest_channel="google_news_hindi_rss" if is_hindi else "google_news_rss",
                is_hindi=is_hindi,
            )
            articles.extend(records)

        return articles

    def fetch_multi_query_google_news(self, max_items_per_query: Optional[int] = None) -> List[Dict]:
        """Fans out across category and sector-cluster Google News RSS queries."""
        queries = [
            "Noida murder OR robbery OR assault OR shooting OR stabbed",
            "Noida theft OR burglary OR snatching OR loot OR fraud",
            "Noida accident OR crash OR collision OR pileup OR hit and run",
            "Noida fire OR blaze OR blast OR collapse OR gas leak",
            "Noida protest OR dharna OR blockade OR clash",
            "Noida Sector 18 OR Sector 15 OR Sector 16 OR Sector 27 OR Sector 38A",
            "Noida Sector 50 OR Sector 51 OR Sector 52 OR Sector 71 OR Sector 76 OR Sector 78",
            "Noida Sector 58 OR Sector 59 OR Sector 61 OR Sector 62 OR Sector 63",
            "Noida Sector 93A OR Sector 100 OR Sector 125 OR Sector 128 OR Sector 135 OR Sector 137 OR Sector 142",
            "Noida Expressway OR DND Flyway OR Botanical Garden OR Sector 150",
        ]

        seen_keys = set()
        collected = []
        for q in queries:
            for rec in self.fetch_google_news_rss(
                query=q,
                max_items=max_items_per_query,
                fetch_unresolved_bodies=self.ingest_config.get("decode_google_news_urls", True),
                max_body_fetches=5,
            ):
                dedup_key = f"{rec['headline'].lower().strip()}||{rec['sector_id']}"
                if dedup_key not in seen_keys:
                    seen_keys.add(dedup_key)
                    collected.append(rec)
        return collected

    def fetch_hindi_noida_rss(self, days: Optional[int] = None) -> List[Dict]:
        """
        Fetches Hindi local Noida news via Google News RSS (`hl=hi`) across incident
        categories and sector groups for significantly broader local sector coverage.
        """
        if days is None:
            days = self.ingest_config.get("lookback_days", 180)
        window = f" when:{days}d"
        hindi_queries = [
            "नोएडा सेक्टर हादसा OR दुर्घटना OR टक्कर OR मौत",
            "नोएडा सेक्टर आग OR भीषण आग OR धमाका OR ब्लास्ट",
            "नोएडा सेक्टर चोरी OR चोर OR नकबजनी OR सेंधमारी",
            "नोएडा सेक्टर लूट OR झपटमारी OR छिनैती OR बदमाश",
            "नोएडा सेक्टर हत्या OR मर्डर OR गोली OR फायरिंग OR चाकू",
            "नोएडा सेक्टर धोखाधड़ी OR ठगी OR साइबर OR गिरफ्तार",
            "नोएडा सेक्टर प्रदर्शन OR जाम OR हंगामा OR बवाल",
            "नोएडा पुलिस मुठभेड़ OR एनकाउंटर सेक्टर",
            "नोएडा एक्सप्रेसवे हादसा OR आग OR जाम",
            'नोएडा "सेक्टर 18" OR "सेक्टर 15" OR "सेक्टर 16" OR "सेक्टर 20" OR "सेक्टर 27"',
            'नोएडा "सेक्टर 37" OR "सेक्टर 39" OR "सेक्टर 49" OR "सेक्टर 50" OR "सेक्टर 51"',
            'नोएडा "सेक्टर 58" OR "सेक्टर 59" OR "सेक्टर 62" OR "सेक्टर 63" OR "सेक्टर 66"',
            'नोएडा "सेक्टर 71" OR "सेक्टर 76" OR "सेक्टर 78" OR "सेक्टर 82" OR "सेक्टर 113"',
            'नोएडा "सेक्टर 125" OR "सेक्टर 126" OR "सेक्टर 128" OR "सेक्टर 137" OR "सेक्टर 142"',
            "नोएडा बरौला OR भंगेल OR मामूरा OR सलारपुर OR छलेरा",
            "नोएडा हरौला OR निठारी OR सरफाबाद OR परथला OR खोड़ा",
        ]
        seen_keys = set()
        collected = []
        decode_gnews = self.ingest_config.get("decode_google_news_urls", True)
        for q in hindi_queries:
            for rec in self.fetch_google_news_rss(
                query=q + window,
                hl="hi",
                ceid="IN:hi",
                is_hindi=True,
                fetch_unresolved_bodies=decode_gnews,
                max_body_fetches=4,
            ):
                dedup_key = f"{rec['headline'].lower().strip()}||{rec['sector_id']}"
                if dedup_key not in seen_keys:
                    seen_keys.add(dedup_key)
                    collected.append(rec)
        return collected

    def fetch_publisher_rss(self, fetch_body: bool = True, timeout: int = 10) -> List[Dict]:
        """
        Fetches direct publisher RSS feeds for Noida/NCR news outlets.
        Dedicated Noida feeds do not require the word 'Noida' in the headline.
        """
        feeds = [
            ("Hindustan Times", "https://www.hindustantimes.com/feeds/rss/cities/noida-news/rssfeed.xml", True),
            ("Times of India", "https://timesofindia.indiatimes.com/rssfeeds/2128839596.cms", False),
            ("Indian Express", "https://indianexpress.com/section/cities/delhi/feed/", False),
        ]

        seen_keys = set()
        articles = []
        for outlet, feed_url, is_noida_dedicated in feeds:
            resp = self._http_get(feed_url, timeout=timeout)
            if resp is None:
                continue
            items = self._parse_rss_xml(resp.content, f"publisher:{outlet}")
            self.stats["raw_publisher_rss_items"] += len(items[:80])

            for item in items[:80]:
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                headline = self._clean_feed_title(title, outlet)
                if not headline:
                    continue

                try:
                    pub_date = parsedate_to_datetime((item.findtext("pubDate") or "").strip()).astimezone().replace(tzinfo=None)
                except Exception:
                    pub_date = datetime.now()

                body = self._clean_rss_description(item.findtext("description") or "")
                combined_preview = f"{headline} {body}".lower()
                if fetch_body and (is_noida_dedicated or "noida" in combined_preview) and not self.gazetteer.resolve_location(combined_preview):
                    lead = self._fetch_article_lead(link)
                    if lead:
                        body = f"{body} {lead}".strip()
                elif body:
                    self.stats["bs4_rss_fallbacks"] += 1

                records = self._evaluate_and_build_records(
                    headline=headline,
                    body=body,
                    outlet=outlet,
                    link=link,
                    pub_date=pub_date,
                    ingest_channel="publisher_rss",
                    require_noida_keyword=False if is_noida_dedicated else None,
                )
                for rec in records:
                    key = f"{rec['headline'].lower().strip()}||{rec['sector_id']}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        articles.append(rec)

        return articles

    @staticmethod
    def _unwrap_bing_url(link: str) -> str:
        """Extracts the direct publisher URL from Bing News RSS apiclick redirect links."""
        if "bing.com/news/apiclick.aspx" in link:
            qs = parse_qs(urlparse(link).query)
            if "url" in qs and qs["url"]:
                return qs["url"][0]
        return link

    def fetch_bing_news_rss(
        self,
        queries: Optional[List[str]] = None,
        fetch_body: bool = True,
        timeout: int = 10,
    ) -> List[Dict]:
        """Fetches Bing News RSS (`format=rss`) and scrapes lead paragraphs from direct URLs."""
        if queries is None:
            queries = [
                "Noida crime robbery murder assault",
                "Noida theft burglary snatching fraud",
                "Noida road accident crash collision",
                "Noida fire blast collapse hazard",
                "Noida protest dharna blockade police",
                "Noida Sector 18 OR Sector 62 OR Sector 63 incident",
                "Noida Sector 50 OR Sector 76 OR Sector 137 OR Expressway",
            ]

        rss_url = "https://www.bing.com/news/search"
        seen_keys = set()
        articles = []

        for q in queries:
            resp = self._http_get(rss_url, params={"q": q, "format": "rss", "mkt": "en-IN"}, timeout=timeout)
            if resp is None:
                continue
            items = self._parse_rss_xml(resp.content, f"bing:{q[:30]}")
            self.stats["raw_bing_news_items"] += len(items[:50])

            for item in items[:50]:
                title = (item.findtext("title") or "").strip()
                raw_link = (item.findtext("link") or "").strip()
                link = self._unwrap_bing_url(raw_link)
                outlet = (item.findtext("source") or "").strip() or urlparse(link).netloc.replace("www.", "") or "Bing News"
                headline = self._clean_feed_title(title, outlet)
                if not headline:
                    continue

                try:
                    pub_date = parsedate_to_datetime((item.findtext("pubDate") or "").strip()).astimezone().replace(tzinfo=None)
                except Exception:
                    pub_date = datetime.now()

                body = self._clean_rss_description(item.findtext("description") or "")
                combined_preview = f"{headline} {body}".lower()

                if fetch_body and not self.gazetteer.resolve_location(combined_preview):
                    if "noida" in combined_preview or self.INCIDENT_TERMS.search(combined_preview):
                        lead = self._fetch_article_lead(link)
                        if lead:
                            body = f"{body} {lead}".strip()
                elif body:
                    self.stats["bs4_rss_fallbacks"] += 1

                records = self._evaluate_and_build_records(
                    headline=headline,
                    body=body,
                    outlet=outlet,
                    link=link,
                    pub_date=pub_date,
                    ingest_channel="bing_news_rss",
                )
                for rec in records:
                    key = f"{rec['headline'].lower().strip()}||{rec['sector_id']}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        articles.append(rec)

        return articles

    def fetch_gdelt_news(self, timespan: Optional[str] = None, max_records: int = 250, timeout: int = 15) -> List[Dict]:
        """Queries the free GDELT 2.0 Doc API (English + Hindi) for historical Noida incidents."""
        if timespan is None:
            days = min(self.ingest_config.get("lookback_days", 180), 90)
            timespan = f"{days}d"

        api_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        gdelt_queries = [
            ("Noida (crime OR accident OR fire OR theft OR robbery OR snatching OR protest OR murder) sourcelang:eng", False),
            ("Noida (sector OR expressway) (police OR arrested OR fire OR crash OR dead) sourcelang:eng", False),
            ("Noida sourcelang:hin", True),
        ]

        seen_keys = set()
        articles = []
        for q_str, is_hi in gdelt_queries:
            params = {
                "query": q_str,
                "mode": "ArtList",
                "maxrecords": str(max_records),
                "timespan": timespan,
                "format": "json",
            }
            resp = self._http_get(api_url, params=params, timeout=timeout)
            if resp is None:
                continue
            try:
                data = resp.json()
            except ValueError as exc:
                self.stats["bs4_parse_failures"] += 1
                logger.warning("GDELT JSON decode failed: %s", exc)
                continue

            raw_articles = data.get("articles", [])
            self.stats["raw_gdelt_items"] += len(raw_articles)

            for item in raw_articles:
                headline = (item.get("title") or "").strip()
                link = (item.get("url") or "").strip()
                outlet = (item.get("domain") or "GDELT News").strip()
                seendate = (item.get("seendate") or "").strip()

                if not headline:
                    continue
                try:
                    pub_date = datetime.strptime(seendate[:15], "%Y%m%dT%H%M%S")
                except Exception:
                    pub_date = datetime.now()

                records = self._evaluate_and_build_records(
                    headline=headline,
                    body="",
                    outlet=outlet,
                    link=link,
                    pub_date=pub_date,
                    ingest_channel="gdelt_doc_api",
                    is_hindi=is_hi,
                )
                for rec in records:
                    key = f"{rec['headline'].lower().strip()}||{rec['sector_id']}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        articles.append(rec)

        return articles

    def _parse_article_page(self, url: str, outlet: str, timeout: int = 6) -> Optional[Dict]:
        """Fetches a direct article page and runs layered BS4 extraction."""
        resp = self._http_get(url, timeout=timeout)
        if resp is None or "text/html" not in resp.headers.get("Content-Type", ""):
            return None
        parsed = self.extract_article_from_html(resp.text, url=url, outlet=outlet)
        if not parsed.get("headline") or len(parsed["headline"]) < 15:
            return None
        if parsed["pub_date"] is None:
            parsed["pub_date"] = datetime.now()
        return parsed

    def fetch_noida_city_pages(self, max_articles: Optional[int] = None, timeout: int = 8) -> List[Dict]:
        """
        Scrapes dedicated Noida city index pages (Times of India, Hindustan Times,
        LiveHindustan, and Amar Ujala Noida) using layered BS4/JSON-LD extraction.
        """
        if max_articles is None:
            max_articles = self.ingest_config.get("max_city_articles", 250)

        index_pages = [
            # English Noida City Pages
            ("Times of India", "https://timesofindia.indiatimes.com/city/noida", "/city/noida/", "https://timesofindia.indiatimes.com", False),
            ("Times of India", "https://timesofindia.indiatimes.com/city/noida/2", "/city/noida/", "https://timesofindia.indiatimes.com", False),
            ("Times of India", "https://timesofindia.indiatimes.com/city/noida/3", "/city/noida/", "https://timesofindia.indiatimes.com", False),
            ("Times of India", "https://timesofindia.indiatimes.com/city/noida/4", "/city/noida/", "https://timesofindia.indiatimes.com", False),
            ("Times of India", "https://timesofindia.indiatimes.com/city/noida/5", "/city/noida/", "https://timesofindia.indiatimes.com", False),
            ("Times of India", "https://timesofindia.indiatimes.com/city/noida/6", "/city/noida/", "https://timesofindia.indiatimes.com", False),
            ("Times of India", "https://timesofindia.indiatimes.com/city/noida/7", "/city/noida/", "https://timesofindia.indiatimes.com", False),
            ("Times of India", "https://timesofindia.indiatimes.com/city/noida/8", "/city/noida/", "https://timesofindia.indiatimes.com", False),
            ("Hindustan Times", "https://www.hindustantimes.com/cities/noida-news", "/cities/noida-news/", "https://www.hindustantimes.com", False),
            ("Hindustan Times", "https://www.hindustantimes.com/cities/noida-news/page-2", "/cities/noida-news/", "https://www.hindustantimes.com", False),
            ("Hindustan Times", "https://www.hindustantimes.com/cities/noida-news/page-3", "/cities/noida-news/", "https://www.hindustantimes.com", False),
            ("Hindustan Times", "https://www.hindustantimes.com/cities/noida-news/page-4", "/cities/noida-news/", "https://www.hindustantimes.com", False),
            # Hindi Noida City Pages
            ("LiveHindustan", "https://www.livehindustan.com/ncr/noida/news", "/ncr/noida/", "https://www.livehindustan.com", True),
            ("LiveHindustan", "https://www.livehindustan.com/ncr/noida/news/page-2", "/ncr/noida/", "https://www.livehindustan.com", True),
            ("Amar Ujala", "https://www.amarujala.com/delhi-ncr/noida", "/delhi-ncr/noida/", "https://www.amarujala.com", True),
            ("Amar Ujala", "https://www.amarujala.com/delhi-ncr/noida/2", "/delhi-ncr/noida/", "https://www.amarujala.com", True),
        ]

        candidate_urls: List[Tuple[str, str, bool]] = []
        seen_urls = set()
        for outlet, page_url, path_prefix, base_origin, is_hi in index_pages:
            resp = self._http_get(page_url, timeout=timeout)
            if resp is None:
                continue
            try:
                soup = BeautifulSoup(resp.text, "html.parser")
            except Exception as exc:
                self.stats["bs4_parse_failures"] += 1
                logger.warning("Failed parsing index page %s: %s", page_url, exc)
                continue

            for a in soup.find_all("a", href=True):
                href = a["href"].split("?")[0].split("#")[0]
                if path_prefix not in href:
                    continue
                if any(skip in href for skip in ("/videos/", "/photos/", "/photostory/", "/page-", "/web-stories/")):
                    continue
                full_url = href if href.startswith("http") else f"{base_origin}{href}"
                if full_url not in seen_urls and len(full_url) > len(base_origin) + len(path_prefix) + 8:
                    seen_urls.add(full_url)
                    candidate_urls.append((outlet, full_url, is_hi))

        self.stats["raw_city_page_urls"] += len(candidate_urls[:max_articles])

        articles = []
        seen_keys = set()
        for outlet, url, is_hi in candidate_urls[:max_articles]:
            parsed = self._parse_article_page(url, outlet)
            if not parsed:
                continue

            records = self._evaluate_and_build_records(
                headline=parsed["headline"],
                body=parsed["body"],
                outlet=outlet,
                link=url,
                pub_date=parsed["pub_date"],
                ingest_channel="city_page_scraper_hindi" if is_hi else "city_page_scraper",
                require_noida_keyword=False,
                is_hindi=is_hi,
            )
            for rec in records:
                key = f"{rec['headline'].lower().strip()}||{rec['sector_id']}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    articles.append(rec)

        return articles

    def collect_all_sources(
        self,
        include_gdelt: bool = True,
        include_publishers: bool = True,
        include_bing: bool = True,
        include_city_pages: bool = True,
        include_hindi: Optional[bool] = None,
    ) -> Dict[str, List[Dict]]:
        """Runs all configured live news channels and returns per-channel results."""
        if include_hindi is None:
            include_hindi = self.ingest_config.get("enable_hindi_feeds", True)

        return {
            "noida_city_pages": self.fetch_noida_city_pages() if include_city_pages else [],
            "publisher_rss": self.fetch_publisher_rss() if include_publishers else [],
            "bing_news_rss": self.fetch_bing_news_rss() if include_bing else [],
            "gdelt_historical": self.fetch_gdelt_news() if include_gdelt else [],
            "google_news_multi": self.fetch_multi_query_google_news(),
            "google_news_hindi": self.fetch_hindi_noida_rss() if include_hindi else [],
        }
