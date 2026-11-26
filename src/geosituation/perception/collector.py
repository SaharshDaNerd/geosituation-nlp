import re
import json
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urlparse, parse_qs
import requests
from bs4 import BeautifulSoup
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
        r"\b(kill(ed|ing|s)?|murder(ed)?|found dead|dead body|(one|two|three|four|five|\d+) dead|death|died|dies|"
        r"body found|shot|shots|fired|gunshot|shoot(ing|out)?|stabb(ed|ing)|knife|"
        r"assault(ed)?|attack(ed)?|rap(e|ed)|molest(ed|ation)?|eve[- ]teasing|harass(ed|ment)?|kidnap(ped|ping)?|abduct(ed|ion)?|"
        r"rob(bed|bery|bers?)|loot(ed)?|thie(f|ves)|theft|stole(n)?|snatch(ed|ing|ers?)?|burglar(y|s|ies)?|"
        r"fraud|scam|cheat(ed|ing)?|duped|conman|cybercrime|cyber fraud|ransom|extort(ion|ed)?|"
        r"arrest(ed|s)?|held|nabbed|detained|custody|booked|fir|case registered|gang|busted|encounter|miscreants?|"
        r"accident|mishap|crash(ed|es)?|collid(e|ed)|collision|mow(ed)? down|hit[- ]and[- ]run|overturn(ed|s)?|"
        r"rams?|rammed|run over|pile-?up|snarl|gridlock|jammed|injur(ed|ies|y)|hurt|"
        r"fire|blaze|gutted|engulfed|short[- ]circuit|explosion|blast|evacuated|"
        r"collapse(d|s)?|caved?[- ]in|electrocut(ed|ion)|drown(ed|s)?|"
        r"waterlog(ged|ging)|flood(ed|ing)?|gas leak|"
        r"protest(s|ers|ed)?|agitation|agitate|demonstration|clash(ed|es)?|ruckus|brawl|scuffle|lathi|blockade|dharna)\b",
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

    # Greater Noida localities that should not be confused with Noida sectors
    GREATER_NOIDA_ONLY = re.compile(
        r"\b(pari chowk|knowledge park|greater noida west|noida extension|bisrakh|ecotech|kasna|surajpur|jewar|"
        r"alpha\s+[12i]|beta\s+[12i]|gamma\s+[12i]|delta\s+[12i])\b",
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

    @staticmethod
    def _clean_rss_description(raw_desc: str) -> str:
        """Strips HTML markup from RSS <description> tags to recover summary text."""
        if not raw_desc:
            return ""
        text = BeautifulSoup(raw_desc, "html.parser").get_text(separator=" ")
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _fetch_article_lead(url: str, timeout: int = 4) -> str:
        """
        Best-effort extraction of the first 3 paragraphs of an article URL.
        Returns empty string on timeout, paywall, or blocked hosts.
        """
        if not url or "news.google.com" in url:
            return ""
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200 or "text/html" not in resp.headers.get("Content-Type", ""):
                return ""
            soup = BeautifulSoup(resp.text, "html.parser")
            paragraphs = [
                p.get_text(" ", strip=True)
                for p in soup.find_all("p")
                if len(p.get_text(strip=True)) > 45
            ]
            return " ".join(paragraphs[:3])
        except Exception:
            return ""

    def _is_valid_noida_incident_text(self, text: str, require_noida_keyword: bool = True) -> bool:
        """Validates that text refers to Noida (not Greater Noida-only/other NCR cities) and an actual incident."""
        low = text.lower()
        if require_noida_keyword and "noida" not in low:
            return False
        if self.GREATER_NOIDA_ONLY.search(low) and not self.gazetteer.regex_sector_pattern.search(low):
            return False
        if self.OTHER_CITIES.search(low):
            return False
        if not self.INCIDENT_TERMS.search(text) or self.EXCLUDE_TERMS.search(text):
            return False
        return True

    def _evaluate_and_build_record(
        self,
        headline: str,
        body: str,
        outlet: str,
        link: str,
        pub_date: datetime,
        ingest_channel: str,
        require_noida_keyword: bool = True,
    ) -> Optional[Dict]:
        """Shared Level 1 validation, rule-assisted categorization, and record builder."""
        combined = f"{headline}. {body}".strip()
        if not self._is_valid_noida_incident_text(combined, require_noida_keyword=require_noida_keyword):
            return None

        record = self.process_raw_article(
            headline=headline,
            body=body,
            source_url=outlet or "News Feed",
            pub_date=pub_date,
        )
        if not record:
            return None

        for cat, pattern in self.HEADLINE_CATEGORY_RULES:
            if pattern.search(headline) or pattern.search(body):
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
        record["ingest_channel"] = ingest_channel
        return record

    def fetch_google_news_rss(self, query: str = "Noida crime OR accident OR fire OR protest",
                              max_items: int = 100, timeout: int = 15) -> List[Dict]:
        """
        Fetches a live Google News RSS search and returns Level 1 records for
        real articles that (a) name Noida, (b) read as a new incident, and
        (c) resolve to a known sector or landmark. Network errors return [].
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

            try:
                pub_date = parsedate_to_datetime((item.findtext("pubDate") or "").strip()).astimezone().replace(tzinfo=None)
            except Exception:
                continue

            desc = self._clean_rss_description(item.findtext("description") or "")
            # Avoid duplicating the headline string if RSS description just echoes the title
            body = "" if desc.lower().startswith(headline[:40].lower()) else desc

            record = self._evaluate_and_build_record(
                headline=headline,
                body=body,
                outlet=outlet or "Google News",
                link=link,
                pub_date=pub_date,
                ingest_channel="google_news_rss",
            )
            if record:
                articles.append(record)

        return articles

    def fetch_multi_query_google_news(self, max_items_per_query: int = 100) -> List[Dict]:
        """
        Fans out across category-specific and sector-targeted Google News RSS queries
        to capture stories across all Noida zones and incident types.
        """
        queries = [
            # Category-focused queries
            "Noida murder OR robbery OR assault OR shooting OR stabbed",
            "Noida theft OR burglary OR snatching OR loot OR fraud",
            "Noida accident OR crash OR collision OR pileup OR hit and run",
            "Noida fire OR blaze OR blast OR collapse OR gas leak",
            "Noida protest OR dharna OR blockade OR clash",
            # Sector-cluster queries (surfaces articles where sector is in the body)
            "Noida Sector 18 OR Sector 15 OR Sector 16 OR Sector 27 OR Sector 38A",
            "Noida Sector 50 OR Sector 51 OR Sector 52 OR Sector 71 OR Sector 76 OR Sector 78",
            "Noida Sector 58 OR Sector 59 OR Sector 61 OR Sector 62 OR Sector 63",
            "Noida Sector 93A OR Sector 100 OR Sector 125 OR Sector 128 OR Sector 135 OR Sector 137 OR Sector 142",
            "Noida Expressway OR DND Flyway OR Botanical Garden OR Sector 150",
        ]

        seen_keys = set()
        collected = []
        for q in queries:
            for rec in self.fetch_google_news_rss(query=q, max_items=max_items_per_query):
                dedup_key = rec["headline"].lower().strip()
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
            try:
                resp = requests.get(feed_url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code != 200:
                    continue
                root = ET.fromstring(resp.content)
            except Exception:
                continue

            for item in root.findall(".//item")[:60]:
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                headline = self._clean_feed_title(title, outlet)
                if not headline:
                    continue

                dedup_key = headline.lower().strip()
                if dedup_key in seen_keys:
                    continue

                try:
                    pub_date = parsedate_to_datetime((item.findtext("pubDate") or "").strip()).astimezone().replace(tzinfo=None)
                except Exception:
                    pub_date = datetime.now()

                body = self._clean_rss_description(item.findtext("description") or "")
                combined_preview = f"{headline} {body}".lower()
                # If RSS description doesn't mention a sector yet, optionally grab lead paragraphs
                if fetch_body and (is_noida_dedicated or "noida" in combined_preview) and not self.gazetteer.resolve_location(combined_preview):
                    lead = self._fetch_article_lead(link)
                    if lead:
                        body = f"{body} {lead}".strip()

                record = self._evaluate_and_build_record(
                    headline=headline,
                    body=body,
                    outlet=outlet,
                    link=link,
                    pub_date=pub_date,
                    ingest_channel="publisher_rss",
                    require_noida_keyword=not is_noida_dedicated,
                )
                if record:
                    seen_keys.add(dedup_key)
                    articles.append(record)

        return articles

    @staticmethod
    def _unwrap_bing_url(link: str) -> str:
        """Extracts the direct publisher URL from Bing News RSS apiclick redirect links."""
        if "bing.com/news/apiclick.aspx" in link:
            qs = parse_qs(urlparse(link).query)
            if "url" in qs and qs["url"]:
                return qs["url"][0]
        return link

    def fetch_bing_news_rss(self, queries: Optional[List[str]] = None,
                            fetch_body: bool = True, timeout: int = 10) -> List[Dict]:
        """
        Fetches Bing News RSS feeds (`format=rss`). Unlike Google News RSS, Bing gives
        direct publisher URLs so lead paragraphs can be scraped to recover sector names.
        """
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
            try:
                resp = requests.get(
                    rss_url,
                    params={"q": q, "format": "rss", "mkt": "en-IN"},
                    timeout=timeout,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if resp.status_code != 200:
                    continue
                root = ET.fromstring(resp.content)
            except Exception:
                continue

            for item in root.findall(".//item")[:40]:
                title = (item.findtext("title") or "").strip()
                raw_link = (item.findtext("link") or "").strip()
                link = self._unwrap_bing_url(raw_link)
                outlet = (item.findtext("source") or "").strip() or urlparse(link).netloc.replace("www.", "") or "Bing News"
                headline = self._clean_feed_title(title, outlet)
                if not headline:
                    continue

                dedup_key = headline.lower().strip()
                if dedup_key in seen_keys:
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

                record = self._evaluate_and_build_record(
                    headline=headline,
                    body=body,
                    outlet=outlet,
                    link=link,
                    pub_date=pub_date,
                    ingest_channel="bing_news_rss",
                    require_noida_keyword=True,
                )
                if record:
                    seen_keys.add(dedup_key)
                    articles.append(record)

        return articles

    def fetch_gdelt_news(self, timespan: str = "30d", max_records: int = 250, timeout: int = 15) -> List[Dict]:
        """
        Queries the free GDELT 2.0 Doc API for historical Noida incident articles
        over the past `timespan` (default 30 days) to populate Level 2 baselines.
        """
        api_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": "Noida (crime OR accident OR fire OR theft OR robbery OR snatching OR protest OR murder) sourcelang:eng",
            "mode": "ArtList",
            "maxrecords": str(max_records),
            "timespan": timespan,
            "format": "json",
        }

        articles = []
        try:
            resp = requests.get(api_url, params=params, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                return articles
            data = resp.json()
        except Exception:
            return articles

        seen_keys = set()
        for item in data.get("articles", []):
            headline = (item.get("title") or "").strip()
            link = (item.get("url") or "").strip()
            outlet = (item.get("domain") or "GDELT News").strip()
            seendate = (item.get("seendate") or "").strip()

            if not headline:
                continue
            dedup_key = headline.lower().strip()
            if dedup_key in seen_keys:
                continue

            try:
                pub_date = datetime.strptime(seendate[:15], "%Y%m%dT%H%M%S")
            except Exception:
                pub_date = datetime.now()

            record = self._evaluate_and_build_record(
                headline=headline,
                body="",
                outlet=outlet,
                link=link,
                pub_date=pub_date,
                ingest_channel="gdelt_doc_api",
            )
            if record:
                seen_keys.add(dedup_key)
                articles.append(record)

        return articles

    def _parse_article_page(self, url: str, outlet: str, timeout: int = 5) -> Optional[Dict]:
        """Fetches a direct article page and extracts headline, body lead, and publish time."""
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200 or "text/html" not in resp.headers.get("Content-Type", ""):
                return None
            soup = BeautifulSoup(resp.text, "html.parser")

            h1 = soup.find("h1")
            headline = h1.get_text(" ", strip=True) if h1 else ""
            if not headline and soup.title:
                headline = self._clean_feed_title(soup.title.get_text(" ", strip=True), outlet)
            if not headline or len(headline) < 15:
                return None

            # Extract publication timestamp from standard meta tags
            pub_date = datetime.now()
            for attr, val in [("property", "article:published_time"), ("name", "publish-date"), ("itemprop", "datePublished")]:
                meta = soup.find("meta", attrs={attr: val})
                if meta and meta.get("content"):
                    try:
                        iso_str = meta["content"].strip().replace("Z", "+00:00")
                        pub_date = datetime.fromisoformat(iso_str).astimezone().replace(tzinfo=None)
                        break
                    except Exception:
                        pass

            paragraphs = [
                p.get_text(" ", strip=True)
                for p in soup.find_all("p")
                if len(p.get_text(strip=True)) > 45
            ]
            body = " ".join(paragraphs[:4])
            return {"headline": headline, "body": body, "pub_date": pub_date}
        except Exception:
            return None

    def fetch_noida_city_pages(self, max_articles: int = 60, timeout: int = 8) -> List[Dict]:
        """
        Scrapes dedicated Noida city index pages (Times of India Noida & Hindustan Times Noida)
        to recover full article bodies and older stories beyond RSS item limits.
        """
        index_pages = [
            ("Times of India", "https://timesofindia.indiatimes.com/city/noida", "/city/noida/", "https://timesofindia.indiatimes.com"),
            ("Times of India", "https://timesofindia.indiatimes.com/city/noida/2", "/city/noida/", "https://timesofindia.indiatimes.com"),
            ("Times of India", "https://timesofindia.indiatimes.com/city/noida/3", "/city/noida/", "https://timesofindia.indiatimes.com"),
            ("Hindustan Times", "https://www.hindustantimes.com/cities/noida-news", "/cities/noida-news/", "https://www.hindustantimes.com"),
            ("Hindustan Times", "https://www.hindustantimes.com/cities/noida-news/page-2", "/cities/noida-news/", "https://www.hindustantimes.com"),
        ]

        candidate_urls = []
        seen_urls = set()
        for outlet, page_url, path_prefix, base_origin in index_pages:
            try:
                resp = requests.get(page_url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code != 200:
                    continue
                soup = BeautifulSoup(resp.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = a["href"].split("?")[0].split("#")[0]
                    if path_prefix not in href:
                        continue
                    if any(skip in href for skip in ("/videos/", "/photos/", "/photostory/", "/page-")):
                        continue
                    full_url = href if href.startswith("http") else f"{base_origin}{href}"
                    if full_url not in seen_urls and len(full_url) > len(base_origin) + len(path_prefix) + 8:
                        seen_urls.add(full_url)
                        candidate_urls.append((outlet, full_url))
            except Exception:
                continue

        articles = []
        seen_headlines = set()
        for outlet, url in candidate_urls[:max_articles]:
            parsed = self._parse_article_page(url, outlet)
            if not parsed:
                continue
            dedup_key = parsed["headline"].lower().strip()
            if dedup_key in seen_headlines:
                continue

            record = self._evaluate_and_build_record(
                headline=parsed["headline"],
                body=parsed["body"],
                outlet=outlet,
                link=url,
                pub_date=parsed["pub_date"],
                ingest_channel="city_page_scraper",
                require_noida_keyword=False,
            )
            if record:
                seen_headlines.add(dedup_key)
                articles.append(record)

        return articles

    def collect_all_sources(self, include_gdelt: bool = True, include_publishers: bool = True,
                            include_bing: bool = True, include_city_pages: bool = True) -> Dict[str, List[Dict]]:
        """
        Runs all configured live news channels and returns per-channel results.
        """
        results = {
            "noida_city_pages": self.fetch_noida_city_pages() if include_city_pages else [],
            "google_news_multi": self.fetch_multi_query_google_news(),
            "bing_news_rss": self.fetch_bing_news_rss() if include_bing else [],
            "publisher_rss": self.fetch_publisher_rss() if include_publishers else [],
            "gdelt_30d": self.fetch_gdelt_news() if include_gdelt else [],
        }
        return results
