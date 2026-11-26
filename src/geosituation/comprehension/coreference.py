import math
import re
from datetime import datetime
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from geosituation.config import DEDUPLICATION_CONFIG

_LOCATION_NOISE_RE = re.compile(
    r"(?:\b(?:noida|sectors?|secs?\.?|sect)\s*[-]?\s*\d{0,3}[a-z]?\b|नोएडा|सेक्टर\s*[-]?\s*\d{0,3}[a-z]?)",
    re.IGNORECASE,
)

# Cross-lingual keyword normalization so Hindi & English articles about the same
# incident align in TF-IDF space (without diluting English-vs-English precision).
_HINDI_TO_EN_MAP = {
    "आग": "fire blaze", "भीषण": "massive", "लपटें": "flames", "चपेट": "engulfed",
    "फैक्ट्री": "factory", "कंपनी": "company", "यूनिट": "unit", "प्लास्टिक": "plastic",
    "सजावट": "decorative", "सजावटी": "decorative", "इमारत": "building", "दुकान": "shop", "दुकानों": "shops",
    "होटल": "hotel", "मॉल": "mall", "अस्पताल": "hospital", "सोसाइटी": "society", "फ्लैट": "flat",
    "चोरी": "theft", "चोर": "thieves", "चोरों": "thieves", "लूट": "robbery loot", "लुटेरे": "robbers",
    "झपटमारी": "snatching", "छिनैती": "snatching", "धोखाधड़ी": "fraud", "ठगी": "fraud scam",
    "क्रिप्टो": "crypto", "साइबर": "cyber", "बैंक": "bank",
    "हत्या": "murder", "मर्डर": "murder", "मौत": "dead", "मृत": "dead", "शव": "body found", "लाश": "body found",
    "गोली": "shot", "फायरिंग": "firing", "चाकू": "stabbed knife", "हमला": "attack", "मारपीट": "assault",
    "गिरफ्तार": "arrested", "दबोचा": "arrested", "मुठभेड़": "encounter", "एनकाउंटर": "encounter",
    "हादसा": "accident", "दुर्घटना": "accident", "टक्कर": "collision", "पलटी": "overturned", "घायल": "injured",
    "बाइक": "bike motorcycle", "मोटरसाइकिल": "motorcycle", "स्कूटी": "scooter", "कार": "car",
    "बस": "bus", "ट्रक": "truck", "डंपर": "dumper", "ऑटो": "auto", "पेड़": "tree",
    "प्रदर्शन": "protest", "जाम": "jam", "हंगामा": "ruckus", "बवाल": "clash",
    "सिलेंडर": "cylinder", "ब्लास्ट": "blast", "धमाका": "blast", "ट्रांसफार्मर": "transformer",
    "लाख": "lakh", "करोड़": "crore", "रुपये": "rs", "मेट्रो": "metro", "एक्सप्रेसवे": "expressway",
    "छात्र": "student", "इंजीनियर": "engineer", "महिला": "woman", "बुजुर्ग": "elderly", "युवक": "youth",
}


def _normalize_for_similarity(text: str) -> str:
    """Strips city/sector boilerplate, normalizes currency/synonyms, and maps Hindi cues to English."""
    cleaned = _LOCATION_NOISE_RE.sub(" ", text or "")
    # Normalize common currency & spelling variants (e.g. 24L -> 24 lakh, cryptocurrency -> crypto)
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s*[lL]\b", r"\1 lakh", cleaned)
    cleaned = re.sub(r"\bcryptocurrenc(?:y|ies)\b", "crypto", cleaned, flags=re.I)
    cleaned = re.sub(r"\bjewellery\b", "jewelry", cleaned, flags=re.I)
    cleaned = re.sub(r"\bduped\b", "fraud", cleaned, flags=re.I)
    cleaned = re.sub(r"\bscam\b", "fraud", cleaned, flags=re.I)
    cleaned = re.sub(r"\b(?:â‚¹|₹)\s*", "", cleaned)

    tokens = []
    for tok in re.split(r"[\s,.;:!?()\"'’‘/\\-]+", cleaned):
        if not tok:
            continue
        if tok in _HINDI_TO_EN_MAP:
            tokens.append(_HINDI_TO_EN_MAP[tok])
        else:
            tokens.append(tok)
    normalized = " ".join(tokens).strip()
    return normalized or text


def _safe_cosine_matrix(texts: List[str]):
    try:
        vectorizer = TfidfVectorizer(max_features=1200, stop_words="english")
        mat = vectorizer.fit_transform(texts)
        return cosine_similarity(mat)
    except ValueError:
        n = len(texts)
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great circle distance between two points on Earth in kilometers."""
    R = 6371.0  # Earth radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class CrossDocumentDeduplicator:
    """
    Level 2 Comprehension:
    Fuses multiple reports of the same real-world incident into a single canonical event.
    Avoids duplicate counting and computes multi-source verification confidence.
    """

    GENERIC_ENTITIES = {"Noida Police", "Traffic Police", "Fire Department", "Fire Tenders", "District Administration"}

    def __init__(self,
                 sim_threshold: float = DEDUPLICATION_CONFIG["sim_threshold"],
                 max_distance_km: float = DEDUPLICATION_CONFIG["max_distance_km"],
                 max_time_window_hours: float = DEDUPLICATION_CONFIG["max_time_window_hours"],
                 same_sector_sim_threshold: float = DEDUPLICATION_CONFIG["same_sector_sim_threshold"],
                 close_dist_km: float = DEDUPLICATION_CONFIG["close_dist_km"],
                 close_sim_threshold: float = DEDUPLICATION_CONFIG["close_sim_threshold"],
                 entity_overlap_sim_floor: float = DEDUPLICATION_CONFIG["entity_overlap_sim_floor"]):
        self.sim_threshold = sim_threshold
        self.max_distance_km = max_distance_km
        self.max_time_window_hours = max_time_window_hours
        self.same_sector_sim_threshold = same_sector_sim_threshold
        self.close_dist_km = close_dist_km
        self.close_sim_threshold = close_sim_threshold
        self.entity_overlap_sim_floor = entity_overlap_sim_floor
        self.last_merge_stats: Dict[str, int] = {}

    def deduplicate(self, reports: List[Dict]) -> List[Dict]:
        """
        Groups raw Level 1 incident reports into canonical deduplicated events.
        """
        self.last_merge_stats = {
            "input_reports": len(reports),
            "output_events": 0,
            "merged_reports": 0,
            "merged_by_high_sim": 0,
            "merged_by_same_sector_sim": 0,
            "merged_by_entity_sim": 0,
            "merged_by_close_proximity": 0,
        }
        if not reports:
            return []

        # Normalize headlines and full texts separately (stripping city/sector boilerplate
        # and bridging Hindi<->English cues) so long bodies don't dilute headline similarity.
        headline_texts = [_normalize_for_similarity(r.get("headline") or r["full_text"]) for r in reports]
        full_texts = [_normalize_for_similarity(r["full_text"]) for r in reports]
        head_sim_matrix = _safe_cosine_matrix(headline_texts)
        full_sim_matrix = _safe_cosine_matrix(full_texts)

        n = len(reports)
        visited = [False] * n
        canonical_events = []

        for i in range(n):
            if visited[i]:
                continue

            cluster_indices = [i]
            visited[i] = True
            rep_i = reports[i]
            time_i = datetime.fromisoformat(rep_i["incident_timestamp"])

            for j in range(i + 1, n):
                if visited[j]:
                    continue
                rep_j = reports[j]

                # 1. Category must match
                if rep_i["category"] != rep_j["category"]:
                    continue

                # 2. Temporal Proximity check
                time_j = datetime.fromisoformat(rep_j["incident_timestamp"])
                time_diff_hours = abs((time_i - time_j).total_seconds()) / 3600.0
                if time_diff_hours > self.max_time_window_hours:
                    continue

                # 3. Spatial Distance check
                dist_km = haversine_distance_km(
                    rep_i["latitude"], rep_i["longitude"],
                    rep_j["latitude"], rep_j["longitude"]
                )
                if dist_km > self.max_distance_km:
                    continue

                # 4. Specific shared entities (excluding ultra-generic agency mentions)
                ent_i = set(
                    rep_i.get("entities", {}).get("roles_and_actors", [])
                    + rep_i.get("entities", {}).get("vehicles", [])
                    + rep_i.get("entities", {}).get("instruments_or_hazards", [])
                ) - self.GENERIC_ENTITIES
                ent_j = set(
                    rep_j.get("entities", {}).get("roles_and_actors", [])
                    + rep_j.get("entities", {}).get("vehicles", [])
                    + rep_j.get("entities", {}).get("instruments_or_hazards", [])
                ) - self.GENERIC_ENTITIES
                shared_entities = ent_i.intersection(ent_j)

                same_sector = (rep_i["sector_id"] == rep_j["sector_id"])
                text_sim = max(float(head_sim_matrix[i][j]), float(full_sim_matrix[i][j]))

                merge_reason = None
                if text_sim >= self.sim_threshold:
                    merge_reason = "merged_by_high_sim"
                elif same_sector and text_sim >= self.same_sector_sim_threshold:
                    merge_reason = "merged_by_same_sector_sim"
                elif same_sector and len(shared_entities) >= 1 and text_sim >= self.entity_overlap_sim_floor:
                    merge_reason = "merged_by_entity_sim"
                elif dist_km <= self.close_dist_km and text_sim >= self.close_sim_threshold:
                    merge_reason = "merged_by_close_proximity"

                if not merge_reason:
                    continue

                # Matches criteria: add to cluster
                self.last_merge_stats[merge_reason] += 1
                self.last_merge_stats["merged_reports"] += 1
                cluster_indices.append(j)
                visited[j] = True

            # Synthesize canonical event from cluster
            canonical_event = self._synthesize_event(cluster_indices, reports)
            canonical_events.append(canonical_event)

        self.last_merge_stats["output_events"] = len(canonical_events)
        return canonical_events

    def _synthesize_event(self, indices: List[int], reports: List[Dict]) -> Dict:
        """Merges cluster reports into one canonical incident tuple."""
        cluster_reps = [reports[idx] for idx in indices]
        
        # Pick primary report: prefer English headline when confidence is tied, then highest confidence
        primary = max(
            cluster_reps,
            key=lambda r: (0 if re.search(r"[\u0900-\u097f]", r.get("headline", "")) else 1, r["model_confidence"]),
        )
        
        # Multi-source Verification Confidence: V(E) = 1 - product(1 - p_i)
        unconfidence_prod = 1.0
        for r in cluster_reps:
            unconfidence_prod *= (1.0 - r["model_confidence"])
        verification_confidence = 1.0 - unconfidence_prod
        verification_confidence = min(0.999, round(verification_confidence, 4))

        # Union of entities
        all_orgs = set()
        all_vehicles = set()
        all_instruments = set()
        all_roles = set()
        for r in cluster_reps:
            ent = r.get("entities", {})
            all_orgs.update(ent.get("organizations", []))
            all_vehicles.update(ent.get("vehicles", []))
            all_instruments.update(ent.get("instruments_or_hazards", []))
            all_roles.update(ent.get("roles_and_actors", []))

        # Sources & Citations
        sources = [
            {
                "report_id": r["report_id"],
                "source_url": r["source_url"],
                "headline": r["headline"],
                "published_at": r["published_at"]
            }
            for r in cluster_reps
        ]

        event_id = f"EVT_{primary['report_id'][4:]}"

        return {
            "event_id": event_id,
            "headline": primary["headline"],
            "summary": primary["full_text"][:280] + ("..." if len(primary["full_text"]) > 280 else ""),
            "category": primary["category"],
            "category_name": primary["category_name"],
            "severity_weight": primary["severity_weight"],
            "sector_id": primary["sector_id"],
            "sector_name": primary["sector_name"],
            "zone": primary["zone"],
            "latitude": primary["latitude"],
            "longitude": primary["longitude"],
            "incident_timestamp": primary["incident_timestamp"],
            "report_count": len(cluster_reps),
            "verification_confidence": verification_confidence,
            "is_multi_source_verified": len(cluster_reps) > 1,
            "entities": {
                "organizations": sorted(list(all_orgs)),
                "vehicles": sorted(list(all_vehicles)),
                "instruments_or_hazards": sorted(list(all_instruments)),
                "roles_and_actors": sorted(list(all_roles))
            },
            "sources": sources
        }
