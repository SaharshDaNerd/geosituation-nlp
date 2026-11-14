import math
from datetime import datetime
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from geosituation.config import DEDUPLICATION_CONFIG

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

    def __init__(self,
                 sim_threshold: float = DEDUPLICATION_CONFIG["sim_threshold"],
                 max_distance_km: float = DEDUPLICATION_CONFIG["max_distance_km"],
                 max_time_window_hours: float = DEDUPLICATION_CONFIG["max_time_window_hours"]):
        self.sim_threshold = sim_threshold
        self.max_distance_km = max_distance_km
        self.max_time_window_hours = max_time_window_hours

    def deduplicate(self, reports: List[Dict]) -> List[Dict]:
        """
        Groups raw Level 1 incident reports into canonical deduplicated events.
        """
        if not reports:
            return []

        # Extract texts for vectorization
        texts = [r["full_text"] for r in reports]
        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)

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

                # Coreference Logic:
                # Two reports refer to the same event if:
                # 1. Same incident category
                # 2. Within temporal window (<= max_time_window_hours)
                # 3. Within spatial radius (<= max_distance_km)
                # 4. Semantic similarity >= sim_threshold OR (Same Sector and shared key entities/keywords)
                
                time_j = datetime.fromisoformat(rep_j["incident_timestamp"])
                time_diff_hours = abs((time_i - time_j).total_seconds()) / 3600.0
                if time_diff_hours > self.max_time_window_hours:
                    continue

                dist_km = haversine_distance_km(
                    rep_i["latitude"], rep_i["longitude"],
                    rep_j["latitude"], rep_j["longitude"]
                )
                if dist_km > self.max_distance_km:
                    continue

                # Check shared entities
                ent_i = set(rep_i.get("entities", {}).get("roles_and_actors", []) + 
                            rep_i.get("entities", {}).get("vehicles", []) +
                            rep_i.get("entities", {}).get("instruments_or_hazards", []))
                ent_j = set(rep_j.get("entities", {}).get("roles_and_actors", []) + 
                            rep_j.get("entities", {}).get("vehicles", []) +
                            rep_j.get("entities", {}).get("instruments_or_hazards", []))
                shared_entities = ent_i.intersection(ent_j)

                same_sector = (rep_i["sector_id"] == rep_j["sector_id"])
                text_sim = float(sim_matrix[i, j])

                is_duplicate = False
                if text_sim >= self.sim_threshold:
                    is_duplicate = True
                elif same_sector and (text_sim >= 0.15 or len(shared_entities) >= 1):
                    is_duplicate = True
                elif dist_km <= 0.8 and text_sim >= 0.20:
                    is_duplicate = True

                if not is_duplicate:
                    continue

                # Matches criteria: add to cluster
                cluster_indices.append(j)
                visited[j] = True

            # Synthesize canonical event from cluster
            canonical_event = self._synthesize_event(cluster_indices, reports)
            canonical_events.append(canonical_event)

        return canonical_events

    def _synthesize_event(self, indices: List[int], reports: List[Dict]) -> Dict:
        """Merges cluster reports into one canonical incident tuple."""
        cluster_reps = [reports[idx] for idx in indices]
        
        # Pick primary report: the one with highest model confidence
        primary = max(cluster_reps, key=lambda r: r["model_confidence"])
        
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
