import math
from datetime import datetime
from typing import Dict, List, Optional
from geosituation.config import TEMPORAL_DECAY_CONFIG, INCIDENT_CATEGORIES

class SituationalIndexCalculator:
    """
    Level 2 Comprehension:
    Applies non-linear temporal decay and severity weighting to aggregate
    incidents into sector-level Situational Activity Index (SAI).
    """

    def __init__(self, half_life_days: float = TEMPORAL_DECAY_CONFIG["half_life_days"]):
        self.half_life_days = half_life_days

    def compute_temporal_weight(self, incident_time_iso: str, current_time: Optional[datetime] = None) -> float:
        """
        Computes exponential decay weight:
        w(t) = exp( - ln(2) * (current_time - incident_time) / half_life )
        """
        if current_time is None:
            current_time = datetime.now()

        incident_dt = datetime.fromisoformat(incident_time_iso)
        delta_days = (current_time - incident_dt).total_seconds() / 86400.0

        if delta_days <= 0:
            return 1.0  # Just happened or future timestamp

        decay = math.exp(- (math.log(2) * delta_days) / self.half_life_days)
        return max(0.01, min(1.0, decay))

    def calculate_sector_indexes(self, canonical_events: List[Dict], sectors: List[Dict], current_time: Optional[datetime] = None) -> Dict[str, Dict]:
        """
        Aggregates events by sector and computes the composite Situational Activity Index (SAI).
        """
        if current_time is None:
            current_time = datetime.now()

        # Initialize structure for every sector in gazetteer
        sector_map = {}
        for s in sectors:
            s_id = s["id"]
            sector_map[s_id] = {
                "sector_id": s_id,
                "sector_name": s["name"],
                "zone": s.get("zone", "Mixed Urban"),
                "latitude": s["lat"],
                "longitude": s["lon"],
                "sai_score": 0.0,
                "raw_incident_count": 0,
                "decayed_incident_count": 0.0,
                "category_breakdown": {cat: 0 for cat in INCIDENT_CATEGORIES if cat != "NON_INCIDENT"},
                "active_events": [],
                "activity_level": "LOW_ACTIVITY_NORMAL"
            }

        # Process each canonical event
        for evt in canonical_events:
            s_id = evt["sector_id"]
            if s_id not in sector_map:
                # Dynamically register unlisted sector
                sector_map[s_id] = {
                    "sector_id": s_id,
                    "sector_name": evt["sector_name"],
                    "zone": evt.get("zone", "Mixed Urban"),
                    "latitude": evt["latitude"],
                    "longitude": evt["longitude"],
                    "sai_score": 0.0,
                    "raw_incident_count": 0,
                    "decayed_incident_count": 0.0,
                    "category_breakdown": {cat: 0 for cat in INCIDENT_CATEGORIES if cat != "NON_INCIDENT"},
                    "active_events": [],
                    "activity_level": "LOW_ACTIVITY_NORMAL"
                }

            s_data = sector_map[s_id]
            t_weight = self.compute_temporal_weight(evt["incident_timestamp"], current_time)
            sev_weight = evt.get("severity_weight", 1.0)
            conf = evt.get("verification_confidence", 0.8)

            # Core SAI formula: SAI = sum( severity * confidence * decay )
            event_impact = sev_weight * conf * t_weight
            s_data["sai_score"] += event_impact
            s_data["raw_incident_count"] += 1
            s_data["decayed_incident_count"] += t_weight

            cat = evt.get("category", "")
            if cat in s_data["category_breakdown"]:
                s_data["category_breakdown"][cat] += 1

            s_data["active_events"].append({
                "event_id": evt["event_id"],
                "headline": evt["headline"],
                "category": cat,
                "timestamp": evt["incident_timestamp"],
                "impact": round(event_impact, 3),
                "temporal_weight": round(t_weight, 3)
            })

        # Finalize scores and activity level classification
        for s_id, s_data in sector_map.items():
            sai = round(s_data["sai_score"], 2)
            s_data["sai_score"] = sai
            s_data["decayed_incident_count"] = round(s_data["decayed_incident_count"], 2)

            # Categorize Situational Activity Level
            if sai < 3.0:
                s_data["activity_level"] = "LOW_ACTIVITY_NORMAL"
                s_data["status_color"] = "#22c55e"  # Green
            elif sai < 8.0:
                s_data["activity_level"] = "MODERATE_ACTIVITY"
                s_data["status_color"] = "#eab308"  # Yellow
            elif sai < 15.0:
                s_data["activity_level"] = "ELEVATED_ATTENTION"
                s_data["status_color"] = "#f97316"  # Orange
            else:
                s_data["activity_level"] = "HIGH_ANOMALY_ALERT"
                s_data["status_color"] = "#ef4444"  # Red

        return sector_map
