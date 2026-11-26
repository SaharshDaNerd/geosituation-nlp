import math
from datetime import datetime
from typing import Dict, List, Optional
from geosituation.config import TEMPORAL_DECAY_CONFIG, INCIDENT_CATEGORIES
from geosituation.comprehension.coreference import haversine_distance_km

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

        # Process direct events inside each sector
        event_impacts = []
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

            event_impacts.append((s_id, evt["latitude"], evt["longitude"], event_impact))

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

        # Add Gaussian spatial spillover from nearby incidents (within 3.0 km, sigma = 1.2 km)
        # so adjacent sectors reflect proximity risk rather than appearing artificially blank.
        spillover_sigma_km = 1.2
        spillover_radius_km = 3.0
        spillover_weight = 0.55

        for s_id, s_data in sector_map.items():
            direct_sai = s_data["sai_score"]
            spillover = 0.0
            s_lat, s_lon = s_data["latitude"], s_data["longitude"]
            for evt_sector, e_lat, e_lon, e_impact in event_impacts:
                if evt_sector == s_id:
                    continue
                dist_km = haversine_distance_km(s_lat, s_lon, e_lat, e_lon)
                if dist_km <= spillover_radius_km:
                    kernel = math.exp(-(dist_km ** 2) / (2.0 * (spillover_sigma_km ** 2)))
                    spillover += e_impact * spillover_weight * kernel

            s_data["direct_sai"] = round(direct_sai, 2)
            s_data["spillover_sai"] = round(spillover, 2)
            s_data["sai_score"] = direct_sai + spillover

        # Compute city-wide peak for relative 0-100 risk normalization
        max_city_sai = max((s["sai_score"] for s in sector_map.values()), default=0.0)
        norm_denom = max(3.5, max_city_sai)

        # Finalize scores, risk score (0-100), and safety/activity classification
        for s_id, s_data in sector_map.items():
            sai = round(s_data["sai_score"], 2)
            s_data["sai_score"] = sai
            s_data["decayed_incident_count"] = round(s_data["decayed_incident_count"], 2)

            if sai <= 0.01:
                risk_score = 5
            else:
                direct_bonus = 10 if s_data["raw_incident_count"] > 0 else 0
                risk_score = min(99, max(8, int(round((sai / norm_denom) * 85 + direct_bonus))))
            s_data["risk_score"] = risk_score

            if s_data["raw_incident_count"] > 0:
                s_data["data_basis"] = "Direct Observed Reports"
            elif s_data["spillover_sai"] >= 0.15:
                s_data["data_basis"] = "Nearby Incident Spillover Estimate"
            else:
                s_data["data_basis"] = "No Recent Reports (Baseline)"

            # Calibrated Safety / Activity Tiers (ensures genuine high-activity sectors surface as Red)
            if sai >= 3.0 or risk_score >= 72:
                s_data["activity_level"] = "HIGH_ANOMALY_ALERT"
                s_data["safety_status"] = "Higher-Risk Zone"
                s_data["status_color"] = "#ef4444"  # Red
            elif sai >= 1.7 or risk_score >= 48:
                s_data["activity_level"] = "ELEVATED_ATTENTION"
                s_data["safety_status"] = "Elevated Risk"
                s_data["status_color"] = "#f97316"  # Orange
            elif sai >= 0.65 or risk_score >= 25:
                s_data["activity_level"] = "MODERATE_ACTIVITY"
                s_data["safety_status"] = "Moderate Caution"
                s_data["status_color"] = "#eab308"  # Yellow
            else:
                s_data["activity_level"] = "LOW_ACTIVITY_NORMAL"
                s_data["safety_status"] = "Relatively Safer"
                s_data["status_color"] = "#22c55e"  # Green

        return sector_map
