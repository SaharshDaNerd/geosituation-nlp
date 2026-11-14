import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class SituationalMomentumProjector:
    """
    Level 3 Projection:
    Calculates short-horizon incident rate trajectory (EWMA / Poisson Projection)
    and compiles the executive Endsley-compliant Situational Awareness Brief.
    """

    def __init__(self, forecast_horizon_days: int = 7):
        self.forecast_horizon_days = forecast_horizon_days

    def project_sector_trajectories(self, sector_map: Dict[str, Dict], anomalies: List[Dict]) -> List[Dict]:
        """
        Projects next-7-day expected incident volume and assigns risk momentum classifications.
        """
        anomaly_map = {a["sector_id"]: a for a in anomalies}
        trajectories = []

        for s_id, s_data in sector_map.items():
            raw_count = s_data["raw_incident_count"]
            sai = s_data["sai_score"]
            decayed = s_data["decayed_incident_count"]
            anom = anomaly_map.get(s_id, {})
            z_score = anom.get("z_score", 0.0)

            # Daily rate estimation based on decayed count
            daily_rate = decayed / 14.0  # Normalized over 2-week active envelope
            projected_7d = round(daily_rate * self.forecast_horizon_days, 1)

            # Determine momentum
            if z_score >= 1.5 or (sai >= 8.0 and projected_7d >= 2.0):
                momentum = "ACCELERATING_ACTIVITY"
                momentum_color = "#ef4444"
                recommendation = "High Priority: Allocate preemptive civic/patrol presence and traffic monitoring."
            elif sai >= 4.0 or projected_7d >= 1.0:
                momentum = "STABLE_MODERATE"
                momentum_color = "#f97316"
                recommendation = "Standard Monitoring: Maintain routine reporting surveillance."
            else:
                momentum = "LOW_STABLE"
                momentum_color = "#22c55e"
                recommendation = "Normal Operations: Minimal reported disruption."

            trajectories.append({
                "sector_id": s_id,
                "sector_name": s_data["sector_name"],
                "current_sai": sai,
                "raw_incidents": raw_count,
                "anomaly_z_score": z_score,
                "projected_7d_incidents": projected_7d,
                "momentum_trajectory": momentum,
                "momentum_color": momentum_color,
                "recommendation": recommendation
            })

        # Sort by SAI score descending
        trajectories.sort(key=lambda x: x["current_sai"], reverse=True)
        return trajectories

    def generate_endsley_executive_summary(self,
                                          canonical_events: List[Dict],
                                          sector_map: Dict[str, Dict],
                                          anomalies: List[Dict],
                                          hotspots: List[Dict],
                                          trajectories: List[Dict]) -> Dict:
        """
        Synthesizes the complete Three-Level Endsley Situational Awareness Summary.
        """
        total_events = len(canonical_events)
        multi_verified = sum(1 for e in canonical_events if e.get("is_multi_source_verified", False))

        # Level 1 Breakdown
        cat_counts = {}
        for e in canonical_events:
            c = e.get("category_name", "Other")
            cat_counts[c] = cat_counts.get(c, 0) + 1

        top_sectors_sai = sorted(sector_map.values(), key=lambda s: s["sai_score"], reverse=True)[:5]
        critical_anomalies = [a for a in anomalies if a.get("is_anomaly", False)]
        accelerating_sectors = [t for t in trajectories if t["momentum_trajectory"] == "ACCELERATING_ACTIVITY"]

        return {
            "timestamp": datetime.now().isoformat(),
            "level_1_perception": {
                "headline": "Current Environmental Cues & Reported Incidents",
                "total_canonical_events": total_events,
                "multi_source_verified_events": multi_verified,
                "category_distribution": cat_counts,
                "latest_incident": canonical_events[0]["headline"] if canonical_events else "No recent incidents"
            },
            "level_2_comprehension": {
                "headline": "Contextual Situation & Sector Baseline Deviations",
                "top_activity_sectors": [
                    {"sector": s["sector_name"], "sai": s["sai_score"], "incidents": s["raw_incident_count"]}
                    for s in top_sectors_sai if s["sai_score"] > 0
                ],
                "active_baseline_anomalies_count": len(critical_anomalies),
                "anomaly_highlights": [a["explanation"] for a in critical_anomalies[:3]],
                "system_data_confidence": "HIGH (Deduplicated multi-source corpus anchored to Noida Gazetteer)"
            },
            "level_3_projection": {
                "headline": "Spatial Trajectory, Cluster Drift & 7-Day Forecast",
                "active_spatial_hotspots_count": len(hotspots),
                "hotspot_trajectories": [h["trajectory_summary"] for h in hotspots],
                "accelerating_risk_sectors": [
                    {"sector": t["sector_name"], "projected_7d": t["projected_7d_incidents"], "momentum": t["momentum_trajectory"]}
                    for t in accelerating_sectors[:4]
                ],
                "operational_guidance": "Focus proactive surveillance on identified cluster drift corridors (Expressway and Commercial Hubs)."
            }
        }
