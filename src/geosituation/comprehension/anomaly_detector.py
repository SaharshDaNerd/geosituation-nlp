import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from geosituation.config import ANOMALY_CONFIG

class SectorAnomalyDetector:
    """
    Level 2 Comprehension:
    Evaluates incident frequency against historical baselines using Poisson Rate Anomaly Z-scores.
    Detects sudden spikes and deviations from normal background reporting.
    """

    def __init__(self,
                 baseline_window_days: int = ANOMALY_CONFIG["baseline_window_days"],
                 recent_window_days: int = ANOMALY_CONFIG["recent_window_days"],
                 z_elevated: float = ANOMALY_CONFIG["z_threshold_elevated"],
                 z_critical: float = ANOMALY_CONFIG["z_threshold_critical"]):
        self.baseline_window_days = baseline_window_days
        self.recent_window_days = recent_window_days
        self.z_elevated = z_elevated
        self.z_critical = z_critical

    def detect_anomalies(self, canonical_events: List[Dict], current_time: Optional[datetime] = None) -> List[Dict]:
        """
        Calculates Poisson Z-scores for each sector and returns list of detected anomalies.
        """
        if current_time is None:
            current_time = datetime.now()

        recent_cutoff = current_time - timedelta(days=self.recent_window_days)
        baseline_cutoff = current_time - timedelta(days=self.baseline_window_days)

        # Aggregate counts per sector
        recent_counts = {}
        baseline_counts = {}
        sector_categories = {}

        for evt in canonical_events:
            s_id = evt["sector_id"]
            evt_dt = datetime.fromisoformat(evt["incident_timestamp"])
            cat = evt.get("category", "UNKNOWN")

            if s_id not in sector_categories:
                sector_categories[s_id] = {}
            sector_categories[s_id][cat] = sector_categories[s_id].get(cat, 0) + 1

            if evt_dt >= recent_cutoff:
                recent_counts[s_id] = recent_counts.get(s_id, 0) + 1
            
            if evt_dt >= baseline_cutoff:
                baseline_counts[s_id] = baseline_counts.get(s_id, 0) + 1

        anomalies = []
        all_sectors = set(recent_counts.keys()).union(set(baseline_counts.keys()))

        for s_id in all_sectors:
            n_recent = recent_counts.get(s_id, 0)
            n_baseline_total = baseline_counts.get(s_id, 0)

            # Historical baseline expected over recent window
            # Scale baseline rate to recent window duration
            scale_ratio = float(self.recent_window_days) / float(self.baseline_window_days)
            expected_recent = max(0.2, n_baseline_total * scale_ratio)

            # Poisson Standardized Z-Score: Z = (Observed - Expected) / sqrt(Expected)
            z_score = (n_recent - expected_recent) / math.sqrt(expected_recent)
            z_score = round(z_score, 2)

            is_anomaly = z_score >= self.z_elevated
            severity = "NORMAL"
            if z_score >= self.z_critical:
                severity = "CRITICAL_SPIKE"
            elif z_score >= self.z_elevated:
                severity = "ELEVATED_SURGE"

            # Dominant category in this sector
            dom_cat = "Mixed Incidents"
            if s_id in sector_categories and sector_categories[s_id]:
                dom_cat = max(sector_categories[s_id].items(), key=lambda x: x[1])[0]

            anomalies.append({
                "sector_id": s_id,
                "z_score": z_score,
                "recent_count": n_recent,
                "expected_count": round(expected_recent, 2),
                "severity": severity,
                "is_anomaly": is_anomaly,
                "dominant_category": dom_cat,
                "explanation": (
                    f"{s_id} exhibits an anomaly Z-score of {z_score}σ "
                    f"({n_recent} reports in {self.recent_window_days}d vs {round(expected_recent, 1)} expected). "
                    f"Predominant pattern: {dom_cat}."
                ) if is_anomaly else f"{s_id} is operating within normal baseline variance (Z={z_score}σ)."
            })

        # Sort by Z-score descending
        anomalies.sort(key=lambda a: a["z_score"], reverse=True)
        return anomalies
