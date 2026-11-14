import math
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np

from geosituation.config import ST_DBSCAN_CONFIG
from geosituation.comprehension.coreference import haversine_distance_km

class STHotspotDetector:
    """
    Level 3 Projection:
    Identifies emerging spatio-temporal incident hotspots and calculates cluster drift velocity.
    """

    def __init__(self,
                 spatial_eps_km: float = ST_DBSCAN_CONFIG["spatial_eps_km"],
                 temporal_eps_days: float = ST_DBSCAN_CONFIG["temporal_eps_days"],
                 min_samples: int = ST_DBSCAN_CONFIG["min_samples"]):
        self.spatial_eps_km = spatial_eps_km
        self.temporal_eps_days = temporal_eps_days
        self.min_samples = min_samples

    def detect_hotspots(self, canonical_events: List[Dict]) -> List[Dict]:
        """
        Runs ST-DBSCAN to discover emerging localized incident clusters and calculate drift.
        """
        if len(canonical_events) < self.min_samples:
            return []

        n = len(canonical_events)
        # Parse timestamps into float days from min timestamp
        times = [datetime.fromisoformat(e["incident_timestamp"]) for e in canonical_events]
        min_time = min(times)
        time_days = [(t - min_time).total_seconds() / 86400.0 for t in times]

        # Build Spatio-Temporal Adjacency Matrix
        adj_matrix = np.zeros((n, n), dtype=bool)
        for i in range(n):
            adj_matrix[i, i] = True
            for j in range(i + 1, n):
                s_dist = haversine_distance_km(
                    canonical_events[i]["latitude"], canonical_events[i]["longitude"],
                    canonical_events[j]["latitude"], canonical_events[j]["longitude"]
                )
                t_dist = abs(time_days[i] - time_days[j])

                if s_dist <= self.spatial_eps_km and t_dist <= self.temporal_eps_days:
                    adj_matrix[i, j] = True
                    adj_matrix[j, i] = True

        # DBSCAN Cluster Discovery
        visited = [False] * n
        cluster_id = 0
        clusters = []

        for i in range(n):
            if visited[i]:
                continue
            neighbors = [j for j in range(n) if adj_matrix[i, j]]

            if len(neighbors) < self.min_samples:
                continue

            # Start new cluster
            cluster_id += 1
            current_cluster = [i]
            visited[i] = True
            queue = [idx for idx in neighbors if idx != i]

            while queue:
                curr = queue.pop(0)
                if not visited[curr]:
                    visited[curr] = True
                    current_cluster.append(curr)
                    curr_neighbors = [k for k in range(n) if adj_matrix[curr, k]]
                    if len(curr_neighbors) >= self.min_samples:
                        for nb in curr_neighbors:
                            if nb not in queue and nb not in current_cluster:
                                queue.append(nb)
                elif curr not in current_cluster:
                    current_cluster.append(curr)

            clusters.append(current_cluster)

        # Analyze each discovered cluster
        hotspot_results = []
        for c_idx, member_indices in enumerate(clusters, 1):
            cluster_events = [canonical_events[idx] for idx in member_indices]
            
            # Sort chronologically
            cluster_events.sort(key=lambda e: e["incident_timestamp"])
            
            lats = [e["latitude"] for e in cluster_events]
            lons = [e["longitude"] for e in cluster_events]
            center_lat = float(np.mean(lats))
            center_lon = float(np.mean(lons))

            # Max radius from centroid
            radius_km = max([
                haversine_distance_km(center_lat, center_lon, e["latitude"], e["longitude"])
                for e in cluster_events
            ])
            radius_km = max(0.5, round(radius_km, 2))

            # Calculate drift trajectory (Centroid of first half vs centroid of second half)
            half = len(cluster_events) // 2
            if half >= 1:
                early_lats = [e["latitude"] for e in cluster_events[:half]]
                early_lons = [e["longitude"] for e in cluster_events[:half]]
                late_lats = [e["latitude"] for e in cluster_events[half:]]
                late_lons = [e["longitude"] for e in cluster_events[half:]]

                e_center_lat, e_center_lon = np.mean(early_lats), np.mean(early_lons)
                l_center_lat, l_center_lon = np.mean(late_lats), np.mean(late_lons)

                drift_dist_km = haversine_distance_km(e_center_lat, e_center_lon, l_center_lat, l_center_lon)
                dt_early = datetime.fromisoformat(cluster_events[0]["incident_timestamp"])
                dt_late = datetime.fromisoformat(cluster_events[-1]["incident_timestamp"])
                duration_days = max(1.0, (dt_late - dt_early).total_seconds() / 86400.0)
                drift_velocity_km_day = round(drift_dist_km / duration_days, 2)

                # Direction bearing
                d_lat = l_center_lat - e_center_lat
                d_lon = l_center_lon - e_center_lon
                angle_rad = math.atan2(d_lon, d_lat)
                bearing_deg = (math.degrees(angle_rad) + 360) % 360
                direction_cardinal = self._bearing_to_cardinal(bearing_deg)
            else:
                drift_velocity_km_day = 0.0
                bearing_deg = 0.0
                direction_cardinal = "STATIONARY"

            # Dominant sector & category
            sectors_involved = list(set([e["sector_id"] for e in cluster_events]))
            categories = [e["category"] for e in cluster_events]
            dom_category = max(set(categories), key=categories.count)

            hotspot_results.append({
                "hotspot_id": f"HOTSPOT_{c_idx:02d}",
                "incident_count": len(cluster_events),
                "center_latitude": round(center_lat, 4),
                "center_longitude": round(center_lon, 4),
                "radius_km": radius_km,
                "sectors_involved": sectors_involved,
                "dominant_category": dom_category,
                "drift_velocity_km_per_day": drift_velocity_km_day,
                "drift_bearing_degrees": round(bearing_deg, 1),
                "drift_direction": direction_cardinal,
                "first_detected": cluster_events[0]["incident_timestamp"],
                "latest_incident": cluster_events[-1]["incident_timestamp"],
                "trajectory_summary": (
                    f"Emerging {dom_category} cluster across {', '.join(sectors_involved[:3])} "
                    f"with {len(cluster_events)} incidents. "
                    f"Drift: {direction_cardinal} at {drift_velocity_km_day} km/day."
                )
            })

        return hotspot_results

    def _bearing_to_cardinal(self, bearing: float) -> str:
        dirs = ["North", "North-East", "East", "South-East", "South", "South-West", "West", "North-West"]
        idx = int((bearing + 22.5) / 45.0) % 8
        return dirs[idx]
