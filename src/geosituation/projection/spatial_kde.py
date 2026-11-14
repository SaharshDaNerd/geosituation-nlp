import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from geosituation.config import NOIDA_BOUNDS, KDE_CONFIG

class SpatialKDEProjector:
    """
    Level 3 Projection:
    Calculates weighted continuous 2D Kernel Density surfaces across the urban landscape.
    Converts discrete point occurrences into continuous spatial risk density contours.
    """

    def __init__(self,
                 bandwidth_km: float = KDE_CONFIG["bandwidth_km"],
                 grid_res: int = KDE_CONFIG["grid_resolution"]):
        self.bandwidth_km = bandwidth_km
        self.grid_res = grid_res
        # Approx conversion: 1 degree latitude ~ 111 km, 1 degree longitude ~ 97 km at lat 28.5N
        self.h_lat = bandwidth_km / 111.0
        self.h_lon = bandwidth_km / 97.0

    def compute_density_grid(self, canonical_events: List[Dict]) -> Dict:
        """
        Generates a 2D interpolated risk density grid.
        Returns coordinate bounds, grid matrix, and top density points for heatmap rendering.
        """
        if not canonical_events:
            return {
                "bounds": NOIDA_BOUNDS,
                "max_density": 0.0,
                "heatmap_points": []
            }

        # Grid coordinates
        lats = np.linspace(NOIDA_BOUNDS["min_lat"], NOIDA_BOUNDS["max_lat"], self.grid_res)
        lons = np.linspace(NOIDA_BOUNDS["min_lon"], NOIDA_BOUNDS["max_lon"], self.grid_res)
        
        # Extract event locations and composite weights
        evt_lats = np.array([e["latitude"] for e in canonical_events])
        evt_lons = np.array([e["longitude"] for e in canonical_events])
        
        # Composite weight = severity_weight * verification_confidence
        weights = np.array([
            e.get("severity_weight", 1.0) * e.get("verification_confidence", 0.8)
            for e in canonical_events
        ])
        
        total_weight = np.sum(weights) if np.sum(weights) > 0 else 1.0

        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        density_grid = np.zeros_like(lat_grid)

        # 2D Gaussian Kernel evaluation
        for i in range(len(canonical_events)):
            d_lat = (lat_grid - evt_lats[i]) / self.h_lat
            d_lon = (lon_grid - evt_lons[i]) / self.h_lon
            sq_dist = d_lat**2 + d_lon**2
            kernel_val = np.exp(-0.5 * sq_dist)
            density_grid += weights[i] * kernel_val

        # Normalize density
        max_density = float(np.max(density_grid))
        if max_density > 0:
            norm_density = density_grid / max_density
        else:
            norm_density = density_grid

        # Format points with significant density for frontend Leaflet heatmap [lat, lon, intensity]
        heatmap_points = []
        for r in range(self.grid_res):
            for c in range(self.grid_res):
                val = float(norm_density[r, c])
                if val >= 0.08:  # Filter out trivial background noise
                    heatmap_points.append([
                        round(float(lats[r]), 4),
                        round(float(lons[c]), 4),
                        round(val, 3)
                    ])

        return {
            "bounds": NOIDA_BOUNDS,
            "max_density": round(max_density, 3),
            "total_incidents": len(canonical_events),
            "heatmap_points": heatmap_points
        }
