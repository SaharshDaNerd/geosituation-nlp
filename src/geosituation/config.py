import os
from pathlib import Path
from pydantic import BaseModel
from typing import Dict

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Lightweight .env loader
def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)

_load_env_file(BASE_DIR / ".env")

# API Keys & Server Config
CARTO_API_KEY = os.getenv("CARTO_API_KEY", "")
MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN", "")
SERVER_HOST = os.getenv("HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("PORT", "8000"))

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GAZETTEER_DIR = DATA_DIR / "gazetteers"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "src" / "geosituation" / "api" / "static"

for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, GAZETTEER_DIR, MODELS_DIR, STATIC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Geographic Bounding Box: Noida, Uttar Pradesh, India
NOIDA_BOUNDS = {
    "min_lat": 28.4500,
    "max_lat": 28.6400,
    "min_lon": 77.2900,
    "max_lon": 77.4600,
    "center_lat": 28.5700,
    "center_lon": 77.3400,
}

# Incident Taxonomy & Category Definitions
INCIDENT_CATEGORIES = {
    "VIOLENT_CRIME": {
        "name": "Violent Crime",
        "description": "Physical assaults, armed robberies, homicides, gang violence, kidnappings",
        "severity_weight": 5.0,
        "color": "#ef4444",  # Red
    },
    "PROPERTY_CRIME": {
        "name": "Property / Financial Crime",
        "description": "Burglary, chain/phone snatching, vehicle theft, shoplifting, extortion, cyber fraud",
        "severity_weight": 2.5,
        "color": "#f97316",  # Orange
    },
    "TRAFFIC_INCIDENT": {
        "name": "Traffic & Road Hazard",
        "description": "Motor vehicle collisions, pedestrian hit-and-runs, expressway pileups, major gridlock",
        "severity_weight": 2.0,
        "color": "#eab308",  # Yellow
    },
    "CIVIC_HAZARD": {
        "name": "Civic & Industrial Hazard",
        "description": "Industrial fires, transformer explosions, building collapses, gas leaks, waterlogging",
        "severity_weight": 3.5,
        "color": "#a855f7",  # Purple
    },
    "PUBLIC_DISRUPTION": {
        "name": "Public Order & Protests",
        "description": "Farmer agitations, highway blockades, unauthorized rallies, strikes, section 144 alerts",
        "severity_weight": 2.0,
        "color": "#3b82f6",  # Blue
    },
    "NON_INCIDENT": {
        "name": "General News / Non-Incident",
        "description": "Infrastructure inauguration, real estate ads, sports, general governance",
        "severity_weight": 0.0,
        "color": "#94a3b8",  # Slate
    }
}

# Level 2 Parameters (Comprehension)
DEDUPLICATION_CONFIG = {
    "sim_threshold": 0.72,        # Semantic text similarity threshold (TF-IDF / Embedding)
    "max_distance_km": 1.8,       # Max distance between reported coordinates for duplicate match
    "max_time_window_hours": 60,  # Max time separation between duplicate news reports (2.5 days)
}

TEMPORAL_DECAY_CONFIG = {
    "half_life_days": 7.0,        # Incident relevance drops by 50% every 7 days
}

ANOMALY_CONFIG = {
    "baseline_window_days": 30,   # Rolling historical baseline window
    "recent_window_days": 7,      # Current evaluation window
    "z_threshold_elevated": 1.5,  # Moderate anomaly threshold
    "z_threshold_critical": 2.5,  # High anomaly threshold
}

# Level 3 Parameters (Projection)
KDE_CONFIG = {
    "bandwidth_km": 1.2,          # Spatial smoothing bandwidth (approx 1 Noida sector width)
    "grid_resolution": 60,        # 60x60 grid interpolation
}

ST_DBSCAN_CONFIG = {
    "spatial_eps_km": 2.2,        # Max spatial clustering radius
    "temporal_eps_days": 5.0,     # Max temporal clustering gap
    "min_samples": 2,             # Minimum reports to form an emerging cluster
}
