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

def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    try:
        return float(val.strip())
    except ValueError:
        return default


# Tunable Strictness Presets (strict | balanced | high_recall)
PIPELINE_PRESETS: Dict[str, Dict] = {
    "strict": {
        "min_text_length": 25,
        "require_noida_keyword": True,
        "allow_rss_summary_only": False,
        "allow_landmark_match": True,
        "min_spatial_confidence": 0.85,
        "min_classifier_confidence": 0.35,
        "allow_multi_sector_expansion": False,
        "max_sectors_per_article": 1,
        "dedup_sim_threshold": 0.72,
        "dedup_same_sector_sim": 0.38,
        "dedup_close_dist_km": 0.8,
        "dedup_close_sim": 0.35,
        "dedup_entity_sim_floor": 0.22,
        "dedup_max_distance_km": 1.8,
        "dedup_max_time_window_hours": 60.0,
        "lookback_days": 60,
        "max_city_articles": 60,
        "max_rss_items": 100,
        "enable_hindi_feeds": False,
        "decode_google_news_urls": True,
    },
    "balanced": {
        "min_text_length": 15,
        "require_noida_keyword": True,
        "allow_rss_summary_only": True,
        "allow_landmark_match": True,
        "min_spatial_confidence": 0.75,
        "min_classifier_confidence": 0.15,
        "allow_multi_sector_expansion": True,
        "max_sectors_per_article": 2,
        "dedup_sim_threshold": 0.68,
        "dedup_same_sector_sim": 0.34,
        "dedup_close_dist_km": 0.8,
        "dedup_close_sim": 0.30,
        "dedup_entity_sim_floor": 0.18,
        "dedup_max_distance_km": 1.8,
        "dedup_max_time_window_hours": 60.0,
        "lookback_days": 180,
        "max_city_articles": 250,
        "max_rss_items": 100,
        "enable_hindi_feeds": True,
        "decode_google_news_urls": True,
    },
    "high_recall": {
        "min_text_length": 10,
        "require_noida_keyword": False,
        "allow_rss_summary_only": True,
        "allow_landmark_match": True,
        "min_spatial_confidence": 0.65,
        "min_classifier_confidence": 0.0,
        "allow_multi_sector_expansion": True,
        "max_sectors_per_article": 3,
        "dedup_sim_threshold": 0.74,
        "dedup_same_sector_sim": 0.30,
        "dedup_close_dist_km": 0.8,
        "dedup_close_sim": 0.30,
        "dedup_entity_sim_floor": 0.16,
        "dedup_max_distance_km": 1.8,
        "dedup_max_time_window_hours": 48.0,
        "lookback_days": 180,
        "max_city_articles": 350,
        "max_rss_items": 100,
        "enable_hindi_feeds": True,
        "decode_google_news_urls": True,
    },
}


def build_pipeline_config(mode: str = "") -> Dict[str, Dict]:
    """Builds grounding, deduplication, and ingestion configs from PIPELINE_MODE + env overrides."""
    selected = (mode or os.getenv("PIPELINE_MODE", "balanced")).strip().lower()
    if selected not in PIPELINE_PRESETS:
        selected = "balanced"
    p = PIPELINE_PRESETS[selected]

    grounding = {
        "pipeline_mode": selected,
        "min_text_length": _env_int("MIN_TEXT_LENGTH", p["min_text_length"]),
        "require_noida_keyword": _env_bool("REQUIRE_NOIDA_KEYWORD", p["require_noida_keyword"]),
        "allow_rss_summary_only": _env_bool("ALLOW_RSS_SUMMARY_ONLY", p["allow_rss_summary_only"]),
        "allow_landmark_match": _env_bool("ALLOW_LANDMARK_MATCH", p["allow_landmark_match"]),
        "min_spatial_confidence": _env_float("MIN_SPATIAL_CONFIDENCE", p["min_spatial_confidence"]),
        "min_classifier_confidence": _env_float("MIN_CLASSIFIER_CONFIDENCE", p["min_classifier_confidence"]),
        "allow_multi_sector_expansion": _env_bool("ALLOW_MULTI_SECTOR_EXPANSION", p["allow_multi_sector_expansion"]),
        "max_sectors_per_article": _env_int("MAX_SECTORS_PER_ARTICLE", p["max_sectors_per_article"]),
    }
    dedup = {
        "sim_threshold": _env_float("DEDUP_SIM_THRESHOLD", p["dedup_sim_threshold"]),
        "same_sector_sim_threshold": _env_float("DEDUP_SAME_SECTOR_SIM", p["dedup_same_sector_sim"]),
        "close_dist_km": _env_float("DEDUP_CLOSE_DIST_KM", p["dedup_close_dist_km"]),
        "close_sim_threshold": _env_float("DEDUP_CLOSE_SIM", p["dedup_close_sim"]),
        "entity_overlap_sim_floor": _env_float("DEDUP_ENTITY_SIM_FLOOR", p["dedup_entity_sim_floor"]),
        "max_distance_km": _env_float("DEDUP_MAX_DISTANCE_KM", p["dedup_max_distance_km"]),
        "max_time_window_hours": _env_float("DEDUP_MAX_TIME_HOURS", p["dedup_max_time_window_hours"]),
    }
    ingest = {
        "lookback_days": _env_int("INGEST_LOOKBACK_DAYS", p["lookback_days"]),
        "max_city_articles": _env_int("MAX_CITY_ARTICLES", p["max_city_articles"]),
        "max_rss_items": _env_int("MAX_RSS_ITEMS", p["max_rss_items"]),
        "enable_hindi_feeds": _env_bool("ENABLE_HINDI_FEEDS", p["enable_hindi_feeds"]),
        "decode_google_news_urls": _env_bool("DECODE_GOOGLE_NEWS_URLS", p["decode_google_news_urls"]),
    }
    return {"grounding": grounding, "deduplication": dedup, "ingest": ingest}


_ACTIVE_CONFIG = build_pipeline_config()
PIPELINE_MODE = _ACTIVE_CONFIG["grounding"]["pipeline_mode"]
GROUNDING_CONFIG = _ACTIVE_CONFIG["grounding"]
DEDUPLICATION_CONFIG = _ACTIVE_CONFIG["deduplication"]
INGEST_CONFIG = _ACTIVE_CONFIG["ingest"]

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
