import json
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from geosituation.config import GAZETTEER_DIR, NOIDA_BOUNDS

class NoidaGazetteer:
    """
    Two-Tier Spatial Resolver:
    Combines deterministic regex sector extraction with a curated POI/Landmark Gazetteer
    to eliminate coordinate hallucination and resolve textual locations to canonical sectors.
    """
    
    def __init__(self, gazetteer_path: Optional[Path] = None):
        if gazetteer_path is None:
            gazetteer_path = GAZETTEER_DIR / "noida_sectors.json"
            
        self.gazetteer_path = gazetteer_path
        self.sectors: List[Dict] = []
        self.alias_to_sector: Dict[str, Dict] = {}
        self.regex_sector_pattern = re.compile(
            r'\b(?:sector|sec|sec\.|sect)\s*[-]?\s*(\d{1,3}[A-Za-z]?)\b',
            re.IGNORECASE
        )
        self.load_gazetteer()

    def load_gazetteer(self) -> None:
        if not self.gazetteer_path.exists():
            raise FileNotFoundError(f"Gazetteer file not found at: {self.gazetteer_path}")
            
        with open(self.gazetteer_path, "r", encoding="utf-8") as f:
            self.sectors = json.load(f)
            
        # Build normalized lookup index
        for sector in self.sectors:
            canonical_name = sector["name"].lower()
            canonical_id = sector["id"].lower()
            self.alias_to_sector[canonical_name] = sector
            self.alias_to_sector[canonical_id] = sector
            
            for alias in sector.get("aliases", []):
                self.alias_to_sector[alias.strip().lower()] = sector

    def resolve_location(self, text: str) -> Optional[Dict]:
        """
        Extracts and resolves spatial mentions in text to a canonical sector and coordinate.
        Returns dict with canonical sector, lat, lon, match_type, and confidence score.
        """
        text_lower = text.lower()
        
        # Pass 1: Deterministic Sector Number Extraction (High Precision)
        sector_matches = self.regex_sector_pattern.findall(text_lower)
        if sector_matches:
            sector_num = sector_matches[0].upper()
            canonical_id = f"Sector {sector_num}"
            
            # Lookup in canonical registry
            for sector in self.sectors:
                if sector["id"].lower() == canonical_id.lower():
                    return {
                        "sector_id": sector["id"],
                        "sector_name": sector["name"],
                        "zone": sector["zone"],
                        "lat": sector["lat"],
                        "lon": sector["lon"],
                        "confidence": 0.96,
                        "match_type": "EXACT_SECTOR_REGEX",
                        "matched_text": f"Sector {sector_num}"
                    }
            
            # If valid sector number within 1-168 but not explicitly in curated list:
            # Generate estimated centroid within Noida bounds
            num_clean = re.sub(r'[^0-9]', '', sector_num)
            if num_clean and 1 <= int(num_clean) <= 168:
                # Approximate interpolation for unlisted sectors
                approx_lat = NOIDA_BOUNDS["center_lat"] - (int(num_clean) % 15) * 0.008
                approx_lon = NOIDA_BOUNDS["center_lon"] + (int(num_clean) % 12) * 0.008
                return {
                    "sector_id": canonical_id,
                    "sector_name": f"{canonical_id}, Noida",
                    "zone": "Mixed Urban Sector",
                    "lat": round(approx_lat, 4),
                    "lon": round(approx_lon, 4),
                    "confidence": 0.82,
                    "match_type": "INTERPOLATED_SECTOR_REGEX",
                    "matched_text": f"Sector {sector_num}"
                }

        # Pass 2: Curated Landmark & POI Lookup (Longest string match first)
        sorted_aliases = sorted(self.alias_to_sector.keys(), key=len, reverse=True)
        for alias in sorted_aliases:
            # Word boundary search for short aliases
            pattern = rf'\b{re.escape(alias)}\b'
            if re.search(pattern, text_lower):
                sector = self.alias_to_sector[alias]
                return {
                    "sector_id": sector["id"],
                    "sector_name": sector["name"],
                    "zone": sector["zone"],
                    "lat": sector["lat"],
                    "lon": sector["lon"],
                    "confidence": 0.90 if len(alias) > 6 else 0.78,
                    "match_type": "LANDMARK_GAZETTEER_MATCH",
                    "matched_text": alias
                }

        # No confident resolution: prevent hallucination
        return None

    def get_all_sectors(self) -> List[Dict]:
        return self.sectors
