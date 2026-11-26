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

    # Real Noida chowks, urban villages, flyovers, industrial phases, hospitals,
    # residential societies, metro stations, and police stations mapped to canonical sectors.
    # Ambiguous names are anchored to their primary/most prominent Noida sector.
    EXTRA_LOCALITY_ALIASES: Dict[str, List[str]] = {
        "Sector 1": ["rajnigandha chowk", "noida phase 1", "phase 1 noida", "gol chakkar"],
        "Sector 2": ["harola", "harola village", "harola market"],
        "Sector 6": ["noida authority", "noida authority office", "jhundpura"],
        "Sector 11": ["metro hospital noida", "jhundpura village"],
        "Sector 12": ["12/22", "12-22", "sector 12-22", "12/22 chowk", "metro hospital sector 12"],
        "Sector 14": ["chilla border", "chilla regulator", "noida gate"],
        "Sector 16": ["film city", "marwah studios", "rajnigandha"],
        "Sector 18": [
            "gip", "great india place", "gardens galleria", "savitri market",
            "atta chakki", "dlf mall", "centre stage mall", "sab mall"
        ],
        "Sector 19": ["max hospital noida", "sanatan dharm mandir noida"],
        "Sector 20": ["sector 20 police station", "sector 20 ps", "nithari", "nithari village"],
        "Sector 21": ["noida stadium", "modi mall", "spice mall", "jalvayu vihar"],
        "Sector 22": ["chaura", "chaura raghunathpur", "chaura village"],
        "Sector 24": ["sector 24 police station", "sector 24 ps", "esic hospital", "ntpc noida"],
        "Sector 25": ["jalvayu vihar sector 25", "spice world mall"],
        "Sector 27": ["kailash hospital", "atta peer", "indira market", "sabzi mandi noida"],
        "Sector 29": ["ganga shopping complex", "brahmaputra market"],
        "Sector 30": ["district hospital noida", "child pgi", "noida district hospital", "dps noida"],
        "Sector 31": ["noida mor", "nithari mor"],
        "Sector 32": ["wave city center", "wave city centre", "city centre", "rto office noida"],
        "Sector 33": ["prakash hospital", "iskcon noida", "iskcon temple noida"],
        "Sector 37": ["mahamaya flyover", "kalindi kunj", "arun vihar", "amity chowk"],
        "Sector 39": ["sector 39 police station", "sector 39 ps", "city hospital noida", "noida district court"],
        "Sector 44": ["chhalera", "chhalera village", "amrapali sapphire"],
        "Sector 45": ["prateek stylome", "amrapali sapphire sector 45", "sadarpur"],
        "Sector 49": ["barola", "barola village", "barola market", "sector 49 police station", "sector 49 ps"],
        "Sector 50": ["neo hospital", "central park sector 50", "alok vihar", "meghdootam park"],
        "Sector 51": ["hoshiarpur", "hoshiyarpur", "kendriya vihar", "cloudnine noida", "shivalik hospital"],
        "Sector 53": ["gijhor", "gijhor village", "gijhor chowk"],
        "Sector 58": ["labour chowk", "khora", "khora colony", "khora market", "sector 58 ps"],
        "Sector 59": ["mamura chowk", "sector 59 metro"],
        "Sector 62": [
            "model town", "noida electronic city", "nh-24 noida", "nh 24 noida",
            "nh-9 noida", "nh 9 noida", "fortis hospital", "jiit", "symphony", "corenthum"
        ],
        "Sector 63": ["phase 3 noida", "noida phase 3", "chhijarsi", "transport nagar noida", "electronic city"],
        "Sector 66": ["mamura", "mamoora", "mamura village"],
        "Sector 71": ["sarfabad", "sarfabad village", "garhi chaukhandi", "kisan chowk"],
        "Sector 74": ["supertech capetown", "capetown society", "north eye"],
        "Sector 75": ["spectrum metro", "golf city", "gardenia gateway"],
        "Sector 76": ["amrapali silicon city", "silicon city", "sethi max royal", "skytech matrott"],
        "Sector 77": ["prateek wisteria", "griha pravesh", "express zenith"],
        "Sector 78": ["mahagun moderne", "hyde park", "antriksh golf view", "nimbus hyde park"],
        "Sector 80": ["phase 2 noida", "noida phase 2", "nsez"],
        "Sector 81": ["salarpur", "salarpur khadar", "nsez metro"],
        "Sector 82": ["bhangel", "bhangel market", "bhangel elevated road", "dadri road", "kendriya vihar 2"],
        "Sector 93A": ["eldeco utopia", "ats greens village", "emerald court", "parsvnath prestige"],
        "Sector 94": ["okhla bird sanctuary", "supernova noida", "kalindi kunj bridge"],
        "Sector 98": ["skymark one", "skymark"],
        "Sector 100": ["lotus boulevard", "century apartments"],
        "Sector 104": ["hajipur", "hazipur", "sector 104 market", "pathways school noida"],
        "Sector 110": ["gejha", "gejha village", "lotus panache", "yatharth hospital"],
        "Sector 113": [
            "sector 113 police station", "sector 113 ps", "sorakha", "sorkha",
            "parthala", "parthala chowk", "parthala flyover", "parthala bridge"
        ],
        "Sector 119": ["amrapali platinum", "gaur grandeur", "unnat fortune"],
        "Sector 120": ["amrapali zodiac", "prateek laurel", "rg residency"],
        "Sector 121": ["cleo county", "homes 121", "garhi chaukhandi chowk"],
        "Sector 125": ["amity university", "amity campus", "amity noida"],
        "Sector 126": ["sector 126 police station", "sector 126 ps", "raipur khadar", "hcl campus"],
        "Sector 128": ["jaypee hospital", "wish town", "jaypee greens"],
        "Sector 129": ["gulshan one29", "gulshan mall"],
        "Sector 135": ["wazidpur", "wajidpur", "expressway police station", "candor techspace"],
        "Sector 137": ["paras tierea", "purvanchal royal park", "logix blossom county", "felix hospital", "gulshan vivante"],
        "Sector 142": ["sector 142 police station", "sector 142 ps", "advant", "advant navis"],
        "Sector 143": ["logix blossom zest", "sikka karnam greens"],
        "Sector 144": ["gulshan botnia"],
        "Sector 150": ["noida sports city", "tata eureka park", "ats pristine", "godrej palm retreat"],
    }
    
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

        # Attach extended chowk, village, and police-station aliases
        by_id = {s["id"].lower(): s for s in self.sectors}
        for sector_id, extra_aliases in self.EXTRA_LOCALITY_ALIASES.items():
            sector = by_id.get(sector_id.lower())
            if sector:
                for alias in extra_aliases:
                    self.alias_to_sector[alias.strip().lower()] = sector

    def resolve_location(self, text: str) -> Optional[Dict]:
        """
        Extracts and resolves spatial mentions in text to a canonical sector and coordinate.
        Returns dict with canonical sector, lat, lon, match_type, and confidence score.
        """
        text_lower = text.lower()
        
        # Pass 1: Deterministic Sector Number Extraction (High Precision)
        # Take the first mentioned sector that exists in the gazetteer. Sectors
        # without a known centroid are never given estimated coordinates.
        by_id = {s["id"].lower(): s for s in self.sectors}
        for sector_num in self.regex_sector_pattern.findall(text_lower):
            sector_num = sector_num.upper()
            canonical_id = f"Sector {sector_num}"
            sector = by_id.get(canonical_id.lower())
            if sector:
                return {
                    "sector_id": sector["id"],
                    "sector_name": sector["name"],
                    "zone": sector["zone"],
                    "lat": sector["lat"],
                    "lon": sector["lon"],
                    "confidence": 0.96,
                    "match_type": "EXACT_SECTOR_REGEX",
                    "matched_text": canonical_id
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
