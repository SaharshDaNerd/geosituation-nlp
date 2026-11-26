import json
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from geosituation.config import GAZETTEER_DIR, GROUNDING_CONFIG, NOIDA_BOUNDS

class NoidaGazetteer:
    """
    Two-Tier Spatial Resolver:
    Combines deterministic regex sector extraction (single, multi-sector lists, and Hindi)
    with a curated POI/Landmark Gazetteer to eliminate coordinate hallucination.
    """

    # Aliases too ambiguous or institutional (e.g. agency names blamed in headlines,
    # or bare tokens matching Greater Noida villages like "Atta Gujran").
    IGNORED_ALIASES = {"atta", "noida authority", "noida authority office"}

    # Real Noida chowks, urban villages, flyovers, industrial phases, hospitals,
    # residential societies, metro stations, and police stations mapped to canonical sectors.
    # Ambiguous names are anchored to their primary/most prominent Noida sector.
    EXTRA_LOCALITY_ALIASES: Dict[str, List[str]] = {
        "Sector 1": ["rajnigandha chowk", "noida phase 1", "phase 1 noida", "gol chakkar", "रजनीगंधा चौक", "फेज 1", "गोल चक्कर"],
        "Sector 2": ["harola", "harola village", "harola market", "हरौला"],
        "Sector 6": ["jhundpura", "noida authority sector 6", "झुंडपुरा"],
        "Sector 11": ["metro hospital noida", "jhundpura village"],
        "Sector 12": ["12/22", "12-22", "sector 12-22", "12/22 chowk", "metro hospital sector 12"],
        "Sector 14": ["chilla border", "chilla regulator", "noida gate", "चिल्ला बॉर्डर"],
        "Sector 16": ["film city", "marwah studios", "rajnigandha", "फिल्म सिटी"],
        "Sector 18": [
            "gip", "great india place", "gardens galleria", "savitri market",
            "atta chakki", "dlf mall", "centre stage mall", "sab mall",
            "अट्टा मार्केट", "जीआईपी मॉल", "डीएलएफ मॉल"
        ],
        "Sector 19": ["max hospital noida", "sanatan dharm mandir noida"],
        "Sector 20": ["sector 20 police station", "sector 20 ps", "nithari", "nithari village", "निठारी"],
        "Sector 21": ["noida stadium", "modi mall", "spice mall", "jalvayu vihar", "नोएडा स्टेडियम"],
        "Sector 22": ["chaura", "chaura raghunathpur", "chaura village", "चौड़ा"],
        "Sector 24": ["sector 24 police station", "sector 24 ps", "esic hospital", "ntpc noida"],
        "Sector 25": ["jalvayu vihar sector 25", "spice world mall"],
        "Sector 27": ["kailash hospital", "atta peer", "indira market", "sabzi mandi noida", "कैलाश अस्पताल"],
        "Sector 29": ["ganga shopping complex", "brahmaputra market", "ब्रह्मपुत्र मार्केट"],
        "Sector 30": ["district hospital noida", "child pgi", "noida district hospital", "dps noida", "जिला अस्पताल नोएडा"],
        "Sector 31": ["noida mor", "nithari mor"],
        "Sector 32": ["wave city center", "wave city centre", "city centre", "rto office noida", "सिटी सेंटर"],
        "Sector 33": ["prakash hospital", "iskcon noida", "iskcon temple noida"],
        "Sector 37": ["mahamaya flyover", "kalindi kunj", "arun vihar", "amity chowk", "महामाया फ्लाईओवर", "कालिंदी कुंज", "बॉटनिकल गार्डन"],
        "Sector 39": ["sector 39 police station", "sector 39 ps", "city hospital noida", "noida district court"],
        "Sector 44": ["chhalera", "chhalera village", "amrapali sapphire", "छलेरा"],
        "Sector 45": ["prateek stylome", "amrapali sapphire sector 45", "sadarpur", "सदरपुर"],
        "Sector 49": ["barola", "barola village", "barola market", "sector 49 police station", "sector 49 ps", "बरौला"],
        "Sector 50": ["neo hospital", "central park sector 50", "alok vihar", "meghdootam park"],
        "Sector 51": ["hoshiarpur", "hoshiyarpur", "kendriya vihar", "cloudnine noida", "shivalik hospital", "होशियारपुर"],
        "Sector 53": ["gijhor", "gijhor village", "gijhor chowk", "गिझौड़"],
        "Sector 58": ["labour chowk", "khora", "khora colony", "khora market", "sector 58 ps", "लेबर चौक", "खोड़ा"],
        "Sector 59": ["mamura chowk", "sector 59 metro", "मामूरा चौक"],
        "Sector 62": [
            "model town", "noida electronic city", "nh-24 noida", "nh 24 noida",
            "nh-9 noida", "nh 9 noida", "fortis hospital", "jiit", "symphony", "corenthum",
            "इलेक्ट्रॉनिक सिटी", "फोर्टिस अस्पताल", "मॉडल टाउन"
        ],
        "Sector 63": ["phase 3 noida", "noida phase 3", "chhijarsi", "transport nagar noida", "फेज 3", "छिजारसी"],
        "Sector 66": ["mamura", "mamoora", "mamura village", "मामूरा"],
        "Sector 71": ["sarfabad", "sarfabad village", "garhi chaukhandi", "kisan chowk", "सरफाबाद", "गढ़ी चौखंडी"],
        "Sector 74": ["supertech capetown", "capetown society", "north eye"],
        "Sector 75": ["spectrum metro", "golf city", "gardenia gateway"],
        "Sector 76": ["amrapali silicon city", "silicon city", "sethi max royal", "skytech matrott"],
        "Sector 77": ["prateek wisteria", "griha pravesh", "express zenith"],
        "Sector 78": ["mahagun moderne", "hyde park", "antriksh golf view", "nimbus hyde park"],
        "Sector 80": ["phase 2 noida", "noida phase 2", "nsez", "फेज 2"],
        "Sector 81": ["salarpur", "salarpur khadar", "nsez metro", "सलारपुर"],
        "Sector 82": ["bhangel", "bhangel market", "bhangel elevated road", "dadri road", "kendriya vihar 2", "भंगेल"],
        "Sector 93A": ["eldeco utopia", "ats greens village", "emerald court", "parsvnath prestige"],
        "Sector 94": ["okhla bird sanctuary", "supernova noida", "kalindi kunj bridge"],
        "Sector 98": ["skymark one", "skymark"],
        "Sector 100": ["lotus boulevard", "century apartments"],
        "Sector 104": ["hajipur", "hazipur", "sector 104 market", "pathways school noida", "हाजीपुर"],
        "Sector 110": ["gejha", "gejha village", "lotus panache", "yatharth hospital", "गेझा"],
        "Sector 113": [
            "sector 113 police station", "sector 113 ps", "sorakha", "sorkha",
            "parthala", "parthala chowk", "parthala flyover", "parthala bridge",
            "परथला", "सोरखा"
        ],
        "Sector 119": ["amrapali platinum", "gaur grandeur", "unnat fortune"],
        "Sector 120": ["amrapali zodiac", "prateek laurel", "rg residency"],
        "Sector 121": ["cleo county", "homes 121", "garhi chaukhandi chowk"],
        "Sector 125": ["amity university", "amity campus", "amity noida", "एमिटी यूनिवर्सिटी"],
        "Sector 126": ["sector 126 police station", "sector 126 ps", "raipur khadar", "hcl campus"],
        "Sector 128": ["jaypee hospital", "wish town", "jaypee greens", "जेपी अस्पताल", "विश टाउन"],
        "Sector 129": ["gulshan one29", "gulshan mall"],
        "Sector 135": ["wazidpur", "wajidpur", "expressway police station", "candor techspace", "वाजिदपुर"],
        "Sector 137": ["paras tierea", "purvanchal royal park", "logix blossom county", "felix hospital", "gulshan vivante"],
        "Sector 142": ["sector 142 police station", "sector 142 ps", "advant", "advant navis"],
        "Sector 143": ["logix blossom zest", "sikka karnam greens"],
        "Sector 144": ["gulshan botnia"],
        "Sector 150": ["noida sports city", "tata eureka park", "ats pristine", "godrej palm retreat"],
        "DND Flyway": ["डीएनडी फ्लाईवे", "डीएनडी टोल"],
        "Noida-Greater Noida Expressway": ["नोएडा एक्सप्रेसवे", "नोएडा-ग्रेटर नोएडा एक्सप्रेसवे"],
    }
    
    def __init__(self, gazetteer_path: Optional[Path] = None):
        if gazetteer_path is None:
            gazetteer_path = GAZETTEER_DIR / "noida_sectors.json"
            
        self.gazetteer_path = gazetteer_path
        self.sectors: List[Dict] = []
        self.by_id: Dict[str, Dict] = {}
        self.alias_to_sector: Dict[str, Dict] = {}
        self.sorted_aliases: List[str] = []
        self.regex_sector_pattern = re.compile(
            r'(?:\b(?:sectors?|secs?\.?|sect)|सेक्टर)\s*[-]?\s*(\d{1,3}[A-Za-z]?)\b',
            re.IGNORECASE
        )
        # Matches multi-sector lists & ranges like "Sectors 15 and 16", "Sector 62-63", "Sec 50, 51 & 52"
        self.multi_sector_pattern = re.compile(
            r'(?:\b(?:sectors?|secs?\.?|sect)|सेक्टर)\s*[-]?\s*'
            r'(\d{1,3}[A-Za-z]?(?:\s*(?:,|/|&|-|\band\b|और)\s*(?:(?:sectors?|secs?\.?|sect)|सेक्टर)?\s*\d{1,3}[A-Za-z]?)*)',
            re.IGNORECASE
        )
        self._sector_token_pattern = re.compile(r'\b(\d{1,3}[A-Za-z]?)\b')
        self.load_gazetteer()

    def load_gazetteer(self) -> None:
        if not self.gazetteer_path.exists():
            raise FileNotFoundError(f"Gazetteer file not found at: {self.gazetteer_path}")
            
        with open(self.gazetteer_path, "r", encoding="utf-8") as f:
            self.sectors = json.load(f)

        self.by_id = {s["id"].lower(): s for s in self.sectors}
        self.alias_to_sector = {}

        # Build normalized lookup index (excluding overly ambiguous/institutional tokens)
        for sector in self.sectors:
            canonical_name = sector["name"].lower()
            canonical_id = sector["id"].lower()
            self.alias_to_sector[canonical_name] = sector
            self.alias_to_sector[canonical_id] = sector
            
            for alias in sector.get("aliases", []):
                norm_alias = alias.strip().lower()
                if norm_alias and norm_alias not in self.IGNORED_ALIASES:
                    self.alias_to_sector[norm_alias] = sector

        # Attach extended chowk, village, and police-station aliases
        for sector_id, extra_aliases in self.EXTRA_LOCALITY_ALIASES.items():
            sector = self.by_id.get(sector_id.lower())
            if sector:
                for alias in extra_aliases:
                    norm_alias = alias.strip().lower()
                    if norm_alias and norm_alias not in self.IGNORED_ALIASES:
                        self.alias_to_sector[norm_alias] = sector

        self.sorted_aliases = sorted(self.alias_to_sector.keys(), key=len, reverse=True)

    def resolve_all_locations(
        self,
        text: str,
        allow_landmark: Optional[bool] = None,
        min_confidence: Optional[float] = None,
        allow_multi_sector: Optional[bool] = None,
    ) -> List[Dict]:
        """
        Extracts all canonical Noida sectors referenced in text (supporting multi-sector lists,
        Hindi 'सेक्टर N', and landmark aliases) ordered by specificity and appearance.
        """
        if not text:
            return []

        if allow_landmark is None:
            allow_landmark = GROUNDING_CONFIG["allow_landmark_match"]
        if min_confidence is None:
            min_confidence = GROUNDING_CONFIG["min_spatial_confidence"]
        if allow_multi_sector is None:
            allow_multi_sector = GROUNDING_CONFIG["allow_multi_sector_expansion"]

        text_lower = text.lower()
        results: List[Dict] = []
        seen_ids = set()

        # Pass 1: Deterministic Sector Number Extraction (single + multi-sector ranges/lists)
        spans = self.multi_sector_pattern.findall(text_lower) if allow_multi_sector else self.regex_sector_pattern.findall(text_lower)
        for span in spans:
            tokens = self._sector_token_pattern.findall(span) if allow_multi_sector else [span]
            for sector_num in tokens:
                sector_num_up = sector_num.upper()
                canonical_id = f"Sector {sector_num_up}"
                sector = self.by_id.get(canonical_id.lower())
                if sector and sector["id"] not in seen_ids:
                    conf = 0.96
                    if conf >= min_confidence:
                        seen_ids.add(sector["id"])
                        results.append({
                            "sector_id": sector["id"],
                            "sector_name": sector["name"],
                            "zone": sector["zone"],
                            "lat": sector["lat"],
                            "lon": sector["lon"],
                            "confidence": conf,
                            "match_type": "EXACT_SECTOR_REGEX",
                            "matched_text": canonical_id,
                        })

        # Pass 2: Curated Landmark & POI Lookup (Longest string match first)
        if allow_landmark:
            for alias in self.sorted_aliases:
                pattern = rf'(?<![\w\u0900-\u097f]){re.escape(alias)}(?![\w\u0900-\u097f])'
                if re.search(pattern, text_lower):
                    sector = self.alias_to_sector[alias]
                    if sector["id"] in seen_ids:
                        continue
                    conf = 0.90 if len(alias) > 6 else 0.78
                    if conf >= min_confidence:
                        seen_ids.add(sector["id"])
                        results.append({
                            "sector_id": sector["id"],
                            "sector_name": sector["name"],
                            "zone": sector["zone"],
                            "lat": sector["lat"],
                            "lon": sector["lon"],
                            "confidence": conf,
                            "match_type": "LANDMARK_GAZETTEER_MATCH",
                            "matched_text": alias,
                        })

        return results

    def resolve_location(self, text: str) -> Optional[Dict]:
        """
        Extracts and resolves the primary spatial mention in text to a canonical sector.
        Returns dict with canonical sector, lat, lon, match_type, and confidence score.
        """
        matches = self.resolve_all_locations(text)
        return matches[0] if matches else None

    def get_all_sectors(self) -> List[Dict]:
        return self.sectors
