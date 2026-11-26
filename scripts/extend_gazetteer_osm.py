"""
Extend the Noida gazetteer with real sector centroids from OpenStreetMap.

For every Noida sector number 1-168 that is not already curated in
data/gazetteers/noida_sectors.json, query Nominatim for "Sector N, Noida" and
accept the result only if:
  * its name is exactly that sector (e.g. "Sector 44"), and
  * it lies inside the Noida / Gautam Buddha Nagar envelope.

Anything that does not pass is skipped (never guessed). Curated entries are
never overwritten. Results are cached in data/raw/osm_sector_cache.json so the
script can be re-run without hitting the API again.

Usage:  uv run python scripts/extend_gazetteer_osm.py
Respects the Nominatim usage policy (1 request / second, identifying UA).
"""
import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
GAZ_PATH = ROOT / "data" / "gazetteers" / "noida_sectors.json"
CACHE_PATH = ROOT / "data" / "raw" / "osm_sector_cache.json"

# Generous envelope around Noida (Sector 1 in the NW to Sector 168 in the SE).
ENVELOPE = {"min_lat": 28.40, "max_lat": 28.66, "min_lon": 77.28, "max_lon": 77.52}
UA = "geosituation-nlp/0.1 (gazetteer builder)"
# Only area-like OSM features describe a whole sector; points such as a bus
# stop named "Sector 38" are rejected.
AREA_TYPES = {"residential", "industrial", "commercial", "retail", "hamlet", "locality",
              "suburb", "neighbourhood", "quarter", "common", "administrative", "village"}


def in_envelope(lat: float, lon: float) -> bool:
    return ENVELOPE["min_lat"] <= lat <= ENVELOPE["max_lat"] and ENVELOPE["min_lon"] <= lon <= ENVELOPE["max_lon"]


def query(sector: str) -> list:
    r = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": f"{sector}, Noida, Uttar Pradesh, India", "format": "jsonv2", "limit": 8, "countrycodes": "in"},
        headers={"User-Agent": UA},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()


def pick(results: list, sector: str):
    want = sector.lower()
    for res in results:
        name = (res.get("name") or "").strip().lower()
        first = (res.get("display_name") or "").split(",")[0].strip().lower()
        if want not in (name, first):
            continue
        if "noida" not in (res.get("display_name") or "").lower():
            continue
        if res.get("type") not in AREA_TYPES:
            continue
        lat, lon = float(res["lat"]), float(res["lon"])
        if in_envelope(lat, lon):
            return {"lat": round(lat, 4), "lon": round(lon, 4), "osm_type": res.get("type"), "display_name": res.get("display_name")}
    return None


def main() -> int:
    gaz = json.loads(GAZ_PATH.read_text(encoding="utf-8"))
    existing = {g["id"].lower() for g in gaz}
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}

    added = 0
    for n in range(1, 169):
        sid = f"Sector {n}"
        if sid.lower() in existing:
            continue
        if sid not in cache:
            try:
                cache[sid] = pick(query(sid), sid)
            except Exception as e:  # network error: leave uncached, try again next run
                print(f"{sid}: error {e}", file=sys.stderr)
                time.sleep(1.1)
                continue
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
            time.sleep(1.1)
        hit = cache[sid]
        if not hit:
            print(f"{sid}: not found in OSM, skipped")
            continue
        gaz.append({
            "id": sid,
            "name": f"{sid}, Noida",
            "zone": "Urban Sector",
            "lat": hit["lat"],
            "lon": hit["lon"],
            "source": "OpenStreetMap Nominatim",
            "aliases": [f"sector {n}", f"sec {n}", f"sector-{n}"],
        })
        existing.add(sid.lower())
        added += 1

    def sort_key(g):
        m = re.match(r"Sector (\d+)", g["id"])
        return (0, int(m.group(1)), g["id"]) if m else (1, 0, g["id"])

    gaz.sort(key=sort_key)
    GAZ_PATH.write_text(json.dumps(gaz, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Added {added} sectors from OpenStreetMap. Gazetteer now has {len(gaz)} entries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
