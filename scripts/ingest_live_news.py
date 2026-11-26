"""
Rebuild the situational database from REAL news (Google News RSS).

* Runs one search per sector in the gazetteer ("Noida Sector N") plus a set of
  city-wide incident searches, restricted to the last N days.
* Every headline goes through the normal Level 1 perception path
  (incident filter -> classifier -> gazetteer). Headlines that are not about
  Noida, are not incidents, or cannot be placed on a known sector are dropped.
* Replaces data/processed/level1_reports.json with only these real reports
  (the synthetic seed corpus is NOT kept), re-runs coreference, then the full
  Level 2 / Level 3 pipeline.

Usage:
    uv run python scripts/ingest_live_news.py            # last 60 days
    uv run python scripts/ingest_live_news.py --days 14
    uv run python scripts/ingest_live_news.py --append   # keep existing reports
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta
from collections import Counter

from geosituation.config import PROCESSED_DATA_DIR
from geosituation.pipeline import SituationalAwarenessPipeline

CITY_QUERIES = [
    "Noida police arrested", "Noida accident", "Noida fire", "Noida murder",
    "Noida robbery OR loot", "Noida theft OR burglary", "Noida snatching",
    "Noida fraud OR scam", "Noida protest", "Noida waterlogging OR collapse",
    "Noida injured", "Noida Expressway accident", "DND flyway accident OR jam",
    "Noida society fire", "Noida road accident killed", "Noida chain snatching",
    "Noida encounter police", "Noida burglary flat", "Noida car theft",
    "Noida residents protest", "Noida body found", "Noida stabbed OR shot",
    "Noida factory fire", "Noida metro station incident", "Noida mall incident",
]
SECTOR_INCIDENT_TERMS = "police OR fire OR accident OR theft OR arrested OR killed OR protest"


def norm(headline: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", headline.lower()).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--append", action="store_true", help="keep existing level1 reports")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    args = ap.parse_args()

    pipeline = SituationalAwarenessPipeline()
    collector = pipeline.collector
    sectors = [s["id"] for s in pipeline.gazetteer.get_all_sectors() if s["id"].startswith("Sector ")]

    # Two searches per sector: everything mentioning it, and an incident-focused one
    queries = [q for s in sectors for q in (f'Noida "{s}"', f'Noida "{s}" {SECTOR_INCIDENT_TERMS}')] + CITY_QUERIES
    window = f" when:{args.days}d"

    reports_path = PROCESSED_DATA_DIR / "level1_reports.json"
    existing = json.loads(reports_path.read_text(encoding="utf-8")) if (args.append and reports_path.exists()) else []
    seen = {norm(r["headline"]) for r in existing}
    collected = []

    for i, q in enumerate(queries, 1):
        recs = collector.fetch_google_news_rss(q + window)
        new = 0
        for r in recs:
            key = norm(r["headline"])
            if key in seen:
                continue
            seen.add(key)
            collected.append(r)
            new += 1
        print(f"[{i:3d}/{len(queries)}] {q[:48]:48s} -> {len(recs):2d} incidents, {new:2d} new", flush=True)
        time.sleep(args.delay)

    if not collected and not existing:
        print("No real incident reports collected (network down?). Existing data left untouched.")
        return 1

    # Retention: keep only reports published within the window (default 60 days)
    cutoff = (datetime.now() - timedelta(days=args.days)).isoformat()
    kept = [r for r in existing + collected if r.get("published_at", "") >= cutoff]
    dropped = len(existing) + len(collected) - len(kept)
    if dropped:
        print(f"Dropped {dropped} reports older than {args.days} days.")
    reports = sorted(kept, key=lambda r: r["published_at"], reverse=True)
    reports_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")

    canonical = pipeline.deduplicator.deduplicate(reports)
    (PROCESSED_DATA_DIR / "canonical_events.json").write_text(json.dumps(canonical, indent=2, ensure_ascii=False), encoding="utf-8")
    state = pipeline.run_pipeline()

    by_sector = Counter(e["sector_id"] for e in canonical)
    print(f"\n{len(reports)} real reports -> {len(canonical)} canonical events across {len(by_sector)} sectors/landmarks.")
    print("Top:", ", ".join(f"{k} ({v})" for k, v in by_sector.most_common(12)))
    print("Hotspots:", len(state["hotspots"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
