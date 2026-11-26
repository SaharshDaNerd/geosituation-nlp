"""
Rebuild the situational database from REAL news across 6 multi-source channels.

Channels:
  1. Direct Noida city index pages (TOI Noida & HT Noida via layered BS4 / JSON-LD)
  2. Direct Publisher RSS feeds (HT Noida, TOI NCR, Indian Express)
  3. Bing News RSS (direct publisher URLs + body extraction)
  4. GDELT 2.0 Doc API (up to 90-day historical backfill)
  5. Google News Hindi RSS (Noida local Hindi reporting)
  6. Google News English RSS (per-sector + city-wide fan-out + redirect decoding)

Usage:
    uv run python scripts/ingest_live_news.py
    uv run python scripts/ingest_live_news.py --mode high_recall --days 90
    uv run python scripts/ingest_live_news.py --append
"""
import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from datetime import datetime, timedelta

from geosituation.config import (
    INGEST_CONFIG,
    PROCESSED_DATA_DIR,
    build_pipeline_config,
)
from geosituation.perception.collector import NewsIncidentCollector
from geosituation.comprehension.coreference import CrossDocumentDeduplicator
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


def norm_key(report: dict) -> str:
    headline = re.sub(r"[^a-z0-9\u0900-\u097f]+", " ", report["headline"].lower()).strip()
    return f"{headline}||{report.get('sector_id', '')}"


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None, help="Lookback window in days")
    ap.add_argument("--append", action="store_true", help="Keep existing level1 reports")
    ap.add_argument("--delay", type=float, default=0.5, help="Seconds between Google News requests")
    ap.add_argument("--mode", choices=["strict", "balanced", "high_recall"], default="", help="Override PIPELINE_MODE")
    args = ap.parse_args()

    cfg = build_pipeline_config(args.mode)
    grounding_cfg = cfg["grounding"]
    dedup_cfg = cfg["deduplication"]
    ingest_cfg = cfg["ingest"]
    days = args.days if args.days is not None else ingest_cfg["lookback_days"]

    print(f"=== GeoSituation-NLP Live Ingestion (mode={grounding_cfg['pipeline_mode']}, days={days}) ===", flush=True)

    pipeline = SituationalAwarenessPipeline()
    collector = NewsIncidentCollector(grounding_config=grounding_cfg, ingest_config=ingest_cfg)
    collector.reset_stats()
    deduplicator = CrossDocumentDeduplicator(**dedup_cfg)

    all_gazetteer_sectors = collector.gazetteer.get_all_sectors()
    sectors = [s["id"] for s in all_gazetteer_sectors if s["id"].startswith("Sector ")]
    queries = [q for s in sectors for q in (f'Noida "{s}"', f'Noida "{s}" {SECTOR_INCIDENT_TERMS}')] + CITY_QUERIES
    window = f" when:{days}d"

    reports_path = PROCESSED_DATA_DIR / "level1_reports.json"
    existing = json.loads(reports_path.read_text(encoding="utf-8")) if (args.append and reports_path.exists()) else []
    seen = {norm_key(r) for r in existing}
    collected = []

    def add_records(recs: list) -> int:
        added = 0
        for r in recs:
            key = norm_key(r)
            if key in seen:
                continue
            seen.add(key)
            collected.append(r)
            added += 1
        return added

    # 1. Direct Noida City Index Pages (TOI Noida + HT Noida)
    print(f"[Source 1/6] Scraping TOI & HT Noida city pages (max_articles={ingest_cfg['max_city_articles']})...", flush=True)
    city_recs = collector.fetch_noida_city_pages(max_articles=ingest_cfg["max_city_articles"])
    print(f"   -> {len(city_recs)} incidents, {add_records(city_recs)} new", flush=True)

    # 2. Direct Publisher RSS Feeds
    print("[Source 2/6] Fetching direct publisher RSS feeds (HT Noida, TOI, IE)...", flush=True)
    pub_recs = collector.fetch_publisher_rss(fetch_body=True)
    print(f"   -> {len(pub_recs)} incidents, {add_records(pub_recs)} new", flush=True)

    # 3. Bing News RSS
    print("[Source 3/6] Fetching Bing News RSS (direct URLs + BS4 extraction)...", flush=True)
    bing_recs = collector.fetch_bing_news_rss(fetch_body=True)
    print(f"   -> {len(bing_recs)} incidents, {add_records(bing_recs)} new", flush=True)

    # 4. GDELT 2.0 Doc API
    gdelt_days = min(days, 90)
    print(f"[Source 4/6] Querying GDELT 2.0 Doc API (timespan={gdelt_days}d)...", flush=True)
    gdelt_recs = collector.fetch_gdelt_news(timespan=f"{gdelt_days}d", max_records=250)
    print(f"   -> {len(gdelt_recs)} incidents, {add_records(gdelt_recs)} new", flush=True)

    # 5. Google News Hindi RSS
    if ingest_cfg.get("enable_hindi_feeds", True):
        print("[Source 5/6] Fetching Google News Hindi RSS (Noida local press)...", flush=True)
        hi_recs = collector.fetch_hindi_noida_rss(days=days)
        print(f"   -> {len(hi_recs)} incidents, {add_records(hi_recs)} new", flush=True)
    else:
        print("[Source 5/6] Hindi RSS skipped (ENABLE_HINDI_FEEDS=false).", flush=True)

    # 6. Google News English RSS fan-out
    print(f"[Source 6/6] Querying Google News RSS ({len(queries)} targeted queries)...", flush=True)
    decode_gnews = ingest_cfg.get("decode_google_news_urls", True)
    for i, q in enumerate(queries, 1):
        is_city = q in CITY_QUERIES
        recs = collector.fetch_google_news_rss(
            q + window,
            fetch_unresolved_bodies=decode_gnews and (is_city or i <= 60),
            max_body_fetches=10 if is_city else 3,
        )
        new = add_records(recs)
        print(f"[{i:3d}/{len(queries)}] {q[:48]:48s} -> {len(recs):2d} incidents, {new:2d} new", flush=True)
        time.sleep(args.delay)

    collector._save_url_cache()

    if not collected and not existing:
        print("No real incident reports collected (network down?). Existing data left untouched.")
        return 1

    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    kept = [r for r in existing + collected if r.get("published_at", "") >= cutoff]
    dropped_old = len(existing) + len(collected) - len(kept)
    reports = sorted(kept, key=lambda r: r["published_at"], reverse=True)
    reports_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")

    canonical = deduplicator.deduplicate(reports)
    (PROCESSED_DATA_DIR / "canonical_events.json").write_text(json.dumps(canonical, indent=2, ensure_ascii=False), encoding="utf-8")
    pipeline.deduplicator = deduplicator
    state = pipeline.run_pipeline()

    # Telemetry & Funnel Audit Summary
    diag = collector.get_diagnostics()
    merge_stats = deduplicator.last_merge_stats
    by_sector = Counter(e["sector_id"] for e in canonical)

    print("\n" + "=" * 78)
    print("   FUNNEL AUDIT & REJECTION / MERGE TELEMETRY")
    print("=" * 78)
    print(
        f"HTTP Stats       : requests={diag.get('http_requests', 0)}, "
        f"200={diag.get('http_200', 0)}, 4xx={diag.get('http_4xx', 0)}, "
        f"5xx={diag.get('http_5xx', 0)}, timeouts={diag.get('http_timeouts', 0)}, "
        f"errors={diag.get('http_errors', 0)}"
    )
    print(
        f"BS4 Extraction   : jsonld={diag.get('bs4_jsonld_hits', 0)}, "
        f"semantic={diag.get('bs4_article_hits', 0)}, p_fallback={diag.get('bs4_paragraph_hits', 0)}, "
        f"meta={diag.get('bs4_meta_fallbacks', 0)}, rss={diag.get('bs4_rss_fallbacks', 0)}, "
        f"failures={diag.get('bs4_parse_failures', 0)}"
    )
    print(
        f"Google Redirects : decoded={diag.get('google_url_decoded', 0)}, "
        f"failed={diag.get('google_url_decode_failed', 0)}"
    )
    print(
        f"Filter Drops     : min_len={diag.get('dropped_min_length', 0)}, "
        f"rss_only={diag.get('dropped_rss_summary_disallowed', 0)}, "
        f"non_noida={diag.get('dropped_non_noida', 0)}, "
        f"greater_noida={diag.get('dropped_greater_noida', 0)}, "
        f"other_city={diag.get('dropped_other_city', 0)}, "
        f"no_cue={diag.get('dropped_no_incident_cue', 0)}, "
        f"excluded={diag.get('dropped_excluded_term', 0)}, "
        f"non_incident={diag.get('dropped_classifier_non_incident', 0)}, "
        f"low_cls_conf={diag.get('dropped_low_classifier_conf', 0)}, "
        f"unresolved_loc={diag.get('dropped_unresolved_location', 0)}, "
        f"expired_window={dropped_old}"
    )
    print(
        f"Level 2 Merge    : in={merge_stats.get('input_reports', 0)} -> "
        f"out={merge_stats.get('output_events', 0)} "
        f"(merged={merge_stats.get('merged_reports', 0)}: "
        f"high_sim={merge_stats.get('merged_by_high_sim', 0)}, "
        f"sector_sim={merge_stats.get('merged_by_same_sector_sim', 0)}, "
        f"entity_sim={merge_stats.get('merged_by_entity_sim', 0)}, "
        f"close={merge_stats.get('merged_by_close_proximity', 0)})"
    )
    print(f"Coverage         : {len(by_sector)} / {len(all_gazetteer_sectors)} gazetteer sectors have direct events.")
    print("Top Sectors      :", ", ".join(f"{k} ({v})" for k, v in by_sector.most_common(12)))
    print(f"Level 3 Hotspots : {len(state['hotspots'])} active clusters")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
