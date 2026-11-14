import json
from datetime import datetime, timedelta
from pathlib import Path
from geosituation.perception.collector import NewsIncidentCollector
from geosituation.comprehension.coreference import CrossDocumentDeduplicator
from geosituation.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

SEED_ARTICLES = [
    # Event 1: Sector 18 Commercial Market Armed Robbery (Reported by 3 different sources)
    {
        "headline": "Armed robbery at jewelry store in Sector 18 Atta Market, four masked men escape with gold worth 40 lakh",
        "body": "Noida: In a daring daylight robbery, four masked assailants stormed into a jewelry showroom in the bustling Atta market of Sector 18. The robbers brandished country-made pistols, threatened security guards and staff, and looted gold ornaments before fleeing on two motorcycles towards Sector 27. Noida police have formed five special teams to trace the miscreants.",
        "source_url": "https://timesofindia.indiatimes.com/city/noida/robbery-sector-18",
        "days_ago": 1.2
    },
    {
        "headline": "Gunmen storm Atta Market jewelry shop in Sector 18, loot gold ornaments at gunpoint",
        "body": "Greater Noida: Tension gripped Sector 18 market on Tuesday afternoon when four armed miscreants raided an upscale jewelry outlet near the metro station. The assailants fired one round in the air to terrorize shopkeepers and escaped on bikes. Senior police officers rushed to the spot with forensic teams.",
        "source_url": "https://hindustantimes.com/ncr/noida-sec-18-robbery-gold",
        "days_ago": 1.1
    },
    {
        "headline": "Sec 18 Atta Market jewelry loot: Noida Police scan CCTV footage of bike-borne robbers",
        "body": "Noida: Investigators are scanning CCTV camera footage after four armed robbers barged into a showroom in Sector 18 market yesterday and decamped with jewelry. Two security guards sustained minor injuries while attempting to shut the glass doors.",
        "source_url": "https://indianexpress.com/cities/delhi/noida-sector-18-atta-loot",
        "days_ago": 0.8
    },

    # Event 2: Expressway Multi-Vehicle Collision (Reported by 2 sources)
    {
        "headline": "Dense fog triggers seven-vehicle pileup on Noida-Greater Noida Expressway near Sector 128, three injured",
        "body": "Noida: Poor morning visibility due to dense smog caused a multi-vehicle collision involving a dumper truck, two private cars, and an SUV on the Noida Expressway near Sector 128 flyover early yesterday morning. Traffic police used cranes to clear damaged vehicles. The injured were shifted to Jaypee Hospital.",
        "source_url": "https://timesofindia.indiatimes.com/city/noida/expressway-pileup-fog",
        "days_ago": 2.5
    },
    {
        "headline": "Massive pileup on Noida Expressway at Sector 128 disrupts morning office commute",
        "body": "Commuters heading towards Delhi faced two-hour delays after a speeding dumper truck rammed into a sedan near Sector 128, triggering a chain collision of five other vehicles. Traffic was diverted onto service lanes.",
        "source_url": "https://hindustantimes.com/ncr/expressway-crash-traffic-jam",
        "days_ago": 2.4
    },

    # Event 3: Sector 62 IT Hub Transformer Fire (Reported by 2 sources)
    {
        "headline": "Major fire in electrical transformer outside IT park in Sector 62, fire tenders rushed to spot",
        "body": "Noida: A high-voltage transformer caught fire with loud explosions near the Fortis Hospital crossing in Sector 62 late yesterday evening. Four fire tenders battled the blaze for over an hour. Power supply was disconnected to prevent damage to nearby commercial complexes.",
        "source_url": "https://timesofindia.indiatimes.com/city/noida/sector-62-fire-transformer",
        "days_ago": 3.0
    },
    {
        "headline": "Transformer blast sparks panic near Sector 62 metro station, thick smoke engulfs road",
        "body": "Noida: Commuters and office workers in Sector 62 witnessed thick plumes of black smoke as an electrical transformer exploded near the institutional area. Fire department personnel contained the fire with chemical foam.",
        "source_url": "https://amarujala.com/delhi-ncr/noida/sec-62-transformer-blaze",
        "days_ago": 2.9
    },

    # Event 4: Sector 137 High-Rise Burglary
    {
        "headline": "Burglaries in two locked high-rise flats in Sector 137 society, cash and laptops stolen",
        "body": "Noida: Burglars targeted two locked apartments in a premium residential society in Sector 137 while the residents were away for the weekend. The thieves broke the front door latches and fled with cash and gold jewelry. Felix Hospital police post is investigating.",
        "source_url": "https://hindustantimes.com/ncr/sec-137-flats-burglary",
        "days_ago": 4.0
    },

    # Event 5: Sector 142 Corporate Park Snatching Surge (Recent Cluster 1)
    {
        "headline": "Bike-borne snatchers target female software engineer outside Advant Navis tower in Sector 142",
        "body": "Noida: Two motorcycle-borne youths snatched a smartphone and handbag from an IT employee while she was walking towards the Sector 142 metro station near Advant Navis Business Park. The miscreants sped away towards the expressway.",
        "source_url": "https://timesofindia.indiatimes.com/city/noida/advant-snatching-sec-142",
        "days_ago": 1.5
    },
    {
        "headline": "Second chain snatching reported near Sector 142 metro station within 24 hours",
        "body": "Noida: In another incident near Advant tower in Sector 142, bike-borne miscreants snatched a gold chain from a commuter waiting for an auto-rickshaw. Police have intensified night patrolling along the service roads.",
        "source_url": "https://hindustantimes.com/ncr/sec-142-chain-snatching-surge",
        "days_ago": 0.5
    },

    # Event 6: Sector 63 Industrial Warehouse Fire
    {
        "headline": "Massive fire destroys garment manufacturing factory warehouse in Sector 63 industrial area",
        "body": "Noida: A devastating fire erupted on the top floor of a garment packaging unit in Sector 63. Ten fire tenders from Phase 3 fire station worked through the night to bring the flames under control. No casualties were reported.",
        "source_url": "https://indianexpress.com/cities/delhi/noida-sector-63-factory-fire",
        "days_ago": 5.2
    },

    # Event 7: Sector 76 Residential Society Car Theft
    {
        "headline": "Luxury SUV stolen from outside society gate in Sector 76, car-lifter gang caught on CCTV",
        "body": "Noida: An expensive SUV was stolen from the roadside parking outside a high-rise society in Sector 76. CCTV footage shows two men using an electronic key cloner to unlock the vehicle within ninety seconds.",
        "source_url": "https://amarujala.com/delhi-ncr/noida/sec-76-suv-stolen",
        "days_ago": 6.0
    },

    # Event 8: Farmers Protest / Road Blockade near DND Flyway
    {
        "headline": "Farmers stage tractor march, block traffic near DND Flyway toll plaza causing five-kilometer jam",
        "body": "Noida: Hundreds of protesting farmers on tractors blocked the entry corridor near the DND flyway and Chilla border demanding enhanced rehabilitation compensation. Heavy police deployment and barricading diverted commuter traffic via Kalindi Kunj.",
        "source_url": "https://timesofindia.indiatimes.com/city/noida/dnd-flyway-farmer-protest",
        "days_ago": 3.8
    },

    # Event 9: Sector 50 Commercial Center Hit and Run
    {
        "headline": "Delivery boy critically injured in hit-and-run crash near Sector 50 market",
        "body": "Noida: A speeding sedan rammed into a delivery executive's motorcycle at the Sector 50 intersection late last night. The car driver fled the scene without stopping. The victim was rushed to a nearby private hospital with multiple fractures.",
        "source_url": "https://hindustantimes.com/ncr/sec-50-delivery-boy-accident",
        "days_ago": 8.5
    },

    # Event 10: Sector 125 Amity University Area Altercation
    {
        "headline": "Clash between two student groups outside Sector 125 university campus, four detained",
        "body": "Noida: Tension flared outside the university campus in Sector 125 following a physical brawl between two student groups over a vehicle parking argument. Police arrived promptly, dispersed the crowd, and detained four youths.",
        "source_url": "https://timesofindia.indiatimes.com/city/noida/sector-125-student-clash",
        "days_ago": 11.0
    },

    # Event 11: Sector 15 Metro Station Phone Snatching
    {
        "headline": "Pedestrian's mobile snatched near Sector 15 metro station entry gate",
        "body": "Noida: An unidentified youth on a scooty snatched an iPhone from a pedestrian outside the Sector 15 metro station near Nayabans before racing towards the Gol Chakkar.",
        "source_url": "https://amarujala.com/delhi-ncr/noida/sec-15-metro-snatching",
        "days_ago": 14.0
    },

    # Control Sample: Non-incident (to prove classifier filters out general news)
    {
        "headline": "Noida Authority announces new sports complex and cricket academy in Sector 21 stadium",
        "body": "Noida: The development authority approved a budget of 45 crore rupees for upgrading sports facilities and constructing a modern Olympic-standard swimming pool at the Sector 21 stadium.",
        "source_url": "https://timesofindia.indiatimes.com/city/noida/sports-complex-sec-21",
        "days_ago": 4.5
    }
]

def seed_database():
    now = datetime.now()
    collector = NewsIncidentCollector()
    raw_reports = []

    print("--- INGESTING & PROCESSING SEED ARTICLES (LEVEL 1: PERCEPTION) ---")
    for item in SEED_ARTICLES:
        pub_date = now - timedelta(days=item["days_ago"])
        record = collector.process_raw_article(
            headline=item["headline"],
            body=item["body"],
            source_url=item["source_url"],
            pub_date=pub_date
        )
        if record:
            raw_reports.append(record)
            print(f"[PERCEIVED] {record['report_id']} | {record['category']} | {record['sector_id']} | {record['headline'][:65]}...")
        else:
            print(f"[FILTERED / NON-INCIDENT] {item['headline'][:65]}...")

    # Save Level 1 Reports
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_reports_path = PROCESSED_DATA_DIR / "level1_reports.json"
    with open(raw_reports_path, "w", encoding="utf-8") as f:
        json.dump(raw_reports, f, indent=2)

    print(f"\nSuccessfully stored {len(raw_reports)} Level 1 incident records at: {raw_reports_path}")

    # Run Level 2 Deduplication
    print("\n--- RUNNING CROSS-DOCUMENT COREFERENCE (LEVEL 2: COMPREHENSION) ---")
    dedup = CrossDocumentDeduplicator()
    canonical_events = dedup.deduplicate(raw_reports)

    print(f"Fused {len(raw_reports)} raw reports into {len(canonical_events)} deduplicated canonical events!")
    for evt in canonical_events:
        verified_flag = "[MULTI-SOURCE VERIFIED]" if evt["is_multi_source_verified"] else "[SINGLE SOURCE]"
        print(f" - {evt['event_id']} | {verified_flag} ({evt['report_count']} reports) | {evt['category']} in {evt['sector_id']} | Conf: {evt['verification_confidence']}")

    canonical_path = PROCESSED_DATA_DIR / "canonical_events.json"
    with open(canonical_path, "w", encoding="utf-8") as f:
        json.dump(canonical_events, f, indent=2)

    print(f"Stored canonical events at: {canonical_path}")

if __name__ == "__main__":
    seed_database()
