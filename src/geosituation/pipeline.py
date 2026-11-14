import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from geosituation.perception.collector import NewsIncidentCollector
from geosituation.perception.gazetteer import NoidaGazetteer
from geosituation.comprehension.coreference import CrossDocumentDeduplicator
from geosituation.comprehension.situational_index import SituationalIndexCalculator
from geosituation.comprehension.anomaly_detector import SectorAnomalyDetector
from geosituation.projection.spatial_kde import SpatialKDEProjector
from geosituation.projection.hotspot_dbscan import STHotspotDetector
from geosituation.projection.momentum import SituationalMomentumProjector
from geosituation.config import PROCESSED_DATA_DIR

class SituationalAwarenessPipeline:
    """
    End-to-End Orchestrator for Endsley's Three-Level Situational Awareness System.
    """

    def __init__(self):
        # Level 1 Modules
        self.collector = NewsIncidentCollector()
        self.gazetteer = NoidaGazetteer()
        
        # Level 2 Modules
        self.deduplicator = CrossDocumentDeduplicator()
        self.sai_calculator = SituationalIndexCalculator()
        self.anomaly_detector = SectorAnomalyDetector()
        
        # Level 3 Modules
        self.kde_projector = SpatialKDEProjector()
        self.hotspot_detector = STHotspotDetector()
        self.momentum_projector = SituationalMomentumProjector()

    def run_pipeline(self, current_time: Optional[datetime] = None) -> Dict:
        """
        Executes the end-to-end pipeline over stored canonical data.
        Returns the unified situational awareness state.
        """
        if current_time is None:
            current_time = datetime.now()

        # Load canonical events
        canonical_path = PROCESSED_DATA_DIR / "canonical_events.json"
        if not canonical_path.exists():
            raise FileNotFoundError(f"Canonical events not found at: {canonical_path}. Run seed script first.")

        with open(canonical_path, "r", encoding="utf-8") as f:
            canonical_events = json.load(f)

        sectors = self.gazetteer.get_all_sectors()

        # --- LEVEL 2: COMPREHENSION ---
        sector_map = self.sai_calculator.calculate_sector_indexes(canonical_events, sectors, current_time)
        anomalies = self.anomaly_detector.detect_anomalies(canonical_events, current_time)

        # --- LEVEL 3: PROJECTION ---
        kde_surface = self.kde_projector.compute_density_grid(canonical_events)
        hotspots = self.hotspot_detector.detect_hotspots(canonical_events)
        trajectories = self.momentum_projector.project_sector_trajectories(sector_map, anomalies)

        executive_summary = self.momentum_projector.generate_endsley_executive_summary(
            canonical_events=canonical_events,
            sector_map=sector_map,
            anomalies=anomalies,
            hotspots=hotspots,
            trajectories=trajectories
        )

        pipeline_state = {
            "last_updated": current_time.isoformat(),
            "executive_summary": executive_summary,
            "canonical_events": canonical_events,
            "sectors": list(sector_map.values()),
            "anomalies": anomalies,
            "hotspots": hotspots,
            "trajectories": trajectories,
            "kde_surface": kde_surface
        }

        # Cache unified state
        state_path = PROCESSED_DATA_DIR / "situational_state.json"
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(pipeline_state, f, indent=2)

        return pipeline_state

    def ingest_new_article(self, headline: str, body: str, source_url: str = "") -> Optional[Dict]:
        """
        Ingests a new raw article, processes through Level 1, updates canonical events,
        and re-triggers comprehension and projection.
        """
        record = self.collector.process_raw_article(
            headline=headline,
            body=body,
            source_url=source_url,
            pub_date=datetime.now()
        )
        if not record:
            return None

        # Append to raw Level 1 records
        raw_reports_path = PROCESSED_DATA_DIR / "level1_reports.json"
        raw_reports = []
        if raw_reports_path.exists():
            with open(raw_reports_path, "r", encoding="utf-8") as f:
                raw_reports = json.load(f)
        raw_reports.insert(0, record)
        with open(raw_reports_path, "w", encoding="utf-8") as f:
            json.dump(raw_reports, f, indent=2)

        # Re-run deduplication
        canonical_events = self.deduplicator.deduplicate(raw_reports)
        canonical_path = PROCESSED_DATA_DIR / "canonical_events.json"
        with open(canonical_path, "w", encoding="utf-8") as f:
            json.dump(canonical_events, f, indent=2)

        # Refresh pipeline state
        return self.run_pipeline()
